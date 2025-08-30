import os
import random
import sys
import argparse
import gc
import pickle
import json
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras import Model
from keras.src.layers import Lambda, Dense, Activation
from keras.src.losses import BinaryFocalCrossentropy, MeanSquaredError
from keras.src.optimizers import AdamW
from keras.src.saving import custom_object_scope
from tensorflow.keras.callbacks import LearningRateScheduler
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dropout, BatchNormalization
from tensorflow.python.client.session import InteractiveSession
from tensorflow.python.keras.backend import set_session
from tensorflow.python.keras.losses import Huber, binary_crossentropy
from tensorflow.python.proto_exports import ConfigProto

from CloudScripts.functions import gaussian_nll, hybrid_loss, calibration_loss, masked_huber_loss, custom_mse, \
    GradientLogger, conf_head_loss, directional_accuracy, RelativeEarlyStopping, mse_directional_loss, \
    directional_mse_loss
from functions import lr_schedule
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from tensorflow.keras import callbacks, backend as K
from tensorflow.keras.utils import plot_model
from modelFunction import create_market_momentum_model
from functions import percent_error_loss, explained_variance

os.environ['PYTHONHASHSEED'] = '0'
random.seed(0)
np.random.seed(0)
tf.random.set_seed(0)

config = ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.9

session = InteractiveSession(config=config)

set_session(session)
print("Using Keras backend:", K.backend())
gc.collect()
K.clear_session()
tf.keras.mixed_precision.set_global_policy('mixed_float16')

def load_prepared_data(filename):
    #Load preprocessed data from an HDF5 file
    with h5py.File(filename, "r") as f:
        X = f["X"][:]
        y = f["y"][:]
        stock_ids = f["stock_ids"][:]
        sector_ids = f["sector_ids"][:]
    return X, y, stock_ids, sector_ids

def _build_datasets(X, y, batch=4096):
    check_data_stats(X, y)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.2, random_state=31)
    train_inputs = {"data_input": X_tr}
    val_inputs   = {"data_input": X_val}

    train_ds = tf.data.Dataset.from_tensor_slices((train_inputs, y_tr)) \
                              .batch(batch).prefetch(tf.data.AUTOTUNE)
    val_ds   = tf.data.Dataset.from_tensor_slices((val_inputs,   y_val)) \
                              .batch(batch).prefetch(tf.data.AUTOTUNE)
    return train_ds, val_ds, X_tr, X_val, y_tr, y_val


def _save_history_plot(hist, fname):
    plt.plot(hist.history["loss"],     label="train")
    plt.plot(hist.history["val_loss"], label="val")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title(fname.split("/")[-1])
    plt.legend()
    plt.savefig(fname)
    plt.close()


def check_data_stats(X, y):
    print("\nInput data (X) statistics:")
    print(f"Shape: {X.shape}")
    print(f"Min: {X.min()}, Max: {X.max()}")
    print(f"Mean: {X.mean()}, Std: {X.std()}")

    for i in range(min(5, X.shape[2])):
        feature = X[:, :, i].flatten()
        print(f"Feature {i}: Min={feature.min():.4f}, Max={feature.max():.4f}, "
              f"Mean={feature.mean():.4f}, Std={feature.std():.4f}")

    print("\nOutput data (y) statistics:")
    print(f"Shape: {y.shape}")
    print(f"Min: {y.min()}, Max: {y.max()}")
    print(f"Mean: {y.mean()}, Std: {y.std()}")

    # Check for NaNs or Infs
    print(f"X contains NaN: {np.isnan(X).any()}")
    print(f"X contains Inf: {np.isinf(X).any()}")
    print(f"y contains NaN: {np.isnan(y).any()}")
    print(f"y contains Inf: {np.isinf(y).any()}")

    return np.isnan(X).any() or np.isinf(X).any() or np.isnan(y).any() or np.isinf(y).any()


#  stage-1  (price model)

def train_price_stage(cfg, sector_to_id, train_ds, val_ds):

    model_price = create_market_momentum_model(
        seq_length=cfg["SEQ_LENGTH"],
        n_features=cfg["N_FEATURES"],
        n_sectors=len(sector_to_id),
        embedding_dim_sector=cfg["EMBEDDING_DIM_SECTOR"],
        future_days=cfg["FUTURE_LENGTH"],
        include_confidence=False
    )

    # compile
    optimiser = Adam(
        learning_rate=1e-3,
        global_clipnorm=1.0,
        amsgrad=True
    )
    #scaled_mse = MeanSquaredError(name="mse_scaled")
    model_price.compile(
        optimizer=optimiser,
        loss= directional_mse_loss,
        metrics=[explained_variance, directional_accuracy]
    )

    # callbacks
    lr_sched = callbacks.LearningRateScheduler(lr_schedule, verbose=1)
    es       = RelativeEarlyStopping(
        monitor="val_loss",
        patience=8,
        min_delta_pct=0.05
    )
    rlr = callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1
    )
    #plot_model(model_price, to_file="../model/Global/price_model_architecture.png", show_shapes=True, show_layer_names=True, expand_nested=True, dpi=200)

    # train
    history = model_price.fit(
        train_ds,
        validation_data=val_ds,
        epochs=100,
        callbacks=[lr_sched, es, rlr],
        shuffle=True,
        verbose=1
    )

    # save
    os.makedirs("../model/Global", exist_ok=True)
    model_price.save("../model/Global/backbone_price.keras")
    _save_history_plot(history, "../model/Global/price_loss.png")

    return model_price



#  stage-2  (confidence head)

def train_conf_stage(cfg, backbone, X_tr, X_val, y_tr, y_val,
                     delta_pct=0.15,                      # hit radius in %
                     fine_tune_backbone=True, ft_epochs=4):


    backbone.trainable = False
    bottleneck = backbone.get_layer("bottleneck").output

    x = Dense(128, activation="relu",
              kernel_regularizer=tf.keras.regularizers.l2(1e-4),
              name="conf_dense1")(bottleneck)
    x = BatchNormalization()(x)
    x = Dropout(0.2, name="conf_dropout1")(x)

    x = Dense(32, activation="relu",
              kernel_regularizer=tf.keras.regularizers.l2(1e-4),
              name="conf_dense2")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.1, name="conf_dropout2")(x)

    logits = Dense(cfg["FUTURE_LENGTH"], activation="linear",
                   name="conf_logit")(x)
    conf_out = Activation("sigmoid", name="confidence")(logits)
    mu_tr  = backbone.predict({"data_input": X_tr},  batch_size=4096)
    mu_val = backbone.predict({"data_input": X_val}, batch_size=4096)

    δ_tr  = (delta_pct / 100.) * np.abs(mu_tr)
    δ_val = (delta_pct / 100.) * np.abs(mu_val)

    hit_tr  = (np.abs(y_tr  - mu_tr) <= δ_tr ).astype("float32")
    hit_val = (np.abs(y_val - mu_val) <= δ_val).astype("float32")

    # focal-loss hyper-params from class balance
    pos_rate = float(hit_tr.mean())
    alpha = min(0.5 + 0.5 * (1 - pos_rate), 0.9)
    gamma = 2.0 if pos_rate > 0.15 else 2.5
    pos_frac = hit_tr.mean()
    class_weight = {0: 1.0, 1: (1.0 - pos_frac) / max(pos_frac, 1e-6)}
    focal_loss = BinaryFocalCrossentropy(alpha=alpha, gamma=gamma)
    bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    optimiser = AdamW(learning_rate=1e-3, weight_decay=1e-4, clipnorm=1.0)

    model_head = Model(backbone.input, conf_out, name="conf_classifier")
    model_head.compile(
        optimizer=optimiser,
        loss=bce,
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="acc", threshold=0.4),
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Precision(name="prec", thresholds=0.4),
            tf.keras.metrics.Recall(name="rec", thresholds=0.4)
        ]
    )
    lr_sched = LearningRateScheduler(lr_schedule, verbose=0)
    es = callbacks.EarlyStopping(monitor="val_loss", patience=5,
                                 min_delta=0.002, restore_best_weights=True)

    hist = model_head.fit({"data_input": X_tr}, hit_tr,
                          validation_data=({"data_input": X_val}, hit_val),
                          epochs=25, batch_size=512,
                          callbacks=[lr_sched, es],
                          class_weight=class_weight)

    if fine_tune_backbone:
        for layer in backbone.layers[-4:]:
            layer.trainable = True

        model_head.compile(optimizer=AdamW(5e-4, weight_decay=1e-4,
                                           clipnorm=1.0),
                           loss=focal_loss, metrics=["auc"])

        model_head.fit({"data_input": X_tr}, hit_tr,
                       validation_data=({"data_input": X_val}, hit_val),
                       epochs=ft_epochs, batch_size=512,
                       callbacks=[lr_sched],
                       class_weight=class_weight)

    inference_model = Model(backbone.input,
                            [backbone.output, conf_out],
                            name="price_and_confidence")
    inference_model.save("../model/Global/model_price_conf.keras")
    _save_history_plot(hist, "../model/Global/confidence_focal.png")
    return inference_model



def main_train(stage_choice="both"):
    gpus = tf.config.list_physical_devices("GPU")
    print("Available GPUs:", gpus)

    # ---- data --------------------------------------------------
    X, y, _, _ = load_prepared_data("../model/Data/processedData.h5")
    print("Loaded prepared data")

    with open("../model/Data/sector_to_id.pkl", "rb") as f:
        sector_to_id = pickle.load(f)
    with open("../model/Data/Globalconfig.json", "r") as f:
        cfg_raw = json.load(f)

    cfg = dict(
        SEQ_LENGTH         = cfg_raw["SEQ_LENGTH"],
        FUTURE_LENGTH      = cfg_raw["FUTURE_LENGTH"],
        EMBEDDING_DIM_SECTOR = cfg_raw["EMBEDDING_DIM_SECTOR"],
        N_FEATURES         = X.shape[2]
    )

    train_ds, val_ds, X_tr, X_val, y_tr, y_val = _build_datasets(X, y)

    if stage_choice in ("price", "both"):
        backbone = train_price_stage(cfg, sector_to_id, train_ds, val_ds)
    else:
        backbone = tf.keras.models.load_model(
            "../model/Global/backbone_price.keras", compile=False, safe_mode=False)

    if stage_choice in ("confidence", "both"):
        _ = train_conf_stage(cfg, backbone, X_tr, X_val, y_tr, y_val)


#CLI
if __name__ == "__main__":
    argp = argparse.ArgumentParser()
    argp.add_argument("--stage",
                      choices=["price", "confidence", "both"],
                      default="both")
    args = argp.parse_args()
    main_train(stage_choice=args.stage)

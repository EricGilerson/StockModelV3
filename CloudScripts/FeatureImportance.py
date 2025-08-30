import os

import numpy as np
import h5py
import pandas as pd
import tensorflow
import tensorflow as tf
from keras.src.saving import load_model
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
from copy import deepcopy
from functions import custom_mse, PositionalEncoding, RecentTrendExtractor, CLSToken
from combinedDataSave import GroupScaler


MODEL_PATH = "../model/Global/model_price_conf.keras"
DATA_PATH = "../model/Data/processedData.h5"
DATA_KEY = "X"
SECTOR_KEY = "sector_ids"
TARGET_KEY = "y"
NUM_REPEATS = 3  # Number of permutations per feature


def loadModel(filename):
    custom_objects = {
        'PositionalEncoding': PositionalEncoding,
        'RecentTrendExtractor': RecentTrendExtractor,
        'CLSToken': CLSToken,
        'tf': tensorflow,
    }
    model = load_model(filename, custom_objects=custom_objects, compile=False, safe_mode=False)
    return model

# LOAD MODEL AND DATA
print("Loading model...")
model = loadModel(MODEL_PATH)

print("Loading data...")
with h5py.File(DATA_PATH, "r") as f:
    X_data = f[DATA_KEY][:]
    sector_data = f[SECTOR_KEY][:]
    y_true = f[TARGET_KEY][:]

print("Data shapes:", X_data.shape, sector_data.shape, y_true.shape)


def evaluate(model, X, sector, y):
    #Evaluate model MSE using only the price output (first output),
    try:
        y_pred_raw = model.predict([X, sector], batch_size=8192, verbose=1)
    except Exception:
        y_pred_raw = model.predict(X, batch_size=8192, verbose=1)

    if isinstance(y_pred_raw, (list, tuple)):
        y_pred = y_pred_raw[0]
    else:
        y_pred = y_pred_raw

    # Compute custom MSE between true y and predicted price
    return custom_mse(y, y_pred)

# PERMUTATION IMPORTANCE (features + sector)

def permutation_importance(model, X, sector, y, num_repeats=3):
    baseline_score = evaluate(model, X, sector, y)
    print(f"Baseline MSE: {baseline_score:.6f}")

    n_features = X.shape[2]
    importances = np.zeros(n_features)

    # Feature importance (per feature in data_input)
    for i in range(n_features):
        print(f"Permuting feature {i + 1}/{n_features}...")
        scores = []

        for _ in range(num_repeats):
            X_permuted = deepcopy(X)
            # Permute values of feature i across samples for each timestep
            for t in range(X.shape[1]):
                perm = np.random.permutation(X.shape[0])
                X_permuted[:, t, i] = X[perm, t, i]

            mse = evaluate(model, X_permuted, sector, y)
            scores.append(mse)

        importances[i] = np.mean(scores) - baseline_score
        print(f"Feature {i}: Importance = {importances[i]:.6f}")

    return importances


importances = permutation_importance(model, X_data, sector_data, y_true, NUM_REPEATS)

# Read feature labels
df = pd.read_csv("../Data/A.csv")
all_columns = df.columns.tolist()
feature_labels = [col for col in all_columns if col not in ["Date", "Close"]]

# PLOT RESULTS
plt.figure(figsize=(14, 6))
bars = plt.bar(feature_labels, importances, color='skyblue')

# Save and show
plt.xticks(rotation=45, ha="right")
plt.xlabel("Feature")
plt.ylabel("Importance (MSE Increase)")
plt.title("Permutation Feature Importance")
plt.grid(True, axis='y')
plt.tight_layout()
os.makedirs("Outputs", exist_ok=True)
plt.savefig("Outputs/permutation_importance.png")
plt.show()

# Save CSV
importances_df = pd.DataFrame({
    "Feature": feature_labels,
    "Importance": importances
})
importances_df.to_csv("Outputs/feature_importance.csv", index=False)

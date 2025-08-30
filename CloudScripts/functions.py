import tensorflow as tf
from keras.src.losses import Huber
from tensorflow.keras.losses import Reduction
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Flatten, Conv1D, Bidirectional, GlobalAveragePooling1D, \
    RepeatVector, Dense, GaussianNoise, LayerNormalization, Add, Concatenate, Lambda, Multiply, Reshape, Dropout, Layer
from tensorflow.keras.regularizers import l2
import numpy as np
from tensorflow.python.keras import callbacks


# Custom Layers & Functions

class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, sequence_length, d_model, **kwargs):
        super(PositionalEncoding, self).__init__(**kwargs)
        self.sequence_length = sequence_length
        self.d_model = d_model

        # Precompute the positional encodings once in float32
        pos_encoding = self.positional_encoding(sequence_length, d_model)
        self.pos_encoding = tf.Variable(
            initial_value=pos_encoding,
            trainable=False,
            dtype=tf.float32,
            name="pos_encoding"
        )

    def get_angles(self, pos, i, d_model):
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        return pos * angle_rates

    def positional_encoding(self, sequence_length, d_model):
        angle_rads = self.get_angles(
            np.arange(sequence_length)[:, np.newaxis],
            np.arange(d_model)[np.newaxis, :],
            d_model
        )
        # Apply sin to even indices, cos to odd indices
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
        pos_encoding = angle_rads[np.newaxis, ...]  # shape: (1, sequence_length, d_model)
        return tf.cast(pos_encoding, dtype=tf.float32)

    def call(self, inputs):
        # Convert pos_encoding to match the inputs' dtype (which may be float16 in mixed precision)
        pos_encoding_cast = tf.cast(self.pos_encoding, inputs.dtype)
        seq_len = tf.shape(inputs)[1]
        return inputs + pos_encoding_cast[:, :seq_len, :]

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config()
        config.update({
            'sequence_length': self.sequence_length,
            'd_model': self.d_model
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

class CLSToken(tf.keras.layers.Layer):
    def __init__(self, dim, **kwargs):
        super(CLSToken, self).__init__(**kwargs)
        self.dim = dim

    def build(self, input_shape):
        self.token = self.add_weight(
            name="cls_token_weight",
            shape=(1, 1, self.dim),
            initializer="random_normal",
            trainable=True
        )
        super(CLSToken, self).build(input_shape)

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        return tf.repeat(self.token, repeats=batch_size, axis=0)

    def get_config(self):
        config = super(CLSToken, self).get_config()
        config.update({"dim": self.dim})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

class RelativeEarlyStopping(callbacks.Callback):
    #Stops when `monitor` has not improved by more than `min_delta_pct`% of its best value for `patience` epochs.
    def __init__(self, monitor="val_loss", patience=8, min_delta_pct=0.2, mode="min"):
        super().__init__()
        self.monitor        = monitor
        self.patience       = patience
        self.min_delta_pct  = min_delta_pct / 100.0
        self.mode           = mode
        self.best           = None
        self.wait           = 0

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get(self.monitor)
        if current is None:
            return

        # initialise best on first call
        if self.best is None:
            self.best = current

        improvement = (self.best - current) if self.mode == "min" else (current - self.best)
        threshold   = abs(self.best) * self.min_delta_pct

        if improvement > threshold:
            self.best = current
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.model.stop_training = True
                if self.params.get("verbose", 1):
                    print(f"\nEpoch {epoch:02d}: early stop (no {self.monitor} improvement > "
                          f"{self.min_delta_pct*100:.2f}% for {self.patience} epochs)")
                # restore best weights
                self.model.set_weights(self.best_weights)

    def on_train_begin(self, logs=None):
        self.wait = 0
        self.best_weights = self.model.get_weights()
def transformer_block(x, num_heads, ff_dim, dropout_rate):
    # Multi-head attention layer
    attn_output = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=ff_dim)(x, x)
    attn_output = Dropout(dropout_rate)(attn_output)
    out1 = LayerNormalization(epsilon=1e-6)(x + attn_output)

    # Feed-forward network
    ffn_output = Dense(ff_dim, activation='relu')(out1)
    ffn_output = Dense(x.shape[-1])(ffn_output)
    ffn_output = Dropout(dropout_rate)(ffn_output)
    out2 = LayerNormalization(epsilon=1e-6)(out1 + ffn_output)
    return out2

class Sign(Layer):
    #Element-wise sign: -1 for x<0, 0 for x==0, 1 for x>0.
    def call(self, inputs):
        return tf.math.sign(inputs)

    def get_config(self):
        base = super().get_config()
        return {**base}
class RecentTrendExtractor(tf.keras.layers.Layer):
    #Extracts the most recent 'steps' time-steps from a sequence.

    def __init__(self, steps, **kwargs):
        super(RecentTrendExtractor, self).__init__(**kwargs)
        self.steps = steps

    def call(self, inputs):
        return inputs[:, -self.steps:, :]

    def get_config(self):
        config = super(RecentTrendExtractor, self).get_config()
        config.update({"steps": self.steps})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

class GradientLogger(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        for i, layer in enumerate(self.model.layers):
            if hasattr(layer, 'kernel'): # Check if the layer has weights
                gradients = tf.GradientTape(self.model.output, layer.kernel)[0]
                if gradients is not None:
                    tf.print(f"Epoch {epoch} Layer {i} Gradient Mean: {tf.reduce_mean(gradients)}")
                    tf.summary.histogram(f"epoch_{epoch}_layer_{i}_gradients", gradients, step=epoch) # For TensorBoard


# Custom Callbacks & Metrics
def lr_schedule(epoch):
    warmup_epochs = 5
    initial_lr = 1e-4
    max_lr = 1e-3
    decay_rate = 0.975
    decay_start = warmup_epochs

    if epoch < warmup_epochs:
        # Linear warm-up
        return initial_lr + (max_lr - initial_lr) * (epoch / warmup_epochs)
    else:
        # Exponential decay after warm-up
        return max_lr * (decay_rate ** (epoch - decay_start))

class LearningRateScheduler(tf.keras.callbacks.Callback):
    def __init__(self, initial_lr, decay_steps, decay_rate):
        super(LearningRateScheduler, self).__init__()
        self.initial_lr = initial_lr
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate

    def on_epoch_begin(self, epoch, logs=None):
        new_lr = self.initial_lr * (self.decay_rate ** (epoch / self.decay_steps))
        tf.keras.backend.set_value(self.model.optimizer.lr, new_lr)

def gaussian_nll(y_true, y_pred):
    future_days = tf.shape(y_true)[-1]
    mu = y_pred[:, :future_days]
    sigma = y_pred[:, future_days:]
    sigma = tf.clip_by_value(sigma, 1e-5, 1e5)  # numerical stability

    # Compute log likelihood of Gaussian
    log_likelihood = 0.5 * tf.math.log(2. * np.pi) + tf.math.log(sigma) + 0.5 * tf.square((y_true - mu) / sigma)
    return tf.reduce_mean(log_likelihood)

def percent_error_loss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = y_pred[:, :1]
    epsilon = 1e-7
    percentage_error = tf.abs((y_true - y_pred) / (tf.maximum(tf.abs(y_true), epsilon)))
    return tf.reduce_mean(percentage_error)

def hybrid_loss(y_true, y_pred):
    future_days = tf.shape(y_true)[-1]
    mu = y_pred[:, :future_days]
    sigma = y_pred[:, future_days:]
    sigma = tf.clip_by_value(sigma, 1e-5, 1e5)

    # NLL component
    log_likelihood = 0.5 * tf.math.log(2. * np.pi) + tf.math.log(sigma) + 0.5 * tf.square((y_true - mu) / sigma)
    nll = tf.reduce_mean(log_likelihood)

    # MSE component
    mse = tf.reduce_mean(tf.square(y_true - mu))

    # Hybrid: favor uncertainty but still reward accuracy
    return nll + 0.1 * mse  # α = 0.1 is a good starting point

@tf.keras.utils.register_keras_serializable()
def directional_mse_loss(y_true, y_pred, delta=1.5):
    y_true_f = _flatten_batch_and_ticker(y_true)[:, 0]     # (N,)
    y_pred_f = _flatten_batch_and_ticker(y_pred)[:, 0]

    #Huber part
    err  = tf.cast(y_true_f - y_pred_f, tf.float32)
    abs_ = tf.abs(err)
    huber = tf.where(
        abs_ <= delta,
        0.5 * tf.square(err),
        delta * abs_ - 0.5 * delta**2
    )
    huber = tf.reduce_mean(huber)

    #directional part (smoothed)
    true_dir = tf.tanh(10.0 * y_true_f)
    pred_dir = tf.tanh(10.0 * y_pred_f)
    dir_loss = tf.reduce_mean(1.0 - true_dir * pred_dir)

    return 0.7 * huber + 0.4 * dir_loss

def explained_variance(y_true, y_pred):
    # Extract predicted mean component
    y_true = _flatten_batch_and_ticker(y_true)[:, 0]    # (B*T,)
    y_pred_mu = _flatten_batch_and_ticker(y_pred)[:, 0]

    y_true = tf.cast(y_true, tf.float32)
    y_pred_mu = tf.cast(y_pred_mu, tf.float32)

    # Replace NaN and Inf with zeros
    y_true = tf.where(tf.math.is_finite(y_true), y_true, 0.0)
    y_pred_mu = tf.where(tf.math.is_finite(y_pred_mu), y_pred_mu, 0.0)

    # Compute residuals and variances
    residual = y_true - y_pred_mu
    error_variance = tf.math.reduce_variance(residual, axis=0)

    # Variance of true values
    true_variance = tf.math.reduce_variance(y_true, axis=0)
    true_variance = tf.where(true_variance < tf.keras.backend.epsilon(),
                             tf.keras.backend.epsilon(),
                             true_variance)

    ev = 1.0 - (error_variance / true_variance)

    return tf.reduce_mean(ev)

def custom_mse(y_true, y_pred):
    y_pred_mu = y_pred[:, :1]

    y_true = tf.cast(y_true, tf.float32)
    y_pred_mu = tf.cast(y_pred_mu, tf.float32)

    # Replace non-finite values (NaN/Inf) with zero
    y_true = tf.where(tf.math.is_finite(y_true), y_true, 0.0)
    y_pred_mu = tf.where(tf.math.is_finite(y_pred_mu), y_pred_mu, 0.0)

    # Compute MSE safely
    diff = y_true - y_pred_mu
    mse = tf.reduce_mean(tf.square(diff))
    return mse
def directional_accuracy(y_true, y_pred, *, ignore_zero=True):

    y_true_f = _flatten_batch_and_ticker(y_true)[:, 0]
    y_pred_f = _flatten_batch_and_ticker(y_pred)[:, 0]

    y_true_f = tf.cast(tf.reshape(y_true_f, [-1]), tf.float32)
    y_pred_f = tf.cast(tf.reshape(y_pred_f, [-1]), tf.float32)

    y_true_f = tf.where(tf.math.is_finite(y_true_f),  y_true_f,  0.0)
    y_pred_f = tf.where(tf.math.is_finite(y_pred_f), y_pred_f, 0.0)

    # compare signs
    same_dir = tf.equal(tf.sign(y_true_f), tf.sign(y_pred_f))

    if ignore_zero:
        mask     = tf.not_equal(y_true_f, 0.0)          # skip zero-moves
        same_dir = tf.logical_and(same_dir, mask)
        denom    = tf.maximum(tf.reduce_sum(tf.cast(mask, tf.float32)), 1.0)
        accuracy = tf.reduce_sum(tf.cast(same_dir, tf.float32)) / denom
    else:
        accuracy = tf.reduce_mean(tf.cast(same_dir, tf.float32))

    return accuracy

@tf.keras.utils.register_keras_serializable()
def mse_directional_loss(alpha=1.0,          # weight on the MSE
                         beta=0.5,           # weight on the direction error
                         ignore_zero=True):
    def loss(y_true, y_pred):
        # split the prediction
        price_hat  = tf.cast(y_pred[:, :1], tf.float32)
        conf_raw   = tf.cast(y_pred[:, 1:], tf.float32)

        conf       = tf.sigmoid(conf_raw)

        # MSE term (heavily penalize high-confidence errors)
        mse_term   = tf.reduce_mean(conf * tf.square(y_true - price_hat))

        # directional-error term
        same_dir   = tf.equal(tf.sign(y_true), tf.sign(price_hat))

        if ignore_zero:
            mask        = tf.not_equal(y_true, 0.0)
            same_dir    = tf.logical_and(same_dir, mask)
            denom       = tf.maximum(tf.reduce_sum(tf.cast(mask, tf.float32)), 1.0)
            dir_acc     = tf.reduce_sum(conf * tf.cast(same_dir, tf.float32)) / denom
        else:
            dir_acc     = tf.reduce_mean(conf * tf.cast(same_dir, tf.float32))

        dir_error  = 1.0 - dir_acc                     # turn accuracy into a loss

        # combine
        return alpha * mse_term + beta * dir_error

    return loss
def calibration_loss(y_true, y_pred):
    # Separate predicted mean and confidence
    y_pred_mu = y_pred[:, :1]
    confidence = y_pred[:, 1:]

    y_true = tf.cast(y_true, tf.float32)
    y_pred_mu = tf.cast(y_pred_mu, tf.float32)
    confidence = tf.cast(confidence, tf.float32)

    y_true = tf.where(tf.math.is_finite(y_true), y_true, 0.0)
    y_pred_mu = tf.where(tf.math.is_finite(y_pred_mu), y_pred_mu, 0.0)
    confidence = tf.where(tf.math.is_finite(confidence), confidence, 0.0)

    confidence = tf.clip_by_value(confidence, 0.0, 1.0)

    error = tf.square(y_true - y_pred_mu)
    return tf.reduce_mean(confidence * error)


def masked_huber_loss(delta=1.0):
    def loss(y_true, y_pred):
        # Extract the mean prediction from the output
        y_pred_mu = y_pred[:, :1]

        # Calculate absolute error
        abs_error = tf.abs(y_true - y_pred_mu)

        # Huber loss calculation
        quadratic = tf.minimum(abs_error, delta)
        linear = abs_error - quadratic

        # Combine quadratic and linear parts
        huber = 0.5 * quadratic ** 2 + delta * linear

        return tf.reduce_mean(huber)

    return loss

def residual_huber(res_true, res_pred, delta=0.002):

    err = tf.abs(res_true - res_pred)
    quad = tf.minimum(err, delta)
    lin  = err - quad
    return tf.reduce_mean(0.5 * tf.square(quad) + delta * lin)

def conf_head_loss(res_true, y_pred, α=0.7, β=0.3):

    return α * residual_huber(res_true, y_pred) + β * calibration_loss(res_true,
                      tf.concat([tf.zeros_like(y_pred), 1. / (1.+y_pred)], axis=-1))

@tf.keras.utils.register_keras_serializable()
def fusion_softmax_fn(x):
    return tf.nn.softmax(x, axis=1)
@tf.keras.utils.register_keras_serializable()
def fusion_token_fn(x):
    return tf.reduce_sum(x, axis=1)
@tf.keras.utils.register_keras_serializable()
def stack_multi_context_fn(x):
    return tf.stack(x, axis=1)
@tf.keras.utils.register_keras_serializable()
def attention_softmax_fn(x):
    return tf.nn.softmax(x, axis=1)
@tf.keras.utils.register_keras_serializable()
def attention_pool_fn(x):
    return tf.reduce_sum(x, axis=1)

def _flatten_batch_and_ticker(t):
    t = tf.convert_to_tensor(t)
    last_dim = tf.shape(t)[-1]            # dynamic D
    new_shape = tf.concat([[-1], [last_dim]], axis=0)
    return tf.reshape(t, new_shape)       # (N, D)
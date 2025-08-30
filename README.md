# StockModel — CloudScripts Pipeline

> End‑to‑end pipeline for building a daily stock movement model with intraday micro‑features, grouped feature scaling, a hybrid Conv/Transformer/LSTM backbone, and evaluation utilities.

![Model Architecture](../model/Global/price_model_architecture.png)

> If the image doesn’t exist yet, generate it by enabling the `plot_model(...)` line in `CloudScripts/runModel.py` (see **Generate the model diagram** below).

---

## Quick Start

1) **Environment**
- Python 3.10+ recommended
- Install dependencies:
  ```bash
  pip install tensorflow keras numpy pandas h5py scikit-learn yfinance alpaca-py python-dotenv matplotlib tqdm joblib
  ```
- Create a `.env` file with your Alpaca keys for SIP minute bars:
  ```env
  APCA_API_KEY_ID=YOUR_KEY
  APCA_API_SECRET_KEY=YOUR_SECRET
  ```

2**Run the pipeline (from `CloudScripts/`):**
```bash
# 1) Build daily + 09:30–09:39 micro-features per ticker
python downloadData.py

# 2) Combine, scale by feature-group, window into sequences, and save H5 datasets
python combinedDataSave.py

# 3) Train stage‑1 (price) and stage‑2 (confidence) models; artifacts land in ../model/Global
python runModel.py

# 4) Evaluate over one or many days; results & plots saved to ./Outputs
python evalOpti.py
python multiEval.py
```

---

## Repository Layout (relative to `CloudScripts/`)

```
CloudScripts/
  downloadData.py
  combinedDataSave.py
  group_scaler.py
  functions.py
  modelFunction.py
  runModel.py
  evalFunctions.py
  evalOpti.py
  multiEval.py
  FeatureImportance.py
  intraday/                  # Cached SIP parquet files (auto‑created)
../Data/
  sectors.csv
  splits.pkl
../model/
  Data/                      # Saved scalers & ID maps; per‑ticker H5 files
  Global/                    # Trained models, loss curves, and architecture PNG
```

> Many scripts write to parent folders (`../Data`, `../model`) to keep training artifacts outside `CloudScripts/`.

---

## What each file does

### `downloadData.py`
Builds the raw dataset per ticker:
- **Daily OHLCV + technicals** via Yahoo Finance
- **Intraday “first 10 minutes” features** from Alpaca SIP (09:30–09:39 ET):
  - `Price_0939`, `atr10m_pct`, `vwap_gap`, `ret939_pct`
- Caches SIP minute bars as monthly parquet files under `CloudScripts/intraday/`.
- Writes a **CSV per ticker** with the final column order ending in `Delta_Close` (your modeling target).

**Inputs**
- `../Data/sectors.csv` (maps tickers → sector)
- `../Data/splits.pkl` (for split handling)

**Outputs**
- `CloudScripts/intraday/*.parquet` (SIP cache)
- `Data/<ticker>.csv` (one file per symbol, later read by the combiner)

---

### `combinedDataSave.py`
Combines all ticker CSVs, applies **grouped scaling**, windows sequences, and writes model‑ready H5 datasets.
- Defines **feature groups** (price, delta, atr, macd, volume, momentum, ratios, micro_pct).
- Applies **per‑group scaling strategies** (e.g., `standard`, `zero_standard`, `log_standard`, `robust`, `minmax`, or `divide_only` for MACD_Hist).
- Cleans NaNs/Inf and **clips outliers**.
- Constructs rolling **sequence windows** (length = `SEQ_LENGTH`) and **future targets** (length = `FUTURE_LENGTH`).
- Saves:
  - A global `processedData.h5` for training.
  - Per‑ticker `../model/Data/Ticker_Data/<TICKER>.h5` for evaluation.
  - Group scalers and ID maps for later **inverse‑scaling**.

**Key class:** `GroupScaler` (shared center/scale for a feature group).

---

### `group_scaler.py`
Lightweight scaler with `transform` and `inverse_transform` matching the per‑group stats saved during preprocessing.

---

### `functions.py`
Custom layers, losses, schedulers, and callbacks used across training and eval:
- **Layers:** `PositionalEncoding`, `CLSToken`, `RecentTrendExtractor`, and `transformer_block`.
- **Losses/metrics:** `directional_mse_loss`, `gaussian_nll`, `hybrid_loss`, `percent_error_loss`, `explained_variance`, `calibration_loss`, `mse_directional_loss`.
- **Training:** `RelativeEarlyStopping`, `lr_schedule`.

---

### `modelFunction.py`
Defines the backbone architecture `create_market_momentum_model(...)`:
- **Projection + positional encoding** on inputs.
- **Market context block:** Conv1D + Transformer fusion.
- **Trend branches:** short (5‑step) and medium (20‑step) trend extractors with Conv1D + pooling.
- **Price‑pattern module:** raw‑price channels with multi‑kernel convs, LSTMs, and learnable positional embeddings.
- **Technical‑feature branch:** parallel short/medium windows over technicals.
- **Fusion:** attention‑pooled token → bottleneck representation.
- **Heads:** stage‑1 price; optional stage‑2 confidence head.

Outputs are configured in training (stage‑1 price; stage‑2 adds confidence).

---

### `runModel.py`
End‑to‑end training script:
- Loads config (`../model/Data/Globalconfig.json`) and cached scalers/ID maps.
- Builds TensorFlow datasets and **trains stage‑1** with `directional_mse_loss` and metrics (`explained_variance`, `directional_accuracy`).
- Saves artifacts to `../model/Global/`:
  - `backbone_price.keras` (stage‑1 model)
  - `price_loss.png` (training curve)
  - `price_model_architecture.png` (if enabled; see below)

**Stage‑2 (confidence head)** is added on top of the bottleneck and trained with regularization and dropout (see code).

---

### `evalFunctions.py`
Utility functions for evaluation/post‑processing:
- **Scaling back to prices:** `unScale(...)` to reconstruct open/pred/true closes from scalers.
- **Trading P&L logic:** `buyShortTrading(...)` with slippage, commissions, optional 2% stop‑loss via daily high/low.
- **Metrics:** `directional_accuracy`, `profit_factor`, `sharpe_ratio`.
- **Plotting:** `plot_cumulative_returns(...)`.

---

### `evalOpti.py`
Single‑pass evaluation over a date window:
- Preloads **all ticker sequences** for the chosen day(s).
- Runs one big model prediction pass, then computes per‑ticker profits and aggregates.
- Fetches S&P‑500 as a baseline and can compute **Top‑N** results by confidence.

Artifacts saved to `./Outputs/`.

---

### `multiEval.py`
Walks backward day‑by‑day from `DATE` in config and repeats the evaluation routine, aggregating results and Top‑N baskets across multiple days. Useful for quick **walk‑forward checks** and visualizations.

---

### `FeatureImportance.py`
Performs **permutation importance** on the trained model:
- Shuffles each feature over samples and measures MSE increase vs baseline.
- Saves `Outputs/permutation_importance.png` for a quick interpretability snapshot.

---

## Generate the model diagram

The architecture image shown at the top is saved during training. If it doesn’t appear after running `runModel.py`, open `CloudScripts/runModel.py` and add the following lines **right after** the model is created (before training):

```python
from tensorflow.keras.utils import plot_model

plot_model(
    model_price,
    to_file="../model/Global/price_model_architecture.png",
    show_shapes=True,
    show_layer_names=True,
    expand_nested=True,
    dpi=200
)
```

You may need Graphviz installed:
```bash
# Windows (choco) / macOS (brew) / Ubuntu (apt)
choco install graphviz
brew install graphviz
sudo apt-get install graphviz
```

---

## Configuration & Paths

- Config JSON: `../model/Data/Globalconfig.json` (includes `SEQ_LENGTH`, `FUTURE_LENGTH`, `DATE`, etc.).
- Preprocessed datasets & scalers:
  - Global H5: `../model/Data/processedData.h5`
  - Per‑ticker H5: `../model/Data/Ticker_Data/<TICKER>.h5`
  - Scalers/IDs: `../model/Data/scalers.pkl`, `../model/Data/sector_to_id.pkl`, `../model/Data/stock_to_id.pkl`
- Trained models, curves, and diagrams: `../model/Global/`

> All `../` paths are **relative to `CloudScripts/`**, i.e., they write to project‑root folders outside this directory.

---

## Notes & Tips

- Ensure Alpaca SIP access is enabled for reliable 09:30–09:39 bars.
- The pipeline is designed to avoid **target leakage** via aligned windows and **grouped scalers** shared only within feature groups.
- For faster experimentation, limit tickers or date ranges in `downloadData.py` and `combinedDataSave.py`.
- Mixed precision is enabled (`float16`) where supported; disable if your environment requires it.

---

## License

This code is provided for research and educational purposes. Verify compliance of data sources (Yahoo Finance, Alpaca) with their terms of service before commercial use.

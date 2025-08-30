import os
import json
import pickle
import random
from datetime import datetime, timedelta

import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from tensorflow.keras import backend as K
import tensorflow as tf
from tensorflow.python.client.session import InteractiveSession
from tensorflow.python.keras.backend import set_session
from tensorflow.python.proto_exports import ConfigProto

from group_scaler import GroupScaler
import matplotlib.dates as mdates
from evalFunctions import *
from config import *

os.environ['PYTHONHASHSEED'] = '0'
random.seed(0)
np.random.seed(0)
tf.random.set_seed(0)

config = ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.8

session = InteractiveSession(config=config)
set_session(session)

K.clear_session()
tf.keras.mixed_precision.set_global_policy("mixed_float16")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

model = loadModel(MODEL_PATH)
with open("../model/Data/Globalconfig.json", "r") as f:
    config = json.load(f)
SEQ_LENGTH = config["SEQ_LENGTH"]
DAY_TO_EVAL_FROM = config["DATE"]

with open("../model/Data/scalers.pkl", "rb") as f:
    scalers = pickle.load(f)

with open("../model/Data/sector_to_id.pkl", "rb") as f:
    sector_to_id = pickle.load(f)
id_to_sector = {v: k for k, v in sector_to_id.items()}

# PRE‑LOAD ALL TICKER DATA
TICKER_DIR = "../model/Data/Ticker_Data"
TICKERS = [f[:-3] for f in os.listdir(TICKER_DIR) if f.endswith(".h5")]

print(f"Loading cached sequences for {len(TICKERS)} tickers …")

ticker_Xy = {}
valid_tickers = []
for tkr in TICKERS:
    try:
        X_tkr, y_tkr = load_individual_prepared_data(os.path.join(TICKER_DIR, f"{tkr}.h5"))

        if X_tkr.shape[0] >= DAYS_BACK + 1:  # Ensure enough data for all days
            ticker_Xy[tkr] = (X_tkr, y_tkr)
            valid_tickers.append(tkr)
    except Exception as e:
        print(f"⚠️  Skipping {tkr}: {e}")

TICKERS = valid_tickers
print(f"Loaded {len(TICKERS)} tickers after filtering")

# PRE-COMPUTE ALL BATCHES
print("Pre-computing batches for all days...")

all_X_batches = []
all_meta_data = []
eval_dates = []

for day_offset in range(DAYS_BACK):
    back_idx = -(day_offset + 1)
    dt_eval = datetime.strptime(DAY_TO_EVAL_FROM, "%Y-%m-%d") - timedelta(days=day_offset)
    eval_day_str = dt_eval.strftime("%Y-%m-%d")

    X_batch = []
    meta_tickers = []
    meta_open_scaled = []
    meta_targets = []
    meta_sectors = []

    for tkr in TICKERS:
        X_all, y_all = ticker_Xy[tkr]
        if X_all.shape[0] < abs(back_idx):
            continue

        X_seq, sector = getInputs(X_all, tkr, back=back_idx)
        X_seq = X_seq[0]  # strip batch dim

        X_batch.append(X_seq)
        meta_tickers.append(tkr)
        meta_open_scaled.append(X_seq[-1, 0])
        meta_targets.append(y_all[back_idx][0])
        meta_sectors.append(sector.item() if hasattr(sector, "item") else sector)

    if X_batch:
        all_X_batches.append(np.asarray(X_batch, dtype=np.float32))
        all_meta_data.append({
            'tickers': meta_tickers,
            'open_scaled': meta_open_scaled,
            'targets': meta_targets,
            'sectors': meta_sectors
        })
        eval_dates.append(eval_day_str)

print(f"Pre-computed {len(all_X_batches)} batches")

# BATCH PREDICTIONS
print("Running batch predictions...")

total_samples = sum(batch.shape[0] for batch in all_X_batches)
combined_X = np.concatenate(all_X_batches, axis=0)

# Single prediction call for ALL samples
print(f"Running single prediction on {total_samples} samples...")
all_preds = model.predict(
    {"data_input": combined_X},
    batch_size=min(128, total_samples),
    verbose=1
)

if isinstance(all_preds, list):
    all_mu_pred, all_conf_pred = all_preds
else:
    all_mu_pred = all_preds[:, :1]
    all_conf_pred = all_preds[:, 1:]

all_pred_scaled = all_mu_pred[:, 0]
all_confidences = all_conf_pred[:, 0]

# PRE-FETCH SP500 DATA
print("Fetching SP500 data...")
start_date = min(eval_dates)
end_date = max(pd.to_datetime(eval_dates) + timedelta(days=1)).strftime("%Y-%m-%d")

sp500_data = yf.download(
    SP500_TICKER,
    start=start_date,
    end=end_date,
    progress=False
)

print("Processing results...")

cumulative_results = []
model_daily_profits = []
sp500_daily_profits = []
topn_model_daily_profits = []
topn_sp500_daily_profits = []
processed_dates = []

pred_idx = 0

for day_idx, eval_day_str in enumerate(eval_dates):
    batch_size = all_X_batches[day_idx].shape[0]
    meta = all_meta_data[day_idx]

    # Extract predictions for this day
    day_pred_scaled = all_pred_scaled[pred_idx:pred_idx + batch_size]
    day_confidences = all_confidences[pred_idx:pred_idx + batch_size]
    pred_idx += batch_size

    # Get SP500 data for this day
    try:
        sp_day_data = sp500_data.loc[eval_day_str]
        sp_open = sp_day_data["Open"]
        sp_close = sp_day_data["Close"]
    except KeyError:
        print(f"Skipping {eval_day_str} — no SP‑500 data available")
        continue

    opens = []
    preds = []
    trues = []
    profits = []

    for i, tkr in enumerate(meta['tickers']):
        open_s = meta['open_scaled'][i]
        pred_s = day_pred_scaled[i]
        true_s = meta['targets'][i]

        open_p, pred_p, true_p = unScale(
            [open_s, pred_s],
            [true_s],
            scalers[tkr],
        )

        profit = buyShortTrading(
            open=open_p,
            prediction=pred_p,
            close=true_p,
            shares=SHARES,
            ticker=tkr,
            day=eval_day_str,
        )

        opens.append(open_p)
        preds.append(pred_p)
        trues.append(true_p)
        profits.append(profit)

    # Create day DataFrame
    df_day = pd.DataFrame({
        "Ticker": meta['tickers'],
        "Open": opens,
        "Close": trues,
        "Prediction": preds,
        "Sector": meta['sectors'],
        "Profit": profits,
        "Confidence": [1 - c for c in day_confidences],
        "Date": eval_day_str,
    })

    cumulative_results.append(df_day)

    # Top-N calculations
    top_df = get_top_variance(TOP_N_CONF, df_day)

    spent_on_topn = 0
    topn_profit = 0
    for _, row in top_df.iterrows():
        open_price = row["Open"]
        if SHARES is None:
            if open_price > BUDGET_PER_TRADE:
                continue
            open_price_val = open_price.item() if isinstance(open_price, np.ndarray) else float(open_price)
            shares_bought = int(BUDGET_PER_TRADE / open_price_val)
        else:
            shares_bought = SHARES
        spent_on_topn += shares_bought * open_price
        topn_profit += row["Profit"]

    topn_model_daily_profits.append(topn_profit)

    # SP500 benchmark (maintain both tracking variables for compatibility)
    sp_shares = spent_on_topn / sp_open
    sp_profit = (sp_close - sp_open) * sp_shares
    sp500_daily_profits.append(sp_profit)
    topn_sp500_daily_profits.append(sp_profit)

    processed_dates.append(eval_day_str)


    print(f"Evaluated {eval_day_str}  (day {day_idx + 1}/{len(eval_dates)})")

# Plot dashboard for every day
for day_idx, day_df in enumerate(cumulative_results):
    eval_day_str = processed_dates[day_idx]
    full_ranked_df = day_df.sort_values("Confidence", ascending=True).reset_index(drop=True)

    plot_trade_signal_dashboard(
        full_ranked_df,
        title=f"Trade Signal Dashboard: {eval_day_str}",
        sort_ascending=True,
    )
    print(f"✔ Dashboard plotted for {eval_day_str}")

# VISUALISATIONS
df_viz = pd.DataFrame({
    "Date": pd.to_datetime(processed_dates),
    "ModelProfit": topn_model_daily_profits,
    "SP500Profit": sp500_daily_profits
})
df_viz = df_viz.sort_values("Date").reset_index(drop=True)

df_viz["ModelCumulative"] = df_viz["ModelProfit"].cumsum()
df_viz["SP500Cumulative"] = df_viz["SP500Profit"].cumsum()

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(
    df_viz["Date"],
    df_viz["ModelCumulative"],
    marker="o",
    label="Model Cumulative Profit"
)
ax.plot(
    df_viz["Date"],
    df_viz["SP500Cumulative"],
    linestyle="--",
    marker="x",
    label="SP500 Cumulative Profit"
)

ax.set_title("Cumulative Profit Comparison (Top N vs. SP500)")
ax.set_xlabel("Date")
ax.set_ylabel("Cumulative Profit ($)")
ax.grid(True)
ax.legend()

locator = mdates.AutoDateLocator()
formatter = mdates.ConciseDateFormatter(locator)
ax.xaxis.set_major_locator(locator)
ax.xaxis.set_major_formatter(formatter)
plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

plt.tight_layout()
plt.show()

# FINAL METRICS
df_all = pd.concat(cumulative_results, ignore_index=True)
print("\nFinal Combined Results:")

df_all['TrueDir'] = np.sign(df_all['Close'] - df_all['Open'])
df_all['PredDir'] = np.sign(df_all['Prediction'] - df_all['Open'])
mask = df_all['TrueDir'] != 0
overall_acc = (df_all.loc[mask, 'TrueDir'] == df_all.loc[mask, 'PredDir']).mean()
print(f"All‑trades directional accuracy: {overall_acc:.4f}")

daily = df_all.groupby('Date').agg({
    'Profit': 'sum',
    'Open': 'sum'
})
daily['PctProfit'] = daily['Profit'] / daily['Open'] * 100

print("\n% profit per day:")
print(f"{daily['PctProfit'].mean():.4f}")

total_actual_profit = daily['Profit'].sum()
print(f"\nTotal actual profit: {total_actual_profit}")

print(f"\nOptimization complete: Processed {len(processed_dates)} days with single model call")
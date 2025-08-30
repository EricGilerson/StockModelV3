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
SEQ_LENGTH       = config["SEQ_LENGTH"]
DAY_TO_EVAL_FROM = config["DATE"]

with open("../model/Data/scalers.pkl", "rb") as f:
    scalers = pickle.load(f)

with open("../model/Data/sector_to_id.pkl", "rb") as f:
    sector_to_id = pickle.load(f)
id_to_sector = {v: k for k, v in sector_to_id.items()}

# PRE‑LOAD ALL TICKER DATA
TICKER_DIR = "../model/Data/Ticker_Data"
TICKERS    = [f[:-3] for f in os.listdir(TICKER_DIR) if f.endswith(".h5")]

print(f"Loading cached sequences for {len(TICKERS)} tickers …")

ticker_Xy = {}
for tkr in TICKERS:
    try:
        X_tkr, y_tkr = load_individual_prepared_data(os.path.join(TICKER_DIR, f"{tkr}.h5"))
        ticker_Xy[tkr] = (X_tkr, y_tkr)
    except Exception as e:
        print(f"Skipping {tkr}: {e}")

TICKERS = list(ticker_Xy.keys())
print(f"Loaded {len(TICKERS)} tickers after filtering")

cumulative_results        = []
model_daily_profits       = []
sp500_daily_profits       = []
topn_model_daily_profits  = []
topn_sp500_daily_profits  = []
processed_dates           = []

current_day_offset = 0
collected_days     = 0

while collected_days < DAYS_BACK:
    back_idx = -(current_day_offset + 1)

    dt_eval      = datetime.strptime(DAY_TO_EVAL_FROM, "%Y-%m-%d") - timedelta(days=current_day_offset)
    eval_day_str = dt_eval.strftime("%Y-%m-%d")

    #Fetch SP‑500 reference
    sp_data = yf.download(
        SP500_TICKER,
        start=eval_day_str,
        end=(dt_eval + timedelta(days=1)).strftime("%Y-%m-%d"),
        progress=False,
    )
    if sp_data.empty:
        print(f"Skipping {eval_day_str} — no SP‑500 data available")
        current_day_offset += 1
        continue

    sp_open  = sp_data["Open"].iloc[0]
    sp_close = sp_data["Close"].iloc[0]

    X_batch           = []
    meta_tickers      = []
    meta_open_scaled  = []
    meta_targets      = []
    meta_sectors      = []

    for tkr in TICKERS:
        X_all, y_all = ticker_Xy[tkr]
        if X_all.shape[0] < abs(back_idx):
            continue

        X_seq, sector = getInputs(X_all, tkr, back=back_idx)
        X_seq = X_seq[0]

        X_batch.append(X_seq)
        meta_tickers.append(tkr)
        meta_open_scaled.append(X_seq[-1, 0])
        meta_targets.append(y_all[back_idx][0])
        meta_sectors.append(sector.item() if hasattr(sector, "item") else sector)

    if not X_batch:
        print(f"Skipping {eval_day_str} — no ticker had enough history")
        current_day_offset += 1
        continue

    X_batch = np.asarray(X_batch, dtype=np.float32)

    #Single forward‑pass
    preds = model.predict(
        {"data_input": X_batch},
        batch_size=len(X_batch),
        verbose=0
    )

    # unpack according to what predict() returned
    if isinstance(preds, list):
        # two separate outputs
        mu_pred, conf_pred = preds
    else:
        mu_pred = preds[:, :1]
        conf_pred = preds[:, 1:]

    pred_scaled = mu_pred[:, 0]
    confidences = conf_pred[:, 0]

    opens      = []
    preds      = []
    trues      = []
    profits    = []

    for i, tkr in enumerate(meta_tickers):
        open_s   = meta_open_scaled[i]
        pred_s   = pred_scaled[i]
        true_s   = meta_targets[i]

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

    # Assemble DataFrame for the day
    df_day = pd.DataFrame({
        "Ticker":      meta_tickers,
        "Open":        opens,
        "Close":       trues,
        "Prediction":  preds,
        "Sector":      meta_sectors,
        "Profit":      profits,
        "Confidence":  [1 - c for c in confidences],
        "Date":        eval_day_str,
    })

    cumulative_results.append(df_day)

    # Top‑N by confidence
    top_df = get_top_variance(TOP_N_CONF, df_day)

    spent_on_topn = 0
    topn_profit   = 0
    for _, row in top_df.iterrows():
        open_price = row["Open"]
        if SHARES is None:
            if open_price > BUDGET_PER_TRADE:
                continue
            open_price_val = open_price.item() if isinstance(open_price, np.ndarray) else float(open_price)
            shares_bought = int(BUDGET_PER_TRADE / open_price_val)
        else:
            shares_bought  = SHARES
        spent_on_topn += shares_bought * open_price
        topn_profit   += row["Profit"]

    topn_model_daily_profits.append(topn_profit)

    sp_shares          = spent_on_topn / sp_open
    sp_profit          = (sp_close - sp_open) * sp_shares
    sp500_daily_profits.append(sp_profit)
    topn_sp500_daily_profits.append(sp_profit)

    full_ranked_df = df_day.sort_values("Confidence", ascending=True).reset_index(drop=True)

    plot_trade_signal_dashboard(
        full_ranked_df,
        title=f"Trade Signal Dashboard: {eval_day_str}",
        sort_ascending=True,
    )
    print(f"✔ Evaluated {eval_day_str}  (day {collected_days+1}/{DAYS_BACK})")
    processed_dates.append(eval_day_str)

    current_day_offset += 1
    collected_days     += 1

# VISUALISATIONS
df = pd.DataFrame({
    "Date": pd.to_datetime(processed_dates),
    "ModelProfit": topn_model_daily_profits,
    "SP500Profit": sp500_daily_profits
})
df = df.sort_values("Date").reset_index(drop=True)

df["ModelCumulative"] = df["ModelProfit"].cumsum()
df["SP500Cumulative"] = df["SP500Profit"].cumsum()

# Plot both cumulative lines
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(
    df["Date"],
    df["ModelCumulative"],
    marker="o",
    label="Model Cumulative Profit"
)
ax.plot(
    df["Date"],
    df["SP500Cumulative"],
    linestyle="--",
    marker="x",
    label="SP500 Cumulative Profit"
)

ax.set_title("Cumulative Profit Comparison (Top N vs. SP500)")
ax.set_xlabel("Date")
ax.set_ylabel("Cumulative Profit ($)")
ax.grid(True)
ax.legend()

locator   = mdates.AutoDateLocator()
formatter = mdates.ConciseDateFormatter(locator)
ax.xaxis.set_major_locator(locator)
ax.xaxis.set_major_formatter(formatter)
plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

plt.tight_layout()
plt.show()


# FINAL METRICS

df_all = pd.concat(cumulative_results, ignore_index=True)
print("\nFinal Combined Results:")

#top_combined = get_top_variance(TOP_N_CONF, df_all)
#print(f"\n— Top {TOP_N_CONF} Confidence Summary —")
#print(f"Total profit:   {top_combined['Profit'].sum():.2f}")
#print(f"Profit Factor:  {profit_factor(top_combined['Profit']):.3f}")
#print(f"Sharpe Ratio:   {sharpe_ratio(top_combined['Profit']):.3f}")
#print(f"Win/Loss Ratio: {win_loss_ratio(top_combined['Profit']):.3f}")
df_all['TrueDir'] = np.sign(df_all['Close']   - df_all['Open'])
df_all['PredDir'] = np.sign(df_all['Prediction'] - df_all['Open'])
mask = df_all['TrueDir'] != 0
overall_acc = (df_all.loc[mask,'TrueDir'] == df_all.loc[mask,'PredDir']).mean()
print(f"All‑trades directional accuracy: {overall_acc:.4f}")

daily = df_all.groupby('Date').agg({
    'Profit': 'sum',
    'Open':   'sum'
}).copy()

daily['PctProfit'] = (
    daily['Profit']
    / daily['Open'].replace(0, np.nan)
    * 100
)

daily['PctProfit'] = daily['PctProfit'].fillna(0)

pct_mean = float(daily['PctProfit'].mean())
total_actual_profit = float(daily['Profit'].sum())

print(f"\n% profit per day: {pct_mean:.2f}")
print(f"\nTotal actual profit: {total_actual_profit:.2f}")
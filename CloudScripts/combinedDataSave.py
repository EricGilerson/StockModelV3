from __future__ import annotations

import os
import numpy as np
import pandas as pd
import h5py
import pickle
import json
from joblib import Parallel, delayed
from sklearn.preprocessing import RobustScaler
from multiprocessing import Manager
from datetime import timedelta

from CloudScripts.group_scaler import GroupScaler

# ─────────────────────────────── constants ────────────────────────────────
SECTORS_DF = pd.read_csv("../Data/sectors.csv")
stock_to_id = {ticker: i for i, ticker in enumerate(SECTORS_DF['Ticker'].unique())}
sector_to_id = {sector: i for i, sector in enumerate(SECTORS_DF['Sector'].unique())}
ticker_to_sector_id = {
    ticker: sector_to_id[SECTORS_DF.loc[SECTORS_DF['Ticker'] == ticker, 'Sector'].iloc[0]]
    for ticker in stock_to_id.keys()
}

feature_groups = {
    "price": ["Price_0939", "MA5"],
    "delta_price": ["Prev_Delta_Close"],
    "atr": ["ATR_14"],
    "macd": ["MACD_Hist"],
    "volume_count": ["OBV"],
    "micro_pct": ["atr10m_pct", "vwap_gap", "ret939_pct"],
    "momentum_pct": ["Stoch_K", "Stoch_D"],
    "ratio": [
        "Gap", "Composite_HL", "Bollinger_Width"
    ],
}

# Scaling recipe per‑group
STRATEGY = {
    "price": "standard",
    "delta_price": "zero_standard",
    "atr": "log_standard",
    "volume_count": "log_standard",
    "ratio": "robust",
    "micro_pct": "minmax",
    "momentum_pct": "minmax",
    "macd": "divide_only",
}



def detect_and_handle_outliers(df, columns, threshold=10.0):
    df_clean = df.copy()
    for col in columns:
        if col not in df_clean.columns:
            continue
        median = df_clean[col].median()
        q1 = df_clean[col].quantile(0.25)
        q3 = df_clean[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = median - threshold * iqr
        upper_bound = median + threshold * iqr
        df_clean[col] = np.clip(df_clean[col], lower_bound, upper_bound)
    return df_clean


def ensure_numeric_dataframe(df):
    df = df.copy()
    for col in df.columns:
        if col == "Date":
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df



def process_single_and_save(
        ticker: str,
        df: pd.DataFrame,
        seq_length: int,
        future_days: int,
        max_features: int,
        stock_to_id: dict,
        sector_to_id: dict,
        ticker_to_sector_id: dict,
        h5_path: str,
        lock,
):
    try:
        #initial cleaning
        df = ensure_numeric_dataframe(df)
        num_cols = df.select_dtypes(include=[np.number]).columns
        df = detect_and_handle_outliers(df, num_cols, threshold=7.0)

        df = df.fillna(df.median(numeric_only=True))
        df[num_cols] = df[num_cols].replace([np.inf, -np.inf], [1e9, -1e9])

        # Use full dataset for fitting scalers
        df_full = df.reset_index(drop=True)

        scalers = {}

        # Store the delta_price scaler for later use on the target
        delta_price_scaler = None

        for group, cols in feature_groups.items():
            cols = [c for c in cols if c in df_full.columns]
            if not cols:
                continue

            Xg = df_full[cols].astype(np.float32).values
            strat = STRATEGY.get(group, "standard")

            if strat == "divide_only":
                q75, q25 = np.nanpercentile(Xg, [75, 25])
                iqr = q75 - q25 or 1.0
                df_full[cols] = df_full[cols].astype(np.float32).values / iqr
                scalers[group] = GroupScaler(0.0, iqr, len(cols), mode="divide_only")
                df_full[cols] = df_full[cols].clip(-10.0, 10.0)

            elif strat in {"standard", "zero_standard"}:
                mu = 0.0 if strat == "zero_standard" else np.nanmean(Xg)
                std = np.nanstd(Xg) or 1.0
                df_full[cols] = (df_full[cols].astype(np.float32).values - mu) / std
                scalers[group] = GroupScaler(mu, std, len(cols), mode=strat)

                if group == "delta_price":
                    delta_price_scaler = (mu, std)

            elif strat == "log_standard":
                Xgs = np.sign(Xg) * np.log1p(np.abs(Xg))
                mu, std = np.nanmean(Xgs), np.nanstd(Xgs) or 1.0
                df_full[cols] = np.sign(df_full[cols]) * np.log1p(np.abs(df_full[cols]))
                df_full[cols] = (df_full[cols] - mu) / std
                df_full[cols] = df_full[cols].clip(-10.0, 10.0)
                sc = GroupScaler(mu, std, len(cols), mode="log_standard", pre="log")
                scalers[group] = sc

            elif strat == "robust":
                med = np.nanmedian(Xg)
                iqr = np.subtract(*np.nanpercentile(Xg, [75, 25])) or 1.0
                df_full[cols] = (df_full[cols] - med) / iqr
                df_full[cols] = df_full[cols].clip(-10.0, 10.0)
                scalers[group] = GroupScaler(med, iqr, len(cols), mode="robust")

            elif strat == "minmax":
                xmin, xmax = np.nanmin(Xg), np.nanmax(Xg)
                rng = xmax - xmin or 1.0
                df_full[cols] = 2.0 * (df_full[cols] - xmin) / rng - 1.0
                scalers[group] = GroupScaler(xmin, rng, len(cols), mode="minmax")
            else:
                raise ValueError(f"Unknown strategy {strat} for group {group}")

        # window and target construction
        tkr = ticker.split('.')[0].replace('.', '-')
        stock_id = stock_to_id[tkr]
        sector_id = ticker_to_sector_id[tkr]

        adj_future = future_days - 1 # number of rows after the window
        n_seq_total = len(df_full) - (seq_length + adj_future) + 1
        if n_seq_total <= 0:
            print(f"[{tkr}] insufficient rows – skipped")
            return None

        feat_cols = [c for v in feature_groups.values() for c in v if c in df_full.columns]
        X_buf, y_buf = [], []

        for s in range(n_seq_total):
            win = df_full[feat_cols].iloc[s: s + seq_length].to_numpy(np.float32)
            tgt_idx = s + seq_length - 1

            # Get raw target value
            tgt_val = df_full['Delta_Close'].iloc[tgt_idx: tgt_idx + future_days].to_numpy(np.float32)

            #tgt_val = (tgt_val > 0).astype(np.float32)
            #delta_price_scaler = None
            # Scale the target using the same scaler as Prev_Delta_Close
            if delta_price_scaler is not None:
                mu, std = delta_price_scaler
                tgt_val = (tgt_val - mu) / std

            if win.shape[1] < max_features:
                pad = np.zeros((seq_length, max_features), dtype=np.float32)
                pad[:, :win.shape[1]] = win
                win = pad
            else:
                win = win[:, :max_features]

            if np.isnan(win).any() or np.isinf(win).any() or np.isnan(tgt_val).any() or np.isinf(tgt_val).any():
                continue

            X_buf.append(win)
            y_buf.append(tgt_val)

        if not X_buf:
            print(f"[{tkr}] no valid windows – skipped")
            return None

        X = np.stack(X_buf)
        y = np.stack(y_buf)

        stock_ids = np.full(len(X), stock_id, np.int32)
        sector_ids = np.full(len(X), sector_id, np.int32)

        #save to HDF5 (thread‑safe)
        os.makedirs('../model/Data/Ticker_Data', exist_ok=True)
        with lock, h5py.File(h5_path, 'a') as h5:
            for name, data in (
                    ('X', X),
                    ('y', y),
                    ('stock_ids', stock_ids),
                    ('sector_ids', sector_ids),
            ):
                if data.size == 0:
                    continue
                if name not in h5:
                    h5.create_dataset(name, data=data, maxshape=(None,) + data.shape[1:],
                                      chunks=True, compression='gzip', compression_opts=9)
                else:
                    dset = h5[name]
                    start = dset.shape[0]
                    dset.resize(start + data.shape[0], axis=0)
                    dset[start:] = data

        # Per‑ticker H5
        with h5py.File(f'../model/Data/Ticker_Data/{tkr}.h5', 'w') as h5t:
            h5t.create_dataset('sequences', data=X)
            h5t.create_dataset('targets', data=y)


        if delta_price_scaler is not None:
            mu, std = delta_price_scaler
            target_scaler = GroupScaler(mu, std, 1)
            scalers[f"{tkr}_target"] = target_scaler

        return {tkr: scalers}

    except Exception as exc:
        print(f"Error processing {ticker}: {exc}")
        import traceback;
        traceback.print_exc()
        return None



def validate_saved_data(h5_path):
    with h5py.File(h5_path, 'r') as f:
        for name in ('X', 'y', 'stock_ids', 'sector_ids'):
            if name not in f:
                continue
            data = f[name][:].astype(np.float32)
            print(f"{name} → shape {data.shape}, NaN? {np.isnan(data).any()}, Inf? {np.isinf(data).any()}")



def load_prepare_and_stream(
        seq_length: int,
        future_days: int,
        cutoff_date: str | None = None,
        n_jobs: int = -1,
        h5_path: str = "../model/Data/processedData.h5",
):
    tickers = [t for t in os.listdir("../Data") if t.endswith(".csv") and t not in ("sectors.csv", "TT.csv", "MTCH.csv", "FAST.csv", "IR.csv")]
    cutoff_dt = pd.to_datetime(cutoff_date) if cutoff_date else None
    data_frames = {}
    max_features = 0

    for ticker in tickers:
        try:
            df = pd.read_csv(f"../Data/{ticker}")
            df['Date'] = pd.to_datetime(df['Date'])
            if cutoff_dt is not None:
                df = df[df['Date'] <= cutoff_dt]
            if df.empty:
                continue
            num_features = len(df.columns) - 2  # exclude Date & Delta_Close
            max_features = max(max_features, num_features)
            data_frames[ticker] = df
        except Exception as e:
            print(f"Error loading {ticker}: {e}")

    if os.path.exists(h5_path):
        os.remove(h5_path)

    with Manager() as manager:
        lock = manager.Lock()
        results = Parallel(n_jobs=n_jobs, verbose=1)(
            delayed(process_single_and_save)(
                ticker, df, seq_length, future_days, max_features,
                stock_to_id, sector_to_id, ticker_to_sector_id, h5_path, lock
            ) for ticker, df in data_frames.items()
        )
        scalers = {k: v for r in results if r for k, v in r.items()}

    validate_saved_data(h5_path)
    return scalers


if __name__ == "__main__":
    SEQ_LENGTH = 30
    FUTURE_DAYS = 1
    CUTOFF_DATE = "2025-05-22"

    os.makedirs('../model/Data/', exist_ok=True)

    print(f"\n▶ Starting ETL (seq={SEQ_LENGTH}, horizon={FUTURE_DAYS})")
    scalers = load_prepare_and_stream(
        seq_length=SEQ_LENGTH,
        future_days=FUTURE_DAYS,
        cutoff_date=CUTOFF_DATE,
        n_jobs=-1,
    )

    #save metadata
    with open("../model/Data/scalers.pkl", "wb") as f:
        pickle.dump(scalers, f)
    with open("../model/Data/stock_to_id.pkl", "wb") as f:
        pickle.dump(stock_to_id, f)
    with open("../model/Data/sector_to_id.pkl", "wb") as f:
        pickle.dump(sector_to_id, f)

    config = {
        "SEQ_LENGTH": SEQ_LENGTH,
        "FUTURE_LENGTH": FUTURE_DAYS,
        "EMBEDDING_DIM_STOCK": 4,
        "EMBEDDING_DIM_SECTOR": 10,
        "DATE": CUTOFF_DATE,
    }
    with open("../model/Data/Globalconfig.json", "w") as f:
        json.dump(config, f)

    print("\n✔ Global data saved with leakage‑free pipeline.")
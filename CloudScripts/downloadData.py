"""
Daily + 09:30–09:39 micro-features builder
• Daily OHLCV + technicals from Yahoo Finance
• First-10-minutes ATR%, VWAP-gap, and ret0939_pct from Alpaca
• Writes one CSV per ticker with Close as *final* column
"""

import os
import pickle
import time, threading
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

from alpaca.data import StockHistoricalDataClient
from dotenv import load_dotenv
from pandas.core.dtypes.common import is_integer_dtype, is_datetime64_any_dtype

load_dotenv()

total_tickers = 0
counter        = 0
counter_lock   = Lock()
final_cols = ['Date', 'Price_0939', 'Prev_Delta_Close', 'Gap',
         'atr10m_pct', 'vwap_gap', 'ret939_pct',
         'MACD_Hist', 'Bollinger_Width',
         'OBV', 'MA5', 'Composite_HL',
         'ATR_14', 'Stoch_K', 'Stoch_D',
         'Delta_Close']

def _bump_progress():
    global counter
    with counter_lock:
        counter += 1
        return counter


#Alpaca client setup
ALP_KEY = os.getenv("APCA_API_KEY_ID")
ALP_SEC = os.getenv("APCA_API_SECRET_KEY")
FEED = None
ALP_OK  = bool(ALP_KEY and ALP_SEC)
if ALP_OK:
    try:
        from alpaca.data.historical import StockHistoricalDataClient as AClient
        from alpaca.data.requests   import StockBarsRequest
        from alpaca.data.timeframe  import TimeFrame
        from alpaca.data.enums import DataFeed, Adjustment

        ALP = AClient(ALP_KEY, ALP_SEC)

        def _try_feed(feed_enum):
            req = StockBarsRequest(
                symbol_or_symbols="AAPL",
                timeframe=TimeFrame.Minute,
                start=dt.datetime(2022, 1, 3, 14, 30, tzinfo=dt.timezone.utc),
                end  =dt.datetime(2022, 1, 3, 14, 31, tzinfo=dt.timezone.utc),
                feed=feed_enum,
            )
            return ALP.get_stock_bars(req).df

        try:
            f = DataFeed.SIP
            test_df = _try_feed(f)
            FEED = f
            print(f"Alpaca OK – feed={f.name}, rows={len(test_df)}")
            print(test_df.head())
        except Exception as e:
            print(e)

    except Exception as e:
        ALP_OK = False
        print(f"Alpaca setup failed: {e}")
else:
    print("Alpaca keys missing: set APCA_API_KEY_ID / APCA_API_SECRET_KEY or .env")

# BULK 09:30–09:39 FETCHER
from dateutil.relativedelta import relativedelta
from itertools import islice
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **kw): return it

def chunks(seq, n):
    it = iter(seq)
    while (batch := list(islice(it, n))):
        yield batch

PARQ_DIR = "intraday"
os.makedirs(PARQ_DIR, exist_ok=True)
CLIENT = AClient(ALP_KEY, ALP_SEC) if ALP_OK else None
from alpaca.trading.client   import TradingClient
from alpaca.trading.requests import GetCalendarRequest

BATCH_SIZE               = 45
MAX_SYMS_PER_CALL        = 45
RATE_LIMIT_CALLS_PER_MIN = 200
RATE_LIMIT_INTERVAL      = 60.0 / RATE_LIMIT_CALLS_PER_MIN
PARQ_DIR                 = Path("intraday")
SECTOR_CSV     = "../Data/sectors.csv"
SPLITS_PKL = "../Data/splits.pkl"

import logging

logger = logging.getLogger(__name__)
#init clients
data_client    = StockHistoricalDataClient(ALP_KEY, ALP_SEC)
trading_client = TradingClient(ALP_KEY, ALP_SEC)

# throttle helper
_last_call_time = 0.0
_throttle_lock = threading.Lock()
def _throttle() -> None:
    global _last_call_time
    with _throttle_lock:
        now      = time.time()
        elapsed  = now - _last_call_time
        to_sleep = max(0.0, RATE_LIMIT_INTERVAL - elapsed)
        if to_sleep:
            time.sleep(to_sleep)
        _last_call_time = time.time()

# helpers
def chunks(seq, n):
    it = iter(seq)
    while (batch := list(islice(it, n))):
        yield batch

#Return all trading‐day midnights (US/Eastern) from start to end (end exclusive).
def _get_trading_days(start: pd.Timestamp, end: pd.Timestamp) -> list[pd.Timestamp]:

    req = GetCalendarRequest(
        start=start.date(),
        end=(end - pd.Timedelta(days=1)).date()
    )
    cal = trading_client.get_calendar(req)
    return [pd.Timestamp(d.date, tz="US/Eastern") for d in cal]
ASOF = pd.Timestamp("today").strftime("%Y-%m-%d")   # "2025-04-27"
# MONTHLY BAR FETCH
def month_bars(symbols, month_start, month_end):
    orig_to_alp = {s: s.replace('-', '.') for s in symbols}
    alp_syms    = list(orig_to_alp.values())
    alp_to_orig = {v: k for k, v in orig_to_alp.items()}

    days = _get_trading_days(month_start, month_end)
    raw_frames = []


    for sym_chunk in chunks(alp_syms, MAX_SYMS_PER_CALL):
        for day in days:
            start_et = day + pd.Timedelta(hours=9, minutes=30)
            end_et   = day + pd.Timedelta(hours=9, minutes=39)

            _throttle()
            req = StockBarsRequest(
                symbol_or_symbols=sym_chunk,
                timeframe=TimeFrame.Minute,
                start=start_et.tz_convert("UTC"),
                end=end_et.tz_convert("UTC"),
                feed=DataFeed.SIP,
                limit=10000,
                adjustment=Adjustment.RAW,
            )
            part = data_client.get_stock_bars(req).df
            if not part.empty:
                raw_frames.append(part)

    if not raw_frames:
        return pd.DataFrame()

    raw = pd.concat(raw_frames)
    df  = (raw.reset_index()
               .rename(columns={'timestamp': 'Date'})
               .set_index('Date')
               .tz_convert('US/Eastern')
               .between_time('09:30', '09:39', inclusive='both')
               .reset_index())

    df['symbol'] = df['symbol'].map(alp_to_orig)
    df['Date']   = df['Date'].dt.normalize()
    return df

def _chunks(seq: list[str], n: int):
    for i in range(0, len(seq), n):
        yield seq[i : i + n]

# Sequential intraday fetch with caching

def bulk_fetch_intraday(
    tickers   : list[str],
    start     : pd.Timestamp,
    end       : pd.Timestamp,
    month_idx : int  = -1,
) -> None:
    """
    Cache 09:30–09:39 ET SIP bars into PARQ_DIR month-by-month.

    month_idx = None: fetch / refresh every month in the range
    month_idx = -1: only refresh the last month
    month_idx = k ≥ 0: only (re)fetch the k-th month since `start`
    """
    if not ALP_OK:
        print("Alpaca keys missing – aborting intraday fetch")
        return

    PARQ_DIR.mkdir(exist_ok=True)

    months_all   : list[tuple[pd.Timestamp, pd.Timestamp]] = []
    month_cursor = start.replace(day=1)
    last_month   = end.replace(day=1)
    total_months = (end.year - start.year) * 12 + end.month - start.month + 1

    for _ in range(total_months):
        is_last   = (month_cursor == last_month)
        month_end = (end + pd.Timedelta(days=1)) if is_last else (
                     month_cursor + relativedelta(months=1))
        months_all.append((month_cursor, month_end))
        month_cursor += relativedelta(months=1)

    # Decide which months to run
    if month_idx is None:
        months_to_run = months_all
    else:
        idx = month_idx if month_idx >= 0 else len(months_all) + month_idx
        months_to_run = [months_all[idx]]

    chunks_of_syms = list(_chunks(tickers, MAX_SYMS_PER_CALL))
    last_start, _  = months_all[-1]

    tasks: list[tuple[list[str], pd.Timestamp, pd.Timestamp, bool]] = []
    for month_start, month_end in months_to_run:
        force = (month_start == last_start) or (month_idx is not None)
        for chunk in chunks_of_syms:
            if not force:
                missing = [
                    s for s in chunk
                    if not (PARQ_DIR / f"{s}_{month_start:%Y%m}.parquet").exists()
                ]
                if not missing:
                    continue
            tasks.append((chunk, month_start, month_end, force))

    for chunk, month_start, month_end, force in tqdm(
        tasks, total=len(tasks), desc="SIP download", unit="task"
    ):
        try:
            fetch_chunk_month(chunk, month_start, month_end, force=force)
        except Exception as e:
            print(f"⚠️ {chunk[:3]}…/{month_start:%Y-%m} force={force} → {e}")

    print("✓ Intraday parquet cache complete")

def fetch_chunk_month(chunk, month_start, month_end, force=False):
    #Pull SIP bars for this chunk/month and save to parquet
    df = month_bars(chunk, month_start, month_end)
    for sym, df_sym in df.groupby("symbol"):
        fname = PARQ_DIR / f"{sym}_{month_start:%Y%m}.parquet"
        if not force and fname.exists():
            continue
        df_sym.to_parquet(fname, index=False)

def bulk_fetch_intraday_thread(tickers, start, end, month_idx=-1):
    months_all = []
    month      = start.replace(day=1)
    last_month = end.replace(day=1)
    total_months = (end.year - start.year) * 12 + end.month - start.month + 1
    for _ in range(total_months):
        is_last   = (month == last_month)
        month_end = (end + pd.Timedelta(days=1)) if is_last else (month + relativedelta(months=1))
        months_all.append((month, month_end))
        month += relativedelta(months=1)

    # pick which months to run
    if month_idx is None:
        months_to_run = months_all
    else:
        # support negative indexing
        idx = month_idx if month_idx >= 0 else len(months_all) + month_idx
        months_to_run = [months_all[idx]]

    # build tasks
    chunks_of_syms   = list(chunks(tickers, MAX_SYMS_PER_CALL))
    last_start, _   = months_all[-1]
    tasks = []
    for month_start, month_end in months_to_run:
        force = (month_start == last_start) or (month_idx is not None)
        for chunk in chunks_of_syms:
            if not force:
                missing = [s for s in chunk
                           if not (PARQ_DIR / f"{s}_{month_start:%Y%m}.parquet").exists()]
                if not missing:
                    continue
            tasks.append((chunk, month_start, month_end, force))

    PARQ_DIR.mkdir(exist_ok=True)
    total_tasks = len(tasks)

    # process in parallel with tqdm
    with ThreadPoolExecutor(max_workers=8) as pool:
        future_to_task = {
            pool.submit(fetch_chunk_month, chunk, ms, me, force): (chunk, ms, force)
            for chunk, ms, me, force in tasks
        }
        for fut in tqdm(as_completed(future_to_task),
                        total=total_tasks,
                        desc="SIP download",
                        unit="task"):
            chunk, month_start, force = future_to_task[fut]
            try:
                fut.result()
            except Exception as e:
                print(f"⚠️ {chunk[:3]}…/{month_start:%Y-%m} force={force} → {e}")

    print("Finished")


MICRO_COLS = ["Price_0939, atr10m_pct", "vwap_gap", "ret939_pct"]

def attach_intraday(df_daily: pd.DataFrame,
                    tkr: str,
                    ticker_splits: dict[pd.Timestamp, float]) -> pd.DataFrame:
    #Add 09:30–09:39 micro-features to df_daily
    # Load this ticker’s parquet month-files
    month_files = sorted(Path(PARQ_DIR).glob(f"{tkr}_*.parquet"))
    if not month_files:
        return df_daily            # nothing cached yet

    intraday = pd.concat(
        (pd.read_parquet(f) for f in month_files),
        ignore_index=True
    )

    if is_integer_dtype(intraday["Date"]):
        intraday["Date"] = pd.to_datetime(
            intraday["Date"], unit="ns", utc=True
        )

    if is_datetime64_any_dtype(intraday["Date"]):
        intraday["Date"] = (
            intraday["Date"]
            .dt.tz_convert("US/Eastern")  # UTC → ET
            .dt.normalize()               # truncate to 00:00
            .dt.tz_localize(None)         # drop tz-info
        )

    # Adjust intraday prices for splits
    if ticker_splits:
        splits_norm = {
            pd.to_datetime(ex_date).normalize(): ratio
            for ex_date, ratio in ticker_splits.items()
        }
        intraday['adj_factor'] = 1.0
        for split_date, ratio in splits_norm.items():
            mask = intraday['Date'] < split_date
            intraday.loc[mask, 'adj_factor'] *= ratio
        for col in ('open', 'high', 'low', 'close', 'vwap'):
            intraday[col] = intraday[col] / intraday['adj_factor']
        intraday.drop(columns=['adj_factor'], inplace=True)

    g           = intraday.groupby("Date")
    first_open  = g.open.first()
    last_close  = g.close.last()
    last_vwap   = g.vwap.last()
    atr10m_pct  = (g.high.mean() - g.low.mean()) / first_open
    vwap_gap    = (last_close - last_vwap) / last_vwap
    ret939_pct  = (last_close - first_open) / first_open

    agg = pd.DataFrame({
        "Price_0939":  last_close,
        "atr10m_pct":  atr10m_pct,
        "vwap_gap":    vwap_gap,
        "ret939_pct":  ret939_pct,
    })

    # Join onto the daily frame
    df2 = df_daily.set_index("Date")
    df2 = df2.drop(columns=[c for c in MICRO_COLS if c in df2.columns])
    df2 = df2.join(agg, how="left")

    return df2.reset_index()



# Indicator helpers
def calculate_rsi(price_series, periods=10):
    #Standard RSI, but shift(1) after computing, so day T sees T-1 data.
    delta = price_series.diff()
    gain = delta.where(delta > 0, 0.0).rolling(window=periods).mean()
    loss = -delta.where(delta < 0, 0.0).rolling(window=periods).mean()
    rs = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))

def calculate_macd(prices, fast=12, slow=26, signal=9):
    #Return MACD histogram as a single Series
    exp1 = prices.ewm(span=fast, adjust=False).mean()
    exp2 = prices.ewm(span=slow, adjust=False).mean()
    macd_line = exp1 - exp2
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return histogram


def calculate_bollinger_width(prices, period=20):
    sma = prices.rolling(window=period).mean()
    std = prices.rolling(window=period).std()
    upper = sma + 2 * std
    lower = sma - 2 * std
    return (upper - lower) / (sma + 1e-9)

def calculate_obv(close_prices, volume):
    #On-Balance Volume
    direction = np.where(close_prices.diff() > 0, 1, -1)
    direction[close_prices.diff() == 0] = 0
    return (direction * volume).cumsum()

def safe_zscore(series):
    return (series - series.mean()) / (series.std() + 1e-9)

# extras
def true_range(high, low, prev_close):
    range1 = high - low
    range2 = (high - prev_close).abs()
    range3 = (low - prev_close).abs()
    return pd.concat([range1, range2, range3], axis=1).max(axis=1)

def calculate_atr(high, low, close, window=14):
    #ATR from (T-1) perspective. We shift later so day T sees up to T-1
    pc = close.shift(1)
    tr = true_range(high, low, pc)
    return tr.rolling(window).mean()

def calculate_stoch_kd(high, low, close, k_period=14, d_period=3):
    #Stochastic K & D.  Will shift(1) afterwards for day T features
    low_min = low.rolling(k_period).min()
    high_max = high.rolling(k_period).max()
    stoch_k = 100 * (close - low_min) / (high_max - low_min + 1e-9)
    stoch_d = stoch_k.rolling(d_period).mean()
    return stoch_k, stoch_d

def load_or_fetch_ipo_and_sectors(batch_size=25, max_workers=8):
    # Scrape S&P 500 list
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    df = pd.read_html(url)[0]
    df['Symbol'] = df['Symbol'].str.replace('.', '-', regex=False)
    tickers = df['Symbol'].tolist()
    sectors = df['GICS Sector'].tolist()
    # Save sectors
    if os.path.exists(SPLITS_PKL):
        with open(SPLITS_PKL, "rb") as f:
            splits = pickle.load(f)
    else:
        splits = {}
        for tkr in tickers:
            try:
                s: pd.Series = yf.Ticker(tkr).splits
                if hasattr(s.index, "tzinfo"):
                    s.index = s.index.tz_convert(None)
                splits[tkr] = s.to_dict()
            except Exception:
                splits[tkr] = {}
        with open(SPLITS_PKL, "wb") as f:
            pickle.dump(splits, f)
    # Save sectors & splits
    pd.DataFrame({"Ticker": tickers, "Sector": sectors}).to_csv(SECTOR_CSV, index=False)
    with open(SPLITS_PKL, "wb") as f:
        pickle.dump(splits, f)

    return tickers, sectors, splits

# Per-ticker build
def build_one(tkr: str, sector: str, start: str, end: str, splits: dict):
    global total_tickers
    current = _bump_progress()
    print(f"{tkr:5s}  ▶  {current}/{total_tickers}")
    df = yf.download(tkr, start=start, end=end,
                     auto_adjust=False, progress=False)
    ticker_splits = splits.get(tkr, {})

    if 'Date' not in df.columns:
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index().rename(columns={'index': 'Date'})
        else:
            df.reset_index(inplace=True)
    df['Date'] = pd.to_datetime(df['Date']).dt.normalize()

    # Flatten any MultiIndex and drop duplicate labels
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.loc[:, ~df.columns.duplicated()].copy()

    # Ensure enough raw data
    if df.empty or len(df) < 50:
        print(f"Skipping {tkr}: insufficient data")
        return
    else:
        df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']].dropna().copy()


        # T-1 shifts
        df['Prev_Open'] = df['Open'].shift(1)
        df['Prev_High'] = df['High'].shift(1)
        df['Prev_Low'] = df['Low'].shift(1)
        df['Prev_Close'] = df['Close'].shift(1)
        df['Prev_Vol'] = df['Volume'].shift(1)
        df['Gap'] = df['Open'] / df['Prev_Close'] - 1  # = open_gap_pct

        #tehincal features
        df['MACD_Hist'] = calculate_macd(df['Close']).shift(1)
        df['Bollinger_Width'] = calculate_bollinger_width(df['Close']).shift(1)
        df['OBV'] = calculate_obv(df['Close'], df['Volume']).shift(1)
        df['MA5'] = df['Close'].rolling(5).mean().shift(1)

        df['Prev_Range'] = (df['Prev_High'] - df['Prev_Low']) / df['Prev_Close']
        df['Body_Ratio'] = (df['Prev_Close'] - df['Prev_Open']) / (
                (df['Prev_High'] - df['Prev_Low']) + 1e-9)
        df['Composite_HL'] = (df['Prev_Range'] + df['Body_Ratio']) / 2
        df['ATR_14'] = calculate_atr(df['High'], df['Low'], df['Close']).shift(1)
        st_k, st_d = calculate_stoch_kd(df['High'], df['Low'], df['Close'])
        df['Stoch_K'] = st_k.shift(1)
        df['Stoch_D'] = st_d.shift(1)

        #09:30–09:39 micro-features
        if ALP_OK:
            df = attach_intraday(df, tkr, ticker_splits)

        df['Delta_Close'] = df['Close'] - df['Price_0939']
        df['Prev_Delta_Close'] = df['Delta_Close'].shift(1)


        df = df[final_cols]

        df = df.dropna()
        out = f"../Data/{tkr}.csv"
        df.round(12).to_csv(out, index=False)
        print(f"✔️ Saved {tkr}")



def main(start="2016-01-04", end="2025-05-23"):
    global total_tickers
    os.makedirs("../Data", exist_ok=True)

    tickers, sectors, splits = load_or_fetch_ipo_and_sectors(batch_size=25)
    total_tickers    = len(tickers)

    START_DATE = pd.Timestamp(start, tz="US/Eastern")
    END_DATE = pd.Timestamp(end, tz="US/Eastern")

    #bulk_fetch_intraday(tickers, start=START_DATE, end=END_DATE, month_idx=None)
    #with ThreadPoolExecutor(max_workers=8) as pool:
    #    for tkr, sec in zip(tickers, sectors):
    #        pool.submit(build_one, tkr, sec, start, (pd.to_datetime(end) + pd.Timedelta(days=1)).strftime('%Y-%m-%d'), splits)
    # --- sequential processing for easier debugging ---
    for tkr, sec in zip(tickers, sectors):
        build_one(tkr, sec, start, (pd.to_datetime(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d"), splits)

if __name__ == "__main__":
    main()

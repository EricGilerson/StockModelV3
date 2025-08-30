import os
import pickle
from datetime import datetime, timedelta

import pandas as pd
import tensorflow as tf
import yfinance as yf
import h5py
from tensorflow.keras.models import load_model
from matplotlib import cm
from scipy.signal import savgol_filter
from scipy.stats import zscore
from tensorflow.python.keras.models import model_from_json
import numpy

from CloudScripts.config import *
from CloudScripts.functions import *

def loadModel(filename):
    custom_objects = {
        'PositionalEncoding': PositionalEncoding,
        'RecentTrendExtractor': RecentTrendExtractor,
        'CLSToken': CLSToken,
        'tf': tf,
        'Sign': Sign
    }
    model = load_model(filename, custom_objects=custom_objects, compile=False, safe_mode=False)
    return model


def loadModelArch(arch_filename, weights_filename):
    # Load the architecture from JSON
    with open(arch_filename, 'r') as json_file:
        json_config = json_file.read()
    custom_objects = {
        'PositionalEncoding': PositionalEncoding,
        'RecentTrendExtractor': RecentTrendExtractor,
        'CLSToken': CLSToken,
    }
    model = model_from_json(json_config, custom_objects=custom_objects)

    # Load the weights
    model.load_weights(weights_filename)
    return model
def load_individual_prepared_data(filename):
    with h5py.File(filename, "r") as f:
        X = f["sequences"][:]
        y = f["targets"][:]
    return X, y

def getInputs(X, ticker, back=-1):
    with open("../model/Data/stock_to_id.pkl", "rb") as f:
        stock_to_id = pickle.load(f)
    with open("../model/Data/sector_to_id.pkl", "rb") as f:
        sector_to_id = pickle.load(f)

    sectors_df = pd.read_csv("../Data/sectors.csv")
    ticker_to_sector_id = {
        ticker: sector_to_id[sectors_df.loc[sectors_df['Ticker'] == ticker, 'Sector'].iloc[0]]
        for ticker in stock_to_id.keys()
    }
    sector_id = ticker_to_sector_id[ticker]
    if abs(back) > len(X):
        raise IndexError(f"Back index {back} out of bounds for ticker {ticker} with {len(X)} samples")

    latest_sequence = X[back]
    sequence = latest_sequence.reshape(1, latest_sequence.shape[-2], latest_sequence.shape[-1])

    sectorid = np.array(sector_id).reshape(1, 1)

    return sequence, sectorid


def unScale(predictions, target, scalers, *, percent=False):
    """
    predictions -> [price0939_scaled, pred_delta_scaled]
    target      -> [true_delta_scaled]
    scalers     -> {"price": RobustScaler, "delta": RobustScaler, ...}
    percent     -> True  if delta is a % move (Close / 09:39 – 1)
                   False if delta is an absolute difference

    Returns unscaled price_0939, pred_close, true_close
    """

    price_s, delta_pred_s = predictions
    price_s       = np.asarray(price_s,       dtype=np.float32)
    delta_pred_s  = np.asarray(delta_pred_s,  dtype=np.float32)
    delta_true_s  = np.asarray(target,        dtype=np.float32)

    # inverse-scale using the scalers you saved per-ticker
    price_scaler  = scalers["price"]         # GroupScaler
    delta_scaler  = scalers["delta_price"]   # GroupScaler

    price0939 = price_scaler.inverse_transform(
                     price_s.reshape(-1, 1)
               )[:, 0]
    delta_pred = delta_scaler.inverse_transform(
                     delta_pred_s.reshape(-1, 1)
                 )[:, 0]
    delta_true = delta_scaler.inverse_transform(
                     delta_true_s.reshape(-1, 1)
                 )[:, 0]

    # rebuild closes
    if percent:
        pred_close = price0939 * (1.0 + delta_pred)
        true_close = price0939 * (1.0 + delta_true)
    else:
        pred_close = price0939 + delta_pred
        true_close = price0939 + delta_true

    return price0939, pred_close, true_close


def directional_accuracy(opens, predictions, closes):
    correct = 0
    total = len(predictions)
    for o, p, c in zip(opens, predictions, closes):
        actual_direction = c - o
        if (p > 0 and actual_direction > 0) or (p < 0 and actual_direction < 0):
            correct += 1
    return correct / total

def total_profit(opens, predictions, closes):
    profit = 0
    for o, p, c in zip(opens, predictions, closes):
        profit += buyShortTrading(o, p, c)
    return profit

def buyShortTrading(open, prediction, close, shares=None, ticker=None, day=None, stopLoss=False, buy=BUDGET_PER_TRADE):

    if shares is None:
        if open > buy:
            return 0.0
        shares = int(buy // open)
    # direction: 1=long, -1=short
    direction = 1 if prediction >= open else -1

    entry_price = open * (1 + SLIPPAGE_PCT * direction)
    exit_price = close * (1 - SLIPPAGE_PCT * direction)

    # gross P/L then subtract commissions
    profit = (exit_price - entry_price) * shares * direction
    profit -= 2 * COMMISSION_PER_TRADE

    # optional 2% stop-loss
    if stopLoss and ticker and day:
        try:
            day_dt = datetime.strptime(day, "%Y-%m-%d")
            nxt_dt = day_dt + timedelta(days=1)
            data = yf.download(ticker,
                               start=day_dt.strftime("%Y-%m-%d"),
                               end=nxt_dt.strftime("%Y-%m-%d"),
                               progress=False)
            if not data.empty:
                low = data['Low'].iloc[0]
                high = data['High'].iloc[0]
                stop_pct = 0.02

                if direction == 1:
                    stop_price = open * (1 - stop_pct)
                    # if price dips below stop, exit there
                    if low < stop_price:
                        exit_sl = stop_price * (1 - SLIPPAGE_PCT)
                        profit = (exit_sl - entry_price) * shares
                        profit -= 2 * COMMISSION_PER_TRADE

                else:  # short
                    stop_price = open * (1 + stop_pct)
                    if high > stop_price:
                        exit_sl = stop_price * (1 + SLIPPAGE_PCT)
                        profit = (entry_price - exit_sl) * shares
                        profit -= 2 * COMMISSION_PER_TRADE

        except Exception as e:
            print(f"Error fetching Yahoo data for stop-loss: {e}")

    return profit

def win_loss_ratio(profits):
    wins, losses = 0, 0
    for profit in profits:
        if profit > 0:
            wins += 1
        elif profit < 0:
            losses += 1
    if losses == 0:
        return float('inf')
    return wins / losses

import numpy as np

def sharpe_ratio(profits):
    if len(profits) < 5:
        return 0
    avg_return = np.mean(profits)
    return_std = np.std(profits)
    if return_std == 0:
        return 0
    sharpe = avg_return / return_std
    return sharpe

def profit_factor(profits):
    gains, losses = 0, 0
    for profit in profits:
        if profit > 0:
            gains += profit
        elif profit < 0:
            losses += abs(profit)
    if losses == 0:
        return float('inf')
    return gains / losses

import matplotlib.pyplot as plt

def plot_cumulative_returns(profits, title=None):
    cumulative_profits = []
    cumulative = 0
    if title is None:
        title = 'Cumulative Returns Over Time'
    for profit in profits:
        cumulative += profit
        cumulative_profits.append(cumulative)

    plt.plot(cumulative_profits)
    plt.title(title)
    plt.xlabel('Trade Number')
    plt.ylabel('Cumulative Profit ($)')
    plt.grid(True)
    plt.show()


def biggest_winners_losers(results, top_n=10):

    if isinstance(results, pd.DataFrame):
        # Sort DataFrame based on 'Profit' column
        sorted_results = results.sort_values(by="Profit", ascending=False)

        # Get top N winners and bottom N losers
        biggest_winners = sorted_results.head(top_n)
        biggest_losers = sorted_results.tail(top_n)

        return biggest_winners, biggest_losers
    else:
        raise TypeError("Expected a Pandas DataFrame as input")

def get_top_percent_difference(n: int, df: pd.DataFrame) -> pd.DataFrame:
    df_copy = df.copy()

    df_copy["Percent_Difference"] = (abs(df_copy["Prediction"] - df_copy["Open"]) /
                                     ((df_copy["Prediction"] + df_copy["Open"]) / 2)) * 100

    sorted_df = df_copy.sort_values(by="Percent_Difference", ascending=False)

    return sorted_df.head(n)

def get_top_variance(n: int, df: pd.DataFrame) -> pd.DataFrame:
    df_copy = df.copy()

    sorted_df = df_copy.sort_values(by="Confidence", ascending=True)

    return sorted_df.head(n)


def sortino_ratio(returns, target_return=0.0):
    downside = np.clip(target_return - np.array(returns), 0, None)
    if len(returns) < 2 or np.std(downside) == 0:
        return 0
    return (np.mean(returns) - target_return) / np.std(downside)

def rolling_sortino(profits, window=25):
    return savgol_filter([
        np.nan if i < window else sortino_ratio(profits[i-window:i])
        for i in range(len(profits))
    ], 11, 2)

def rolling_hit_rate(profits, window=25):
    return savgol_filter([
        np.nan if i < window else np.mean(np.array(profits[i-window:i]) > 0)
        for i in range(len(profits))
    ], 11, 2)

# --- Main Visualization ---
def plot_trade_signal_dashboard(df, title='Trade Signal Dashboard', sort_ascending=True):
    # Sort by confidence
    sorted_df = df.sort_values(by="Confidence", ascending=sort_ascending).reset_index(drop=True)
    profits = sorted_df["Profit"].values
    confidence = sorted_df["Confidence"].values
    trades = np.arange(len(profits))
    cumulative_profits = np.cumsum(profits)

    # Derivatives for signal arrows
    delta = np.gradient(cumulative_profits)
    smooth_delta = savgol_filter(delta, 11, 2)
    accel = np.gradient(smooth_delta)

    # Metrics
    hit_rate_vals = rolling_hit_rate(profits, window=10)

    # Normalize confidence for color mapping
    norm_conf = (confidence - confidence.min()) / (confidence.max() - confidence.min())
    colors = cm.viridis(norm_conf)

    fig, axs = plt.subplots(3, 1, figsize=(14, 12), gridspec_kw={'height_ratios': [3, 1, 1]})
    fig.suptitle(title, fontsize=16, fontweight='bold')

    # Plot 1: Cumulative Profit
    ax1 = axs[0]
    ax1.plot(cumulative_profits, label='Cumulative Profit ($)', color='tab:blue', linewidth=2)
    ax1.scatter(trades, cumulative_profits, c=colors, s=15, label='Confidence Gradient')
    ax1.set_ylabel('Cumulative Profit ($)', fontsize=12)
    ax1.set_xlabel('Trade Rank (by Confidence)', fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.4)

    # Inflection points (2nd derivative threshold)
    inflection_idx = np.where(np.abs(accel) > np.std(accel) * 1.5)[0]
    for i in inflection_idx:
        ax1.annotate('↑' if accel[i] > 0 else '↓',
                     xy=(trades[i], cumulative_profits[i]),
                     textcoords="offset points",
                     xytext=(0, -15 if accel[i] > 0 else 10),
                     ha='center',
                     fontsize=10,
                     color='green' if accel[i] > 0 else 'red')

    lines1, labels1 = ax1.get_legend_handles_labels()
    ax1.legend(lines1, labels1, loc='upper left', fontsize=10)

    # Plot 2: Profit per Trade
    ax2 = axs[1]
    ax2.plot(trades, profits, color='tab:gray', label='Profit per Trade', alpha=0.6)
    ax2.axhline(0, color='black', linestyle='--', linewidth=0.8)
    ax2.set_ylabel('Profit', fontsize=11)
    ax2.set_xlabel('Trade Rank', fontsize=11)
    ax2.grid(True, linestyle=':', alpha=0.3)
    ax2.legend(loc='upper right', fontsize=9)

    # Plot 3: Rolling Hit Rate
    ax3 = axs[2]
    ax3.plot(hit_rate_vals, label='Rolling Hit Rate', color='tab:green', linewidth=2)
    ax3.set_ylabel('Hit Rate', fontsize=11)
    ax3.set_xlabel('Trade Rank', fontsize=11)
    ax3.grid(True, linestyle='--', alpha=0.3)
    ax3.legend(loc='upper left', fontsize=10)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()



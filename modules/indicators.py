from __future__ import annotations

import numpy as np
import pandas as pd


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    if "Close" not in out:
        out["Close"] = 0.0
    if "High" not in out:
        out["High"] = out["Close"]
    if "Low" not in out:
        out["Low"] = out["Close"]
    if "Open" not in out:
        out["Open"] = out["Close"]
    close = pd.to_numeric(out["Close"], errors="coerce").ffill().fillna(0.0)
    high = pd.to_numeric(out["High"], errors="coerce").fillna(close)
    low = pd.to_numeric(out["Low"], errors="coerce").fillna(close)

    out["MA20"] = close.rolling(20).mean()
    out["EMA50"] = close.ewm(span=50, adjust=False).mean()
    out["EMA200"] = close.ewm(span=200, adjust=False).mean()

    std = close.rolling(20).std()
    out["BB_MID"] = out["MA20"]
    out["BB_UPPER"] = out["MA20"] + 2 * std
    out["BB_LOWER"] = out["MA20"] - 2 * std
    out["BB_WIDTH"] = ((out["BB_UPPER"] - out["BB_LOWER"]) / out["BB_MID"].replace(0, np.nan) * 100).fillna(0.0)

    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    out["RSI14"] = 100 - (100 / (1 + rs))

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    out["MACD"] = ema12 - ema26
    out["MACD_SIGNAL"] = out["MACD"].ewm(span=9, adjust=False).mean()
    out["MACD_HIST"] = out["MACD"] - out["MACD_SIGNAL"]

    prev_close = close.shift(1)
    tr = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    out["ATR14"] = tr.rolling(14).mean()
    out["ATR_PCT"] = (out["ATR14"] / close.replace(0, np.nan) * 100).fillna(0.0)
    out["CANDLE_RANGE"] = high - low
    out["CANDLE_RANGE_PCT"] = (out["CANDLE_RANGE"] / close.replace(0, np.nan) * 100).fillna(0.0)
    bb_rising = out["BB_WIDTH"].diff().rolling(3).sum().fillna(0.0) > 0
    atr_rising = out["ATR14"].diff().rolling(3).sum().fillna(0.0) > 0
    out["BB_EXPANDING"] = ((out["BB_WIDTH"] > out["BB_WIDTH"].rolling(20).mean() * 1.1) | bb_rising).fillna(False)
    out["ATR_EXPANDING"] = ((out["ATR14"] > out["ATR14"].rolling(20).mean() * 1.1) | atr_rising).fillna(False)
    out["CANDLE_EXPANSION"] = (out["CANDLE_RANGE"] > out["CANDLE_RANGE"].rolling(20).mean() * 1.5).fillna(False)

    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    atr = tr.rolling(14).mean()
    plus_di = 100 * pd.Series(plus_dm, index=out.index).rolling(14).mean() / atr
    minus_di = 100 * pd.Series(minus_dm, index=out.index).rolling(14).mean() / atr
    dx = ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)) * 100
    out["ADX14"] = dx.rolling(14).mean()

    if "Volume" in out:
        vol_ma = out["Volume"].rolling(20).mean()
        out["VOLUME_MA20"] = vol_ma
        out["VOLUME_SPIKE"] = out["Volume"] > vol_ma * 1.5
    else:
        out["VOLUME_SPIKE"] = False

    out["HIGHER_HIGH"] = out["High"] > out["High"].shift(1).rolling(5).max()
    out["LOWER_LOW"] = out["Low"] < out["Low"].shift(1).rolling(5).min()
    out["BOS_UP"] = (out["Close"] > out["High"].shift(1).rolling(10).max()) & out["VOLUME_SPIKE"]
    out["BOS_DOWN"] = (out["Close"] < out["Low"].shift(1).rolling(10).min()) & out["VOLUME_SPIKE"]
    out["LIQUIDITY_SWEEP_UP"] = (out["High"] > out["High"].shift(1).rolling(10).max()) & (out["Close"] < out["Open"])
    out["LIQUIDITY_SWEEP_DOWN"] = (out["Low"] < out["Low"].shift(1).rolling(10).min()) & (out["Close"] > out["Open"])
    return out


def support_resistance(df: pd.DataFrame, window: int = 50) -> dict:
    if df.empty:
        return {"support": None, "resistance": None}
    recent = df.tail(window)
    return {"support": float(recent["Low"].min()), "resistance": float(recent["High"].max())}


def candle_reject(df: pd.DataFrame) -> str:
    if df.empty:
        return "Chưa đủ dữ liệu."
    row = df.iloc[-1]
    body = abs(row["Close"] - row["Open"])
    upper = row["High"] - max(row["Close"], row["Open"])
    lower = min(row["Close"], row["Open"]) - row["Low"]
    if upper > body * 2 and row["Close"] < row["Open"]:
        return "Nến có râu trên dài, dấu hiệu bị từ chối ở vùng giá cao."
    if lower > body * 2 and row["Close"] > row["Open"]:
        return "Nến có râu dưới dài, dấu hiệu lực mua đỡ giá."
    return "Chưa có mẫu hình từ chối giá rõ ràng."

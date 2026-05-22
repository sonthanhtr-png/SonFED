from __future__ import annotations

from datetime import datetime
from typing import Dict

import pandas as pd
import streamlit as st
import yfinance as yf


INTERVAL_MAP = {
    "M15": "15m",
    "H1": "1h",
    "H4": "4h",
    "D1": "1d",
    "15m": "15m",
    "1h": "1h",
    "4h": "4h",
    "1d": "1d",
}


@st.cache_data(ttl=180, show_spinner=False)
def fetch_ohlcv(ticker: str, period: str = "6mo", interval: str = "1h") -> pd.DataFrame:
    """Lấy dữ liệu thị trường qua yfinance và chuẩn hóa cột."""
    try:
        yf_interval = INTERVAL_MAP.get(interval, interval)
        data = yf.download(ticker, period=period, interval=yf_interval, progress=False, auto_adjust=False)
        if data is None or data.empty:
            return pd.DataFrame()
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        data = data.rename(columns=str.title)
        keep = [c for c in ["Open", "High", "Low", "Close", "Adj Close", "Volume"] if c in data.columns]
        data = data[keep].copy()
        data.index = pd.to_datetime(data.index)
        data = data.dropna(subset=["Close"])
        return data
    except Exception:
        return pd.DataFrame()


def fetch_market_bundle(tickers: Dict[str, str], period: str, interval: str) -> Dict[str, pd.DataFrame]:
    return {name: fetch_ohlcv(ticker, period, interval) for name, ticker in tickers.items()}


def data_status(bundle: Dict[str, pd.DataFrame]) -> dict:
    ok = [k for k, v in bundle.items() if not v.empty]
    missing = [k for k, v in bundle.items() if v.empty]
    return {"ok": ok, "missing": missing, "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

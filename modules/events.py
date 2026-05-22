from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from .utils import ROOT


IMPORTANT_EVENTS = {"CPI", "Core CPI", "PPI", "NFP", "Nonfarm", "FOMC", "Powell Speech", "Core PCE", "PCE"}


def load_events(path: str = "events.csv") -> pd.DataFrame:
    p = ROOT / path
    if not p.exists():
        return pd.DataFrame(columns=["time", "event", "impact", "note"])
    try:
        df = pd.read_csv(p)
        df["time"] = pd.to_datetime(df["time"], errors="coerce")
        return df.dropna(subset=["time"])
    except Exception:
        return pd.DataFrame(columns=["time", "event", "impact", "note"])


def event_risk(df: pd.DataFrame, minutes_before: int = 60, minutes_after: int = 30) -> dict:
    now = datetime.now()
    if df.empty:
        return {"blocked": False, "message": "Không có lịch tin.", "events": []}
    start = now - timedelta(minutes=minutes_after)
    end = now + timedelta(minutes=minutes_before)
    mask = (df["time"] >= start) & (df["time"] <= end) & (df["event"].isin(IMPORTANT_EVENTS))
    rows = df.loc[mask].sort_values("time")
    if rows.empty:
        return {"blocked": False, "message": "Không có tin lớn gần thời điểm hiện tại.", "events": []}
    names = ", ".join(rows["event"].astype(str).tolist())
    return {"blocked": True, "message": f"Sắp hoặc vừa có tin lớn: {names}. Nên né giao dịch.", "events": rows.to_dict("records")}

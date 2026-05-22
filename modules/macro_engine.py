from __future__ import annotations

from typing import Dict

import pandas as pd

from .indicators import add_indicators
from .utils import latest_float, pct_change


def pressure_index(bundle: Dict[str, pd.DataFrame]) -> dict:
    points = 0
    details = []

    weights = {
        "DXY": 20,
        "US10Y": 25,
        "US02Y": 10,
        "VIX": 15,
        "OIL": 10,
    }
    for key, weight in weights.items():
        change = pct_change(bundle.get(key, pd.DataFrame()).get("Close", pd.Series(dtype=float)))
        if change > 0:
            points += weight
            details.append(f"{key} tăng: +{weight}")
        else:
            details.append(f"{key} không tăng: +0")

    nasdaq_change = pct_change(bundle.get("NASDAQ", pd.DataFrame()).get("Close", pd.Series(dtype=float)))
    if nasdaq_change < 0:
        points += 10
        details.append("Nasdaq giảm: +10")

    gold = bundle.get("GOLD", pd.DataFrame())
    if not gold.empty:
        g = add_indicators(gold)
        last = g.iloc[-1]
        if latest_float(g["Close"]) < latest_float(g["MA20"]):
            points += 10
            details.append("Vàng dưới MA20: +10")

    points = int(max(0, min(100, points)))
    if points <= 30:
        text = "Môi trường thiên về nới lỏng, hỗ trợ vàng."
    elif points <= 60:
        text = "Môi trường trung tính."
    elif points <= 80:
        text = "Môi trường nghiêng về FED thắt chặt, vàng chịu áp lực."
    else:
        text = "Rủi ro cao, thị trường biến động mạnh."
    dxy_change = pct_change(bundle.get("DXY", pd.DataFrame()).get("Close", pd.Series(dtype=float)))
    return {"score": points, "interpretation": text, "details": details, "dxy_change": dxy_change}


def gold_bias(bundle: Dict[str, pd.DataFrame]) -> str:
    dxy = pct_change(bundle.get("DXY", pd.DataFrame()).get("Close", pd.Series(dtype=float)))
    y10 = pct_change(bundle.get("US10Y", pd.DataFrame()).get("Close", pd.Series(dtype=float)))
    vix = pct_change(bundle.get("VIX", pd.DataFrame()).get("Close", pd.Series(dtype=float)))

    if dxy > 0 and y10 > 0:
        base = "DXY và lợi suất 10 năm cùng tăng, bất lợi cho vàng."
    elif dxy < 0 and y10 < 0:
        base = "DXY và lợi suất 10 năm cùng giảm, có lợi cho vàng."
    else:
        base = "Vàng đang chịu hai lực kéo ngược nhau."
    if vix > 1:
        base += " VIX tăng mạnh nên vàng có thể được hỗ trợ bởi nhu cầu trú ẩn."
    return base

from __future__ import annotations

import os
from typing import Dict

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

FRED_SERIES = {
    "CPI": "CPIAUCSL",
    "Core CPI": "CPILFESL",
    "PCE": "PCEPI",
    "Fed Funds Rate": "FEDFUNDS",
}


def fetch_fred_latest() -> Dict[str, dict]:
    api_key = os.getenv("FRED_API_KEY", "")
    if not api_key:
        return {"enabled": False, "message": "Chưa cấu hình FRED_API_KEY.", "data": {}}
    results: Dict[str, dict] = {}
    for name, series_id in FRED_SERIES.items():
        try:
            resp = requests.get(
                "https://api.stlouisfed.org/fred/series/observations",
                params={
                    "series_id": series_id,
                    "api_key": api_key,
                    "file_type": "json",
                    "sort_order": "desc",
                    "limit": 1,
                },
                timeout=10,
            )
            resp.raise_for_status()
            observations = resp.json().get("observations", [])
            if observations:
                value = observations[0].get("value")
                results[name] = {"date": observations[0].get("date"), "value": float(value)}
        except Exception:
            results[name] = {"date": None, "value": None}
    return {"enabled": True, "message": "Đã kết nối FRED.", "data": results}


def fred_to_frame(payload: Dict[str, dict]) -> pd.DataFrame:
    rows = []
    for name, item in payload.get("data", {}).items():
        rows.append({"Chỉ số": name, "Ngày": item.get("date"), "Giá trị": item.get("value")})
    return pd.DataFrame(rows)

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def load_json(path: str | Path, default: Any) -> Any:
    try:
        p = Path(path)
        if not p.is_absolute():
            p = ROOT / p
        if not p.exists():
            return default
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return default


def save_json(path: str | Path, payload: Any) -> None:
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def resolve_shared_dir(config: dict) -> Path:
    configured = Path(config.get("paths", {}).get("shared_dir", "C:\\SonFED\\shared"))
    try:
        configured.mkdir(parents=True, exist_ok=True)
        return configured
    except Exception:
        fallback = ROOT / config.get("paths", {}).get("fallback_shared_dir", "shared")
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback


def pct_change(series) -> float:
    try:
        clean = series.dropna()
        if len(clean) < 2 or clean.iloc[-2] == 0:
            return 0.0
        return float((clean.iloc[-1] - clean.iloc[-2]) / abs(clean.iloc[-2]) * 100)
    except Exception:
        return 0.0


def latest_float(series, default: float = 0.0) -> float:
    try:
        clean = series.dropna()
        if clean.empty:
            return default
        return float(clean.iloc[-1])
    except Exception:
        return default

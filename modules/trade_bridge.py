from __future__ import annotations

from pathlib import Path

from .utils import load_json, save_json


def read_shared(shared_dir: Path) -> dict:
    return {
        "signal": load_json(shared_dir / "signal.json", {}),
        "trade_status": load_json(shared_dir / "trade_status.json", {}),
        "risk_status": load_json(shared_dir / "risk_status.json", {}),
        "bot_log": load_json(shared_dir / "bot_log.json", []),
    }


def write_bot_log(shared_dir: Path, message: str) -> None:
    log = load_json(shared_dir / "bot_log.json", [])
    if not isinstance(log, list):
        log = []
    log.append(message)
    save_json(shared_dir / "bot_log.json", log[-200:])

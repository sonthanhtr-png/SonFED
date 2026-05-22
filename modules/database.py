from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from .utils import ROOT


def connect(db_path: str = "data/sonfed.db") -> sqlite3.Connection:
    p = ROOT / db_path
    p.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(p)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            symbol TEXT,
            action TEXT,
            confidence INTEGER,
            payload TEXT
        )
        """
    )
    conn.commit()
    return conn


def log_signal(signal: dict) -> None:
    conn = connect()
    try:
        conn.execute(
            "INSERT INTO signals(created_at, symbol, action, confidence, payload) VALUES (?, ?, ?, ?, ?)",
            (
                signal.get("time"),
                signal.get("symbol"),
                signal.get("action"),
                signal.get("confidence"),
                json.dumps(signal, ensure_ascii=False),
            ),
        )
        conn.commit()
    finally:
        conn.close()


def recent_signals(limit: int = 50):
    conn = connect()
    try:
        return conn.execute(
            "SELECT created_at, symbol, action, confidence, payload FROM signals ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
    finally:
        conn.close()

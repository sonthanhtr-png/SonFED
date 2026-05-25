from __future__ import annotations

from datetime import datetime
from typing import Any

import pandas as pd


BASE_CAPITAL = 200.0


def _num(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _side(position: dict[str, Any]) -> str:
    return str(position.get("type_name", position.get("type", "NONE")) or "NONE").upper()


def position_age_minutes(position: dict[str, Any]) -> int:
    opened = position.get("open_time") or position.get("time")
    if not opened:
        return 0
    try:
        dt = pd.to_datetime(opened, errors="coerce").to_pydatetime()
    except Exception:
        return 0
    return max(0, int((datetime.now() - dt).total_seconds() // 60))


def position_totals(positions: list[dict[str, Any]]) -> dict[str, Any]:
    buy = [p for p in positions if _side(p) == "BUY"]
    sell = [p for p in positions if _side(p) == "SELL"]
    return {
        "buy_orders": len(buy),
        "sell_orders": len(sell),
        "buy_volume": round(sum(_num(p.get("lot", p.get("volume"))) for p in buy), 2),
        "sell_volume": round(sum(_num(p.get("lot", p.get("volume"))) for p in sell), 2),
        "floating_pnl": round(sum(_num(p.get("profit")) for p in positions), 2),
    }


def enrich_positions(positions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for pos in positions:
        rows.append(
            {
                "ticket": pos.get("ticket"),
                "symbol": pos.get("symbol", "XAUUSD"),
                "type": _side(pos),
                "lot": _num(pos.get("lot", pos.get("volume"))),
                "entry": _num(pos.get("entry", pos.get("price_open"))),
                "current": _num(pos.get("current_price", pos.get("price_current"))),
                "profit": _num(pos.get("profit")),
                "sl": _num(pos.get("sl")),
                "tp": _num(pos.get("tp")),
                "age_minutes": position_age_minutes(pos),
            }
        )
    return rows


def account_statistics(account: dict[str, Any], positions: list[dict[str, Any]], base_capital: float = BASE_CAPITAL) -> dict[str, Any]:
    balance = _num(account.get("balance"))
    equity = _num(account.get("equity"), balance)
    profit_percent = ((balance - base_capital) / base_capital * 100) if base_capital > 0 and balance else 0.0
    return {
        "balance": round(balance, 2),
        "equity": round(equity, 2),
        "drawdown_percent": round(_num(account.get("drawdown_percent")), 2),
        "profit_percent": round(profit_percent, 2),
        **position_totals(positions),
    }

from __future__ import annotations

from datetime import datetime
from typing import Any


def normalize_trade_status(payload: dict[str, Any] | None) -> dict[str, Any]:
    if not payload:
        return {
            "connected": False,
            "message": "Chưa nhận được trạng thái lệnh từ SonEXEC.",
            "updated_at": None,
            "account": {},
            "positions": [],
        }

    positions = payload.get("positions")
    if isinstance(positions, list):
        normalized_positions = positions
    else:
        # Tương thích file trade_status.json cũ chỉ có một vị thế tổng quát.
        position = payload.get("position", "NONE")
        normalized_positions = []
        if position and position != "NONE":
            normalized_positions.append({
                "ticket": payload.get("ticket", 0),
                "symbol": payload.get("symbol", "XAUUSD"),
                "type": position,
                "lot": payload.get("lot", 0),
                "entry": payload.get("entry_price", payload.get("entry", 0)),
                "current_price": payload.get("current_price", 0),
                "sl": payload.get("sl"),
                "tp": payload.get("tp"),
                "profit": payload.get("profit", 0),
                "open_time": payload.get("open_time"),
                "trailing_enabled": payload.get("trailing_enabled", False),
                "trailing_mode": payload.get("trailing_mode", ""),
                "trailing_distance": payload.get("trailing_distance", 0),
            })

    account = payload.get("account", {})
    if not account:
        account = {
            "drawdown_percent": payload.get("drawdown_percent", 0),
            "equity": payload.get("equity"),
            "balance": payload.get("balance"),
            "free_margin": payload.get("free_margin"),
        }

    return {
        "connected": True,
        "message": "Đã nhận trạng thái lệnh từ SonEXEC.",
        "updated_at": payload.get("updated_at", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
        "account": account,
        "positions": normalized_positions,
        "raw": payload,
    }


def position_table_rows(feedback: dict[str, Any], adjustments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_ticket = {str(item.get("ticket")): item for item in adjustments}
    rows = []
    for pos in feedback.get("positions", []):
        adj = by_ticket.get(str(pos.get("ticket")), {})
        rows.append({
            "Ticket": pos.get("ticket"),
            "Symbol": pos.get("symbol"),
            "Loại": pos.get("type"),
            "Lot": pos.get("lot"),
            "Entry": pos.get("entry"),
            "Giá hiện tại": pos.get("current_price"),
            "SL": pos.get("sl"),
            "TP": pos.get("tp"),
            "Lãi/lỗ": pos.get("profit"),
            "Trailing": "Bật" if pos.get("trailing_enabled") else "Tắt",
            "Đánh giá SonFED": adj.get("reason", "Chưa có đánh giá riêng cho lệnh này."),
            "Đề xuất": adj.get("action", "HOLD_POSITION"),
            "Độ tin cậy": adj.get("ai_confidence", adj.get("confidence", 0)),
        })
    return rows

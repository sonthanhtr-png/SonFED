from __future__ import annotations

from typing import Any


def normalize_risk_status(payload: dict[str, Any] | None) -> dict[str, Any]:
    if not payload:
        return {
            "connected": False,
            "message": "Chưa nhận được trạng thái rủi ro từ SonEXEC.",
            "spread_points": None,
            "allow": False,
            "reason": "Chưa kiểm tra",
        }
    # SonEXEC ghi format mới: {trade_locked: bool, reasons: list[str]}
    # Format cũ (mt5_trade_bot): {allow: bool, reason: str}
    if "trade_locked" in payload:
        locked = bool(payload.get("trade_locked", False))
        reasons = payload.get("reasons", [])
        allow = not locked
        reason = "; ".join(reasons) if reasons else (
            "Đủ điều kiện giao dịch." if allow else "Giao dịch đang bị khóa."
        )
    else:
        allow = bool(payload.get("allow", False))
        reason = payload.get("reason", "")
    return {
        "connected": True,
        "message": reason,
        "spread_points": payload.get("spread_points"),
        "allow": allow,
        "reason": reason,
        "raw": payload,
    }

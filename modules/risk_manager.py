from __future__ import annotations

from datetime import datetime


def assess_auto_trade(signal: dict, config: dict, event_risk: dict, trade_status: dict | None = None, risk_status: dict | None = None) -> dict:
    trade_cfg = config.get("trade", {})
    reasons = []
    allowed = True

    min_confidence = min(trade_cfg.get("min_confidence", 70), 50) if signal.get("scalp_accepted") else trade_cfg.get("min_confidence", 70)
    if signal.get("confidence", 0) < min_confidence:
        allowed = False
        reasons.append("Độ tin cậy dưới ngưỡng cho phép.")
    if not signal.get("allow_auto_trade", False):
        allowed = False
        reasons.append("Tín hiệu không cho phép tự động giao dịch.")
    if event_risk.get("blocked"):
        allowed = False
        reasons.append(event_risk.get("message", "Có tin lớn gần thời điểm hiện tại."))
    if signal.get("conflict", False):
        allowed = False
        reasons.append("Tín hiệu vĩ mô và kỹ thuật đang mâu thuẫn.")

    status = trade_status or {}
    if status.get("drawdown_percent", 0) > trade_cfg.get("max_drawdown_percent", 5):
        allowed = False
        reasons.append("Drawdown vượt giới hạn an toàn.")
    if status.get("open_positions", 0) > trade_cfg.get("max_open_positions", 3):
        allowed = False
        reasons.append("Số lệnh đang mở vượt giới hạn.")

    risk = risk_status or {}
    if risk.get("spread_points", 0) > trade_cfg.get("max_spread_points", 35):
        allowed = False
        reasons.append("Spread đang cao.")

    if signal.get("action") == "WAIT":
        allowed = False
        reasons.append("Tín hiệu hiện tại là chờ.")

    return {
        "allow_auto_trade": allowed,
        "reasons": reasons or ["Đủ điều kiện an toàn."],
        "checked_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

from __future__ import annotations


def smart_alerts(gold_analysis: dict, macro: dict, mtf: dict, event_risk: dict) -> list[str]:
    alerts = []
    text = " ".join(gold_analysis.get("items", []))
    if "dải dưới Bollinger" in text:
        alerts.append("Không nên SELL đuổi sát BB dưới.")
    if "dải trên Bollinger" in text:
        alerts.append("Không nên BUY đuổi quá xa MA20.")
    if "lệch pha" in mtf.get("summary", ""):
        alerts.append("M15, H1 và H4 đang lệch pha, giảm khối lượng hoặc chờ xác nhận.")
    if event_risk.get("blocked"):
        alerts.append(event_risk.get("message", "Có tin lớn, nên né giao dịch."))
    if macro.get("score", 50) >= 81:
        alerts.append("Rủi ro cao, thị trường có thể biến động mạnh.")
    return alerts or ["Chưa có cảnh báo rủi ro đặc biệt."]

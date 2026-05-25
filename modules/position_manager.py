from __future__ import annotations


def _safe_float(value, default=0.0):
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def summarize_position(trade_status: dict, latest_context: dict | None = None) -> dict:
    if not trade_status:
        return {
            "summary": "Chưa có trạng thái lệnh từ bot MT5.",
            "position": "NONE",
            "open_positions": 0,
            "drawdown_percent": 0,
            "floating_profit": 0,
        }
    position = trade_status.get("position", "NONE")
    profit = _safe_float(trade_status.get("profit"))
    drawdown = _safe_float(trade_status.get("drawdown_percent"))
    open_positions = int(trade_status.get("open_positions", 1 if position not in {"NONE", ""} else 0))
    context = latest_context or {}
    pressure = context.get("pressure", 50)
    regime = context.get("regime", "")

    if position == "SELL" and ("tăng" in regime.lower() or pressure < 35):
        summary = "SELL đang gặp rủi ro. Không mở thêm SELL mới, ưu tiên chờ reject hoặc cân nhắc giảm vị thế."
    elif position == "BUY" and ("giảm" in regime.lower() or pressure > 65):
        summary = "BUY đang gặp rủi ro. Không mở thêm BUY mới, ưu tiên bảo toàn vốn."
    elif position in {"BUY", "SELL"}:
        summary = f"Lệnh {position} hiện tại chưa có cảnh báo đảo chiều nghiêm trọng."
    else:
        summary = "Không có vị thế mở."

    return {
        "summary": summary,
        "position": position,
        "open_positions": open_positions,
        "drawdown_percent": drawdown,
        "floating_profit": profit,
    }


def ai_trade_summary(trade_status: dict, gold_analysis: dict, macro: dict) -> str:
    position = trade_status.get("position", "NONE") if trade_status else "NONE"
    if position == "SELL":
        if "tăng" in gold_analysis.get("regime", "").lower():
            return "SELL hiện tại đang gặp áp lực do kỹ thuật ngắn hạn cải thiện. Ưu tiên chờ phản ứng tại kháng cự trước khi quyết định đóng hoặc giảm SELL."
        return "SELL hiện tại vẫn phù hợp hơn nếu H1 chưa đảo chiều tăng lớn. Không nên mở thêm khi giá sát hỗ trợ."
    if position == "BUY":
        if macro.get("score", 50) > 65:
            return "BUY hiện tại chịu áp lực từ môi trường FED thiên về thắt chặt. Cần giảm rủi ro nếu giá mất MA20."
        return "BUY hiện tại có thể tiếp tục giữ nếu giá còn trên MA20 và không có tin lớn gần thời điểm hiện tại."
    return "Chưa có vị thế mở, ưu tiên chờ tín hiệu rõ thay vì giao dịch liên tục."

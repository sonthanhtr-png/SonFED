from __future__ import annotations

import pandas as pd

from .indicators import add_indicators, candle_reject, support_resistance
from .market_regime_engine import detect_market_regime
from .utils import latest_float


def detect_regime(df: pd.DataFrame) -> str:
    return detect_market_regime(df).get("label", "Chưa rõ")


def build_ai_analysis_text(gold_analysis: dict, market_regime: dict, volatility: dict, macro: dict | None = None, mtf: dict | None = None) -> str:
    levels = gold_analysis.get("levels", {}) if isinstance(gold_analysis, dict) else {}
    resistance = levels.get("resistance")
    support = levels.get("support")
    macro = macro or {}
    mtf = mtf or {}
    macro_notes = []
    if macro.get("dxy_trend") or macro.get("dxy"):
        macro_notes.append(f"DXY: {macro.get('dxy_trend', macro.get('dxy'))}.")
    if macro.get("us10y_trend") or macro.get("us10y"):
        macro_notes.append(f"US10Y: {macro.get('us10y_trend', macro.get('us10y'))}.")
    if not macro_notes:
        macro_notes.append("Chưa có đủ dữ liệu macro, ưu tiên đọc technical và volatility.")

    bias = market_regime.get("bias_text", market_regime.get("bias", "WAIT"))
    if market_regime.get("bias") == "SELL" and resistance:
        invalidation = f"Nếu giá vượt lại vùng kháng cự {resistance:.2f} và giữ được phía trên, hủy bias SELL."
    elif market_regime.get("bias") == "BUY" and support:
        invalidation = f"Nếu giá thủng vùng hỗ trợ {support:.2f} và không hồi lại, hủy bias BUY."
    else:
        invalidation = "Nếu cấu trúc giá đổi hướng hoặc volatility mở rộng ngược kịch bản, chuyển về WAIT."

    risk_note = "Không nên vào full lot khi volatility tăng mạnh." if volatility.get("score", 0) >= 45 else "Có thể theo dõi tín hiệu nhưng vẫn cần quản trị SL/TP rõ ràng."
    mtf_summary = mtf.get("summary", "Chưa có xác nhận đa khung thời gian.")
    return "\n".join(
        [
            f"Volatility: {volatility.get('level', 'Chưa rõ')} ({volatility.get('score', 0)}/100).",
            " ".join(volatility.get("notes", [])) or "Chưa có ghi chú volatility rõ ràng.",
            f"Market regime: {market_regime.get('label', 'Chưa rõ')}.",
            f"Cấu trúc: {market_regime.get('structure', 'Chưa rõ')}. Momentum: {market_regime.get('momentum', 'Chưa rõ')}.",
            "Macro: " + " ".join(macro_notes),
            f"Đa khung thời gian: {mtf_summary}",
            f"Bias hiện tại: {bias} với xác suất {market_regime.get('probability', 0)}%.",
            f"Điều kiện hủy kịch bản: {invalidation}",
            f"Cảnh báo: {risk_note}",
        ]
    )


def analyze_gold(df: pd.DataFrame) -> dict:
    if df.empty:
        return {
            "summary": "Không có dữ liệu vàng.",
            "ai_analysis": "Chưa thể phân tích vì thiếu dữ liệu.",
            "items": [],
            "levels": {},
            "data": df,
            "regime": "Chưa đủ dữ liệu",
            "market_regime": detect_market_regime(df),
            "volatility": {},
        }

    data = add_indicators(df)
    last = data.iloc[-1]
    prev = data.iloc[-2] if len(data) > 1 else last
    regime = detect_market_regime(data)
    volatility = regime.get("volatility", {})
    items = []

    if last["Close"] < last["MA20"] and last["MA20"] < prev["MA20"]:
        items.append("Giá dưới MA20 và MA20 dốc xuống, nhịp ngắn hạn nghiêng giảm.")
    elif last["Close"] > last["MA20"] and last["MA20"] > prev["MA20"]:
        items.append("Giá trên MA20 và MA20 dốc lên, nhịp ngắn hạn đang cải thiện.")
    else:
        items.append("Giá đang dao động quanh MA20, hướng ngắn hạn chưa đủ rõ.")

    items.extend(volatility.get("notes", []))
    if volatility.get("score", 0) >= 45:
        items.append("Volatility đang mở rộng, không xem đây là sideway bình thường.")

    if last["Close"] <= last["BB_LOWER"]:
        items.append("Vàng chạm dải dưới Bollinger Bands, có thể hồi kỹ thuật nhưng cần nến xác nhận.")
    if last["Close"] >= last["BB_UPPER"]:
        items.append("Vàng sát dải trên Bollinger Bands, tránh BUY đuổi khi chưa có retest.")

    rsi = latest_float(data["RSI14"])
    if rsi < 30:
        items.append("RSI dưới 30, trạng thái quá bán.")
    elif rsi > 70:
        items.append("RSI trên 70, trạng thái quá mua.")
    else:
        items.append(f"RSI ở mức {rsi:.1f}, chưa vào vùng quá mua/quá bán.")

    if bool(last.get("BOS_UP", False)):
        items.append("Có breakout tăng kèm volume spike.")
    if bool(last.get("BOS_DOWN", False)):
        items.append("Có breakdown giảm kèm volume spike.")
    if bool(last.get("LIQUIDITY_SWEEP_UP", False)):
        items.append("Có dấu hiệu quét thanh khoản phía trên rồi bị đẩy xuống.")
    if bool(last.get("LIQUIDITY_SWEEP_DOWN", False)):
        items.append("Có dấu hiệu quét thanh khoản phía dưới rồi bật lên.")

    items.append(candle_reject(data))
    levels = support_resistance(data)
    summary = (
        f"Giá hiện tại {last['Close']:.2f}. "
        f"Trạng thái thị trường: {regime['label']}. "
        f"Volatility: {volatility.get('level', 'Chưa rõ')} ({volatility.get('score', 0)}/100). "
        f"Bias hiện tại: {regime.get('bias_text', 'WAIT')}. "
        f"Xác suất: {regime.get('probability', 0)}%. "
        f"Risk level: {regime.get('risk_level', 'Trung bình')}."
    )
    ai_analysis = build_ai_analysis_text({"levels": levels}, regime, volatility)
    return {
        "summary": summary,
        "ai_analysis": ai_analysis,
        "items": items,
        "levels": levels,
        "data": data,
        "regime": regime["label"],
        "market_regime": regime,
        "volatility": volatility,
    }


def multi_timeframe_summary(frames: dict[str, pd.DataFrame]) -> dict:
    trends = {}
    regimes = {}
    for name, df in frames.items():
        regime = detect_market_regime(df)
        trends[name] = regime.get("label", "Chưa rõ")
        regimes[name] = regime

    h4 = trends.get("H4", "")
    h1 = trends.get("H1", "")
    m15 = trends.get("M15", "")
    if "Bear" in h4 and "Bear" in h1 and "Bull" in m15:
        text = "Vàng đang hồi kỹ thuật trong xu hướng giảm lớn."
    elif "Bull" in h4 and "Bull" in h1 and "Bear" in m15:
        text = "Vàng đang điều chỉnh ngắn hạn trong xu hướng tăng lớn."
    elif "Volatile Range" in {m15, h1, h4}:
        text = "Có khung thời gian đang vào Volatile Range, tránh xem thị trường là sideway yếu."
    elif len(set(trends.values())) > 1:
        text = "Các khung thời gian đang lệch pha, nên chờ xác nhận rõ hơn."
    else:
        text = "Các khung thời gian tương đối đồng thuận."
    return {"trends": trends, "regimes": regimes, "summary": text}

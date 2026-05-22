from __future__ import annotations

import pandas as pd

from .indicators import add_indicators, candle_reject, support_resistance
from .utils import latest_float


def detect_regime(df: pd.DataFrame) -> str:
    if df.empty or len(df) < 60:
        return "Chưa đủ dữ liệu"
    data = add_indicators(df)
    last = data.iloc[-1]
    if last["ADX14"] > 25 and last["EMA50"] > last["EMA200"] and last["Close"] > last["MA20"]:
        return "Xu hướng tăng rõ"
    if last["ADX14"] > 25 and last["EMA50"] < last["EMA200"] and last["Close"] < last["MA20"]:
        return "Xu hướng giảm rõ"
    if last["ADX14"] < 20:
        return "Đi ngang"
    return "Chưa rõ xu hướng"


def analyze_gold(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"summary": "Không có dữ liệu vàng.", "items": [], "levels": {}, "data": df}
    data = add_indicators(df)
    last = data.iloc[-1]
    prev = data.iloc[-2] if len(data) > 1 else last
    items = []

    if last["Close"] < last["MA20"] and last["MA20"] < prev["MA20"]:
        items.append("Giá dưới MA20 và MA20 dốc xuống, xu hướng ngắn hạn đang giảm.")
    elif last["Close"] > last["MA20"] and last["MA20"] > prev["MA20"]:
        items.append("Giá trên MA20 và MA20 dốc lên, xu hướng ngắn hạn đang cải thiện.")
    else:
        items.append("Giá đang dao động quanh MA20, tín hiệu ngắn hạn chưa rõ.")

    if last["Close"] <= last["BB_LOWER"]:
        items.append("Vàng đang chạm dải dưới Bollinger Bands, có khả năng hồi kỹ thuật ngắn hạn nhưng chưa xác nhận đảo chiều tăng.")
    if last["Close"] >= last["BB_UPPER"]:
        items.append("Vàng đang sát dải trên Bollinger Bands, không nên BUY đuổi quá xa MA20.")
    if last["BB_WIDTH"] > prev["BB_WIDTH"] and last["Close"] < last["MA20"]:
        items.append("Bollinger Bands mở rộng theo hướng giảm, đà giảm còn mạnh.")

    rsi = latest_float(data["RSI14"])
    if rsi < 30:
        items.append("RSI dưới 30, trạng thái quá bán.")
    elif rsi > 70:
        items.append("RSI trên 70, trạng thái quá mua.")
    else:
        items.append(f"RSI ở mức {rsi:.1f}, chưa vào vùng quá mua hoặc quá bán.")

    if latest_float(data["ADX14"]) > 25:
        items.append("ADX trên 25, xu hướng hiện tại khá mạnh.")
    else:
        items.append("ADX chưa cao, thị trường có thể thiếu hướng đi rõ.")

    if last["EMA50"] < last["EMA200"]:
        items.append("EMA50 dưới EMA200, xu hướng lớn vẫn đang giảm.")
    else:
        items.append("EMA50 trên EMA200, xu hướng lớn đang nghiêng về tăng.")

    if bool(last.get("BOS_UP", False)):
        items.append("Có phá cấu trúc tăng kèm volume tăng.")
    if bool(last.get("BOS_DOWN", False)):
        items.append("Có phá cấu trúc giảm kèm volume tăng.")
    if bool(last.get("LIQUIDITY_SWEEP_UP", False)):
        items.append("Có dấu hiệu quét thanh khoản phía trên rồi bị đẩy xuống.")
    if bool(last.get("LIQUIDITY_SWEEP_DOWN", False)):
        items.append("Có dấu hiệu quét thanh khoản phía dưới rồi bật lên.")

    items.append(candle_reject(data))
    regime = detect_regime(data)
    levels = support_resistance(data)
    summary = f"Giá hiện tại {last['Close']:.2f}. Trạng thái thị trường: {regime}."
    return {"summary": summary, "items": items, "levels": levels, "data": data, "regime": regime}


def multi_timeframe_summary(frames: dict[str, pd.DataFrame]) -> dict:
    trends = {}
    for name, df in frames.items():
        trends[name] = detect_regime(df)
    h4 = trends.get("H4", "")
    h1 = trends.get("H1", "")
    m15 = trends.get("M15", "")
    if "giảm" in h4.lower() and "giảm" in h1.lower() and "tăng" in m15.lower():
        text = "Vàng đang hồi kỹ thuật trong xu hướng giảm lớn."
    elif "tăng" in h4.lower() and "tăng" in h1.lower() and "giảm" in m15.lower():
        text = "Vàng đang điều chỉnh ngắn hạn trong xu hướng tăng lớn."
    elif len(set(trends.values())) > 1:
        text = "Các khung thời gian đang lệch pha, nên chờ xác nhận rõ hơn."
    else:
        text = "Các khung thời gian tương đối đồng thuận."
    return {"trends": trends, "summary": text}

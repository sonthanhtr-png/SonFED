from __future__ import annotations

import pandas as pd

from .indicators import support_resistance
from .utils import latest_float


def _levels(df: pd.DataFrame) -> tuple[float, float, float]:
    levels = support_resistance(df)
    price = latest_float(df["Close"]) if not df.empty else 0.0
    atr = latest_float(df.get("ATR14", pd.Series(dtype=float)), 10.0) or max(price * 0.002, 5)
    return price, float(levels.get("support") or price - atr), float(levels.get("resistance") or price + atr)


def build_strategies(gold_analysis: dict, macro: dict, mtf: dict, event_risk: dict, config: dict) -> list[dict]:
    df = gold_analysis.get("data", pd.DataFrame())
    if df.empty:
        return []
    last = df.iloc[-1]
    price, support, resistance = _levels(df)
    atr = latest_float(df["ATR14"], max(price * 0.002, 5))
    enabled = config.get("features", {}).get("strategies", {})
    pressure = macro.get("score", 50)
    regime = gold_analysis.get("regime", "Chưa rõ xu hướng")
    rows = []

    def add(name, condition, probability, entry, tp, sl, alert):
        if enabled.get(name, True):
            rows.append({
                "strategy": name,
                "condition": condition,
                "probability": int(probability),
                "entry": entry,
                "take_profit": round(float(tp), 2),
                "stop_loss": round(float(sl), 2),
                "alert": alert,
            })

    add(
        "Hồi kỹ thuật",
        "Giá chạm dải dưới Bollinger Bands hoặc RSI quá bán, sau đó giữ lại trên MA20 M15.",
        62 if last["Close"] <= last["BB_LOWER"] or last["RSI14"] < 35 else 45,
        f"{price - atr * 0.2:.2f}-{price + atr * 0.2:.2f}",
        price + atr * 1.2,
        price - atr * 0.8,
        "Không BUY nếu H1 vẫn giảm mạnh và chưa có nến xác nhận.",
    )
    add(
        "SELL theo xu hướng",
        "EMA50 dưới EMA200, giá dưới MA20, ADX trên 25 và nhịp hồi bị từ chối.",
        72 if "giảm" in regime.lower() and pressure >= 60 else 52,
        f"{resistance - atr * 0.3:.2f}-{resistance:.2f}",
        support,
        resistance + atr * 0.6,
        "Không nên SELL đuổi sát BB dưới. Chờ hồi hoặc reject.",
    )
    add(
        "Breakout",
        "Đóng nến trên kháng cự, volume tăng và giữ được phía trên.",
        68 if bool(last.get("BOS_UP", False)) else 48,
        f"Trên {resistance:.2f}",
        resistance + atr * 1.5,
        resistance - atr * 0.7,
        f"Nếu vàng break {resistance:.2f} và giữ được phía trên, hạn chế SELL đuổi.",
    )
    add(
        "Reject",
        "Râu nến dài tại kháng cự, volume bán tăng và bị đẩy xuống.",
        66 if bool(last.get("LIQUIDITY_SWEEP_UP", False)) else 50,
        f"{resistance - atr * 0.2:.2f}-{resistance:.2f}",
        price - atr,
        resistance + atr * 0.5,
        f"Nếu vàng reject tại {resistance:.2f} với volume bán tăng, có thể cân nhắc SELL theo xu hướng chính.",
    )
    add(
        "Sideway",
        "ADX thấp, giá kẹt giữa hỗ trợ và kháng cự.",
        60 if latest_float(df["ADX14"]) < 20 else 40,
        f"{support:.2f}-{resistance:.2f}",
        (support + resistance) / 2,
        support - atr * 0.5,
        "Ưu tiên biên độ ngắn, không mở lệnh khi giá ở giữa vùng.",
    )
    add(
        "Né tin",
        "Có CPI, Core CPI, PCE, Nonfarm, FOMC hoặc Powell Speech gần thời điểm hiện tại.",
        90 if event_risk.get("blocked") else 10,
        "Không vào lệnh",
        price,
        price,
        event_risk.get("message", "Không có tin lớn gần thời điểm hiện tại."),
    )
    add(
        "Đảo chiều tăng",
        "Quét thanh khoản dưới, RSI thoát quá bán, MACD cải thiện và giá lấy lại MA20.",
        64 if bool(last.get("LIQUIDITY_SWEEP_DOWN", False)) and last["Close"] > last["MA20"] else 44,
        f"Trên {last['MA20']:.2f}",
        price + atr * 1.4,
        price - atr * 0.8,
        "Không tự BUY nếu kỹ thuật chưa xác nhận bằng nến đóng trên MA20.",
    )
    add(
        "Tiếp diễn giảm",
        "Giá phá hỗ trợ, volume tăng và vĩ mô gây áp lực cho vàng.",
        70 if bool(last.get("BOS_DOWN", False)) and pressure >= 60 else 50,
        f"Dưới {support:.2f}",
        support - atr * 1.3,
        support + atr * 0.7,
        "Chỉ SELL khi breakdown rõ, tránh vào lệnh giữa vùng nhiễu.",
    )
    return sorted(rows, key=lambda x: x["probability"], reverse=True)

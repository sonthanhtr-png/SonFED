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
    market_regime = gold_analysis.get("market_regime", {})
    volatility = gold_analysis.get("volatility", {})
    volatility_score = int(volatility.get("score", 0) or 0)
    scalp_pressure = sum(
        [
            bool(volatility.get("atr_expanding")),
            bool(volatility.get("bb_expanding")),
            bool(volatility.get("candle_expansion")),
            bool(volatility.get("volume_spike")),
            bool(volatility.get("momentum_expanding")),
        ]
    )
    scalp_ready = volatility_score >= 25 or scalp_pressure >= 2
    high_vol_penalty = 4 if volatility_score >= 90 else 0
    scalp_bonus = min(14, scalp_pressure * 3 + (4 if volatility_score >= 35 else 0))
    rows = []

    def add(name, condition, probability, entry, tp, sl, alert, action="WAIT", scalp=False):
        if enabled.get(name, True):
            adjusted_probability = int(probability) - high_vol_penalty + (scalp_bonus if scalp else 0)
            adjusted_probability = max(10, min(88, adjusted_probability))
            effective_action = action
            if action in {"BUY", "SELL"} and not scalp and adjusted_probability < 58:
                effective_action = "WAIT"
            rows.append({
                "strategy": name,
                "condition": condition,
                "probability": adjusted_probability,
                "action": effective_action,
                "scalp": bool(scalp),
                "scalp_ready": bool(scalp_ready),
                "momentum_score": int(scalp_pressure),
                "entry": entry,
                "take_profit": round(float(tp), 2),
                "stop_loss": round(float(sl), 2),
                "alert": alert,
                "volatility": volatility.get("level", "Chưa rõ"),
                "risk_level": "Cao" if volatility_score >= 85 else "Trung bình",
            })

    close_above_ma = bool(last["Close"] > last["MA20"])
    close_below_ma = bool(last["Close"] < last["MA20"])
    macd_positive = bool(last["MACD_HIST"] > 0)
    macd_negative = bool(last["MACD_HIST"] < 0)
    macd_delta = latest_float(df["MACD_HIST"].diff(), 0.0)
    bull_candle = bool(last["Close"] > last["Open"])
    bear_candle = bool(last["Close"] < last["Open"])
    buy_flow = macd_positive and (close_above_ma or bull_candle or macd_delta > 0)
    sell_flow = macd_negative and (close_below_ma or bear_candle or macd_delta < 0)
    buy_momentum = buy_flow and scalp_ready
    sell_momentum = sell_flow and scalp_ready
    bos_up = bool(last.get("BOS_UP", False))
    bos_down = bool(last.get("BOS_DOWN", False))
    sweep_up = bool(last.get("LIQUIDITY_SWEEP_UP", False))
    sweep_down = bool(last.get("LIQUIDITY_SWEEP_DOWN", False))

    breakout_action = "BUY" if bos_up else "SELL" if bos_down else "WAIT"
    reject_action = "BUY" if sweep_down else "SELL" if sweep_up else "WAIT"

    add(
        "Scalp BUY momentum M15",
        "M15 giữ trên MA20, MACD dương, ATR/BB/volume đang mở rộng. Ưu tiên ăn nhịp ngắn, không chờ H4 hoàn hảo.",
        54 if buy_momentum else 30,
        f"{price - atr * 0.15:.2f}-{price + atr * 0.10:.2f}",
        price + atr * 0.65,
        price - atr * 0.60,
        "Scalp BUY: vào sớm theo momentum, BE nhanh khi có lời nhỏ.",
        action="BUY" if buy_momentum else "WAIT",
        scalp=True,
    )
    add(
        "Scalp SELL momentum M15",
        "M15 nằm dưới MA20, MACD âm, ATR/BB/volume đang mở rộng. Ưu tiên ăn nhịp ngắn, không chờ H4 hoàn hảo.",
        54 if sell_momentum else 30,
        f"{price - atr * 0.10:.2f}-{price + atr * 0.15:.2f}",
        price - atr * 0.65,
        price + atr * 0.60,
        "Scalp SELL: vào sớm theo momentum, BE nhanh khi có lời nhỏ.",
        action="SELL" if sell_momentum else "WAIT",
        scalp=True,
    )
    add(
        "Scalp breakout M15",
        "Giá phá vùng gần nhất kèm volatility expansion. Chấp nhận RR nhỏ nếu lực chạy còn tốt.",
        56 if breakout_action != "WAIT" and scalp_ready else 30,
        f"{price - atr * 0.10:.2f}-{price + atr * 0.10:.2f}",
        price + atr * 0.75 if breakout_action == "BUY" else price - atr * 0.75 if breakout_action == "SELL" else price,
        price - atr * 0.65 if breakout_action == "BUY" else price + atr * 0.65 if breakout_action == "SELL" else price,
        "Scalp breakout: ưu tiên phản ứng nhanh, trailing sớm sau khi giá chạy.",
        action=breakout_action,
        scalp=True,
    )
    add(
        "Scalp reject M15",
        "Quét thanh khoản rồi bật/reject rõ. Ưu tiên ăn nhịp hồi ngắn thay vì chờ setup hoàn hảo.",
        55 if reject_action != "WAIT" else 30,
        f"{price - atr * 0.15:.2f}-{price + atr * 0.15:.2f}",
        price + atr * 0.65 if reject_action == "BUY" else price - atr * 0.65 if reject_action == "SELL" else price,
        price - atr * 0.60 if reject_action == "BUY" else price + atr * 0.60 if reject_action == "SELL" else price,
        "Scalp reject: vào theo phản ứng giá, thoát nhanh nếu momentum yếu lại.",
        action=reject_action,
        scalp=True,
    )

    add(
        "Hồi kỹ thuật",
        "Giá chạm dải dưới Bollinger Bands hoặc RSI quá bán, sau đó giữ lại trên MA20 M15.",
        62 if last["Close"] <= last["BB_LOWER"] or last["RSI14"] < 35 else 45,
        f"{price - atr * 0.2:.2f}-{price + atr * 0.2:.2f}",
        price + atr * 1.2,
        price - atr * 0.8,
        "Không BUY nếu H1 vẫn giảm mạnh và chưa có nến xác nhận.",
        action="BUY",
    )
    add(
        "SELL theo xu hướng",
        "EMA50 dưới EMA200, giá dưới MA20, ADX trên 25 và nhịp hồi bị từ chối.",
        72 if market_regime.get("bias") == "SELL" and pressure >= 55 else 52,
        f"{resistance - atr * 0.3:.2f}-{resistance:.2f}",
        support,
        resistance + atr * 0.6,
        "Không nên SELL đuổi sát BB dưới. Chờ hồi hoặc reject.",
        action="SELL",
    )
    add(
        "Breakout",
        "Đóng nến trên kháng cự, volume tăng và giữ được phía trên.",
        72 if bool(last.get("BOS_UP", False)) and volatility_score >= 35 else 48,
        f"Trên {resistance:.2f}",
        resistance + atr * 1.5,
        resistance - atr * 0.7,
        f"Nếu vàng break {resistance:.2f} và giữ được phía trên, hạn chế SELL đuổi.",
        action="BUY",
    )
    add(
        "Reject",
        "Râu nến dài tại kháng cự, volume bán tăng và bị đẩy xuống.",
        66 if bool(last.get("LIQUIDITY_SWEEP_UP", False)) else 50,
        f"{resistance - atr * 0.2:.2f}-{resistance:.2f}",
        price - atr,
        resistance + atr * 0.5,
        f"Nếu vàng reject tại {resistance:.2f} với volume bán tăng, có thể cân nhắc SELL theo xu hướng chính.",
        action="SELL",
    )
    add(
        "Sideway",
        "ADX thấp, giá kẹt giữa hỗ trợ và kháng cự.",
        60 if latest_float(df["ADX14"]) < 20 else 40,
        f"{support:.2f}-{resistance:.2f}",
        (support + resistance) / 2,
        support - atr * 0.5,
        "Ưu tiên biên độ ngắn, không mở lệnh khi giá ở giữa vùng.",
        action=market_regime.get("bias", "WAIT"),
    )
    add(
        "Né tin",
        "Có CPI, Core CPI, PCE, Nonfarm, FOMC hoặc Powell Speech gần thời điểm hiện tại.",
        90 if event_risk.get("blocked") else 10,
        "Không vào lệnh",
        price,
        price,
        event_risk.get("message", "Không có tin lớn gần thời điểm hiện tại."),
        action="WAIT",
    )
    add(
        "Đảo chiều tăng",
        "Quét thanh khoản dưới, RSI thoát quá bán, MACD cải thiện và giá lấy lại MA20.",
        64 if bool(last.get("LIQUIDITY_SWEEP_DOWN", False)) and last["Close"] > last["MA20"] else 44,
        f"Trên {last['MA20']:.2f}",
        price + atr * 1.4,
        price - atr * 0.8,
        "Không tự BUY nếu kỹ thuật chưa xác nhận bằng nến đóng trên MA20.",
        action="BUY",
    )
    add(
        "Tiếp diễn giảm",
        "Giá phá hỗ trợ, volume tăng và vĩ mô gây áp lực cho vàng.",
        72 if bool(last.get("BOS_DOWN", False)) and pressure >= 55 and volatility_score >= 35 else 50,
        f"Dưới {support:.2f}",
        support - atr * 1.3,
        support + atr * 0.7,
        "Chỉ SELL khi breakdown rõ, tránh vào lệnh giữa vùng nhiễu.",
        action="SELL",
    )
    def sort_key(row: dict) -> tuple[int, int, int]:
        if event_risk.get("blocked") and row["strategy"] == "Né tin":
            return (3, 0, row["probability"])
        actionable = row.get("action") in {"BUY", "SELL"}
        scalp = bool(row.get("scalp")) and actionable
        return (2 if actionable else 0, 1 if scalp else 0, row["probability"])

    return sorted(rows, key=sort_key, reverse=True)

from __future__ import annotations

import logging

import pandas as pd

from .indicators import add_indicators, support_resistance
from .utils import latest_float

logger = logging.getLogger(__name__)


def _slope(series: pd.Series, periods: int = 5) -> float:
    try:
        clean = series.dropna()
        if len(clean) <= periods or clean.iloc[-periods] == 0:
            return 0.0
        return float((clean.iloc[-1] - clean.iloc[-periods]) / abs(clean.iloc[-periods]) * 100)
    except Exception:
        return 0.0


def analyze_volatility(df: pd.DataFrame) -> dict:
    if df.empty or len(df) < 30:
        return {
            "level": "Chưa đủ dữ liệu",
            "score": 0,
            "bb_width": 0.0,
            "atr": 0.0,
            "atr_pct": 0.0,
            "range_pct": 0.0,
            "candle_expansion": False,
            "volume_spike": False,
            "bb_expanding": False,
            "atr_expanding": False,
            "notes": ["Chưa đủ dữ liệu để đo volatility."],
        }

    data = add_indicators(df)
    if "CANDLE_RANGE" not in data and {"High", "Low"}.issubset(data.columns):
        data["CANDLE_RANGE"] = data["High"] - data["Low"]
    if "CANDLE_RANGE_PCT" not in data and {"CANDLE_RANGE", "Close"}.issubset(data.columns):
        data["CANDLE_RANGE_PCT"] = data["CANDLE_RANGE"] / data["Close"].replace(0, pd.NA) * 100
    if "ATR_PCT" not in data and {"ATR14", "Close"}.issubset(data.columns):
        data["ATR_PCT"] = data["ATR14"] / data["Close"].replace(0, pd.NA) * 100
    for col, default in {
        "BB_WIDTH": 0.0,
        "ATR14": 0.0,
        "ATR_PCT": 0.0,
        "CANDLE_RANGE": 0.0,
        "CANDLE_RANGE_PCT": 0.0,
        "BB_EXPANDING": False,
        "ATR_EXPANDING": False,
        "CANDLE_EXPANSION": False,
        "VOLUME_SPIKE": False,
        "MACD_HIST": 0.0,
    }.items():
        if col not in data:
            data[col] = default
    last = data.iloc[-1]
    bb_width = latest_float(data["BB_WIDTH"])
    bb_avg = latest_float(data["BB_WIDTH"].rolling(20).mean())
    atr = latest_float(data["ATR14"])
    atr_avg = latest_float(data["ATR14"].rolling(20).mean())
    atr_pct = latest_float(data["ATR_PCT"])
    range_pct = latest_float(data["CANDLE_RANGE_PCT"])
    range_avg = latest_float(data["CANDLE_RANGE"].rolling(20).mean())

    bb_expanding = bool(last.get("BB_EXPANDING", False)) or (bb_avg > 0 and bb_width > bb_avg * 1.1)
    atr_expanding = bool(last.get("ATR_EXPANDING", False)) or (atr_avg > 0 and atr > atr_avg * 1.1)
    candle_expansion = bool(last.get("CANDLE_EXPANSION", False)) or (
        range_avg > 0 and latest_float(data["CANDLE_RANGE"]) > range_avg * 1.5
    )
    volume_spike = bool(last.get("VOLUME_SPIKE", False))
    momentum_expanding = abs(latest_float(data["MACD_HIST"])) > abs(latest_float(data["MACD_HIST"].rolling(20).mean()))

    score = 0
    if bb_expanding:
        score += 25
    if atr_expanding:
        score += 25
    if candle_expansion:
        score += 25
    if volume_spike:
        score += 15
    if momentum_expanding:
        score += 10
    if atr_pct >= 0.45 or range_pct >= 0.7:
        score += 10
    score = min(score, 100)

    if bb_expanding and atr_expanding and candle_expansion:
        score = max(score, 70)

    if score >= 70:
        level = "Biến động mạnh"
    elif score >= 45:
        level = "Volatile Range"
    elif score >= 25:
        level = "Expansion"
    else:
        level = "Sideway yếu"

    notes = []
    notes.append("BB mở rộng mạnh." if bb_expanding else "BB chưa mở rộng rõ.")
    notes.append("ATR tăng so với trung bình." if atr_expanding else "ATR chưa tăng rõ.")
    notes.append("Biên nến đang mở rộng." if candle_expansion else "Biên nến chưa đột biến.")
    if volume_spike:
        notes.append("Volume spike xuất hiện.")
    if momentum_expanding:
        notes.append("Momentum đang mở rộng.")

    return {
        "level": level,
        "score": int(score),
        "bb_width": round(float(bb_width or 0), 5),
        "atr": round(float(atr or 0), 2),
        "atr_pct": round(float(atr_pct or 0), 2),
        "range_pct": round(float(range_pct or 0), 2),
        "candle_expansion": candle_expansion,
        "volume_spike": volume_spike,
        "bb_expanding": bb_expanding,
        "atr_expanding": atr_expanding,
        "momentum_expanding": momentum_expanding,
        "notes": notes,
    }


def detect_market_regime(df: pd.DataFrame) -> dict:
    if df.empty or len(df) < 60:
        return {
            "label": "Chưa đủ dữ liệu",
            "score": 0,
            "bias": "WAIT",
            "bias_text": "Chờ thêm dữ liệu",
            "probability": 0,
            "risk_level": "Trung bình",
            "trend_score": 0,
            "structure": "Chưa rõ",
            "momentum": "Chưa rõ",
            "volatility": analyze_volatility(df),
        }

    data = add_indicators(df)
    last = data.iloc[-1]
    recent = data.tail(20)
    volatility = analyze_volatility(data)
    ma20_slope = _slope(data["MA20"], 5)
    ema50_slope = _slope(data["EMA50"], 8)
    macd_hist_slope = _slope(data["MACD_HIST"], 3)

    higher_highs = int(recent["HIGHER_HIGH"].tail(10).sum())
    lower_lows = int(recent["LOWER_LOW"].tail(10).sum())
    closes = data["Close"].dropna()
    higher_lows = bool(len(recent) >= 8 and recent["Low"].iloc[-1] > recent["Low"].rolling(6).min().iloc[-2])
    lower_highs = bool(len(recent) >= 8 and recent["High"].iloc[-1] < recent["High"].rolling(6).max().iloc[-2])

    bull_points = 0
    bear_points = 0
    if ma20_slope > 0.04:
        bull_points += 15
    if ma20_slope < -0.04:
        bear_points += 15
    if ema50_slope > 0.03:
        bull_points += 15
    if ema50_slope < -0.03:
        bear_points += 15
    if higher_highs >= 2 and higher_lows:
        bull_points += 25
    if lower_lows >= 2 and lower_highs:
        bear_points += 25
    if last["Close"] > last["MA20"] and last["MACD_HIST"] > 0:
        bull_points += 20
    if last["Close"] < last["MA20"] and last["MACD_HIST"] < 0:
        bear_points += 20
    if bool(last.get("BOS_UP", False)):
        bull_points += 25
    if bool(last.get("BOS_DOWN", False)):
        bear_points += 25

    direction_score = bull_points - bear_points
    trend_strength = min(100, abs(direction_score) + int(latest_float(data["ADX14"]) or 0))
    is_breakout = bool(last.get("BOS_UP", False) or last.get("BOS_DOWN", False))
    exhausted = (
        (last["RSI14"] > 72 and last["Close"] >= last["BB_UPPER"])
        or (last["RSI14"] < 28 and last["Close"] <= last["BB_LOWER"])
        or bool(last.get("LIQUIDITY_SWEEP_UP", False))
        or bool(last.get("LIQUIDITY_SWEEP_DOWN", False))
    )

    if is_breakout and direction_score > 0:
        label = "Bull Expansion"
    elif is_breakout and direction_score < 0:
        label = "Bear Expansion"
    elif exhausted:
        label = "Exhaustion"
    elif trend_strength >= 65 and abs(direction_score) >= 35:
        label = "Strong Trend"
    elif trend_strength >= 40 and abs(direction_score) >= 20:
        label = "Weak Trend"
    elif volatility["score"] >= 45:
        label = "Volatile Range"
    else:
        label = "Quiet Range"

    if direction_score >= 15:
        bias = "BUY"
        bias_text = "BUY nhẹ" if direction_score < 45 else "BUY"
    elif direction_score <= -15:
        bias = "SELL"
        bias_text = "SELL nhẹ" if direction_score > -45 else "SELL"
    else:
        bias = "WAIT"
        bias_text = "WAIT"

    scalp_pressure = sum(
        [
            bool(volatility.get("atr_expanding")),
            bool(volatility.get("bb_expanding")),
            bool(volatility.get("candle_expansion")),
            bool(volatility.get("volume_spike")),
            bool(volatility.get("momentum_expanding")),
        ]
    )
    scalp_ready = scalp_pressure >= 2 or volatility.get("score", 0) >= 25
    if bias == "WAIT" and scalp_ready:
        if last["Close"] > last["MA20"] and last["MACD_HIST"] > 0:
            bias = "BUY"
            bias_text = "Scalp BUY"
            direction_score = max(direction_score, 12)
        elif last["Close"] < last["MA20"] and last["MACD_HIST"] < 0:
            bias = "SELL"
            bias_text = "Scalp SELL"
            direction_score = min(direction_score, -12)

    probability = min(84, 48 + int(abs(direction_score) * 0.5) + min(12, int(trend_strength / 8)))
    if scalp_ready and bias in {"BUY", "SELL"}:
        probability = max(probability, 52 + min(12, scalp_pressure * 3))
    if volatility["score"] >= 85:
        probability = max(45, probability - 8)
    if exhausted:
        probability = max(48, probability - 4)

    risk_level = "Cao" if volatility["score"] >= 85 or exhausted else "Trung bình"
    if volatility["score"] < 25 and label == "Quiet Range":
        risk_level = "Thấp"

    structure = "HH/HL nghiêng tăng" if direction_score > 0 else "LL/LH nghiêng giảm" if direction_score < 0 else "Cấu trúc chưa rõ"
    momentum = "Momentum tăng" if macd_hist_slope > 0 and last["MACD_HIST"] > 0 else "Momentum giảm" if macd_hist_slope < 0 and last["MACD_HIST"] < 0 else "Momentum trung tính"

    return {
        "label": label,
        "score": int(trend_strength),
        "bias": bias,
        "bias_text": bias_text,
        "probability": int(probability),
        "risk_level": risk_level,
        "trend_score": int(direction_score),
        "structure": structure,
        "momentum": momentum,
        "scalp_ready": bool(scalp_ready),
        "scalp_pressure": int(scalp_pressure),
        "ma20_slope": round(ma20_slope, 3),
        "ema50_slope": round(ema50_slope, 3),
        "volatility": volatility,
    }


def build_decision(gold_analysis: dict, macro: dict, mtf: dict, strategies: list[dict]) -> dict:
    df = gold_analysis.get("data", pd.DataFrame())
    if df.empty:
        return {"action": "WAIT", "winrate": 0, "tp": None, "sl": None, "rr": None, "reason": "Chưa đủ dữ liệu."}

    regime = gold_analysis.get("market_regime") or gold_analysis.get("regime") or {}
    if not isinstance(regime, dict) or not regime:
        logger.warning("Không có market_regime, AI Decision Box có thể không chính xác.")
        regime = detect_market_regime(df)
    volatility = regime.get("volatility", {})
    best = strategies[0] if strategies else {}
    price = latest_float(df["Close"])
    levels = support_resistance(df)
    atr = latest_float(df["ATR14"], max(price * 0.002, 5))
    support = float(levels.get("support") or price - atr)
    resistance = float(levels.get("resistance") or price + atr)
    action = regime.get("bias", "WAIT")
    best_action = best.get("action")
    best_probability = int(best.get("probability", 0) or 0)
    scalp_best = bool(best.get("scalp"))
    if best_action in {"BUY", "SELL"} and best_probability >= (50 if scalp_best else 58):
        action = best_action

    if macro.get("score", 50) >= 85 and action == "BUY":
        action = "WAIT"
    if macro.get("score", 50) <= 15 and action == "SELL":
        action = "WAIT"
    if volatility.get("score", 0) >= 90 and action in {"BUY", "SELL"}:
        action = "WAIT" if best_probability < 68 else action

    use_best_levels = best_action == action
    if action == "BUY":
        tp = (best.get("take_profit") if use_best_levels else None) or price + atr * (0.85 if scalp_best else 1.3)
        sl = (best.get("stop_loss") if use_best_levels else None) or price - atr * (0.65 if scalp_best else 0.8)
    elif action == "SELL":
        tp = (best.get("take_profit") if use_best_levels else None) or price - atr * (0.85 if scalp_best else 1.3)
        sl = (best.get("stop_loss") if use_best_levels else None) or price + atr * (0.65 if scalp_best else 0.8)
    else:
        tp = resistance
        sl = support

    risk = abs(price - float(sl)) if sl else 0
    reward = abs(float(tp) - price) if tp else 0
    rr = round(reward / risk, 2) if risk else None
    winrate = int(min(86, max(45, best_probability or regime.get("probability", 50))))
    if volatility.get("score", 0) >= 85:
        winrate = max(42, winrate - 7)

    reason_parts = [
        f"Regime: {regime.get('label', 'Chưa rõ')}.",
        f"Volatility: {volatility.get('level', 'Chưa rõ')} ({volatility.get('score', 0)}/100).",
        f"{regime.get('structure', '')}. {regime.get('momentum', '')}.",
    ]
    if action == "SELL":
        reason_parts.append("Ưu tiên scalp SELL theo momentum M15, chốt nhịp ngắn và quản lý lệnh nhanh.")
        reason_parts.append(f"Nếu break xác nhận trên {resistance:.2f} thì hủy bias SELL.")
    elif action == "BUY":
        reason_parts.append("Ưu tiên scalp BUY theo momentum M15, chốt nhịp ngắn và quản lý lệnh nhanh.")
        reason_parts.append(f"Nếu breakdown dưới {support:.2f} thì hủy bias BUY.")
    else:
        reason_parts.append("Chỉ đứng ngoài khi risk quá cao hoặc momentum chưa đủ rõ cho scalp.")

    return {
        "action": action,
        "winrate": winrate,
        "tp": round(float(tp), 2) if tp else None,
        "sl": round(float(sl), 2) if sl else None,
        "rr": rr,
        "reason": " ".join(reason_parts),
        "risk_level": regime.get("risk_level", "Trung bình"),
    }

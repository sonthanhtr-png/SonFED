from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

import pandas as pd

from modules.utils import latest_float

logger = logging.getLogger(__name__)

ACTION_MAP = {
    "tighten_trailing": "ADJUST_TRAILING",
    "hold_with_atr_trailing": "HOLD_POSITION",
    "close_position": "CLOSE_POSITION",
    "move_to_break_even": "MOVE_TO_BREAKEVEN",
    "move_to_breakeven": "MOVE_TO_BREAKEVEN",
    "hold_position": "HOLD_POSITION",
    "disable_new_entry": "DISABLE_NEW_ENTRY",
    "partial_close": "PARTIAL_CLOSE",
    "reduce_position": "PARTIAL_CLOSE",
    "lock_profit": "LOCK_PROFIT",
}

VALID_ADJUSTMENT_ACTIONS = {
    "ADJUST_TRAILING",
    "HOLD_POSITION",
    "CLOSE_POSITION",
    "MOVE_TO_BREAKEVEN",
    "DISABLE_NEW_ENTRY",
    "PARTIAL_CLOSE",
    "LOCK_PROFIT",
    "ADJUST_SL",
    "ADJUST_TP",
}


def normalize_adjustment_action(action: str | None) -> str:
    raw = str(action or "HOLD_POSITION").strip()
    if not raw:
        return "HOLD_POSITION"
    key = raw.lower().replace("-", "_").replace(" ", "_")
    normalized = ACTION_MAP.get(key, raw.upper().replace("-", "_").replace(" ", "_"))
    if normalized == "MOVE_TO_BREAK_EVEN":
        normalized = "MOVE_TO_BREAKEVEN"
    if normalized not in VALID_ADJUSTMENT_ACTIONS:
        logger.warning("Action adjustment khong hop le tu SonFED: %s", raw)
        return "HOLD_POSITION"
    return normalized


def _num(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _side(position: dict[str, Any]) -> str:
    return str(position.get("type", position.get("type_name", "NONE"))).upper()


def evaluate_position_state(position: dict[str, Any], gold_df: pd.DataFrame, signal: dict, market_regime: dict, mtf: dict) -> dict:
    current = _num(position.get("current_price"), latest_float(gold_df["Close"]) if not gold_df.empty else 0.0)
    entry = _num(position.get("entry", position.get("price_open")), current)
    side = _side(position)
    profit = _num(position.get("profit"))
    atr = latest_float(gold_df.get("ATR14", pd.Series(dtype=float)), max(current * 0.002, 5)) if not gold_df.empty else 5
    volatility_score = int(signal.get("volatility_score", market_regime.get("volatility", {}).get("score", 0)) or 0)
    confidence = int(signal.get("confidence", 0) or 0)
    h1 = str(mtf.get("trends", {}).get("H1", ""))
    h4 = str(mtf.get("trends", {}).get("H4", ""))
    direction = "BUY" if "BUY" in side else "SELL" if "SELL" in side else "NONE"
    points_profit = (current - entry) if direction == "BUY" else (entry - current) if direction == "SELL" else 0.0
    trend_still_valid = (
        direction == "SELL" and ("Bear" in h1 or "Bear" in h4 or "giảm" in h1.lower() or "giảm" in h4.lower())
    ) or (
        direction == "BUY" and ("Bull" in h1 or "Bull" in h4 or "tăng" in h1.lower() or "tăng" in h4.lower())
    )
    return {
        "ticket": position.get("ticket"),
        "symbol": position.get("symbol", signal.get("symbol", "XAUUSD")),
        "side": direction,
        "entry": entry,
        "current": current,
        "profit": profit,
        "points_profit": points_profit,
        "atr": atr,
        "confidence": confidence,
        "volatility_score": volatility_score,
        "market_regime": signal.get("market_regime", market_regime.get("label", "Chưa rõ")),
        "trend_still_valid": trend_still_valid,
        "current_signal": signal.get("action", "WAIT"),
    }


def calculate_atr_trailing(position_state: dict, side: str, multiplier: float = 1.5) -> float | None:
    current = position_state.get("current")
    atr = position_state.get("atr")
    if not current or not atr:
        return None
    return round(current - atr * multiplier, 2) if side == "BUY" else round(current + atr * multiplier, 2)


def calculate_structure_trailing(gold_df: pd.DataFrame, side: str, lookback: int = 20) -> float | None:
    if gold_df.empty:
        return None
    recent = gold_df.tail(lookback)
    if side == "BUY":
        return round(float(recent["Low"].min()), 2)
    if side == "SELL":
        return round(float(recent["High"].max()), 2)
    return None


def calculate_break_even(position_state: dict, buffer_points: float = 0.0) -> float | None:
    entry = position_state.get("entry")
    side = position_state.get("side")
    if not entry:
        return None
    if side == "BUY":
        return round(entry + buffer_points, 2)
    if side == "SELL":
        return round(entry - buffer_points, 2)
    return None


def select_trailing_strategy(position_state: dict, policy: dict | None = None) -> dict:
    strategy = (policy or {}).get("position_management_strategy", "AI thích nghi")
    volatility = int(position_state.get("volatility_score", 0))
    if strategy == "Bảo toàn vốn":
        return {"mode": "atr_trailing", "atr_multiplier": 0.9, "aggressiveness": "chặt"}
    if strategy == "Break-even":
        return {"mode": "break_even", "atr_multiplier": 1.0, "aggressiveness": "nhanh"}
    if strategy == "Bám xu hướng":
        return {"mode": "atr_trailing", "atr_multiplier": 1.1, "aggressiveness": "nhanh"}
    if volatility >= 70:
        return {"mode": "atr_trailing", "atr_multiplier": 0.9, "aggressiveness": "chặt"}
    if volatility >= 45:
        return {"mode": "atr_trailing", "atr_multiplier": 1.0, "aggressiveness": "nhanh"}
    if position_state.get("trend_still_valid"):
        return {"mode": "atr_trailing", "atr_multiplier": 1.1, "aggressiveness": "nhanh"}
    return {"mode": "atr_trailing", "atr_multiplier": 1.0, "aggressiveness": "nhanh"}


def apply_conservative_mode(position_state: dict, gold_df: pd.DataFrame, reason: str) -> dict:
    side = position_state["side"]
    new_sl = calculate_atr_trailing(position_state, side, 1.2)
    partial = 50 if position_state.get("profit", 0) > 0 else 30
    return {
        "adjustment_mode": "conservative",
        "action": "tighten_trailing",
        "partial_close_percent": partial,
        "new_sl_mode": "atr_trailing",
        "new_sl": new_sl,
        "atr_multiplier": 1.2,
        "reason": reason,
    }


def apply_trend_follow_mode(position_state: dict, gold_df: pd.DataFrame, reason: str) -> dict:
    side = position_state["side"]
    trailing = select_trailing_strategy(position_state, {"position_management_strategy": "Bám xu hướng"})
    return {
        "adjustment_mode": "trend_follow",
        "action": "hold_with_atr_trailing",
        "partial_close_percent": 0,
        "new_sl_mode": trailing["mode"],
        "new_sl": calculate_atr_trailing(position_state, side, trailing["atr_multiplier"]),
        "atr_multiplier": trailing["atr_multiplier"],
        "reason": reason,
    }


def apply_exit_mode(position_state: dict, reason: str) -> dict:
    return {
        "adjustment_mode": "exit",
        "action": "close_position",
        "partial_close_percent": 100,
        "new_sl_mode": None,
        "new_sl": None,
        "atr_multiplier": None,
        "reason": reason,
    }


def generate_position_adjustment(
    position: dict[str, Any],
    gold_df: pd.DataFrame,
    signal: dict,
    previous_signal: str | None,
    market_regime: dict,
    mtf: dict,
    policy: dict | None = None,
) -> dict:
    state = evaluate_position_state(position, gold_df, signal, market_regime, mtf)
    side = state["side"]
    current_signal = signal.get("action", "WAIT")
    previous = previous_signal or side
    strategy = (policy or {}).get("position_management_strategy", "AI thích nghi")
    reason = "Lệnh đang mở được đánh giá theo regime, volatility, momentum và trạng thái tín hiệu mới."
    scalp_profit_threshold = max(0.8, min(2.0, state["atr"] * 0.18))
    opposite_signal = (side == "BUY" and current_signal == "SELL") or (side == "SELL" and current_signal == "BUY")

    if opposite_signal:
        if state["profit"] > 0 or state["points_profit"] > 0:
            action = {
                "adjustment_mode": "scalp_flip",
                "action": "partial_close",
                "partial_close_percent": 50,
                "new_sl_mode": "break_even",
                "new_sl": calculate_break_even(state),
                "atr_multiplier": None,
                "reason": "Tín hiệu M15 đổi chiều. Chốt một phần nhanh, kéo SL về hòa vốn và không cố giữ lệnh vì scalp ưu tiên phản ứng.",
            }
        else:
            action = apply_exit_mode(state, "Tín hiệu M15 đổi chiều khi lệnh chưa có lợi thế. Đóng để tránh biến scalp thành lệnh gồng.")
    elif state["points_profit"] >= scalp_profit_threshold:
        trailing = select_trailing_strategy(state, policy)
        action = {
            "adjustment_mode": "scalp_profit_lock",
            "action": "partial_close" if state["points_profit"] >= scalp_profit_threshold * 1.6 else "move_to_break_even",
            "partial_close_percent": 30 if state["points_profit"] >= scalp_profit_threshold * 1.6 else 0,
            "new_sl_mode": "break_even" if state["points_profit"] < scalp_profit_threshold * 1.6 else trailing["mode"],
            "new_sl": calculate_break_even(state) if state["points_profit"] < scalp_profit_threshold * 1.6 else calculate_atr_trailing(state, side, trailing["atr_multiplier"]),
            "atr_multiplier": None if state["points_profit"] < scalp_profit_threshold * 1.6 else trailing["atr_multiplier"],
            "reason": "Scalp đang có lợi nhuận nhỏ. Ưu tiên khóa lời sớm, dời BE/trailing nhanh thay vì chờ RR lớn.",
        }
    elif previous in {"BUY", "SELL"} and current_signal == "WAIT":
        if state["profit"] > 0 or state["points_profit"] > 0:
            action = {
                "adjustment_mode": "scalp_wait_lock",
                "action": "partial_close" if state["points_profit"] >= scalp_profit_threshold * 0.7 else "move_to_break_even",
                "partial_close_percent": 30 if state["points_profit"] >= scalp_profit_threshold * 0.7 else 0,
                "new_sl_mode": "break_even",
                "new_sl": calculate_break_even(state),
                "atr_multiplier": None,
                "reason": "AI chuyển sang WAIT khi lệnh đã có lời. Khóa lời sớm, có thể chốt một phần nhỏ và kéo SL về BE thay vì giữ quá lâu.",
            }
        elif state["trend_still_valid"] and strategy in {"Bám xu hướng", "AI thích nghi"}:
            action = apply_trend_follow_mode(
                state,
                gold_df,
                "AI chuyển sang WAIT nhưng H1/H4 vẫn còn xu hướng. Giữ lệnh, bám ATR trailing và không mở thêm vị thế mới.",
            )
        elif state["volatility_score"] >= 75 or state["confidence"] < 50:
            action = apply_exit_mode(state, "Regime xấu đi mạnh hoặc confidence collapse. Ưu tiên đóng vị thế để bảo toàn vốn.")
        else:
            action = apply_conservative_mode(
                state,
                gold_df,
                "Momentum suy yếu hoặc volatility tăng. Siết trailing, khóa lợi nhuận và giảm một phần vị thế nếu đang lời.",
            )
    elif current_signal == side and state["trend_still_valid"]:
        action = apply_trend_follow_mode(state, gold_df, "Tín hiệu vẫn cùng chiều với vị thế. Giữ trend lớn và trailing theo ATR.")
    else:
        action = {
            "adjustment_mode": "hold",
            "action": "hold_position",
            "partial_close_percent": 0,
            "new_sl_mode": "atr_trailing",
            "new_sl": calculate_atr_trailing(state, side, 1.5),
            "atr_multiplier": 1.5,
            "reason": reason,
        }

    action["action"] = normalize_adjustment_action(action.get("action"))
    return {
        "symbol": state["symbol"],
        "ticket": state["ticket"],
        "current_signal": current_signal,
        "position_side": side,
        "floating_profit": state["profit"],
        "market_regime": state["market_regime"],
        "risk_level": signal.get("risk_level", "Trung bình"),
        "ai_confidence": signal.get("confidence", 0),
        "confidence": signal.get("confidence", 0),
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        **action,
    }


def build_position_adjustment_payload(
    trade_feedback: dict,
    gold_df: pd.DataFrame,
    signal: dict,
    previous_signal: str | None,
    market_regime: dict,
    mtf: dict,
    policy: dict | None = None,
) -> dict:
    adjustments = [
        generate_position_adjustment(pos, gold_df, signal, previous_signal, market_regime, mtf, policy)
        for pos in trade_feedback.get("positions", [])
    ]
    return {
        "symbol": signal.get("symbol", "XAUUSD"),
        "current_signal": signal.get("action", "WAIT"),
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "adjustments": adjustments,
        "ai_position_analysis": build_ai_position_analysis(adjustments, signal, mtf),
        "created_by": "SonFED",
    }


def build_ai_position_analysis(adjustments: list[dict], signal: dict, mtf: dict) -> str:
    if not adjustments:
        return "Chưa có lệnh mở để quản lý. SonFED chỉ theo dõi tín hiệu và không đề xuất điều chỉnh vị thế."
    first = adjustments[0]
    return "\n".join(
        [
            f"AI hiện tại: {signal.get('action', 'WAIT')} với confidence {signal.get('confidence', 0)}%.",
            f"Market regime: {first.get('market_regime', 'Chưa rõ')}.",
            f"H1/H4: {mtf.get('trends', {}).get('H1', 'Chưa rõ')} / {mtf.get('trends', {}).get('H4', 'Chưa rõ')}.",
            "Chiến lược hiện tại:",
            f"- {first.get('reason', '')}",
            "Entry chỉ là khởi đầu. Lợi nhuận dài hạn đến từ position management và khả năng bảo toàn vốn khi thị trường thay đổi.",
        ]
    )

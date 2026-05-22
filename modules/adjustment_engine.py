from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from .utils import latest_float, pct_change, save_json

ADJUSTMENT_ACTIONS = {
    "HOLD_POSITION",
    "ADJUST_SL",
    "ADJUST_TP",
    "ADJUST_TRAILING",
    "MOVE_TO_BREAKEVEN",
    "REDUCE_POSITION",
    "CLOSE_POSITION",
    "DISABLE_NEW_ENTRY",
    "ENABLE_TRAILING",
    "DISABLE_TRAILING",
    "HEDGE_POSITION",
}


def _confidence(base: int, profit: float, pressure: int) -> int:
    value = base
    if profit < 0:
        value += 5
    if pressure >= 80:
        value += 5
    return int(max(1, min(95, value)))


def _safe_number(value: Any, fallback: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return fallback


def analyze_position(position: dict[str, Any], gold_df: pd.DataFrame, macro: dict, mtf: dict, config: dict) -> dict[str, Any]:
    ticket = position.get("ticket", 0)
    symbol = position.get("symbol", "XAUUSD")
    side = str(position.get("type", "NONE")).upper()
    entry = _safe_number(position.get("entry"))
    current = _safe_number(position.get("current_price"), latest_float(gold_df["Close"]) if not gold_df.empty else entry)
    profit = _safe_number(position.get("profit"))
    pressure = int(macro.get("score", 50))
    require_confirm = bool(config.get("trade", {}).get("require_adjustment_confirm", True))

    last = gold_df.iloc[-1] if not gold_df.empty else {}
    ma20 = _safe_number(last.get("MA20", current) if hasattr(last, "get") else current, current)
    atr = _safe_number(last.get("ATR14", max(current * 0.002, 5)) if hasattr(last, "get") else 5, 5)
    resistance = float(gold_df["High"].tail(50).max()) if not gold_df.empty else current + atr
    support = float(gold_df["Low"].tail(50).min()) if not gold_df.empty else current - atr
    volume_spike = bool(last.get("VOLUME_SPIKE", False)) if hasattr(last, "get") else False
    dxy_change = float(macro.get("dxy_change", 0.0))

    action = "HOLD_POSITION"
    reduce_percent = 0
    new_sl = position.get("sl")
    new_tp = position.get("tp")
    trailing_mode = position.get("trailing_mode") or "ATR"
    atr_multiplier = 1.5
    confidence = 55
    reason = "Lệnh hiện tại chưa có tín hiệu rủi ro rõ, ưu tiên giữ và theo dõi phản ứng giá."

    if side == "SELL":
        sell_under_pressure = current > ma20 and (volume_spike or dxy_change < 0)
        strong_break = current > resistance
        reject_near_resistance = current >= resistance - atr * 0.35 and current < resistance and profit >= 0
        if strong_break:
            # Nguy hiểm nhất: giá phá hẳn kháng cự → ưu tiên cao nhất
            action = "CLOSE_POSITION" if profit < 0 else "REDUCE_POSITION"
            reduce_percent = 50 if action == "REDUCE_POSITION" else 0
            new_sl = None if action == "CLOSE_POSITION" else min(position.get("sl") or current + atr, current + atr * 1.2)
            new_tp = None
            confidence = _confidence(78, profit, pressure)
            reason = "Giá đã phá mạnh vùng kháng cự gần. Không mở thêm SELL và cần giảm rủi ro lệnh hiện tại."
        elif sell_under_pressure:
            # MA20 bị vượt + lực mua tăng → giảm vị thế
            action = "REDUCE_POSITION"
            reduce_percent = 50 if current > ma20 + atr * 0.3 else 30
            new_sl = min(position.get("sl") or current + atr, current + atr)
            new_tp = support
            confidence = _confidence(72, profit, pressure)
            reason = "Lệnh SELL đang gặp áp lực vì M15 đã vượt MA20 và lực mua ngắn hạn tăng. Nên giảm vị thế hoặc chờ reject rõ hơn."
        elif reject_near_resistance:
            # Giá phản ứng tại kháng cự → trailing bảo vệ
            action = "ADJUST_TRAILING"
            new_sl = current + atr * 0.5
            new_tp = support
            confidence = _confidence(68, profit, pressure)
            reason = "Giá đang phản ứng tại kháng cự, có thể giữ SELL nhưng nên trailing theo ATR."
        elif profit > 0:
            # Đang có lời, chưa nguy hiểm → dời về hòa vốn
            action = "MOVE_TO_BREAKEVEN"
            new_sl = min(entry, current + atr * 0.8)
            new_tp = support
            confidence = _confidence(70, profit, pressure)
            reason = "SELL đang có lợi nhuận. Nên dời SL về hòa vốn và bật trailing theo ATR để bảo vệ thành quả."

    elif side == "BUY":
        buy_supported = current > ma20 and pressure < 70
        reject_at_resistance = current >= resistance - atr * 0.25 and volume_spike
        macro_against_buy = pressure >= 65 and profit < 0
        if macro_against_buy:
            # Nguy hiểm nhất: vĩ mô chống BUY + đang lỗ → ưu tiên cao nhất
            action = "REDUCE_POSITION"
            reduce_percent = 50
            new_sl = max(position.get("sl") or entry - atr, entry - atr)
            new_tp = None
            confidence = _confidence(76, profit, pressure)
            reason = "BUY đang gặp rủi ro vì môi trường vĩ mô bất lợi cho vàng. Nên giảm vị thế hoặc đóng lệnh nếu giá mất MA20."
        elif reject_at_resistance:
            # Giá bị đẩy xuống tại kháng cự → giảm vị thế
            action = "REDUCE_POSITION"
            reduce_percent = 50
            new_sl = max(position.get("sl") or entry, entry)
            new_tp = resistance
            confidence = _confidence(73, profit, pressure)
            reason = "Giá chạm vùng kháng cự và có dấu hiệu bị bán xuống. Nên giảm một phần vị thế BUY hoặc dời SL về hòa vốn."
        elif profit > 0:
            # Đang có lời → trailing bảo vệ
            action = "ADJUST_TRAILING"
            reduce_percent = 0
            new_sl = max(position.get("sl") or entry, entry)
            new_tp = resistance
            confidence = _confidence(70, profit, pressure)
            reason = "BUY đang có lợi thế. Có thể giữ lệnh, dời SL về hòa vốn và trailing theo ATR."
        elif buy_supported:
            # Giá giữ MA20, vĩ mô chưa bất lợi → bật trailing
            action = "ENABLE_TRAILING"
            reduce_percent = 0
            new_sl = position.get("sl")
            new_tp = resistance
            confidence = _confidence(66, profit, pressure)
            reason = "Giá giữ trên MA20, BUY còn lợi thế. Nên bật trailing để bảo vệ lợi nhuận."

    if action not in ADJUSTMENT_ACTIONS:
        action = "HOLD_POSITION"

    return {
        "ticket": ticket,
        "symbol": symbol,
        "action": action,
        "reduce_percent": reduce_percent,
        "new_sl": round(float(new_sl), 2) if new_sl not in (None, "") else None,
        "new_tp": round(float(new_tp), 2) if new_tp not in (None, "") else None,
        "new_trailing_mode": trailing_mode,
        "atr_multiplier": atr_multiplier,
        "confidence": confidence,
        "reason": reason,
        "require_user_confirm": require_confirm,
        "created_by": "SonFED",
    }


def create_trade_adjustments(trade_feedback: dict, gold_df: pd.DataFrame, macro: dict, mtf: dict, config: dict, event_risk: dict) -> dict:
    adjustments = []
    for position in trade_feedback.get("positions", []):
        adjustments.append(analyze_position(position, gold_df, macro, mtf, config))

    if event_risk.get("blocked"):
        adjustments.append({
            "ticket": None,
            "symbol": config.get("trade", {}).get("symbol", "XAUUSD"),
            "action": "DISABLE_NEW_ENTRY",
            "reduce_percent": 0,
            "new_sl": None,
            "new_tp": None,
            "new_trailing_mode": None,
            "atr_multiplier": None,
            "confidence": 90,
            "reason": "Sắp có tin quan trọng, hạn chế vào lệnh đuổi và tắt mở lệnh mới.",
            "require_user_confirm": True,
            "created_by": "SonFED",
        })

    return {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "adjustments": adjustments,
        "created_by": "SonFED",
    }


def write_trade_adjustment(payload: dict, shared_dir: Path) -> None:
    save_json(shared_dir / "trade_adjustment.json", payload)

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from .risk_manager import assess_auto_trade
from .utils import save_json

ALLOWED_ACTIONS = {"BUY", "SELL", "WAIT", "CLOSE_BUY", "CLOSE_SELL", "REDUCE_POSITION"}


def create_signal(
    strategies: list[dict],
    gold_analysis: dict,
    macro: dict,
    mtf: dict,
    config: dict,
    event_risk: dict,
    trade_status: dict | None = None,
) -> dict:
    best = strategies[0] if strategies else {}
    confidence = int(best.get("probability", 0))
    pressure = macro.get("score", 50)
    market_regime = gold_analysis.get("market_regime", {})
    volatility = gold_analysis.get("volatility", {})
    volatility_score = int(volatility.get("score", 0) or 0)
    conflict = False

    action = "WAIT"
    if event_risk.get("blocked"):
        action = "WAIT"
    elif market_regime.get("bias") in {"BUY", "SELL"} and confidence >= 58:
        action = market_regime.get("bias")
    elif confidence >= 70 and "SELL" in best.get("strategy", "").upper():
        action = "SELL"
    elif confidence >= 68 and best.get("strategy") in {"Hồi kỹ thuật", "Đảo chiều tăng", "Breakout"}:
        action = "BUY"

    if pressure >= 61 and action == "BUY":
        conflict = True
        action = "WAIT"
    if pressure <= 30 and action == "SELL":
        conflict = True
        action = "WAIT"
    if volatility_score >= 75 and confidence < 72:
        conflict = True
        action = "WAIT"

    reason = best.get("alert", "Chưa đủ điều kiện tạo tín hiệu giao dịch.")
    if conflict:
        reason = "Không đủ điều kiện auto trade full size. Chờ xác nhận reject/breakdown rõ hơn vì volatility hoặc vĩ mô đang rủi ro."

    risk_level = "Cao" if event_risk.get("blocked") or macro.get("score", 0) >= 81 or volatility_score >= 70 else "Trung bình"
    signal = {
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "symbol": config.get("trade", {}).get("symbol", "XAUUSD"),
        "action": action if action in ALLOWED_ACTIONS else "WAIT",
        "mode": config.get("trade", {}).get("mode", "Manual"),
        "confidence": confidence,
        "winrate": market_regime.get("probability", confidence),
        "risk_level": risk_level,
        "entry_zone": best.get("entry", "Chờ"),
        "take_profit": best.get("take_profit"),
        "stop_loss": best.get("stop_loss"),
        "strategy": best.get("strategy", "Chờ tín hiệu"),
        "reason": reason,
        "allow_auto_trade": bool(config.get("trade", {}).get("allow_auto_trade", False) and action in {"BUY", "SELL"} and risk_level != "Cao"),
        "conflict": conflict,
        "mtf_summary": mtf.get("summary", ""),
        "market_regime": market_regime.get("label", gold_analysis.get("regime", "Chưa rõ")),
        "market_regime_score": market_regime.get("score", 0),
        "volatility": volatility.get("level", "Chưa rõ"),
        "volatility_score": volatility_score,
    }
    risk_check = assess_auto_trade(signal, config, event_risk, trade_status)
    signal["risk_check"] = risk_check
    if not risk_check["allow_auto_trade"]:
        signal["allow_auto_trade"] = False
    return signal


def write_signal(signal: dict, shared_dir: Path) -> None:
    save_json(shared_dir / "signal.json", signal)
    save_json(Path("data") / "signal.json", signal)

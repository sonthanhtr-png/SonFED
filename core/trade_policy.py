from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from modules.utils import save_json


@dataclass
class AITradePolicy:
    symbol: str = "XAUUSD"
    allow_buy: bool = True
    allow_sell: bool = True
    max_buy_orders: int = 3
    max_sell_orders: int = 3
    default_lot: float = 0.03
    max_buy_volume: float = 0.09
    max_sell_volume: float = 0.09
    min_confidence: int = 70
    min_rr: float = 1.2
    max_spread: int = 35
    filter_high_volatility: bool = True
    filter_important_news: bool = True
    allow_sonexec_read_signal: bool = False
    allow_auto_execution: bool = False
    allow_auto_adjustment: bool = False
    position_management_strategy: str = "AI thích nghi"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_policy(policy: AITradePolicy) -> AITradePolicy:
    policy.symbol = policy.symbol.strip() or "XAUUSD"
    policy.max_buy_orders = int(max(0, min(20, policy.max_buy_orders)))
    policy.max_sell_orders = int(max(0, min(20, policy.max_sell_orders)))
    policy.default_lot = float(max(0.01, min(10.0, policy.default_lot)))
    policy.max_buy_volume = float(max(0.0, min(100.0, policy.max_buy_volume)))
    policy.max_sell_volume = float(max(0.0, min(100.0, policy.max_sell_volume)))
    policy.min_confidence = int(max(1, min(100, policy.min_confidence)))
    policy.min_rr = float(max(0.1, min(10.0, policy.min_rr)))
    policy.max_spread = int(max(1, min(5000, policy.max_spread)))
    if policy.position_management_strategy not in {"Bảo toàn vốn", "Break-even", "Bám xu hướng", "AI thích nghi"}:
        policy.position_management_strategy = "AI thích nghi"
    return policy


def policy_from_config(config: dict[str, Any]) -> AITradePolicy:
    trade = config.setdefault("trade", {})
    raw = config.setdefault("trade_policy", {})
    policy = AITradePolicy(
        symbol=str(trade.get("symbol") or raw.get("symbol") or "XAUUSD"),
        allow_buy=bool(raw.get("allow_buy", True)),
        allow_sell=bool(raw.get("allow_sell", True)),
        max_buy_orders=int(raw.get("max_buy_orders", 3)),
        max_sell_orders=int(raw.get("max_sell_orders", 3)),
        default_lot=float(raw.get("default_lot", trade.get("default_lot", 0.03))),
        max_buy_volume=float(raw.get("max_buy_volume", 0.09)),
        max_sell_volume=float(raw.get("max_sell_volume", 0.09)),
        min_confidence=int(raw.get("min_confidence", trade.get("min_confidence", 70))),
        min_rr=float(raw.get("min_rr", 1.2)),
        max_spread=int(raw.get("max_spread", trade.get("max_spread_points", 35))),
        filter_high_volatility=bool(raw.get("filter_high_volatility", True)),
        filter_important_news=bool(raw.get("filter_important_news", True)),
        allow_sonexec_read_signal=bool(raw.get("allow_sonexec_read_signal", False)),
        allow_auto_execution=bool(raw.get("allow_auto_execution", trade.get("allow_auto_trade", False))),
        allow_auto_adjustment=bool(raw.get("allow_auto_adjustment", False)),
        position_management_strategy=str(raw.get("position_management_strategy", "AI thích nghi")),
    )
    return validate_policy(policy)


def save_policy_to_config(config: dict[str, Any], policy: AITradePolicy) -> dict[str, Any]:
    policy = validate_policy(policy)
    config.setdefault("trade", {})
    config["trade"]["symbol"] = policy.symbol
    config["trade"]["min_confidence"] = policy.min_confidence
    config["trade"]["max_spread_points"] = policy.max_spread
    config["trade"]["allow_auto_trade"] = policy.allow_auto_execution
    config["trade"]["default_lot"] = policy.default_lot
    config["trade_policy"] = policy.to_dict()
    return config


def _count_positions(positions: list[dict[str, Any]], side: str, symbol: str) -> int:
    side = side.upper()
    symbol = symbol.upper()
    count = 0
    for pos in positions:
        pos_side = str(pos.get("type", "")).upper()
        pos_symbol = str(pos.get("symbol", "")).upper()
        if side in pos_side and (not pos_symbol or pos_symbol == symbol):
            count += 1
    return count


def _sum_position_volume(positions: list[dict[str, Any]], side: str, symbol: str) -> float:
    side = side.upper()
    symbol = symbol.upper()
    total = 0.0
    for pos in positions:
        pos_side = str(pos.get("type", pos.get("type_name", ""))).upper()
        pos_symbol = str(pos.get("symbol", "")).upper()
        if side in pos_side and (not pos_symbol or pos_symbol == symbol):
            try:
                total += float(pos.get("lot", pos.get("volume", 0)) or 0)
            except Exception:
                continue
    return total


def build_market_state(
    ai_decision: dict[str, Any],
    signal: dict[str, Any],
    gold_analysis: dict[str, Any],
    event_risk: dict[str, Any],
    risk_feedback: dict[str, Any],
    trade_feedback: dict[str, Any],
) -> dict[str, Any]:
    positions = trade_feedback.get("positions", []) if trade_feedback.get("connected") else []
    return {
        "initial_decision": ai_decision.get("action", signal.get("action", "WAIT")),
        "confidence": int(signal.get("confidence", ai_decision.get("winrate", 0)) or 0),
        "winrate": int(ai_decision.get("winrate", signal.get("winrate", 0)) or 0),
        "rr": ai_decision.get("rr"),
        "tp": ai_decision.get("tp", signal.get("take_profit")),
        "sl": ai_decision.get("sl", signal.get("stop_loss")),
        "risk_level": signal.get("risk_level", "Trung bình"),
        "regime": signal.get("market_regime", gold_analysis.get("regime", "Chưa rõ")),
        "volatility": signal.get("volatility", gold_analysis.get("volatility", {}).get("level", "Chưa rõ")),
        "volatility_score": int(signal.get("volatility_score", gold_analysis.get("volatility", {}).get("score", 0)) or 0),
        "spread": risk_feedback.get("spread_points"),
        "news_blocked": bool(event_risk.get("blocked")),
        "buy_orders": _count_positions(positions, "BUY", signal.get("symbol", "XAUUSD")),
        "sell_orders": _count_positions(positions, "SELL", signal.get("symbol", "XAUUSD")),
        "buy_volume": _sum_position_volume(positions, "BUY", signal.get("symbol", "XAUUSD")),
        "sell_volume": _sum_position_volume(positions, "SELL", signal.get("symbol", "XAUUSD")),
    }


def apply_policy_to_signal(signal: dict[str, Any], policy: AITradePolicy, market_state: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    policy = validate_policy(policy)
    updated = dict(signal)
    original_decision = market_state.get("initial_decision", updated.get("action", "WAIT"))
    decision = original_decision if original_decision in {"BUY", "SELL", "WAIT"} else "WAIT"
    reasons: list[str] = []

    if decision == "BUY" and not policy.allow_buy:
        reasons.append("BUY đang bị tắt trong Chính sách giao dịch AI.")
    if decision == "SELL" and not policy.allow_sell:
        reasons.append("SELL đang bị tắt trong Chính sách giao dịch AI.")
    if decision == "BUY" and market_state.get("buy_orders", 0) >= policy.max_buy_orders:
        reasons.append("Số lệnh BUY đã đạt giới hạn trong Chính sách giao dịch AI.")
    if decision == "SELL" and market_state.get("sell_orders", 0) >= policy.max_sell_orders:
        reasons.append("Số lệnh SELL đã đạt giới hạn trong Chính sách giao dịch AI.")
    if decision == "BUY" and float(market_state.get("buy_volume") or 0) + policy.default_lot > policy.max_buy_volume:
        reasons.append("Tổng volume BUY sẽ vượt giới hạn trong Chính sách giao dịch AI.")
    if decision == "SELL" and float(market_state.get("sell_volume") or 0) + policy.default_lot > policy.max_sell_volume:
        reasons.append("Tổng volume SELL sẽ vượt giới hạn trong Chính sách giao dịch AI.")

    confidence = int(market_state.get("confidence") or 0)
    winrate = int(market_state.get("winrate") or confidence)
    if max(confidence, winrate) < policy.min_confidence:
        reasons.append("Độ tin cậy chưa đạt ngưỡng tối thiểu.")

    rr = market_state.get("rr")
    if decision in {"BUY", "SELL"} and (rr is None or float(rr) < policy.min_rr):
        reasons.append("Tỷ lệ RR chưa đạt yêu cầu.")

    spread = market_state.get("spread")
    if spread is not None and float(spread) > policy.max_spread:
        reasons.append("Spread vượt ngưỡng cho phép.")

    if policy.filter_high_volatility and int(market_state.get("volatility_score") or 0) >= 70:
        reasons.append("Biến động thị trường quá mạnh, không phù hợp để vào lệnh tự động.")

    if policy.filter_important_news and market_state.get("news_blocked"):
        reasons.append("Đang gần thời điểm tin tức quan trọng.")

    if reasons:
        decision = "WAIT"

    reason = " ".join(reasons) if reasons else updated.get("reason", "")
    updated.update(
        {
            "symbol": policy.symbol,
            "decision": decision,
            "action": decision,
            "confidence": confidence,
            "winrate": winrate,
            "take_profit": market_state.get("tp"),
            "stop_loss": market_state.get("sl"),
            "tp": market_state.get("tp"),
            "sl": market_state.get("sl"),
            "rr": market_state.get("rr"),
            "lot": policy.default_lot,
            "risk_level": market_state.get("risk_level"),
            "regime": market_state.get("regime"),
            "market_regime": market_state.get("regime"),
            "volatility": market_state.get("volatility"),
            "policy": policy.to_dict(),
            "reason": reason,
            "allow_auto_trade": bool(policy.allow_sonexec_read_signal and policy.allow_auto_execution and decision in {"BUY", "SELL"} and not reasons),
            "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "policy_blocked": bool(reasons),
        }
    )
    result = {
        "initial_decision": original_decision,
        "final_decision": decision,
        "blocked": bool(reasons),
        "reasons": reasons,
        "message": reason,
    }
    return updated, result


def write_signal_if_allowed(signal: dict[str, Any], policy: AITradePolicy, shared_dir: Path) -> bool:
    if not policy.allow_sonexec_read_signal:
        return False
    save_json(shared_dir / "signal.json", signal)
    save_json(Path("data") / "signal.json", signal)
    return True

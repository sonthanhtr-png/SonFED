from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from .utils import save_json


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _normalize_signal(value: Any) -> str:
    text = str(value or "WAIT").upper()
    return text if text in {"BUY", "SELL", "WAIT"} else "WAIT"


def _normalize_regime(label: str, volatility_score: int) -> str:
    text = str(label or "").upper()
    if "EXHAUST" in text or "REVERSAL" in text or "ĐẢO" in text:
        return "REVERSAL"
    if "STRONG TREND" in text or "BULL EXPANSION" in text or "BEAR EXPANSION" in text:
        return "TREND"
    if "WEAK TREND" in text:
        return "WEAK_TREND"
    if "VOLATILE" in text or volatility_score >= 70:
        return "VOLATILE"
    if "TREND" in text:
        return "TREND"
    return "SIDEWAY"


def _normalize_volatility(score: int) -> str:
    if score >= 90:
        return "EXTREME"
    if score >= 65:
        return "HIGH"
    if score >= 30:
        return "NORMAL"
    return "LOW"


def _normalize_momentum(signal: dict[str, Any], regime: dict[str, Any]) -> str:
    score = _as_int(signal.get("momentum_score", regime.get("scalp_pressure", 0)))
    if score >= 3:
        return "STRONG"
    if score >= 1:
        return "NORMAL"
    text = str(regime.get("momentum", "")).lower()
    if "tăng" in text or "giảm" in text or "strong" in text:
        return "NORMAL"
    return "WEAK"


def _macro_bias(pressure_index: int) -> str:
    if pressure_index <= 35:
        return "BUY"
    if pressure_index >= 65:
        return "SELL"
    return "NEUTRAL"


def _multi_tf_alignment(mtf: dict[str, Any]) -> bool:
    summary = str(mtf.get("summary", "")).lower()
    if "lệch pha" in summary or "lech pha" in summary:
        return False
    if "đồng thuận" in summary or "dong thuan" in summary:
        return True

    regimes = mtf.get("regimes", {})
    biases: list[str] = []
    if isinstance(regimes, dict):
        for key in ("M15", "H1", "H4"):
            item = regimes.get(key, {})
            bias = str(item.get("bias", "")).upper() if isinstance(item, dict) else ""
            if bias in {"BUY", "SELL"}:
                biases.append(bias)
    return len(biases) >= 2 and len(set(biases)) == 1


def _is_high_risk(risk_level: Any) -> bool:
    text = str(risk_level or "").lower()
    return "cao" in text or "high" in text


def _execution_mode(
    signal: str,
    confidence: int,
    regime: str,
    volatility: str,
    momentum: str,
    pressure_index: int,
    aligned: bool,
    risk_level: Any,
    scalp_accepted: bool,
) -> str:
    if signal == "WAIT":
        return "SAFE"
    if pressure_index > 80 or volatility == "EXTREME" or not aligned or _is_high_risk(risk_level):
        return "SAFE"
    if regime == "TREND" and momentum == "STRONG" and confidence >= 55:
        return "TREND"
    if regime in {"SIDEWAY", "VOLATILE", "WEAK_TREND"} or scalp_accepted:
        return "SCALP"
    if volatility == "HIGH":
        return "SAFE"
    return "SCALP"


def _recommended_trailing(execution_mode: str, volatility: str, regime: str) -> str:
    if execution_mode == "SAFE":
        return "SAFE"
    if execution_mode == "TREND" and volatility in {"HIGH", "EXTREME"}:
        return "ATR"
    if execution_mode == "TREND" or regime == "TREND":
        return "TREND"
    return "FAST"


def _recommended_risk_mode(
    signal: str,
    confidence: int,
    volatility: str,
    pressure_index: int,
    aligned: bool,
    momentum: str,
    risk_level: Any,
) -> str:
    if signal == "WAIT" or volatility in {"HIGH", "EXTREME"} or pressure_index > 80 or not aligned or _is_high_risk(risk_level):
        return "SAFE"
    if confidence >= 75 and momentum == "STRONG" and aligned:
        return "AGGRESSIVE"
    return "NORMAL"


def build_ai_state(
    signal: dict[str, Any],
    gold_analysis: dict[str, Any],
    macro: dict[str, Any],
    mtf: dict[str, Any],
    ai_decision: dict[str, Any] | None = None,
    policy_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    regime_data = gold_analysis.get("market_regime", {}) if isinstance(gold_analysis, dict) else {}
    volatility_data = regime_data.get("volatility", {}) if isinstance(regime_data, dict) else {}
    volatility_score = _as_int(signal.get("volatility_score", volatility_data.get("score", 0)))
    pressure = _as_int(macro.get("score", signal.get("pressure_index", 50)), 50)
    action = _normalize_signal(signal.get("action"))
    confidence = _as_int(signal.get("confidence", 0))
    regime = _normalize_regime(str(regime_data.get("label", signal.get("market_regime", ""))), volatility_score)
    volatility = _normalize_volatility(volatility_score)
    momentum = _normalize_momentum(signal, regime_data if isinstance(regime_data, dict) else {})
    aligned = _multi_tf_alignment(mtf)
    scalp_accepted = bool(signal.get("scalp_accepted"))
    risk_level = signal.get("risk_level", regime_data.get("risk_level", "MEDIUM") if isinstance(regime_data, dict) else "MEDIUM")

    execution = _execution_mode(
        action,
        confidence,
        regime,
        volatility,
        momentum,
        pressure,
        aligned,
        risk_level,
        scalp_accepted,
    )
    trailing = _recommended_trailing(execution, volatility, regime)
    risk_mode = _recommended_risk_mode(action, confidence, volatility, pressure, aligned, momentum, risk_level)

    return {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "schema_version": 1,
        "source": "SonFED",
        "signal": action,
        "confidence": confidence,
        "market_regime": regime,
        "volatility": volatility,
        "volatility_score": volatility_score,
        "momentum": momentum,
        "momentum_score": _as_int(signal.get("momentum_score", 0)),
        "risk_level": "HIGH" if _is_high_risk(risk_level) else str(risk_level or "MEDIUM").upper(),
        "pressure_index": pressure,
        "multi_tf_alignment": bool(aligned),
        "macro_bias": _macro_bias(pressure),
        "execution_mode": execution,
        "recommended_trailing": trailing,
        "recommended_risk_mode": risk_mode,
        "strategy": signal.get("strategy", ""),
        "reason": signal.get("reason", ""),
        "allow_auto_trade": bool(signal.get("allow_auto_trade", False) and action in {"BUY", "SELL"}),
        "scalp_accepted": scalp_accepted,
        "policy_blocked": bool((policy_result or {}).get("blocked", False)),
        "policy_reasons": (policy_result or {}).get("reasons", []),
        "ai_decision": ai_decision or {},
        "mtf_summary": mtf.get("summary", ""),
    }


def write_ai_state(ai_state: dict[str, Any], shared_dir: Path) -> None:
    save_json(shared_dir / "ai_state.json", ai_state)
    heartbeat = {
        "timestamp": ai_state.get("timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
        "source": "SonFED",
        "status": "RUNNING",
        "signal": ai_state.get("signal", "WAIT"),
        "confidence": ai_state.get("confidence", 0),
        "market_regime": ai_state.get("market_regime", "SIDEWAY"),
        "execution_mode": ai_state.get("execution_mode", "SAFE"),
    }
    risk_state = {
        "timestamp": heartbeat["timestamp"],
        "source": "SonFED",
        "risk_level": ai_state.get("risk_level", "MEDIUM"),
        "recommended_risk_mode": ai_state.get("recommended_risk_mode", "SAFE"),
        "pressure_index": ai_state.get("pressure_index", 50),
        "volatility": ai_state.get("volatility", "NORMAL"),
        "multi_tf_alignment": ai_state.get("multi_tf_alignment", False),
        "allow_auto_trade": ai_state.get("allow_auto_trade", False),
    }
    save_json(shared_dir / "heartbeat.json", heartbeat)
    save_json(shared_dir / "risk_state.json", risk_state)
    save_json(Path("data") / "ai_state.json", ai_state)
    save_json(Path("data") / "heartbeat.json", heartbeat)

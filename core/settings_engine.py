from __future__ import annotations

import math
from typing import Any

from modules.utils import ROOT, load_json, save_json


SETTINGS_PATH = ROOT / "data" / "sonfed_settings.json"

AI_MODES = ["An toàn", "Cân bằng", "Chủ động", "Tấn công"]

POSITION_STRATEGIES = [
    "Bảo toàn vốn",
    "Dời SL về hòa vốn",
    "Bám xu hướng",
    "AI tự thích nghi",
]

STRATEGY_ALIASES = {
    "Break-even": "Dời SL về hòa vốn",
    "AI thích nghi": "AI tự thích nghi",
}

AI_MODE_PRESETS: dict[str, dict[str, Any]] = {
    "An toàn": {
        "min_ai_confidence": 80,
        "min_rr": 1.8,
        "max_spread": 30,
        "avoid_high_volatility": True,
        "avoid_news": True,
        "position_strategy": "Bảo toàn vốn",
    },
    "Cân bằng": {
        "min_ai_confidence": 70,
        "min_rr": 1.2,
        "max_spread": 35,
        "avoid_high_volatility": True,
        "avoid_news": True,
        "position_strategy": "AI tự thích nghi",
    },
    "Chủ động": {
        "min_ai_confidence": 60,
        "min_rr": 1.0,
        "max_spread": 45,
        "avoid_high_volatility": False,
        "avoid_news": True,
        "position_strategy": "Bám xu hướng",
    },
    "Tấn công": {
        "min_ai_confidence": 50,
        "min_rr": 0.8,
        "max_spread": 60,
        "avoid_high_volatility": False,
        "avoid_news": False,
        "position_strategy": "AI tự thích nghi",
    },
}

ADVANCED_SETTING_KEYS = tuple(next(iter(AI_MODE_PRESETS.values())).keys())
OBSOLETE_SETTING_KEYS = {
    "max_total_buy_lot",
    "max_total_sell_lot",
    "min_ai_confidence",
    "min_rr",
    "max_spread",
    "avoid_high_volatility",
    "avoid_news",
    "position_strategy",
}

DEFAULT_SETTINGS: dict[str, Any] = {
    "default_lot": 0.03,
    "max_buy_orders": 3,
    "max_sell_orders": 3,
    "ai_mode": "Chủ động",
    "allow_sonexec_signal_read": False,
    "enable_auto_execution": False,
    "enable_position_management": False,
    "advanced_settings": None,
}


def get_default_settings() -> dict[str, Any]:
    return dict(DEFAULT_SETTINGS)


def _to_float(value: Any, default: float, minimum: float, maximum: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = default
    if not math.isfinite(number):
        number = default
    return float(max(minimum, min(maximum, number)))


def _to_int(value: Any, default: int, minimum: int, maximum: int) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        number = default
    return int(max(minimum, min(maximum, number)))


def _to_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on", "bật", "bat", "có", "co"}:
            return True
        if normalized in {"0", "false", "no", "off", "tắt", "tat", "không", "khong"}:
            return False
    return default


def _to_strategy(value: Any, default: str) -> str:
    strategy = STRATEGY_ALIASES.get(str(value), str(value))
    return strategy if strategy in POSITION_STRATEGIES else default


def _to_ai_mode(value: Any, default: str) -> str:
    mode = str(value)
    return mode if mode in AI_MODES else default


def validate_advanced_settings(settings: dict[str, Any] | None = None, ai_mode: str = "Cân bằng") -> dict[str, Any]:
    raw = settings if isinstance(settings, dict) else {}
    preset = AI_MODE_PRESETS.get(ai_mode, AI_MODE_PRESETS["Cân bằng"])
    return {
        "min_ai_confidence": _to_int(raw.get("min_ai_confidence"), preset["min_ai_confidence"], 1, 100),
        "min_rr": round(_to_float(raw.get("min_rr"), preset["min_rr"], 0.1, 10.0), 2),
        "max_spread": _to_int(raw.get("max_spread"), preset["max_spread"], 1, 5000),
        "avoid_high_volatility": _to_bool(raw.get("avoid_high_volatility"), preset["avoid_high_volatility"]),
        "avoid_news": _to_bool(raw.get("avoid_news"), preset["avoid_news"]),
        "position_strategy": _to_strategy(raw.get("position_strategy"), preset["position_strategy"]),
    }


def get_ai_mode_settings(ai_mode: str) -> dict[str, Any]:
    mode = _to_ai_mode(ai_mode, DEFAULT_SETTINGS["ai_mode"])
    return validate_advanced_settings(AI_MODE_PRESETS[mode], mode)


def get_effective_advanced_settings(settings: dict[str, Any]) -> dict[str, Any]:
    validated = validate_settings(settings)
    advanced = validated.get("advanced_settings")
    if isinstance(advanced, dict):
        return advanced
    return get_ai_mode_settings(validated["ai_mode"])


def _migrate_legacy_advanced_settings(raw: dict[str, Any], ai_mode: str) -> dict[str, Any] | None:
    if isinstance(raw.get("advanced_settings"), dict):
        return validate_advanced_settings(raw["advanced_settings"], ai_mode)

    if not any(key in raw for key in OBSOLETE_SETTING_KEYS):
        return None

    migrated = validate_advanced_settings(raw, ai_mode)
    if migrated == get_ai_mode_settings(ai_mode):
        return None
    return migrated


def validate_settings(settings: dict[str, Any] | None = None) -> dict[str, Any]:
    raw = settings if isinstance(settings, dict) else {}
    defaults = get_default_settings()
    ai_mode = _to_ai_mode(raw.get("ai_mode"), defaults["ai_mode"])
    return {
        "default_lot": round(_to_float(raw.get("default_lot"), defaults["default_lot"], 0.01, 10.0), 2),
        "max_buy_orders": _to_int(raw.get("max_buy_orders"), defaults["max_buy_orders"], 0, 20),
        "max_sell_orders": _to_int(raw.get("max_sell_orders"), defaults["max_sell_orders"], 0, 20),
        "ai_mode": ai_mode,
        "allow_sonexec_signal_read": _to_bool(raw.get("allow_sonexec_signal_read"), defaults["allow_sonexec_signal_read"]),
        "enable_auto_execution": _to_bool(raw.get("enable_auto_execution"), defaults["enable_auto_execution"]),
        "enable_position_management": _to_bool(raw.get("enable_position_management"), defaults["enable_position_management"]),
        "advanced_settings": _migrate_legacy_advanced_settings(raw, ai_mode),
    }


def load_sonfed_settings() -> dict[str, Any]:
    raw = load_json(SETTINGS_PATH, {})
    if not isinstance(raw, dict):
        raw = {}

    merged = dict(raw)
    for key in OBSOLETE_SETTING_KEYS:
        merged.pop(key, None)
    validated = validate_settings(raw)
    merged.update(validated)

    if not SETTINGS_PATH.exists() or any(merged.get(key) != raw.get(key) for key in merged) or any(key in raw for key in OBSOLETE_SETTING_KEYS):
        save_json(SETTINGS_PATH, merged)

    return merged


def save_sonfed_settings(settings: dict[str, Any]) -> dict[str, Any]:
    current = load_json(SETTINGS_PATH, {})
    if not isinstance(current, dict):
        current = {}

    for key in OBSOLETE_SETTING_KEYS:
        current.pop(key, None)
    validated = validate_settings(settings)
    current.update(validated)
    save_json(SETTINGS_PATH, current)
    return current

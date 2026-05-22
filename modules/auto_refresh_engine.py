from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

from .utils import ROOT, load_json, save_json


STATE_PATH = ROOT / "data" / "auto_refresh_state.json"
SNAPSHOT_PATH = ROOT / "data" / "market_snapshots.json"
SETTINGS_PATH = ROOT / "data" / "sonfed_settings.json"
INTERVAL_OPTIONS = [1, 3, 5, 15, 30]
DEFAULT_INTERVAL = 5


def ensure_auto_refresh_config(config: dict) -> dict:
    auto = config.setdefault("auto_refresh", {})
    auto.setdefault("enabled", False)
    auto.setdefault("interval_minutes", DEFAULT_INTERVAL)
    if auto["interval_minutes"] not in INTERVAL_OPTIONS:
        auto["interval_minutes"] = DEFAULT_INTERVAL
    return config


def load_state() -> dict:
    state = load_json(STATE_PATH, {})
    return state if isinstance(state, dict) else {}


def save_state(state: dict) -> None:
    save_json(STATE_PATH, state)


def load_sonfed_settings() -> dict:
    settings = load_json(SETTINGS_PATH, {})
    if not isinstance(settings, dict):
        settings = {}
    settings.setdefault("auto_refresh_enabled", False)
    settings.setdefault("refresh_interval_minutes", DEFAULT_INTERVAL)
    settings.setdefault("telegram_enabled", False)
    settings.setdefault("auto_trade_enabled", False)
    if settings["refresh_interval_minutes"] not in INTERVAL_OPTIONS:
        settings["refresh_interval_minutes"] = DEFAULT_INTERVAL
    return settings


def save_sonfed_settings(settings: dict) -> None:
    clean = load_sonfed_settings()
    clean.update(settings)
    if clean["refresh_interval_minutes"] not in INTERVAL_OPTIONS:
        clean["refresh_interval_minutes"] = DEFAULT_INTERVAL
    save_json(SETTINGS_PATH, clean)


def get_auto_refresh_enabled(default: bool = False) -> bool:
    return bool(load_sonfed_settings().get("auto_refresh_enabled", default))


def set_auto_refresh_enabled(value: bool) -> None:
    settings = load_sonfed_settings()
    settings["auto_refresh_enabled"] = bool(value)
    save_sonfed_settings(settings)


def parse_time(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except Exception:
        return None


def fmt_time(value: str | None) -> str:
    dt = parse_time(value)
    return dt.strftime("%H:%M:%S %d/%m/%Y") if dt else "Chưa có"


def prepare_refresh(config: dict, manual_refresh: bool = False) -> dict:
    config = ensure_auto_refresh_config(config)
    auto = config.get("auto_refresh", {})
    state = load_state()
    now = datetime.now()
    interval = int(auto.get("interval_minutes", DEFAULT_INTERVAL))
    next_update = parse_time(state.get("next_update_time"))

    due = bool(manual_refresh)
    if auto.get("enabled"):
        due = due or next_update is None or now >= next_update

    return {
        "enabled": bool(auto.get("enabled")),
        "interval_minutes": interval,
        "due": due,
        "state": state,
        "last_update_time": state.get("last_update_time"),
        "next_update_time": state.get("next_update_time"),
    }


def _latest_value(bundle: dict, key: str) -> float | None:
    try:
        df = bundle.get(key)
        if df is None or df.empty:
            return None
        return float(df["Close"].dropna().iloc[-1])
    except Exception:
        return None


def build_snapshot(
    bundle: dict,
    macro: dict,
    gold_analysis: dict,
    mtf: dict,
    signal: dict,
    market_summary: str,
) -> dict:
    levels = gold_analysis.get("levels", {})
    return {
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "gold_price": _latest_value(bundle, "GOLD"),
        "dxy": _latest_value(bundle, "DXY"),
        "us10y": _latest_value(bundle, "US10Y"),
        "vix": _latest_value(bundle, "VIX"),
        "oil": _latest_value(bundle, "OIL"),
        "nasdaq": _latest_value(bundle, "NASDAQ"),
        "sp500": _latest_value(bundle, "SP500"),
        "pressure_index": macro.get("score"),
        "action": signal.get("action", "WAIT"),
        "strategy": signal.get("strategy", "Chờ tín hiệu"),
        "confidence": signal.get("confidence", 0),
        "summary": market_summary,
        "regime": gold_analysis.get("regime", ""),
        "m15_trend": mtf.get("trends", {}).get("M15", ""),
        "h1_trend": mtf.get("trends", {}).get("H1", ""),
        "support": levels.get("support"),
        "resistance": levels.get("resistance"),
    }


def load_snapshots(limit: int = 200) -> list[dict]:
    rows = load_json(SNAPSHOT_PATH, [])
    if not isinstance(rows, list):
        return []
    return rows[-limit:]


def append_snapshot(snapshot: dict, limit: int = 500) -> None:
    rows = load_snapshots(limit)
    rows.append(snapshot)
    save_json(SNAPSHOT_PATH, rows[-limit:])


def signal_key(signal: dict) -> str:
    return "|".join(
        str(signal.get(key, ""))
        for key in ("action", "strategy", "confidence", "entry_zone", "risk_level")
    )


def detect_changes(previous: dict | None, snapshot: dict, signal: dict, macro: dict, event_risk: dict, risk_feedback: dict) -> list[str]:
    if not previous:
        return ["Đã tạo ảnh chụp thị trường đầu tiên."]

    changes: list[str] = []
    prev_action = previous.get("action", "WAIT")
    action = snapshot.get("action", "WAIT")
    if prev_action != action and {prev_action, action} & {"BUY", "SELL"}:
        changes.append(f"Tín hiệu đổi từ {prev_action} sang {action}.")

    prev_pressure = int(previous.get("pressure_index") or 0)
    pressure = int(snapshot.get("pressure_index") or 0)
    if prev_pressure <= 70 < pressure:
        changes.append(f"SonFED Pressure Index vượt 70: {pressure}/100.")

    price = snapshot.get("gold_price")
    resistance = snapshot.get("resistance")
    prev_price = previous.get("gold_price")
    if price and resistance and prev_price and prev_price <= resistance < price:
        changes.append(f"Giá vàng break kháng cự quan trọng {resistance:.2f}.")

    summary_text = f"{snapshot.get('summary', '')} {signal.get('reason', '')}".lower()
    if "reject" in summary_text or "quét thanh khoản" in summary_text:
        changes.append("Giá có dấu hiệu reject hoặc quét vùng quan trọng.")

    if abs(float(macro.get("dxy_change") or 0)) >= 0.2 and float(macro.get("dxy_change") or 0) > 0:
        changes.append(f"DXY tăng mạnh: {float(macro.get('dxy_change') or 0):.2f}%.")

    # US10Y không được macro trả riêng, so sánh trực tiếp snapshot hiện tại với lần trước.
    prev_us10y = previous.get("us10y")
    us10y = snapshot.get("us10y")
    if prev_us10y and us10y and prev_us10y != 0:
        us10y_change = (us10y - prev_us10y) / abs(prev_us10y) * 100
        if us10y_change >= 0.5:
            changes.append(f"US10Y tăng mạnh: {us10y_change:.2f}%.")

    if event_risk.get("blocked"):
        changes.append(event_risk.get("message", "Sắp có tin lớn."))

    if risk_feedback.get("connected") and not risk_feedback.get("allow", True):
        changes.append(f"Lệnh SonEXEC đang gặp rủi ro: {risk_feedback.get('reason', '')}")

    return changes


def should_send_telegram(changes: list[str], state: dict, signal: dict) -> tuple[bool, str]:
    important = [c for c in changes if "đầu tiên" not in c.lower()]
    if not important:
        return False, ""
    key = signal_key(signal) + "|" + "|".join(important)
    if state.get("last_telegram_key") == key:
        return False, key
    last_sent = parse_time(state.get("last_telegram_time"))
    if last_sent and datetime.now() - last_sent < timedelta(minutes=10):
        return False, key
    return True, key


def finalize_refresh(
    refresh: dict,
    snapshot: dict,
    signal: dict,
    changes: list[str],
    telegram_key: str | None = None,
    telegram_sent: bool = False,
) -> dict:
    state = dict(refresh.get("state") or {})
    now = datetime.now()
    interval = int(refresh.get("interval_minutes", DEFAULT_INTERVAL))

    previous_action = state.get("current_action", "Chưa có")
    state.update(
        {
            "last_update_time": now.isoformat(timespec="seconds"),
            "next_update_time": (now + timedelta(minutes=interval)).isoformat(timespec="seconds"),
            "previous_action": previous_action,
            "current_action": signal.get("action", "WAIT"),
            "last_signal_key": signal_key(signal),
            "last_changes": changes,
            "last_snapshot": snapshot,
        }
    )
    if telegram_sent and telegram_key:
        state["last_telegram_key"] = telegram_key
        state["last_telegram_time"] = now.isoformat(timespec="seconds")

    append_snapshot(snapshot)
    save_state(state)
    return state


def build_market_summary(
    snapshot: dict,
    macro: dict,
    gold_analysis: dict,
    mtf: dict,
    bias: str,
    signal: dict,
    changes: list[str] | None = None,
) -> str:
    changes = changes or []
    return "\n".join(
        [
            f"Cập nhật lúc: {datetime.now().strftime('%H:%M')}",
            "",
            f"Giá vàng hiện tại: {snapshot.get('gold_price') or 'N/A'}",
            f"SonFED Pressure Index: {macro.get('score')}/100",
            f"Trạng thái thị trường: {gold_analysis.get('regime', 'Chưa rõ')}",
            f"Volatility: {gold_analysis.get('volatility', {}).get('level', 'Chưa rõ')}",
            f"Xu hướng M15: {mtf.get('trends', {}).get('M15', 'Chưa rõ')}",
            f"Xu hướng H1: {mtf.get('trends', {}).get('H1', 'Chưa rõ')}",
            f"DXY: {snapshot.get('dxy') or 'N/A'}",
            f"US10Y: {snapshot.get('us10y') or 'N/A'}",
            "",
            "Kết luận:",
            gold_analysis.get("summary", ""),
            bias,
            "",
            "Chiến lược hiện tại:",
            f"{signal.get('action', 'WAIT')} - {signal.get('strategy', 'Chờ tín hiệu')} - winrate {signal.get('winrate', signal.get('confidence', 0))}%.",
            "",
            "Lưu ý:",
            " ".join(changes) if changes else signal.get("reason", "Chưa có thay đổi quan trọng."),
        ]
    )


def dashboard_summary(refresh: dict) -> dict:
    state = refresh.get("state") or {}
    return {
        "enabled": "Bật" if refresh.get("enabled") else "Tắt",
        "interval": f"{refresh.get('interval_minutes', DEFAULT_INTERVAL)} phút",
        "last_update": fmt_time(state.get("last_update_time")),
        "next_update": fmt_time(state.get("next_update_time")),
        "current_action": state.get("current_action", "Chưa có"),
        "previous_action": state.get("previous_action", "Chưa có"),
        "changes": state.get("last_changes", []),
    }

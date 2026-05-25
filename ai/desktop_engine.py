from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yfinance as yf

from modules.ai_state_bridge import build_ai_state, write_ai_state
from modules.events import event_risk, load_events
from modules.gold_analyzer import analyze_gold
from modules.indicators import add_indicators
from modules.macro_engine import gold_bias, pressure_index
from modules.market_regime_engine import build_decision
from modules.mtf_engine import analyze_mtf
from modules.signal_engine import create_signal, write_signal
from modules.strategy_engine import build_strategies
from modules.trade_bridge import read_shared
from modules.utils import latest_float, load_json, save_json
from shared.file_bus import DEFAULT_SHARED_DIR, SharedFileBus


ROOT = Path(__file__).resolve().parents[1]
SIGNAL_HISTORY_PATH = ROOT / "data" / "signal_history.json"
YF_CACHE_DIR = ROOT / "data" / "yf_cache"

YF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
if hasattr(yf, "set_tz_cache_location"):
    yf.set_tz_cache_location(str(YF_CACHE_DIR))


INTERVAL_MAP = {
    "M15": "15m",
    "H1": "1h",
    "H4": "4h",
    "D1": "1d",
    "15m": "15m",
    "1h": "1h",
    "4h": "4h",
    "1d": "1d",
}


def fetch_ohlcv(ticker: str, period: str = "6mo", interval: str = "1h") -> pd.DataFrame:
    try:
        yf_interval = INTERVAL_MAP.get(interval, interval)
        data = yf.download(ticker, period=period, interval=yf_interval, progress=False, auto_adjust=False, threads=False)
        if data is None or data.empty:
            return pd.DataFrame()
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        data = data.rename(columns=str.title)
        keep = [col for col in ["Open", "High", "Low", "Close", "Adj Close", "Volume"] if col in data.columns]
        data = data[keep].copy()
        data.index = pd.to_datetime(data.index)
        return data.dropna(subset=["Close"])
    except Exception:
        return pd.DataFrame()


def fetch_market_bundle(tickers: dict[str, str], period: str, interval: str) -> dict[str, pd.DataFrame]:
    return {name: fetch_ohlcv(ticker, period, interval) for name, ticker in tickers.items()}


def data_status(bundle: dict[str, pd.DataFrame]) -> dict[str, Any]:
    ok = [key for key, frame in bundle.items() if not frame.empty]
    missing = [key for key, frame in bundle.items() if frame.empty]
    return {"ok": ok, "missing": missing, "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _latest(frame: pd.DataFrame) -> float:
    if frame is None or frame.empty or "Close" not in frame:
        return 0.0
    return _safe_float(frame["Close"].dropna().iloc[-1], 0.0) if not frame["Close"].dropna().empty else 0.0


def _change_pct(frame: pd.DataFrame) -> float:
    if frame is None or frame.empty or "Close" not in frame:
        return 0.0
    close = frame["Close"].dropna()
    if len(close) < 2:
        return 0.0
    previous = _safe_float(close.iloc[-2])
    current = _safe_float(close.iloc[-1])
    return ((current - previous) / previous * 100) if previous else 0.0


def _bias_for_indicator(name: str, change_pct: float, frame: pd.DataFrame | None = None) -> str:
    name = name.upper()
    if name == "GOLD":
        if frame is not None and not frame.empty and {"Close", "MA20"}.issubset(frame.columns):
            close = _safe_float(frame["Close"].iloc[-1])
            ma20 = _safe_float(frame["MA20"].iloc[-1])
            if close > ma20 and change_pct >= 0:
                return "BUY"
            if close < ma20 and change_pct <= 0:
                return "SELL"
        return "WAIT"
    if abs(change_pct) < 0.05:
        return "WAIT"
    if name in {"DXY", "US10Y", "US02Y", "OIL", "NASDAQ"}:
        return "SELL" if change_pct > 0 else "BUY"
    if name == "VIX":
        return "BUY" if change_pct > 0 else "WAIT"
    return "WAIT"


RADAR_EXPLANATIONS = {
    "Gold": "Giá vàng hiện tại. Trên MA20 và momentum tăng thường ưu tiên BUY; dưới MA20 và momentum giảm thường ưu tiên SELL.",
    "DXY": "DXY là sức mạnh USD. DXY tăng thường gây áp lực SELL vàng; DXY giảm thường hỗ trợ BUY vàng.",
    "US10Y": "Lợi suất Mỹ 10 năm. US10Y tăng làm chi phí nắm giữ vàng cao hơn; US10Y giảm thường hỗ trợ vàng.",
    "VIX": "Chỉ số sợ hãi. VIX tăng mạnh có thể hỗ trợ vàng do nhu cầu trú ẩn; VIX giảm thường trung tính hoặc giảm nhu cầu trú ẩn.",
    "Oil": "Giá dầu ảnh hưởng kỳ vọng lạm phát. Dầu tăng có thể khiến FED hawkish hơn, thường bất lợi cho vàng.",
    "Nasdaq": "Đại diện khẩu vị rủi ro. Nasdaq tăng mạnh là risk-on, vàng có thể yếu; Nasdaq giảm có thể hỗ trợ trú ẩn.",
    "CPI": "Lạm phát tiêu dùng Mỹ. Cao hơn kỳ vọng thường bất lợi cho vàng; thấp hơn kỳ vọng hỗ trợ BUY vàng.",
    "Core CPI": "Lạm phát lõi. Core CPI cao thường gây áp lực SELL vàng; thấp thường hỗ trợ BUY vàng.",
    "PCE": "Chỉ số lạm phát FED rất quan tâm. PCE cao bất lợi cho vàng; PCE thấp hỗ trợ vàng.",
    "Nonfarm": "Bảng lương phi nông nghiệp. Số liệu mạnh có thể đẩy USD/yield tăng; số liệu yếu hỗ trợ vàng.",
    "FED Rate": "Lãi suất FED. Lãi suất cao hoặc kỳ vọng tăng lãi suất thường bất lợi cho vàng.",
    "Powell Speech": "Phát biểu của Chủ tịch FED. Hawkish bất lợi cho vàng; dovish hỗ trợ vàng.",
    "GDP": "GDP Mỹ mạnh có thể hỗ trợ USD/yield, bất lợi cho vàng; GDP yếu thường hỗ trợ kỳ vọng FED mềm hơn.",
    "Unemployment": "Thất nghiệp tăng có thể hỗ trợ kỳ vọng giảm lãi suất; thất nghiệp giảm mạnh thường bất lợi cho vàng.",
}


def _current_status_text(name: str, bias: str, change_pct: float = 0.0) -> str:
    direction = "tăng" if change_pct > 0 else "giảm" if change_pct < 0 else "trung tính"
    if bias == "BUY":
        return f"Hiện tại {name} {direction}, đang hỗ trợ BUY vàng."
    if bias == "SELL":
        return f"Hiện tại {name} {direction}, đang tạo áp lực SELL vàng."
    return f"Hiện tại {name} chưa tạo thiên hướng rõ."


def _build_forex_radar(bundle: dict[str, pd.DataFrame], gold_df: pd.DataFrame) -> list[dict[str, Any]]:
    specs = [
        ("Gold", "GOLD"),
        ("DXY", "DXY"),
        ("US10Y", "US10Y"),
        ("VIX", "VIX"),
        ("Oil", "OIL"),
        ("Nasdaq", "NASDAQ"),
    ]
    rows: list[dict[str, Any]] = []
    for label, key in specs:
        frame = gold_df if key == "GOLD" else bundle.get(key, pd.DataFrame())
        change = _change_pct(frame)
        bias = _bias_for_indicator(key, change, frame)
        rows.append(
            {
                "name": label,
                "value": _latest(frame),
                "change_pct": round(change, 2),
                "bias": bias,
                "explanation": f"{RADAR_EXPLANATIONS[label]}\n{_current_status_text(label, bias, change)}",
            }
        )
    return rows


def _build_fed_radar(events_df: pd.DataFrame, event_state: dict[str, Any], macro: dict[str, Any]) -> list[dict[str, Any]]:
    names = ["CPI", "Core CPI", "PCE", "Nonfarm", "FED Rate", "Powell Speech", "GDP", "Unemployment"]
    rows: list[dict[str, Any]] = []
    blocked = bool(event_state.get("blocked"))
    pressure = _safe_float(macro.get("score"), 50)
    for name in names:
        related = pd.DataFrame()
        if events_df is not None and not events_df.empty:
            mask = events_df.astype(str).apply(lambda col: col.str.contains(name, case=False, na=False)).any(axis=1)
            related = events_df[mask].head(1)
        expected = "-"
        actual = "-"
        if not related.empty:
            expected = str(related.iloc[0].get("forecast", related.iloc[0].get("expected", "-")) or "-")
            actual = str(related.iloc[0].get("actual", "-") or "-")
        bias = "WAIT"
        if blocked:
            bias = "WAIT"
        elif pressure >= 65 and name in {"CPI", "Core CPI", "PCE", "Nonfarm", "FED Rate", "GDP"}:
            bias = "SELL"
        elif pressure <= 35 and name in {"CPI", "Core CPI", "PCE", "Nonfarm", "FED Rate", "Unemployment"}:
            bias = "BUY"
        rows.append(
            {
                "name": name,
                "expected": expected,
                "actual": actual,
                "bias": bias,
                "explanation": f"{RADAR_EXPLANATIONS[name]}\n{_current_status_text(name, bias)}",
            }
        )
    return rows


def _signal_history_key(signal: dict[str, Any]) -> str:
    return "|".join(str(signal.get(key, "")) for key in ("time", "action", "confidence", "strategy"))


def _load_signal_history(limit: int = 200) -> list[dict[str, Any]]:
    rows = load_json(SIGNAL_HISTORY_PATH, [])
    if not isinstance(rows, list):
        return []
    return rows[-limit:]


def _append_signal_history(signal: dict[str, Any], limit: int = 200) -> list[dict[str, Any]]:
    rows = _load_signal_history(limit)
    key = _signal_history_key(signal)
    if rows and rows[-1].get("key") == key:
        return rows
    rows.append(
        {
            "key": key,
            "timestamp": signal.get("time", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
            "signal": signal.get("action", "WAIT"),
            "confidence": signal.get("confidence", 0),
            "reason": signal.get("reason", ""),
        }
    )
    rows = rows[-limit:]
    save_json(SIGNAL_HISTORY_PATH, rows)
    return rows


def _load_config() -> dict[str, Any]:
    config = load_json(ROOT / "config.json", {})
    config.setdefault("paths", {})
    config["paths"]["shared_dir"] = str(DEFAULT_SHARED_DIR)
    config.setdefault("app", {})
    config.setdefault("tickers", {})
    return config


def _frame(config: dict[str, Any], key: str, period: str, interval: str) -> pd.DataFrame:
    ticker = config.get("tickers", {}).get(key, "")
    return add_indicators(fetch_ohlcv(ticker, period, interval)) if ticker else pd.DataFrame()


class SonFEDDesktopEngine:
    def __init__(self, shared_dir: str | Path = DEFAULT_SHARED_DIR) -> None:
        self.shared = SharedFileBus(shared_dir)
        self.config = _load_config()
        self.shared_dir = self.shared.shared_dir

    def refresh(self) -> dict[str, Any]:
        period = self.config.get("app", {}).get("default_period", "5d")
        timeframe = self.config.get("app", {}).get("default_timeframe", "15m")
        bundle = fetch_market_bundle(self.config.get("tickers", {}), period, timeframe)
        gold_df = add_indicators(bundle.get("GOLD", pd.DataFrame()))
        gold_analysis = analyze_gold(gold_df)
        macro = pressure_index(bundle)
        bias = gold_bias(bundle)
        events_path = Path(self.config.get("paths", {}).get("events", "events.csv"))
        if not events_path.is_absolute():
            events_path = ROOT / events_path
        events_df = load_events(events_path)
        erisk = event_risk(events_df)
        mtf_frames = {
            "M15": _frame(self.config, "GOLD", "5d", "15m"),
            "H1": _frame(self.config, "GOLD", "1mo", "1h"),
            "H4": _frame(self.config, "GOLD", "6mo", "4h"),
        }
        mtf = analyze_mtf(mtf_frames)
        strategies = build_strategies(gold_df, gold_analysis, macro, mtf, erisk)
        trade_status = read_shared(self.shared_dir).get("trade_status", {})
        ai_decision = build_decision(gold_analysis, macro, mtf, strategies)
        signal = create_signal(strategies, gold_analysis, macro, mtf, self.config, erisk, trade_status)
        ai_state = build_ai_state(signal, gold_analysis, macro, mtf, ai_decision, {})
        signal_history = _append_signal_history(signal)
        radar_forex = _build_forex_radar(bundle, gold_df)
        radar_fed = _build_fed_radar(events_df, erisk, macro)

        market_state = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "source": "SonFED",
            "symbol": signal.get("symbol", "XAUUSD"),
            "price": latest_float(gold_df["Close"]) if not gold_df.empty else 0.0,
            "signal": signal.get("action", "WAIT"),
            "confidence": signal.get("confidence", 0),
            "pressure_index": macro.get("score", 50),
            "macro_bias": bias,
            "market_regime": ai_state.get("market_regime", "SIDEWAY"),
            "volatility": ai_state.get("volatility", "NORMAL"),
            "momentum": ai_state.get("momentum", "WEAK"),
            "execution_mode": ai_state.get("execution_mode", "SAFE"),
            "recommended_trailing": ai_state.get("recommended_trailing", "SAFE"),
            "data_status": data_status(bundle),
            "macro_details": macro.get("details", []),
            "mtf_summary": mtf.get("summary", ""),
            "reason": signal.get("reason", ""),
            "radar_forex": radar_forex,
            "radar_fed": radar_fed,
        }
        risk_state = {
            "timestamp": market_state["timestamp"],
            "source": "SonFED",
            "signal": signal.get("action", "WAIT"),
            "confidence": signal.get("confidence", 0),
            "risk_level": ai_state.get("risk_level", "MEDIUM"),
            "recommended_risk_mode": ai_state.get("recommended_risk_mode", "SAFE"),
            "allow_auto_trade": bool(signal.get("allow_auto_trade", False)),
            "pressure_index": macro.get("score", 50),
            "reason": signal.get("reason", ""),
        }

        write_signal(signal, self.shared_dir)
        write_ai_state(ai_state, self.shared_dir)
        self.shared.write("market_state.json", market_state)
        self.shared.write("risk_state.json", risk_state)
        self.shared.heartbeat(
            "RUNNING",
            {
                "signal": signal.get("action", "WAIT"),
                "confidence": signal.get("confidence", 0),
                "market_regime": ai_state.get("market_regime", "SIDEWAY"),
                "execution_mode": ai_state.get("execution_mode", "SAFE"),
            },
        )
        save_json(ROOT / "data/market_state.json", market_state)
        return {
            "signal": signal,
            "ai_state": ai_state,
            "market_state": market_state,
            "risk_state": risk_state,
            "macro": macro,
            "gold": gold_df.tail(80),
            "radar_forex": radar_forex,
            "radar_fed": radar_fed,
            "signal_history": signal_history,
            "strategies": strategies[:5],
            "mtf": mtf,
            "events": erisk,
            "events_table": events_df.head(50).to_dict("records") if events_df is not None and not events_df.empty else [],
        }

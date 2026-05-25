from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv

from core.trade_policy import (
    AITradePolicy,
    apply_policy_to_signal,
    build_market_state,
    policy_from_config,
    save_policy_to_config,
    write_signal_if_allowed,
)
from core.settings_engine import (
    AI_MODES,
    get_default_settings as get_default_sonfed_settings,
    get_ai_mode_settings,
    get_effective_advanced_settings,
    load_sonfed_settings as load_persistent_sonfed_settings,
    save_sonfed_settings as save_persistent_sonfed_settings,
    validate_settings as validate_persistent_sonfed_settings,
)
from core.trade_management_engine import build_position_adjustment_payload, normalize_adjustment_action
from modules.auto_refresh_engine import (
    INTERVAL_OPTIONS,
    build_market_summary,
    build_snapshot,
    dashboard_summary,
    detect_changes,
    ensure_auto_refresh_config,
    finalize_refresh,
    fmt_time,
    get_auto_refresh_enabled,
    load_state,
    load_sonfed_settings,
    prepare_refresh,
    save_sonfed_settings,
    set_auto_refresh_enabled,
    should_send_telegram,
    signal_key,
)
from modules.alerts import smart_alerts
from modules.ai_state_bridge import build_ai_state, write_ai_state
from modules.data_fetcher import data_status, fetch_market_bundle, fetch_ohlcv
from modules.database import log_signal, recent_signals
from modules.events import event_risk, load_events
from modules.fred_client import fetch_fred_latest
from modules.gold_analyzer import analyze_gold
from modules.indicators import add_indicators
from modules.market_regime_engine import build_decision
from modules.macro_engine import gold_bias, pressure_index
from modules.mtf_engine import analyze_mtf
from modules.position_manager import ai_trade_summary, summarize_position
from modules.performance_engine import build_account_report, build_ai_change_alert, build_orders_report
from modules.signal_engine import create_signal
from modules.smartmoney_engine import smartmoney_notes
from modules.strategy_engine import build_strategies
from modules.telegram_engine import build_alert, send_telegram, send_telegram_queued
from modules.trade_bridge import read_shared
from modules.trade_statistics import account_statistics, enrich_positions
from modules.utils import ROOT, latest_float, load_json, pct_change, resolve_shared_dir, save_json
from modules.adjustment_engine import write_trade_adjustment
from modules.position_feedback import normalize_trade_status, position_table_rows
from modules.risk_feedback import normalize_risk_status

load_dotenv()

SIGNAL_HISTORY_PATH = ROOT / "data" / "signal_history.json"
TELEGRAM_MONITOR_STATE_PATH = ROOT / "data" / "telegram_monitor_state.json"

TRADING_MODES = ["Hướng dẫn sử dụng", "Bán tự động", "Tự động", "AI hỗ trợ"]
LEGACY_MODE_MAP = {
    "Manual": "Hướng dẫn sử dụng",
    "Semi Auto": "Bán tự động",
    "Auto": "Tự động",
    "AI Assisted": "AI hỗ trợ",
}

st.set_page_config(page_title="SonFED", page_icon="🟡", layout="wide")


def load_config() -> dict:
    return ensure_telegram_config(ensure_auto_refresh_config(load_json("config.json", {})))


def ensure_telegram_config(config: dict) -> dict:
    telegram = config.setdefault("telegram", {})
    telegram.setdefault("enabled", False)
    telegram.setdefault("cycle_minutes", 30)
    telegram.setdefault("report_enabled", True)
    telegram.setdefault("order_alerts_enabled", True)
    telegram.setdefault("performance_report_enabled", True)
    telegram.setdefault("drawdown_alerts_enabled", True)
    telegram.setdefault("ai_bias_alerts_enabled", True)
    telegram.setdefault("report_interval_minutes", 15)
    telegram.setdefault("drawdown_alert_percent", 5.0)
    telegram.setdefault("profit_target_percent", 10.0)
    telegram.setdefault("cooldown_seconds", 300)
    telegram.setdefault("base_capital", 200.0)
    return config


def save_config(config: dict) -> None:
    save_json("config.json", config)


def safe_float(value, default=0.0):
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def metric_value(df: pd.DataFrame) -> str:
    if df.empty:
        return "N/A"
    return f"{safe_float(df['Close'].dropna().iloc[-1]):.2f}"


REGIME_LABELS = {
    "Quiet Range": "Sideway yếu",
    "Volatile Range": "Sideway biến động mạnh",
    "Bull Expansion": "Bứt phá tăng mạnh",
    "Bear Expansion": "Bứt phá giảm mạnh",
    "Strong Trend": "Xu hướng mạnh",
    "Weak Trend": "Xu hướng yếu",
    "Exhaustion": "Dấu hiệu kiệt sức",
}

ACTION_LABELS = {
    "BUY": "Ưu tiên mua",
    "SELL": "Ưu tiên bán",
    "WAIT": "Đứng ngoài chờ rõ hơn",
    "HOLD_POSITION": "Giữ lệnh",
    "ADJUST_TRAILING": "Siết trailing stop",
    "MOVE_TO_BREAKEVEN": "Dời stop loss về hòa vốn",
    "PARTIAL_CLOSE": "Chốt lời một phần",
    "CLOSE_POSITION": "Đóng lệnh",
    "LOCK_PROFIT": "Khóa lợi nhuận",
    "DISABLE_NEW_ENTRY": "Không mở thêm lệnh mới",
}

def vi_regime(value: str | None) -> str:
    raw = str(value or "Chưa rõ")
    return REGIME_LABELS.get(raw, raw)


def vi_action(value: str | None) -> str:
    raw = str(value or "WAIT").upper()
    return ACTION_LABELS.get(raw, raw)


def render_confidence_guide() -> None:
    with st.expander("Giải thích nhanh: Độ tin cậy AI"):
        st.dataframe(
            pd.DataFrame(
                [
                    {"Mức": "0-40%", "Ý nghĩa": "Rất yếu", "Cách dùng": "Không nên giao dịch"},
                    {"Mức": "40-55%", "Ý nghĩa": "Yếu", "Cách dùng": "Chỉ phù hợp scalping mạnh"},
                    {"Mức": "55-70%", "Ý nghĩa": "Trung bình", "Cách dùng": "Có thể dùng intraday, cần xác nhận thêm"},
                    {"Mức": "70-85%", "Ý nghĩa": "Mạnh", "Cách dùng": "Khá an toàn cho người mới"},
                    {"Mức": "85-100%", "Ý nghĩa": "Rất mạnh", "Cách dùng": "Market rõ xu hướng"},
                ]
            ),
            use_container_width=True,
            hide_index=True,
        )
        st.caption("Độ tin cậy AI không đảm bảo chắc thắng. Đây chỉ là mức AI tự tin với phân tích hiện tại.")


def render_wait_explanation() -> None:
    with st.expander("WAIT nghĩa là gì?"):
        st.info(
            "WAIT không có nghĩa thị trường chắc chắn đảo chiều. WAIT nghĩa là AI không còn đủ chắc chắn "
            "để tiếp tục BUY hoặc SELL mạnh."
        )
        st.write("Ví dụ: SELL mạnh → momentum yếu đi → biến động tăng → AI chuyển sang WAIT.")
        st.write("Lúc này SonEXEC có thể giữ lệnh, dời stop loss, trailing stop hoặc bảo toàn vốn thay vì đóng lệnh ngay.")


def render_operating_system_help() -> None:
    with st.expander("Hệ thống SonFED hoạt động thế nào?"):
        st.write("SonFED không chỉ là app báo BUY/SELL. Hệ thống vận hành theo 5 bước:")
        st.write("1. Thu thập dữ liệu: giá vàng, USD, lợi suất trái phiếu, Nasdaq và biến động thị trường.")
        st.write("2. AI phân tích: xu hướng, sideway, breakout, biến động mạnh/yếu.")
        st.write("3. AI quyết định: BUY, SELL hoặc WAIT.")
        st.write("4. Chính sách giao dịch: lọc tín hiệu yếu, tránh spread cao, tránh tin tức mạnh và market nguy hiểm.")
        st.write("5. SonEXEC thực thi: vào lệnh, dời SL theo giá, dời SL về hòa vốn, chốt lời và quản lý rủi ro.")


def render_quick_decision_explanation(signal: dict, ai_decision: dict, gold_analysis: dict, macro: dict, mtf: dict) -> None:
    action = signal.get("action", ai_decision.get("action", "WAIT"))
    regime = gold_analysis.get("market_regime", {})
    reasons = []
    if macro.get("score", 50) >= 60:
        reasons.append("áp lực vĩ mô đang cao")
    if macro.get("score", 50) <= 40:
        reasons.append("áp lực vĩ mô đang thấp hơn")
    if regime.get("momentum"):
        reasons.append(str(regime.get("momentum")).lower())
    if mtf.get("summary"):
        reasons.append(mtf["summary"].lower())
    if not reasons:
        reasons.append("AI chưa thấy lợi thế đủ rõ")
    st.info(f"{vi_action(action)}: AI đánh giá như vậy vì " + "; ".join(reasons[:3]) + ".")


def make_gold_chart(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if df.empty:
        return fig
    data = add_indicators(df)
    fig.add_trace(go.Candlestick(x=data.index, open=data["Open"], high=data["High"], low=data["Low"], close=data["Close"], name="Giá"))
    for col, name in [("MA20", "MA20"), ("EMA50", "EMA50"), ("EMA200", "EMA200"), ("BB_UPPER", "BB trên"), ("BB_LOWER", "BB dưới")]:
        if col in data:
            fig.add_trace(go.Scatter(x=data.index, y=data[col], name=name, mode="lines"))
    fig.update_layout(height=560, margin=dict(l=10, r=10, t=35, b=10), xaxis_rangeslider_visible=False)
    return fig


def _calculate_gold_strength(data: pd.DataFrame, market_regime: dict | None = None) -> dict:
    if data.empty:
        return {"buy": 0, "sell": 0, "dominance": "WAIT", "momentum": 0, "trend": 0, "volatility": 0}
    market_regime = market_regime or {}
    last = data.iloc[-1]
    buy_points = 0
    sell_points = 0
    close = latest_float(data["Close"])
    ma20 = latest_float(data.get("MA20", pd.Series(dtype=float)))
    ema50 = latest_float(data.get("EMA50", pd.Series(dtype=float)))
    ema200 = latest_float(data.get("EMA200", pd.Series(dtype=float)))
    macd_hist = latest_float(data.get("MACD_HIST", pd.Series(dtype=float)))
    macd_delta = latest_float(data.get("MACD_HIST", pd.Series(dtype=float)).diff()) if "MACD_HIST" in data else 0.0
    rsi = latest_float(data.get("RSI14", pd.Series(dtype=float)), 50.0)
    adx = latest_float(data.get("ADX14", pd.Series(dtype=float)), 0.0)
    volatility = int(market_regime.get("volatility", {}).get("score", 0) or 0)

    if close > ma20:
        buy_points += 18
    elif close < ma20:
        sell_points += 18
    if close > ema50:
        buy_points += 12
    elif close < ema50:
        sell_points += 12
    if ema50 > ema200:
        buy_points += 12
    elif ema50 < ema200:
        sell_points += 12
    if macd_hist > 0:
        buy_points += 18
    elif macd_hist < 0:
        sell_points += 18
    if macd_delta > 0:
        buy_points += 8
    elif macd_delta < 0:
        sell_points += 8
    if 52 <= rsi <= 70:
        buy_points += 10
    elif 30 <= rsi <= 48:
        sell_points += 10
    elif rsi > 72:
        sell_points += 6
    elif rsi < 28:
        buy_points += 6
    if bool(last.get("BOS_UP", False)) or bool(last.get("LIQUIDITY_SWEEP_DOWN", False)):
        buy_points += 16
    if bool(last.get("BOS_DOWN", False)) or bool(last.get("LIQUIDITY_SWEEP_UP", False)):
        sell_points += 16
    if market_regime.get("bias") == "BUY":
        buy_points += 12
    elif market_regime.get("bias") == "SELL":
        sell_points += 12

    total = max(1, buy_points + sell_points)
    buy = int(round(buy_points / total * 100))
    sell = max(0, 100 - buy)
    dominance = "BUY" if buy >= sell + 12 else "SELL" if sell >= buy + 12 else "WAIT"
    momentum = min(100, int(abs(macd_hist) / max(abs(latest_float(data.get("ATR14", pd.Series(dtype=float)), 1.0)), 0.01) * 100))
    trend = min(100, int(abs(buy - sell) + min(adx, 40)))
    return {"buy": buy, "sell": sell, "dominance": dominance, "momentum": momentum, "trend": trend, "volatility": volatility}


def _recent_signal_points(data: pd.DataFrame, max_points: int = 10) -> tuple[pd.DataFrame, pd.DataFrame]:
    if data.empty:
        return pd.DataFrame(), pd.DataFrame()
    recent = data.tail(80).copy()
    macd_delta = recent["MACD_HIST"].diff() if "MACD_HIST" in recent else pd.Series(0, index=recent.index)
    bos_up = recent["BOS_UP"].astype(bool) if "BOS_UP" in recent else pd.Series(False, index=recent.index)
    bos_down = recent["BOS_DOWN"].astype(bool) if "BOS_DOWN" in recent else pd.Series(False, index=recent.index)
    sweep_up = recent["LIQUIDITY_SWEEP_UP"].astype(bool) if "LIQUIDITY_SWEEP_UP" in recent else pd.Series(False, index=recent.index)
    sweep_down = recent["LIQUIDITY_SWEEP_DOWN"].astype(bool) if "LIQUIDITY_SWEEP_DOWN" in recent else pd.Series(False, index=recent.index)
    volume_spike = recent["VOLUME_SPIKE"].astype(bool) if "VOLUME_SPIKE" in recent else pd.Series(False, index=recent.index)
    buy_mask = (
        bos_up
        | sweep_down
        | (volume_spike & (recent["Close"] > recent["MA20"]) & (recent["MACD_HIST"] > 0) & (macd_delta > 0))
    )
    sell_mask = (
        bos_down
        | sweep_up
        | (volume_spike & (recent["Close"] < recent["MA20"]) & (recent["MACD_HIST"] < 0) & (macd_delta < 0))
    )
    return recent.loc[buy_mask].tail(max_points), recent.loc[sell_mask].tail(max_points)


def make_ai_gold_chart(df: pd.DataFrame, gold_analysis: dict, signal: dict) -> go.Figure:
    fig = go.Figure()
    if df.empty:
        return fig
    data = add_indicators(df).tail(180)
    market_regime = gold_analysis.get("market_regime", {})
    strength = _calculate_gold_strength(data, market_regime)
    dominance = strength["dominance"]
    bg = "#f0fdf4" if dominance == "BUY" else "#fef2f2" if dominance == "SELL" else "#f9fafb"
    ribbon = "rgba(22, 163, 74, 0.18)" if dominance == "BUY" else "rgba(220, 38, 38, 0.16)" if dominance == "SELL" else "rgba(55, 65, 81, 0.10)"

    fig.add_trace(
        go.Candlestick(
            x=data.index,
            open=data["Open"],
            high=data["High"],
            low=data["Low"],
            close=data["Close"],
            name="XAUUSD",
            increasing_line_color="#16a34a",
            increasing_fillcolor="#22c55e",
            decreasing_line_color="#dc2626",
            decreasing_fillcolor="#ef4444",
            whiskerwidth=0.55,
        )
    )
    indicator_style = {
        "MA20": ("MA20", "rgba(37, 99, 235, 0.38)", 1.2),
        "EMA50": ("EMA50", "rgba(124, 58, 237, 0.30)", 1.0),
        "EMA200": ("EMA200", "rgba(17, 24, 39, 0.24)", 1.0),
        "BB_UPPER": ("BB trên", "rgba(107, 114, 128, 0.18)", 1.0),
        "BB_LOWER": ("BB dưới", "rgba(107, 114, 128, 0.18)", 1.0),
    }
    for col, (name, color, width) in indicator_style.items():
        if col in data:
            fig.add_trace(go.Scatter(x=data.index, y=data[col], name=name, mode="lines", line=dict(color=color, width=width)))

    buy_points, sell_points = _recent_signal_points(data)
    if not buy_points.empty:
        fig.add_trace(
            go.Scatter(
                x=buy_points.index,
                y=buy_points["Low"] - data["ATR14"].fillna(0).reindex(buy_points.index).fillna(0) * 0.25,
                mode="markers+text",
                name="BUY signal",
                text=["BUY"] * len(buy_points),
                textposition="bottom center",
                marker=dict(symbol="triangle-up", size=16, color="#16a34a", line=dict(color="white", width=1)),
                textfont=dict(color="#166534", size=12),
            )
        )
    if not sell_points.empty:
        fig.add_trace(
            go.Scatter(
                x=sell_points.index,
                y=sell_points["High"] + data["ATR14"].fillna(0).reindex(sell_points.index).fillna(0) * 0.25,
                mode="markers+text",
                name="SELL signal",
                text=["SELL"] * len(sell_points),
                textposition="top center",
                marker=dict(symbol="triangle-down", size=16, color="#dc2626", line=dict(color="white", width=1)),
                textfont=dict(color="#991b1b", size=12),
            )
        )

    fig.add_shape(type="rect", xref="paper", yref="paper", x0=0, x1=1, y0=0.935, y1=1, fillcolor=ribbon, line=dict(width=0), layer="below")
    fig.add_annotation(
        xref="paper",
        yref="paper",
        x=0.012,
        y=0.968,
        showarrow=False,
        text=f"{dominance} ZONE · BUY {strength['buy']}% / SELL {strength['sell']}%",
        font=dict(size=13, color="#111827"),
        align="left",
    )
    fig.add_annotation(
        xref="paper",
        yref="paper",
        x=0.99,
        y=0.968,
        showarrow=False,
        text=f"Signal: {signal.get('action', 'WAIT')}",
        font=dict(size=13, color="#15803d" if signal.get("action") == "BUY" else "#b91c1c" if signal.get("action") == "SELL" else "#374151"),
        align="right",
    )
    fig.update_layout(
        height=640,
        margin=dict(l=8, r=8, t=42, b=8),
        xaxis_rangeslider_visible=False,
        plot_bgcolor=bg,
        paper_bgcolor="#ffffff",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="rgba(107,114,128,0.16)", zeroline=False)
    return fig


def _bar_blocks(value: int, total: int = 10) -> str:
    filled = max(0, min(total, round(value / 100 * total)))
    return "█" * filled + "░" * (total - filled)


def _status_from_score(value: int, metric: str) -> tuple[str, str]:
    if value >= 70:
        return ("Mạnh", "buy" if metric != "risk" else "risk")
    if value >= 40:
        return ("Trung bình", "wait")
    return ("Yếu", "sell" if metric == "momentum" else "wait")


def _regime_action(label: str) -> tuple[str, str, str]:
    raw = str(label or "Chưa rõ")
    if "Bull" in raw or "BUY" in raw or "tăng" in raw.lower():
        return "BUY", "🟢", "Xu hướng tăng"
    if "Bear" in raw or "SELL" in raw or "giảm" in raw.lower():
        return "SELL", "🔴", "Xu hướng giảm"
    if "Exhaustion" in raw:
        return "RISK", "⚠️", "Quá mua/bán"
    if "Volatile" in raw:
        return "RISK", "🟠", "Biến động mạnh"
    if "Quiet" in raw or "Range" in raw:
        return "WAIT", "⚫", "Sideway"
    return "WAIT", "⚫", raw


def _render_strength_panel(strength: dict) -> None:
    buy = int(strength.get("buy", 0))
    sell = int(strength.get("sell", 0))
    momentum_label, momentum_state = _status_from_score(int(strength.get("momentum", 0)), "momentum")
    trend_label, trend_state = _status_from_score(int(strength.get("trend", 0)), "trend")
    volatility_label, volatility_state = _status_from_score(int(strength.get("volatility", 0)), "risk")
    left, right = st.columns([1.1, 1.0])
    with left:
        with st.container(border=True):
            st.markdown("**Trend Bar**")
            st.success(f"BUY  {_bar_blocks(buy)}  {buy}%")
            st.error(f"SELL {_bar_blocks(sell)}  {sell}%")
    with right:
        with st.container(border=True):
            st.markdown("**Momentum Radar**")
            cols = st.columns(3)
            cols[0].metric("Momentum", f"{strength.get('momentum', 0)}%", momentum_label)
            cols[1].metric("Trend strength", f"{strength.get('trend', 0)}%", trend_label)
            cols[2].metric("Volatility", f"{strength.get('volatility', 0)}%", volatility_label)


def _render_mtf_heatmap(mtf: dict) -> None:
    trends = mtf.get("trends", {})
    regimes = mtf.get("regimes", {})
    st.markdown("**Heatmap đa khung**")
    cols = st.columns(5)
    for tf in ["M1", "M5", "M15", "H1", "H4"]:
        label = trends.get(tf) or (regimes.get(tf, {}) or {}).get("label", "Chưa nạp")
        action, icon, text = _regime_action(label)
        with cols[["M1", "M5", "M15", "H1", "H4"].index(tf)]:
            with st.container(border=True):
                st.caption(tf)
                _show_action_status(action, f"{icon} {text}")


def _render_signal_strip(data: pd.DataFrame, signal: dict) -> None:
    if data.empty:
        return
    last = data.iloc[-1]
    items = [
        ("BOS", "BUY" if bool(last.get("BOS_UP", False)) else "SELL" if bool(last.get("BOS_DOWN", False)) else "WAIT"),
        ("Thanh khoản", "BUY" if bool(last.get("LIQUIDITY_SWEEP_DOWN", False)) else "SELL" if bool(last.get("LIQUIDITY_SWEEP_UP", False)) else "WAIT"),
        ("Động lượng", signal.get("action", "WAIT") if signal.get("scalp_accepted") else "WAIT"),
        ("Hành động", signal.get("action", "WAIT")),
    ]
    cols = st.columns(4)
    for idx, (label, action) in enumerate(items):
        action = str(action or "WAIT").upper()
        with cols[idx]:
            _show_action_status(action, f"{_action_text(action)}\n\n{label}: {_bias_label(action)}")

    summary = []
    liquidity = dict(items).get("Thanh khoản", "WAIT")
    momentum = dict(items).get("Động lượng", "WAIT")
    bos = dict(items).get("BOS", "WAIT")
    action = dict(items).get("Hành động", "WAIT")
    summary.append(f"Thanh khoản đang nghiêng về {liquidity}.")
    summary.append(f"Động lượng ngắn hạn {'hỗ trợ ' + momentum if momentum in {'BUY', 'SELL'} else 'chưa xác nhận rõ'}.")
    summary.append(f"BOS hiện là {bos}.")
    summary.append("Cần quản lý lệnh chặt." if action in {"BUY", "SELL"} and "WAIT" in {bos, momentum} else f"Hành động hiện tại: {action}.")
    st.info(" ".join(summary))


def render_ai_gold_technical_tab(gold_df: pd.DataFrame, gold_analysis: dict, signal: dict, mtf: dict, signal_history: list[dict]) -> None:
    data = add_indicators(gold_df) if not gold_df.empty else gold_df
    strength = _calculate_gold_strength(data, gold_analysis.get("market_regime", {}))
    _render_strength_panel(strength)
    st.plotly_chart(make_ai_gold_chart(gold_df, gold_analysis, signal), use_container_width=True, key="technical_ai_gold_chart")
    _render_signal_strip(data, signal)
    render_trade_signal_timeline(signal_history, limit=15, compact=True)
    _render_mtf_heatmap(mtf)


def normalize_trading_mode(mode: str) -> str:
    return LEGACY_MODE_MAP.get(mode, mode if mode in TRADING_MODES else "Hướng dẫn sử dụng")


def get_trading_mode_config(mode: str) -> dict:
    mode = normalize_trading_mode(mode)
    configs = {
        "Hướng dẫn sử dụng": {"write_signal": False, "allow_auto_trade_toggle": False, "allow_auto_refresh": False, "log_signal": False},
        "Bán tự động": {"write_signal": False, "allow_auto_trade_toggle": False, "allow_auto_refresh": False, "log_signal": False},
        "Tự động": {"write_signal": True, "allow_auto_trade_toggle": True, "allow_auto_refresh": True, "log_signal": True},
        "AI hỗ trợ": {"write_signal": True, "allow_auto_trade_toggle": True, "allow_auto_refresh": False, "log_signal": True},
    }
    return configs[mode]


def get_ai_trade_policy(config: dict) -> AITradePolicy:
    return policy_from_config(config)


MAIN_SONFED_SETTING_KEYS = (
    "default_lot",
    "max_buy_orders",
    "max_sell_orders",
    "ai_mode",
    "allow_sonexec_signal_read",
    "enable_auto_execution",
    "enable_position_management",
)
ADVANCED_SETTING_KEYS = (
    "min_ai_confidence",
    "min_rr",
    "max_spread",
    "avoid_high_volatility",
    "avoid_news",
    "position_strategy",
)
POSITION_STRATEGY_OPTIONS = ["Bảo toàn vốn", "Dời SL về hòa vốn", "Bám xu hướng", "AI tự thích nghi"]
AI_MODE_DESCRIPTIONS = {
    "An toàn": "Ít lệnh hơn, ưu tiên bảo toàn vốn.",
    "Cân bằng": "Phù hợp sử dụng hằng ngày, cân bằng giữa cơ hội và rủi ro.",
    "Chủ động": "Nhiều tín hiệu hơn, chấp nhận biến động cao hơn.",
    "Tấn công": "Rủi ro cao, chỉ dùng khi đã hiểu hệ thống.",
}
DISPLAY_TO_ENGINE_STRATEGY = {
    "Dời SL về hòa vốn": "Break-even",
    "AI tự thích nghi": "AI thích nghi",
}
ENGINE_TO_DISPLAY_STRATEGY = {
    "Break-even": "Dời SL về hòa vốn",
    "AI thích nghi": "AI tự thích nghi",
}


def collect_advanced_settings_from_state() -> dict:
    preset = get_ai_mode_settings(st.session_state.get("ai_mode", get_default_sonfed_settings()["ai_mode"]))
    return {
        "min_ai_confidence": st.session_state.get("min_ai_confidence", preset["min_ai_confidence"]),
        "min_rr": st.session_state.get("min_rr", preset["min_rr"]),
        "max_spread": st.session_state.get("max_spread", preset["max_spread"]),
        "avoid_high_volatility": st.session_state.get("avoid_high_volatility", preset["avoid_high_volatility"]),
        "avoid_news": st.session_state.get("avoid_news", preset["avoid_news"]),
        "position_strategy": st.session_state.get("position_strategy", preset["position_strategy"]),
    }


def sync_advanced_settings_state(settings: dict) -> None:
    for key in ADVANCED_SETTING_KEYS:
        st.session_state[key] = settings[key]


def initialize_sonfed_settings_state(settings: dict) -> None:
    clean = validate_persistent_sonfed_settings(settings)
    for key in MAIN_SONFED_SETTING_KEYS:
        value = clean[key]
        if key not in st.session_state:
            st.session_state[key] = value
    if "advanced_settings" not in st.session_state:
        st.session_state["advanced_settings"] = clean.get("advanced_settings")

    effective_advanced = get_effective_advanced_settings(clean)
    for key, value in effective_advanced.items():
        if key not in st.session_state:
            st.session_state[key] = value


def sync_ai_mode_preset_to_advanced_state() -> None:
    if st.session_state.get("advanced_settings") is None:
        sync_advanced_settings_state(get_ai_mode_settings(st.session_state.get("ai_mode", "Cân bằng")))


def mark_advanced_settings_manual() -> None:
    st.session_state["advanced_settings"] = collect_advanced_settings_from_state()


def collect_sonfed_settings_from_state() -> dict:
    defaults = get_default_sonfed_settings()
    settings = {key: st.session_state.get(key, defaults[key]) for key in MAIN_SONFED_SETTING_KEYS}
    settings["advanced_settings"] = None
    if st.session_state.get("advanced_settings") is not None:
        settings["advanced_settings"] = collect_advanced_settings_from_state()
    return settings


def sync_sonfed_settings_state(settings: dict) -> None:
    clean = validate_persistent_sonfed_settings(settings)
    for key in MAIN_SONFED_SETTING_KEYS:
        st.session_state[key] = clean[key]
    st.session_state["advanced_settings"] = clean.get("advanced_settings")
    sync_advanced_settings_state(get_effective_advanced_settings(clean))


def apply_sonfed_settings_to_policy(policy: AITradePolicy, settings: dict) -> AITradePolicy:
    clean = validate_persistent_sonfed_settings(settings)
    advanced = get_effective_advanced_settings(clean)
    policy.default_lot = safe_float(clean.get("default_lot"), 0.03)
    policy.allow_buy = True
    policy.allow_sell = True
    policy.max_buy_orders = int(clean["max_buy_orders"])
    policy.max_sell_orders = int(clean["max_sell_orders"])
    policy.max_buy_volume = round(policy.default_lot * policy.max_buy_orders, 2)
    policy.max_sell_volume = round(policy.default_lot * policy.max_sell_orders, 2)
    policy.min_confidence = int(advanced["min_ai_confidence"])
    policy.min_rr = safe_float(advanced.get("min_rr"), 0.8)
    policy.max_spread = int(advanced["max_spread"])
    policy.filter_high_volatility = bool(advanced["avoid_high_volatility"])
    policy.filter_important_news = bool(advanced["avoid_news"])
    policy.allow_sonexec_read_signal = bool(clean["allow_sonexec_signal_read"])
    policy.allow_auto_execution = bool(clean["enable_auto_execution"])
    policy.allow_auto_adjustment = bool(clean["enable_position_management"])
    policy.position_management_strategy = DISPLAY_TO_ENGINE_STRATEGY.get(
        advanced["position_strategy"],
        advanced["position_strategy"],
    )
    return policy


def persist_policy_settings_snapshot(settings: dict) -> None:
    config = load_config()
    policy = apply_sonfed_settings_to_policy(get_ai_trade_policy(config), settings)
    save_policy_to_config(config, policy)
    save_config(config)


def save_sonfed_settings_from_state() -> None:
    try:
        saved_settings = save_persistent_sonfed_settings(collect_sonfed_settings_from_state())
        sync_sonfed_settings_state(saved_settings)
        persist_policy_settings_snapshot(saved_settings)
        st.session_state["_sonfed_settings_message"] = "Đã lưu cấu hình SonFED"
        st.session_state.pop("_sonfed_settings_error", None)
    except Exception as exc:
        st.session_state["_sonfed_settings_error"] = f"Không thể lưu cấu hình: {exc}"
        st.session_state.pop("_sonfed_settings_message", None)


def restore_default_sonfed_settings() -> None:
    try:
        defaults = get_default_sonfed_settings()
        saved_settings = save_persistent_sonfed_settings(defaults)
        sync_sonfed_settings_state(saved_settings)
        persist_policy_settings_snapshot(saved_settings)
        st.session_state["_sonfed_settings_message"] = "Đã khôi phục mặc định"
        st.session_state.pop("_sonfed_settings_error", None)
    except Exception as exc:
        st.session_state["_sonfed_settings_error"] = f"Không thể lưu cấu hình: {exc}"
        st.session_state.pop("_sonfed_settings_message", None)


def render_ai_trade_policy(config: dict, mode_config: dict) -> AITradePolicy:
    st.sidebar.expander("Hệ thống hoạt động thế nào?").write(
        "SonFED là bộ não phân tích thị trường và tạo tín hiệu. SonEXEC là bộ máy thực thi, "
        "vào lệnh và quản lý lệnh đang mở. Người mới nên để Auto Trade tắt cho đến khi hiểu rõ rủi ro."
    )
    settings = load_persistent_sonfed_settings()
    initialize_sonfed_settings_state(settings)
    policy = apply_sonfed_settings_to_policy(get_ai_trade_policy(config), collect_sonfed_settings_from_state())
    with st.sidebar.expander("Chính sách giao dịch AI"):
        st.number_input(
            "Khối lượng cơ bản mỗi lệnh",
            min_value=0.01,
            max_value=10.0,
            step=0.01,
            key="default_lot",
            help="Ví dụ 0.03 nghĩa là mỗi lệnh SonEXEC mở sẽ dùng 0.03 lot nếu Auto Trade được bật. Người mới nên dùng lot nhỏ.",
        )
        st.number_input(
            "Số lệnh BUY tối đa",
            min_value=0,
            max_value=20,
            step=1,
            key="max_buy_orders",
            help="Giới hạn số lệnh mua để tránh mở quá nhiều lệnh cùng chiều.",
        )
        st.number_input(
            "Số lệnh SELL tối đa",
            min_value=0,
            max_value=20,
            step=1,
            key="max_sell_orders",
            help="Giới hạn số lệnh bán để tránh dồn quá nhiều rủi ro một phía.",
        )
        ai_mode = st.selectbox(
            "Chế độ AI",
            AI_MODES,
            key="ai_mode",
            on_change=sync_ai_mode_preset_to_advanced_state,
            help="Người dùng chọn mức độ rủi ro. AI tự xử lý chi tiết.",
        )
        st.caption(AI_MODE_DESCRIPTIONS.get(ai_mode, "AI tự chọn chính sách phù hợp."))
        st.toggle(
            "Cho phép SonEXEC đọc tín hiệu",
            key="allow_sonexec_signal_read",
            disabled=not mode_config["write_signal"],
            help="Khi bật, SonFED sẽ gửi BUY/SELL/WAIT sang SonEXEC qua signal.json.",
        )
        st.toggle(
            "Tự động vào lệnh bằng SonEXEC",
            key="enable_auto_execution",
            disabled=not mode_config["allow_auto_trade_toggle"],
            help="Khi bật, SonEXEC có thể tự vào lệnh bằng tiền thật nếu tín hiệu và kiểm tra rủi ro đều đạt.",
        )
        if st.session_state.get("enable_auto_execution"):
            st.error("Chỉ bật tự động vào lệnh khi đã hiểu rõ hệ thống và rủi ro.")
        st.toggle(
            "Tự động quản lý lệnh đang mở",
            key="enable_position_management",
            disabled=not mode_config["write_signal"],
            help="Tính năng này không mở lệnh mới. SonEXEC có thể dời stop loss, khóa lợi nhuận, trailing stop hoặc chốt lời một phần cho lệnh đang mở.",
        )

        lot = safe_float(st.session_state.get("default_lot"), 0.03)
        max_sell_orders = int(st.session_state.get("max_sell_orders", 3))
        st.info(f"Nếu mỗi lệnh {lot:.2f} lot và tối đa {max_sell_orders} lệnh SELL, tổng SELL tối đa là {lot * max_sell_orders:.2f} lot.")

        with st.expander("Cài đặt nâng cao"):
            st.slider(
                "Độ tin cậy AI tối thiểu",
                1,
                100,
                format="%d%%",
                key="min_ai_confidence",
                on_change=mark_advanced_settings_manual,
                help="AI chỉ vào lệnh khi đủ tự tin.",
            )
            st.number_input(
                "Tỷ lệ RR tối thiểu",
                min_value=0.1,
                max_value=10.0,
                step=0.1,
                key="min_rr",
                on_change=mark_advanced_settings_manual,
                help="RR là tỷ lệ lợi nhuận/rủi ro tối thiểu.",
            )
            st.number_input(
                "Spread tối đa",
                min_value=1,
                max_value=5000,
                key="max_spread",
                on_change=mark_advanced_settings_manual,
                help="Spread cao làm lệnh vừa vào đã bất lợi.",
            )
            st.toggle(
                "Tránh thị trường biến động mạnh",
                key="avoid_high_volatility",
                on_change=mark_advanced_settings_manual,
            )
            st.toggle(
                "Tránh giao dịch gần tin tức mạnh",
                key="avoid_news",
                on_change=mark_advanced_settings_manual,
            )
            st.selectbox(
                "Chiến lược quản lý lệnh",
                POSITION_STRATEGY_OPTIONS,
                key="position_strategy",
                on_change=mark_advanced_settings_manual,
                help="Nếu không chỉnh thủ công, AI tự chọn theo Chế độ AI.",
            )
        if st.session_state.get("advanced_settings") is not None:
            st.warning("Bạn đang dùng cấu hình nâng cao thủ công.")
        st.caption("Cấu hình được lưu cục bộ trên máy.")
        action_cols = st.columns(2)
        action_cols[0].button("Lưu cấu hình", use_container_width=True, on_click=save_sonfed_settings_from_state)
        action_cols[1].button("Khôi phục mặc định", use_container_width=True, on_click=restore_default_sonfed_settings)
        if st.session_state.get("_sonfed_settings_message"):
            st.success(st.session_state.pop("_sonfed_settings_message"))
        if st.session_state.get("_sonfed_settings_error"):
            st.error(st.session_state.pop("_sonfed_settings_error"))
        policy = apply_sonfed_settings_to_policy(policy, collect_sonfed_settings_from_state())
    if not mode_config["write_signal"]:
        policy.allow_sonexec_read_signal = False
        policy.allow_auto_adjustment = False
    if not mode_config["allow_auto_trade_toggle"]:
        policy.allow_auto_execution = False
    save_policy_to_config(config, policy)
    return policy


def apply_trade_policy(signal: dict, policy: AITradePolicy, market_state: dict) -> tuple[dict, dict]:
    return apply_policy_to_signal(signal, policy, market_state)


def render_policy_status(policy: AITradePolicy) -> None:
    cols = st.columns(5)
    cols[0].metric("BUY", "Được phép" if policy.allow_buy else "Đang tắt")
    cols[1].metric("SELL", "Được phép" if policy.allow_sell else "Đang tắt")
    cols[2].metric("Tự động vào lệnh", "Bật" if policy.allow_auto_execution else "Tắt")
    cols[3].metric("Gửi tín hiệu sang SonEXEC", "Bật" if policy.allow_sonexec_read_signal else "Tắt")
    cols[4].metric("Tự động quản lý lệnh", "Bật" if policy.allow_auto_adjustment else "Tắt")


def render_policy_warning(policy_result: dict) -> None:
    if policy_result.get("blocked"):
        st.warning(
            f"AI ban đầu nghiêng về {policy_result.get('initial_decision', 'WAIT')}, "
            f"nhưng tín hiệu đã chuyển thành WAIT vì {policy_result.get('message', '')}"
        )


def render_position_management_panel(trade_feedback: dict, adjustments_payload: dict, signal: dict, gold_analysis: dict) -> None:
    positions = trade_feedback.get("positions", [])
    buy_volume = sum(safe_float(p.get("lot", p.get("volume", 0))) for p in positions if "BUY" in str(p.get("type", p.get("type_name", ""))).upper())
    sell_volume = sum(safe_float(p.get("lot", p.get("volume", 0))) for p in positions if "SELL" in str(p.get("type", p.get("type_name", ""))).upper())
    floating_profit = sum(safe_float(p.get("profit", 0)) for p in positions)
    trailing_modes = sorted({str(p.get("trailing_mode", "")) for p in positions if p.get("trailing_mode")})

    st.subheader("Quản lý lệnh đang mở")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Lệnh đang mở", len(positions))
    c2.metric("Tổng khối lượng BUY", f"{buy_volume:.2f}")
    c3.metric("Tổng khối lượng SELL", f"{sell_volume:.2f}")
    c4.metric("Lãi/lỗ đang chạy", f"{floating_profit:,.2f}")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Cách dời SL", ", ".join(trailing_modes) if trailing_modes else "Chưa bật")
    c6.metric("Trạng thái thị trường", vi_regime(gold_analysis.get("regime", "Chưa rõ")))
    c7.metric("Mức rủi ro", signal.get("risk_level", "N/A"))
    c8.metric("Độ tin cậy AI", f"{signal.get('confidence', 0)}%")

    st.subheader("Chiến lược quản lý lệnh")
    st.info("SonFED chỉ chọn hướng xử lý. SonEXEC mới là nơi dời SL theo giá, dời SL về hòa vốn, chốt lời một phần, khóa lợi nhuận và quản lý từng lệnh.")

    st.subheader("Đề xuất xử lý lệnh")
    adjustments = adjustments_payload.get("adjustments", [])
    if adjustments:
        display_rows = []
        for item in adjustments:
            row = dict(item)
            row["action"] = vi_action(normalize_adjustment_action(item.get("action", "")))
            display_rows.append(row)
        st.dataframe(pd.DataFrame(display_rows), use_container_width=True)
    else:
        st.info("Chưa có lệnh mở để đề xuất điều chỉnh.")

    st.subheader("AI phân tích lệnh đang mở")
    st.write(adjustments_payload.get("ai_position_analysis", "Chưa có phân tích vị thế."))


def render_ai_decision_box(signal: dict, ai_decision: dict) -> None:
    st.subheader("AI Decision Box")
    d1, d2, d3, d4, d5 = st.columns(5)
    d1.metric("Quyết định AI", vi_action(signal.get("decision", signal.get("action", "WAIT"))))
    d2.metric("Xác suất", f"{ai_decision.get('winrate', signal.get('confidence', 0))}%")
    d3.metric("Chốt lời dự kiến", ai_decision.get("tp") if ai_decision.get("tp") is not None else "N/A")
    d4.metric("Cắt lỗ dự kiến", ai_decision.get("sl") if ai_decision.get("sl") is not None else "N/A")
    d5.metric("Tỷ lệ lời/lỗ", ai_decision.get("rr") if ai_decision.get("rr") is not None else "N/A")


def _fmt_number(value: object, digits: int = 2) -> str:
    if value in {None, "N/A"}:
        return "N/A"
    number = safe_float(value, None)
    if number is None:
        return "N/A"
    return f"{number:.{digits}f}"


def _fmt_market_value(df: pd.DataFrame) -> str:
    if df is None or df.empty or "Close" not in df:
        return "N/A"
    return _fmt_number(latest_float(df["Close"]), 2)


def _fmt_change(change: float) -> str:
    change = safe_float(change)
    if abs(change) < 0.01:
        return "0.00%"
    sign = "+" if change > 0 else ""
    return f"{sign}{change:.2f}%"


def _state_class(action: str | None) -> str:
    action = str(action or "WAIT").upper()
    if action == "BUY":
        return "buy"
    if action == "SELL":
        return "sell"
    if action in {"CAO", "HIGH", "RISK"}:
        return "risk"
    return "wait"


def _bias_label(action: str, short: bool = False) -> str:
    if action == "BUY":
        return "BUY" if short else "Lợi BUY vàng"
    if action == "SELL":
        return "SELL" if short else "Lợi SELL vàng"
    if action == "RISK":
        return "RISK" if short else "Rủi ro cao"
    return "WAIT" if short else "Trung tính"


def _action_text(action: str | None, buy_text: str = "Ưu tiên BUY", sell_text: str = "Ưu tiên SELL", wait_text: str = "WAIT") -> str:
    action = str(action or "WAIT").upper()
    if action == "BUY":
        return f"🟢 {buy_text}"
    if action == "SELL":
        return f"🔴 {sell_text}"
    if action == "RISK":
        return "🟠 Rủi ro cao"
    return f"⚫ {wait_text}"


def _show_action_status(action: str | None, text: str) -> None:
    action = str(action or "WAIT").upper()
    if action == "BUY":
        st.success(text)
    elif action == "SELL":
        st.error(text)
    elif action == "RISK":
        st.warning(text)
    else:
        st.info(text)


RADAR_EXPLANATIONS = {
    "Gold": {
        "intro": "Giá vàng hiện tại.",
        "impact": "Giá trên MA20 và momentum tăng thường ủng hộ BUY.",
        "up": "Giá tăng hoặc giữ trên MA20 → ưu tiên BUY nếu momentum còn mở rộng.",
        "down": "Giá giảm hoặc nằm dưới MA20 → ưu tiên SELL nếu momentum yếu đi.",
    },
    "DXY": {
        "intro": "DXY là chỉ số sức mạnh đồng USD.",
        "impact": "Vàng thường đi ngược USD.",
        "up": "DXY tăng → bất lợi cho vàng → ưu tiên SELL.",
        "down": "DXY giảm → hỗ trợ vàng → ưu tiên BUY.",
    },
    "US10Y": {
        "intro": "US10Y là lợi suất trái phiếu Mỹ 10 năm.",
        "impact": "Lợi suất cao làm chi phí nắm giữ vàng tăng.",
        "up": "US10Y tăng → bất lợi cho vàng → ưu tiên SELL.",
        "down": "US10Y giảm → hỗ trợ vàng → ưu tiên BUY.",
    },
    "VIX": {
        "intro": "VIX là chỉ số sợ hãi của thị trường.",
        "impact": "VIX tăng mạnh có thể hỗ trợ vàng nhờ nhu cầu trú ẩn.",
        "up": "VIX tăng → risk-off → có thể ưu tiên BUY vàng.",
        "down": "VIX giảm → nhu cầu trú ẩn yếu hơn → vàng dễ bị SELL.",
    },
    "Oil": {
        "intro": "Oil phản ánh áp lực năng lượng và kỳ vọng lạm phát.",
        "impact": "Dầu tăng có thể khiến FED hawkish hơn, thường bất lợi cho vàng.",
        "up": "Oil tăng → lạm phát kỳ vọng tăng → nghiêng SELL vàng.",
        "down": "Oil giảm → áp lực lạm phát dịu lại → hỗ trợ BUY vàng.",
    },
    "Nasdaq": {
        "intro": "Nasdaq đại diện khẩu vị rủi ro của thị trường.",
        "impact": "Nasdaq mạnh thường là risk-on, làm nhu cầu trú ẩn giảm.",
        "up": "Nasdaq tăng → risk-on → vàng dễ yếu, ưu tiên SELL.",
        "down": "Nasdaq giảm → risk-off → hỗ trợ BUY vàng.",
    },
    "CPI": {
        "intro": "CPI là lạm phát tiêu dùng Mỹ.",
        "impact": "CPI cao hơn kỳ vọng khiến FED có thể giữ lãi suất cao.",
        "up": "CPI cao hơn kỳ vọng → bất lợi cho vàng → ưu tiên SELL.",
        "down": "CPI thấp hơn kỳ vọng → hỗ trợ kỳ vọng giảm lãi suất → ưu tiên BUY.",
    },
    "Core CPI": {
        "intro": "Core CPI là lạm phát lõi, loại bỏ thực phẩm và năng lượng.",
        "impact": "Core CPI được thị trường dùng để đo áp lực lạm phát bền vững.",
        "up": "Core CPI cao hơn kỳ vọng → gây áp lực SELL vàng.",
        "down": "Core CPI thấp hơn kỳ vọng → hỗ trợ BUY vàng.",
    },
    "PCE": {
        "intro": "PCE là chỉ số lạm phát FED rất quan tâm.",
        "impact": "PCE ảnh hưởng trực tiếp đến kỳ vọng lãi suất.",
        "up": "PCE cao hơn kỳ vọng → bất lợi cho vàng → ưu tiên SELL.",
        "down": "PCE thấp hơn kỳ vọng → có lợi cho vàng → ưu tiên BUY.",
    },
    "Nonfarm": {
        "intro": "Nonfarm là bảng lương phi nông nghiệp Mỹ.",
        "impact": "Việc làm mạnh có thể kéo USD và lợi suất tăng.",
        "up": "Nonfarm mạnh hơn kỳ vọng → bất lợi cho vàng → ưu tiên SELL.",
        "down": "Nonfarm yếu hơn kỳ vọng → FED mềm hơn → hỗ trợ BUY vàng.",
    },
    "FED Rate": {
        "intro": "FED Rate là lãi suất điều hành của FED.",
        "impact": "Lãi suất cao làm vàng kém hấp dẫn hơn tài sản sinh lời.",
        "up": "Kỳ vọng tăng hoặc giữ lãi suất cao → ưu tiên SELL vàng.",
        "down": "Kỳ vọng giảm lãi suất → hỗ trợ BUY vàng.",
    },
    "Powell Speech": {
        "intro": "Powell Speech là phát biểu của Chủ tịch FED.",
        "impact": "Giọng điệu của Powell có thể đổi kỳ vọng lãi suất rất nhanh.",
        "up": "Hawkish/cứng rắn → bất lợi cho vàng → ưu tiên SELL.",
        "down": "Dovish/mềm mỏng → hỗ trợ vàng → ưu tiên BUY.",
    },
    "GDP": {
        "intro": "GDP đo sức khỏe tăng trưởng của kinh tế Mỹ.",
        "impact": "GDP mạnh có thể hỗ trợ USD và lợi suất.",
        "up": "GDP cao hơn kỳ vọng → thường bất lợi cho vàng.",
        "down": "GDP thấp hơn kỳ vọng → có thể hỗ trợ BUY vàng.",
    },
    "Unemployment": {
        "intro": "Unemployment phản ánh sức khỏe thị trường lao động Mỹ.",
        "impact": "Thất nghiệp tăng làm kỳ vọng FED mềm hơn.",
        "up": "Thất nghiệp cao hơn kỳ vọng → hỗ trợ BUY vàng.",
        "down": "Thất nghiệp thấp hơn kỳ vọng → có thể gây áp lực SELL vàng.",
    },
}


def get_radar_explanation(indicator_name: str, value: object, change_pct: object, bias: str) -> str:
    info = RADAR_EXPLANATIONS.get(indicator_name, {
        "intro": f"{indicator_name} là chỉ số theo dõi thị trường.",
        "impact": "Chỉ số này được dùng để đánh giá áp lực BUY/SELL lên vàng.",
        "up": "Chỉ số tăng có thể thay đổi bias của vàng.",
        "down": "Chỉ số giảm có thể thay đổi bias của vàng.",
    })
    try:
        change_value = float(change_pct)
    except (TypeError, ValueError):
        change_value = 0.0

    macro_names = {"CPI", "Core CPI", "PCE", "Nonfarm", "FED Rate", "Powell Speech", "GDP", "Unemployment"}
    if abs(change_value) < 0.01:
        movement = "chưa tạo thiên hướng rõ"
    elif indicator_name in macro_names:
        movement = "cao hơn kỳ vọng" if change_value > 0 else "thấp hơn kỳ vọng"
    else:
        movement = "đang tăng" if change_value > 0 else "đang giảm"

    bias = str(bias or "WAIT").upper()
    if bias == "BUY":
        current = f"Hiện tại {indicator_name} {movement}, đây là yếu tố hỗ trợ BUY vàng."
    elif bias == "SELL":
        current = f"Hiện tại {indicator_name} {movement}, đây là áp lực SELL đối với vàng."
    elif bias == "RISK":
        current = f"Hiện tại {indicator_name} tạo rủi ro cao, nên giảm khối lượng và quản lý lệnh chặt."
    else:
        current = f"Hiện tại {indicator_name} {movement}, chưa tạo thiên hướng rõ."

    value_text = ""
    if value is not None and str(value).strip() and str(value).strip().upper() != "N/A":
        value_text = f"Giá trị: {value}."
    lines = [info["intro"], info["impact"], info["up"], info["down"], value_text, current]
    return "\n".join(line for line in lines if line)


def render_radar_card(name: str, value: object, change: object, bias: str, explanation: str) -> None:
    bias = str(bias or "WAIT").upper()
    display_value = value if value is not None and str(value).strip() else "N/A"
    with st.container(border=True):
        st.metric(f"{name} ?", display_value, change, help=explanation)
        _show_action_status(bias, f"{_action_text(bias)}: {_bias_label(bias)}")


def _macro_pressure_action(score: object) -> str:
    try:
        value = int(float(score))
    except (TypeError, ValueError):
        value = 50
    if value <= 30:
        return "BUY"
    if value >= 61:
        return "SELL"
    return "WAIT"


def _fred_summary(payload: dict | None) -> str:
    payload = payload or {}
    if not payload.get("enabled"):
        return "FRED: Chưa kết nối"
    data = payload.get("data", {})
    if not data:
        return "FRED: Chưa có dữ liệu mới"
    parts = []
    for name, item in data.items():
        value = item.get("value")
        date = item.get("date") or "N/A"
        value_text = _fmt_number(value, 2) if value is not None else "N/A"
        parts.append(f"{name}: {value_text} ({date})")
    return " · ".join(parts[:4])


def _market_bias(key: str, change: float, gold_df: pd.DataFrame) -> tuple[str, str]:
    threshold = 0.03
    if key == "GOLD":
        if gold_df is not None and not gold_df.empty and {"Close", "MA20"}.issubset(gold_df.columns):
            close = latest_float(gold_df["Close"])
            ma20 = latest_float(gold_df["MA20"])
            if close > ma20:
                return "BUY", "Giá trên MA20"
            if close < ma20:
                return "SELL", "Giá dưới MA20"
        if change > threshold:
            return "BUY", "Giá tăng"
        if change < -threshold:
            return "SELL", "Giá giảm"
        return "WAIT", "Đi ngang"
    if key in {"DXY", "US10Y", "OIL"}:
        if change > threshold:
            return "SELL", "Tăng gây áp lực"
        if change < -threshold:
            return "BUY", "Giảm hỗ trợ vàng"
        return "WAIT", "Trung tính"
    if key == "VIX":
        if change > 0.5:
            return "BUY", "Risk-off hỗ trợ vàng"
        if change < -0.5:
            return "SELL", "Risk-on giảm trú ẩn"
        return "WAIT", "Trung tính"
    if key == "NASDAQ":
        if change > threshold:
            return "SELL", "Risk-on"
        if change < -threshold:
            return "BUY", "Risk-off"
        return "WAIT", "Trung tính"
    return "WAIT", "Trung tính"


def _build_forex_cards(bundle: dict, gold_df: pd.DataFrame) -> list[dict]:
    specs = [
        ("Gold", "GOLD"),
        ("DXY", "DXY"),
        ("US10Y", "US10Y"),
        ("VIX", "VIX"),
        ("Oil", "OIL"),
        ("Nasdaq", "NASDAQ"),
    ]
    cards = []
    for label, key in specs:
        df = gold_df if key == "GOLD" else bundle.get(key, pd.DataFrame())
        change = pct_change(df.get("Close", pd.Series(dtype=float))) if isinstance(df, pd.DataFrame) else 0.0
        action, note = _market_bias(key, change, gold_df)
        cards.append(
            {
                "name": label,
                "value": _fmt_market_value(df),
                "change": _fmt_change(change),
                "change_pct": change,
                "action": action,
                "note": note,
            }
        )
    return cards


def _parse_event_number(value: object) -> float | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).replace("%", "").replace(",", "").strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _event_value(row: pd.Series, names: tuple[str, ...], default: str) -> str:
    for name in names:
        if name in row and not pd.isna(row[name]) and str(row[name]).strip():
            return str(row[name])
    return default


def _find_event(events_df: pd.DataFrame, aliases: tuple[str, ...]) -> pd.Series | None:
    if events_df is None or events_df.empty or "event" not in events_df:
        return None
    pattern = "|".join(aliases)
    rows = events_df[events_df["event"].astype(str).str.contains(pattern, case=False, na=False, regex=True)].copy()
    if rows.empty:
        return None
    if "time" in rows:
        rows["time"] = pd.to_datetime(rows["time"], errors="coerce")
        now = pd.Timestamp.now()
        upcoming = rows[rows["time"] >= now].sort_values("time")
        if not upcoming.empty:
            return upcoming.iloc[0]
        return rows.sort_values("time", ascending=False).iloc[0]
    return rows.iloc[0]


def _macro_impact(label: str, expected: str, actual: str, row: pd.Series | None, erisk: dict) -> tuple[str, str]:
    exp_num = _parse_event_number(expected)
    act_num = _parse_event_number(actual)
    if exp_num is not None and act_num is not None:
        hot = act_num > exp_num
        if label == "Unemployment":
            action = "BUY" if hot else "SELL"
        else:
            action = "SELL" if hot else "BUY"
        return action, _bias_label(action, short=True)
    if row is not None and erisk.get("blocked"):
        blocked_events = " ".join(str(item.get("event", "")) for item in erisk.get("events", []))
        if str(row.get("event", "")) in blocked_events:
            return "RISK", "NEWS"
    return "WAIT", "Chờ"


def _build_macro_cards(events_df: pd.DataFrame, erisk: dict) -> list[dict]:
    specs = [
        ("CPI", ("CPI",)),
        ("Core CPI", ("Core CPI",)),
        ("PCE", ("PCE", "Core PCE")),
        ("Nonfarm", ("Nonfarm", "NFP")),
        ("FED Rate", ("FED Rate", "FOMC")),
        ("Powell Speech", ("Powell Speech", "Powell")),
        ("GDP", ("GDP",)),
        ("Unemployment", ("Unemployment", "Jobless")),
    ]
    cards = []
    for label, aliases in specs:
        row = _find_event(events_df, aliases)
        expected = _event_value(row, ("expected", "forecast", "consensus", "estimate"), "Chờ dữ liệu") if row is not None else "Chờ dữ liệu"
        actual = _event_value(row, ("actual", "result", "real"), "Chưa công bố") if row is not None else "Chưa công bố"
        event_time = ""
        if row is not None and "time" in row and not pd.isna(row["time"]):
            event_time = pd.to_datetime(row["time"]).strftime("%d/%m %H:%M")
        action, impact = _macro_impact(label, expected, actual, row, erisk)
        exp_num = _parse_event_number(expected)
        act_num = _parse_event_number(actual)
        change_pct = (act_num - exp_num) if exp_num is not None and act_num is not None else 0.0
        cards.append(
            {
                "name": label,
                "expected": expected,
                "actual": actual,
                "impact": impact,
                "action": action,
                "change_pct": change_pct,
                "time": event_time or "Không có lịch",
            }
        )
    return cards


def _build_market_status(
    signal: dict,
    gold_analysis: dict,
    macro: dict,
    forex_cards: list[dict],
    erisk: dict,
    policy_result: dict,
    risk_fb: dict,
    mtf: dict,
) -> list[tuple[str, str]]:
    action = str(signal.get("action", "WAIT")).upper()
    strategy = str(signal.get("strategy", ""))
    regime = gold_analysis.get("market_regime", {})
    volatility_score = int(signal.get("volatility_score", gold_analysis.get("volatility", {}).get("score", 0)) or 0)
    rows: list[tuple[str, str]] = []
    pressure_score = int(macro.get("score", 50) or 50)
    pressure_action = _macro_pressure_action(pressure_score)
    rows.append((f"SonFED Pressure Index {pressure_score}/100: {_bias_label(pressure_action)}", pressure_action))
    if action in {"BUY", "SELL"}:
        strength = "mạnh" if signal.get("scalp_accepted") or int(signal.get("momentum_score", 0) or 0) >= 3 else "vừa"
        rows.append((f"Momentum {action} {strength}", action))
    else:
        rows.append(("Momentum chưa đủ rõ để vào mới", "WAIT"))

    for item in forex_cards:
        if item["name"] in {"DXY", "US10Y", "VIX"}:
            if item["action"] == "BUY":
                rows.append((f"{item['name']} hỗ trợ BUY vàng", "BUY"))
            elif item["action"] == "SELL":
                rows.append((f"{item['name']} gây áp lực SELL vàng", "SELL"))

    if "breakout" in strategy.lower():
        rows.append(("Breakout M15 đã xác nhận", action if action in {"BUY", "SELL"} else "WAIT"))
    else:
        rows.append(("Chưa breakout xác nhận", "WAIT"))

    if volatility_score >= 70:
        rows.append((f"Volatility mở rộng mạnh: {volatility_score}/100", "RISK"))
    elif volatility_score >= 35:
        rows.append((f"Volatility đủ cho scalp: {volatility_score}/100", action if action in {"BUY", "SELL"} else "WAIT"))
    else:
        rows.append((f"Volatility thấp: {volatility_score}/100", "WAIT"))

    if regime.get("momentum"):
        rows.append((str(regime.get("momentum")), action if action in {"BUY", "SELL"} else "WAIT"))
    if erisk.get("blocked"):
        rows.append(("Tin vĩ mô đang gần, ưu tiên giảm rủi ro", "RISK"))
    if policy_result.get("blocked"):
        rows.append(("Policy đang khóa entry mới", "RISK"))
    if risk_fb.get("connected") and not risk_fb.get("allow", True):
        rows.append(("SonEXEC risk đang khóa giao dịch", "RISK"))
    if mtf.get("summary"):
        rows.append((compact_reason(mtf.get("summary", ""), 90), "WAIT"))
    return rows[:8]


def render_ai_trading_radar_overview(
    config: dict,
    refresh_info: dict,
    signal: dict,
    ai_decision: dict,
    gold_analysis: dict,
    macro: dict,
    bias: str,
    mtf: dict,
    bundle: dict,
    gold_df: pd.DataFrame,
    events_df: pd.DataFrame,
    erisk: dict,
    policy_result: dict,
    risk_fb: dict,
    signal_history: list[dict],
    fred_payload: dict | None = None,
) -> None:
    action = str(signal.get("action", ai_decision.get("action", "WAIT")) or "WAIT").upper()
    confidence = signal.get("confidence", ai_decision.get("winrate", 0))
    rr = signal.get("rr", ai_decision.get("rr", "N/A"))
    tp = signal.get("take_profit", ai_decision.get("tp", "N/A"))
    sl = signal.get("stop_loss", ai_decision.get("sl", "N/A"))
    risk = signal.get("risk_level", "N/A")
    mode = normalize_trading_mode(config.get("trade", {}).get("mode", "Hướng dẫn sử dụng"))
    forex_cards = _build_forex_cards(bundle, gold_df)
    macro_cards = _build_macro_cards(events_df, erisk)
    status_rows = _build_market_status(signal, gold_analysis, macro, forex_cards, erisk, policy_result, risk_fb, mtf)
    refresh_state = "ON" if refresh_info.get("enabled") else "OFF"

    st.subheader("AI Decision Box")
    with st.container(border=True):
        _show_action_status(action, f"{_action_text(action, buy_text='BUY', sell_text='SELL', wait_text='WAIT')} · M15 Scalp · {mode} · Refresh {refresh_state}")
        cols = st.columns(5)
        cols[0].metric("Confidence", f"{confidence}%")
        cols[1].metric("RR", rr if rr is not None else "N/A")
        cols[2].metric("TP", _fmt_number(tp) if tp not in {None, "N/A"} else "N/A")
        cols[3].metric("SL", _fmt_number(sl) if sl not in {None, "N/A"} else "N/A")
        cols[4].metric("Risk", risk)

    st.subheader("Radar Forex")
    forex_cols = st.columns(6)
    for idx, card in enumerate(forex_cards):
        with forex_cols[idx]:
            explanation = get_radar_explanation(card["name"], card["value"], card.get("change_pct", 0.0), card["action"])
            render_radar_card(card["name"], card["value"], card["change"], card["action"], explanation)

    st.subheader("Radar FED / Vĩ Mô")
    for start in range(0, len(macro_cards), 4):
        cols = st.columns(4)
        for idx, card in enumerate(macro_cards[start:start + 4]):
            with cols[idx]:
                explanation = get_radar_explanation(card["name"], card["actual"], card.get("change_pct", 0.0), card["action"])
                render_radar_card(card["name"], card["actual"], f"Kỳ vọng: {card['expected']}", card["action"], explanation)
                st.caption(card["time"])

    st.subheader("Áp lực BUY/SELL")
    try:
        pressure_score = int(float(macro.get("score", 50) or 50))
    except (TypeError, ValueError):
        pressure_score = 50
    pressure_action = _macro_pressure_action(pressure_score)
    pressure_cols = st.columns(3)
    with pressure_cols[0]:
        with st.container(border=True):
            st.markdown("**SonFED Pressure Index**")
            st.metric("Điểm áp lực", f"{pressure_score}/100")
            _show_action_status(pressure_action, f"{_action_text(pressure_action)}: {macro.get('interpretation', 'Trạng thái vĩ mô trung tính')}")
    with pressure_cols[1]:
        with st.container(border=True):
            st.markdown("**Macro Bias**")
            st.caption("BUY / SELL / WAIT cho vàng")
            _show_action_status(pressure_action, f"{_bias_label(pressure_action)}: {compact_reason(bias, 160)}")
    with pressure_cols[2]:
        with st.container(border=True):
            st.markdown("**FRED**")
            st.caption("Dữ liệu vĩ mô Mỹ mới nhất")
            st.info(_fred_summary(fred_payload))

    st.subheader("AI Market Status")
    for start in range(0, len(status_rows), 2):
        cols = st.columns(2)
        for idx, (text, row_action) in enumerate(status_rows[start:start + 2]):
            with cols[idx]:
                _show_action_status(row_action, text)
    render_trade_signal_timeline(signal_history, limit=15, compact=True)


def _signal_history_key(signal: dict) -> str:
    return "|".join(
        str(signal.get(key, ""))
        for key in ("action", "strategy", "confidence", "entry_zone", "risk_level", "rr")
    )


def load_signal_history(limit: int = 200) -> list[dict]:
    rows = load_json(SIGNAL_HISTORY_PATH, [])
    if not isinstance(rows, list):
        return []
    return [row for row in rows if isinstance(row, dict)][-limit:]


def append_signal_history(signal: dict, limit: int = 200) -> list[dict]:
    rows = load_signal_history(limit)
    key = _signal_history_key(signal)
    if rows and rows[-1].get("key") == key:
        return rows
    reason = signal.get("reason") or signal.get("strategy") or ""
    row = {
        "timestamp": signal.get("updated_at") or signal.get("time") or pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "signal": str(signal.get("action", "WAIT")).upper(),
        "confidence": int(signal.get("confidence", 0) or 0),
        "reason": " ".join(str(reason).split())[:240],
        "strategy": signal.get("strategy", ""),
        "rr": signal.get("rr"),
        "key": key,
        "replay_id": key,
        "accuracy": None,
    }
    rows.append(row)
    rows = rows[-limit:]
    save_json(SIGNAL_HISTORY_PATH, rows)
    return rows


def _signal_icon(action: str | None) -> str:
    action = str(action or "WAIT").upper()
    if action == "BUY":
        return "🟢 BUY"
    if action == "SELL":
        return "🔴 SELL"
    return "⚫ WAIT"


def _format_signal_time(value: object) -> str:
    try:
        return pd.to_datetime(value).strftime("%d/%m/%Y %H:%M")
    except Exception:
        return str(value or "N/A")


def render_trade_signal_timeline(history: list[dict], limit: int = 15, compact: bool = False) -> None:
    rows = list(reversed(history[-limit:]))
    if not rows:
        st.info("Chưa có lịch sử tín hiệu.")
        return
    with st.container(border=True):
        st.markdown("**Trade Signal Timeline**")
        st.caption(f"{len(rows)} tín hiệu gần nhất · đã sẵn sàng cho replay, accuracy và performance tracking")
        for row in rows:
            action = str(row.get("signal", "WAIT")).upper()
            reason = compact_reason(row.get("reason", ""), 130 if compact else 180)
            cols = st.columns([1.45, 1.0, 0.85, 3.0])
            cols[0].caption(_format_signal_time(row.get("timestamp")))
            with cols[1]:
                _show_action_status(action, _signal_icon(action))
            cols[2].metric("Confidence", f"{row.get('confidence', 0)}%")
            cols[3].caption(reason or row.get("strategy", "Không có lý do."))


def compact_reason(text: str, limit: int = 150) -> str:
    clean = " ".join(str(text or "").split())
    if len(clean) <= limit:
        return clean
    return clean[: limit - 3].rstrip() + "..."


def process_sonfed_telegram_monitoring(config: dict, trade_status: dict, performance_status: dict, signal: dict, risk_feedback: dict, changes: list[str]) -> None:
    telegram = ensure_telegram_config(config).get("telegram", {})
    if not telegram.get("enabled"):
        return
    cooldown = int(telegram.get("cooldown_seconds", 300) or 300)
    interval_seconds = max(60, int(telegram.get("report_interval_minutes", 15) or 15) * 60)
    bucket = int(time.time() // interval_seconds)

    if telegram.get("report_enabled", True):
        message = build_account_report(performance_status or trade_status, config)
        send_telegram_queued(message, enabled=True, event_key=f"sonfed-account-report|{bucket}", cooldown_seconds=interval_seconds)

    account = trade_status.get("account", {}) if isinstance(trade_status, dict) else {}
    drawdown = safe_float(account.get("drawdown_percent"))
    if telegram.get("drawdown_alerts_enabled", True) and drawdown >= safe_float(telegram.get("drawdown_alert_percent"), 5.0):
        send_telegram_queued(
            f"⚠️ DRAWDOWN ALERT\n\nDrawdown hiện tại: {drawdown:.2f}%",
            enabled=True,
            event_key=f"sonfed-drawdown|{int(drawdown)}",
            cooldown_seconds=cooldown,
        )

    if risk_feedback.get("connected") and not risk_feedback.get("allow", True):
        send_telegram_queued(
            f"⚠️ RISK ALERT\n\n{risk_feedback.get('reason', 'SonEXEC đang khóa giao dịch.')}",
            enabled=True,
            event_key=f"sonfed-risk|{risk_feedback.get('reason', '')}",
            cooldown_seconds=cooldown,
        )

    state = load_json(TELEGRAM_MONITOR_STATE_PATH, {})
    if not isinstance(state, dict):
        state = {}
    positions = performance_status.get("positions") or trade_status.get("positions", [])
    current_tickets = {str(pos.get("ticket")) for pos in positions if pos.get("ticket") is not None}
    previous_tickets = set(state.get("open_tickets", []))
    if telegram.get("order_alerts_enabled", True):
        for ticket in sorted(current_tickets - previous_tickets):
            pos = next((item for item in positions if str(item.get("ticket")) == ticket), {})
            send_telegram_queued(
                "⚠️ NEW OPEN POSITION\n\n" + build_orders_report([pos]),
                enabled=True,
                event_key=f"sonfed-ticket-open|{ticket}",
                cooldown_seconds=cooldown,
            )
        for ticket in sorted(previous_tickets - current_tickets):
            send_telegram_queued(
                f"⚠️ POSITION CLOSED\n\nLệnh #{ticket} đã không còn mở trên SonEXEC.",
                enabled=True,
                event_key=f"sonfed-ticket-closed|{ticket}",
                cooldown_seconds=cooldown,
            )
    state["open_tickets"] = sorted(current_tickets)
    current_action = str(signal.get("action", "WAIT"))
    previous_action = str(state.get("last_ai_action", current_action))
    if telegram.get("ai_bias_alerts_enabled", True) and previous_action != current_action:
        message = build_ai_change_alert(previous_action, current_action, str(signal.get("reason", "")))
        send_telegram_queued(message, enabled=True, event_key=f"sonfed-ai-change|{previous_action}|{current_action}|{signal.get('reason', '')}", cooldown_seconds=cooldown)
    state["last_ai_action"] = current_action
    state["last_changes"] = changes[-10:]
    save_json(TELEGRAM_MONITOR_STATE_PATH, state)


def _money_text(value: object) -> str:
    number = safe_float(value)
    sign = "+" if number > 0 else ""
    return f"{sign}{number:.2f}$"


def _pct_text(value: object) -> str:
    number = safe_float(value)
    sign = "+" if number > 0 else ""
    return f"{sign}{number:.1f}%"


def render_statistics_tab(performance_status: dict, trade_status: dict, signal: dict, risk_fb: dict, config: dict) -> None:
    telegram = ensure_telegram_config(config).get("telegram", {})
    account = performance_status.get("account") or trade_status.get("account", {})
    positions = performance_status.get("positions") or trade_status.get("positions", [])
    stats = performance_status.get("statistics") or account_statistics(account, positions, safe_float(telegram.get("base_capital"), 200.0))
    updated_at = performance_status.get("updated_at") or trade_status.get("updated_at") or "Chưa có dữ liệu"
    position_rows = enrich_positions(positions)

    st.subheader("Thống kê SonFED")
    st.caption("Nguồn dữ liệu: SonEXEC cập nhật sang shared/performance_status.json, SonFED tổng hợp và gửi Telegram.")
    if not performance_status:
        st.warning("Chưa nhận được performance_status.json từ SonEXEC. Đang dùng dữ liệu trade_status hiện có.")

    with st.container(border=True):
        st.markdown("**Account Command Center**")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Balance", f"{safe_float(stats.get('balance')):,.2f}$")
        c2.metric("Equity", f"{safe_float(stats.get('equity')):,.2f}$")
        c3.metric("Floating PnL", _money_text(stats.get("floating_pnl", 0)))
        c4.metric("Profit trên vốn 200$", _pct_text(stats.get("profit_percent", 0)))
        c5.metric("Drawdown", _pct_text(stats.get("drawdown_percent", 0)))

    left, right = st.columns(2)
    with left:
        with st.container(border=True):
            st.markdown("**Exposure BUY/SELL**")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("BUY Orders", int(safe_float(stats.get("buy_orders"))))
            c2.metric("SELL Orders", int(safe_float(stats.get("sell_orders"))))
            c3.metric("BUY Volume", f"{safe_float(stats.get('buy_volume')):.2f}")
            c4.metric("SELL Volume", f"{safe_float(stats.get('sell_volume')):.2f}")
    with right:
        with st.container(border=True):
            st.markdown("**Bot Health**")
            c1, c2, c3 = st.columns(3)
            c1.metric("Cập nhật", updated_at)
            c2.metric("AI Signal", signal.get("action", "WAIT"))
            c3.metric("Confidence", f"{signal.get('confidence', 0)}%")
            if risk_fb.get("connected") and not risk_fb.get("allow", True):
                st.error(f"Risk đang khóa: {risk_fb.get('reason', '')}")
            else:
                st.success("Risk hiện tại ổn hoặc chưa có cảnh báo từ SonEXEC.")

    with st.container(border=True):
        st.markdown("**Performance**")
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("Winrate", _pct_text(stats.get("winrate", 0)))
        c2.metric("RR Avg", f"{safe_float(stats.get('rr_avg')):.2f}")
        c3.metric("Win", int(safe_float(stats.get("wins"))))
        c4.metric("Loss", int(safe_float(stats.get("losses"))))
        c5.metric("Today PnL", _money_text(stats.get("today_pnl", 0)))
        c6.metric("Week PnL", _money_text(stats.get("week_pnl", 0)))

    best = stats.get("best_strategy", {}) if isinstance(stats.get("best_strategy"), dict) else {}
    worst = stats.get("worst_strategy", {}) if isinstance(stats.get("worst_strategy"), dict) else {}
    s1, s2 = st.columns(2)
    with s1:
        with st.container(border=True):
            st.markdown("**Best Strategy**")
            st.success(best.get("name", "Chưa có dữ liệu"))
            st.metric("Winrate", _pct_text(best.get("winrate", 0)))
            st.metric("Profit", _money_text(best.get("profit", 0)))
    with s2:
        with st.container(border=True):
            st.markdown("**Worst Strategy**")
            st.warning(worst.get("name", "Chưa có dữ liệu"))
            st.metric("Winrate", _pct_text(worst.get("winrate", 0)))
            st.metric("Profit", _money_text(worst.get("profit", 0)))

    with st.container(border=True):
        st.markdown("**Lệnh đang mở**")
        if position_rows:
            st.dataframe(pd.DataFrame(position_rows), use_container_width=True, hide_index=True)
        else:
            st.info("Không có lệnh đang mở.")

    with st.container(border=True):
        st.markdown("**Telegram Monitor**")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Telegram", "ON" if telegram.get("enabled") else "OFF")
        c2.metric("Report", "ON" if telegram.get("report_enabled") else "OFF")
        c3.metric("Cooldown", f"{int(telegram.get('cooldown_seconds', 300))}s")
        c4.metric("Chu kỳ", f"{int(telegram.get('report_interval_minutes', 15))} phút")
        st.caption("Nếu chưa gửi được Telegram, mở bot và gửi /start một lần để SonFED tự nhận chat_id.")


def build_compact_ai_status(signal: dict, ai_decision: dict, gold_analysis: dict, adjustments_payload: dict, policy_result: dict) -> str:
    adjustments = adjustments_payload.get("adjustments", [])
    active_adjustments = [
        item for item in adjustments
        if normalize_adjustment_action(item.get("action", "")) not in {"HOLD_POSITION", "DISABLE_NEW_ENTRY"}
    ]
    if active_adjustments:
        item = active_adjustments[0]
        return f"AI ưu tiên {vi_action(normalize_adjustment_action(item.get('action', ''))).lower()} theo volatility hiện tại."

    action = signal.get("action", ai_decision.get("action", "WAIT"))
    volatility_score = int(signal.get("volatility_score", gold_analysis.get("volatility", {}).get("score", 0)) or 0)
    if policy_result.get("blocked"):
        return "AI chưa mở vị thế mới vì " + compact_reason(policy_result.get("message", "risk chưa đạt điều kiện vận hành."))
    if action == "WAIT" and volatility_score >= 70:
        return "AI đang đánh giá thị trường biến động mạnh và chưa đủ xác nhận để mở vị thế mới."
    if action == "WAIT":
        return "AI đang đứng ngoài, chờ xác nhận rõ hơn trước khi mở vị thế mới."
    return f"AI ưu tiên {action} với xác suất {ai_decision.get('winrate', signal.get('confidence', 0))}% và RR {ai_decision.get('rr', 'N/A')}."


def build_critical_alerts(signal: dict, policy_result: dict, erisk: dict, risk_fb: dict, mtf: dict) -> list[str]:
    alerts: list[str] = []
    if policy_result.get("blocked"):
        alerts.append(compact_reason(policy_result.get("message", ""), 180))
    if risk_fb.get("connected") and not risk_fb.get("allow", True):
        alerts.append("SonEXEC đang khóa risk: " + compact_reason(risk_fb.get("reason", ""), 140))
    if erisk.get("blocked"):
        alerts.append(compact_reason(erisk.get("message", "Sắp có tin mạnh."), 160))

    risk_level = str(signal.get("risk_level", "")).lower()
    volatility_score = int(signal.get("volatility_score", 0) or 0)
    if "cao" in risk_level:
        alerts.append("Risk đang ở mức cao.")
    if volatility_score >= 70:
        alerts.append(f"Volatility bất thường: {volatility_score}/100.")

    spread = risk_fb.get("spread_points")
    max_spread = signal.get("policy", {}).get("max_spread")
    if spread is not None and max_spread is not None and safe_float(spread) > safe_float(max_spread):
        alerts.append(f"Spread bất thường: {safe_float(spread):.0f} điểm.")

    mtf_summary = str(mtf.get("summary", "")).lower()
    if any(token in mtf_summary for token in ("lệch", "ngược", "conflict", "không đồng thuận")):
        alerts.append("Đa khung thời gian chưa đồng thuận.")
    return [item for item in dict.fromkeys(alerts) if item][:5]


def render_critical_alerts(signal: dict, policy_result: dict, erisk: dict, risk_fb: dict, mtf: dict) -> None:
    alerts = build_critical_alerts(signal, policy_result, erisk, risk_fb, mtf)
    if not alerts:
        st.success("Không có cảnh báo nghiêm trọng.")
        return
    for item in alerts:
        st.warning(item)


def render_auto_refresh_compact(refresh_info: dict) -> None:
    st.subheader("Auto refresh")
    summary = dashboard_summary(refresh_info)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Trạng thái", summary["enabled"])
    c2.metric("Chu kỳ", summary["interval"])
    c3.metric("Tín hiệu hiện tại", summary["current_action"])
    c4.metric("Tín hiệu trước", summary["previous_action"])
    st.caption(f"Cập nhật cuối: {summary['last_update']} | Tiếp theo: {summary['next_update']}")


def render_sonexec_status_compact(trade_feedback: dict, risk_fb: dict) -> None:
    st.subheader("SonEXEC status")
    if trade_feedback.get("connected"):
        st.success(compact_reason(trade_feedback.get("message", "Đã nhận trạng thái từ SonEXEC."), 140))
    else:
        st.warning(compact_reason(trade_feedback.get("message", "Chưa nhận trạng thái từ SonEXEC."), 140))

    account = trade_feedback.get("account", {})
    if account:
        c1, c2, c3 = st.columns(3)
        c1.metric("Balance", f"{safe_float(account.get('balance')):,.2f}")
        c2.metric("Equity", f"{safe_float(account.get('equity')):,.2f}")
        c3.metric("Drawdown", f"{safe_float(account.get('drawdown_percent')):.2f}%")
    else:
        st.info("MT5 chưa kết nối hoặc chưa có dữ liệu tài khoản.")

    if risk_fb.get("connected"):
        if risk_fb.get("allow"):
            st.success("Risk OK: " + compact_reason(risk_fb.get("reason", ""), 120))
        else:
            st.error("Risk khóa: " + compact_reason(risk_fb.get("reason", ""), 120))


def render_position_management_compact(trade_feedback: dict, adjustments_payload: dict, signal: dict, gold_analysis: dict) -> None:
    positions = trade_feedback.get("positions", [])
    buy_volume = sum(safe_float(p.get("lot", p.get("volume", 0))) for p in positions if "BUY" in str(p.get("type", p.get("type_name", ""))).upper())
    sell_volume = sum(safe_float(p.get("lot", p.get("volume", 0))) for p in positions if "SELL" in str(p.get("type", p.get("type_name", ""))).upper())
    floating_profit = sum(safe_float(p.get("profit", 0)) for p in positions)

    st.subheader("Trạng thái quản lý lệnh")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Lệnh mở", len(positions))
    c2.metric("BUY lot", f"{buy_volume:.2f}")
    c3.metric("SELL lot", f"{sell_volume:.2f}")
    c4.metric("P/L", f"{floating_profit:,.2f}")

    c5, c6, c7 = st.columns(3)
    c5.metric("Risk", signal.get("risk_level", "N/A"))
    c6.metric("Market", vi_regime(gold_analysis.get("regime", "Chưa rõ")))
    c7.metric("Confidence", f"{signal.get('confidence', 0)}%")

    adjustments = adjustments_payload.get("adjustments", [])
    if not adjustments:
        st.info("Chưa có điều chỉnh lệnh mới.")
        return
    rows = []
    for item in adjustments:
        rows.append(
            {
                "Lệnh": item.get("ticket", "N/A"),
                "Hành động": vi_action(normalize_adjustment_action(item.get("action", ""))),
                "Độ tin cậy": item.get("ai_confidence", item.get("confidence", 0)),
                "Lý do": compact_reason(item.get("reason", ""), 120),
            }
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def render_guided_mode_ui(
    config: dict,
    refresh_info: dict,
    signal: dict,
    ai_decision: dict,
    gold_analysis: dict,
    macro: dict,
    mtf: dict,
    changes: list[str],
    policy: AITradePolicy,
    policy_result: dict,
    bundle: dict,
    gold_df: pd.DataFrame,
    bias: str,
    erisk: dict,
    risk_fb: dict,
    trade_feedback: dict,
    adjustments_payload: dict,
) -> None:
    render_status_strip(config, refresh_info, signal)
    render_user_guide_mode(config)
    st.plotly_chart(make_gold_chart(gold_df), use_container_width=True, key="overview_guided_gold_chart")
    st.subheader("Kết luận nhanh")
    st.write(gold_analysis["summary"])
    render_ai_decision_box(signal, ai_decision)
    render_policy_status(policy)
    render_policy_warning(policy_result)
    render_quick_decision_explanation(signal, ai_decision, gold_analysis, macro, mtf)
    st.write(ai_decision["reason"])
    st.subheader("Phân tích AI")
    st.write(gold_analysis.get("ai_analysis", "Chưa có phân tích AI."))
    st.write(bias)
    st.info(mtf["summary"])
    st.subheader("Cảnh báo risk")
    for alert in smart_alerts(gold_analysis, macro, mtf, erisk):
        st.warning(alert)
    render_auto_refresh_compact(refresh_info)


def render_semi_auto_mode_ui(
    config: dict,
    refresh_info: dict,
    signal: dict,
    ai_decision: dict,
    gold_analysis: dict,
    macro: dict,
    mtf: dict,
    changes: list[str],
    policy: AITradePolicy,
    policy_result: dict,
    bundle: dict,
    gold_df: pd.DataFrame,
    bias: str,
    erisk: dict,
    risk_fb: dict,
    trade_feedback: dict,
    adjustments_payload: dict,
) -> None:
    render_status_strip(config, refresh_info, signal)
    st.warning("Bán tự động: AI đưa tín hiệu, người dùng xác nhận thủ công.")
    render_ai_decision_box(signal, ai_decision)
    st.info(build_compact_ai_status(signal, ai_decision, gold_analysis, adjustments_payload, policy_result))
    render_policy_status(policy)
    render_critical_alerts(signal, policy_result, erisk, risk_fb, mtf)
    st.subheader("Lý do AI quyết định")
    st.write(compact_reason(ai_decision.get("reason", signal.get("reason", "")), 420))
    st.plotly_chart(make_gold_chart(gold_df), use_container_width=True, key="overview_semi_gold_chart")
    render_auto_refresh_compact(refresh_info)


def render_auto_mode_ui(
    config: dict,
    refresh_info: dict,
    signal: dict,
    ai_decision: dict,
    gold_analysis: dict,
    macro: dict,
    mtf: dict,
    changes: list[str],
    policy: AITradePolicy,
    policy_result: dict,
    bundle: dict,
    gold_df: pd.DataFrame,
    bias: str,
    erisk: dict,
    risk_fb: dict,
    trade_feedback: dict,
    adjustments_payload: dict,
) -> None:
    render_status_strip(config, refresh_info, signal)
    render_ai_decision_box(signal, ai_decision)
    st.subheader("Trạng thái AI")
    st.info(build_compact_ai_status(signal, ai_decision, gold_analysis, adjustments_payload, policy_result))
    st.subheader("Risk quan trọng")
    render_critical_alerts(signal, policy_result, erisk, risk_fb, mtf)
    render_auto_refresh_compact(refresh_info)
    render_sonexec_status_compact(trade_feedback, risk_fb)
    render_position_management_compact(trade_feedback, adjustments_payload, signal, gold_analysis)
    if changes:
        st.subheader("Tín hiệu mới")
        for item in changes[:4]:
            st.warning(compact_reason(item, 150))


def render_ai_assistant_mode_ui(
    config: dict,
    refresh_info: dict,
    signal: dict,
    ai_decision: dict,
    gold_analysis: dict,
    macro: dict,
    mtf: dict,
    changes: list[str],
    policy: AITradePolicy,
    policy_result: dict,
    bundle: dict,
    gold_df: pd.DataFrame,
    bias: str,
    erisk: dict,
    risk_fb: dict,
    trade_feedback: dict,
    adjustments_payload: dict,
) -> None:
    render_ai_decision_box(signal, ai_decision)
    st.info(build_compact_ai_status(signal, ai_decision, gold_analysis, adjustments_payload, policy_result))
    st.subheader("Cảnh báo quan trọng")
    render_critical_alerts(signal, policy_result, erisk, risk_fb, mtf)
    st.subheader("Điều chỉnh lệnh")
    render_position_management_compact(trade_feedback, adjustments_payload, signal, gold_analysis)
    st.subheader("Tín hiệu mới")
    if changes:
        for item in changes[:3]:
            st.warning(compact_reason(item, 150))
    else:
        st.success("Chưa có tín hiệu mới cần chú ý.")


def render_dashboard_explanation() -> None:
    st.subheader("📊 Dashboard gồm những gì?")
    rows = [
        ("Giá vàng", "Giá XAU/USD hiện tại. Đây là giá thị trường vàng so với USD."),
        ("DXY", "Chỉ số sức mạnh đồng USD. USD mạnh thường gây áp lực giảm lên vàng."),
        ("US10Y", "Lợi suất trái phiếu Mỹ 10 năm. Yield tăng thường bất lợi cho vàng."),
        ("Pressure Index", "Điểm áp lực thị trường do AI đánh giá. Điểm càng cao, thị trường càng nguy hiểm hoặc biến động mạnh."),
        ("Biểu đồ giá", "Hiển thị xu hướng giá vàng, MA20, EMA50, EMA200 và Bollinger Bands."),
        ("Trạng thái vận hành", "Cho biết app đang ở chế độ nào, auto refresh, Telegram và auto trade có bật không."),
    ]
    for title, body in rows:
        with st.container(border=True):
            st.markdown(f"**{title}**")
            st.write(body)


def render_trading_mode_help() -> None:
    st.subheader("📈 Các chế độ giao dịch")
    data = [
        {
            "Chế độ": "Hướng dẫn sử dụng",
            "Dùng để": "Học cách dùng app, xem dashboard, hiểu AI Decision Box.",
            "Không làm": "Không auto trade, không ghi signal, không gửi lệnh.",
            "Khuyến nghị": "Người mới bắt đầu.",
        },
        {
            "Chế độ": "Bán tự động",
            "Dùng để": "SonFED phân tích và đưa BUY / SELL / WAIT.",
            "Không làm": "Người dùng vẫn tự quyết định vào lệnh.",
            "Khuyến nghị": "Người mới đã hiểu cơ bản.",
        },
        {
            "Chế độ": "Tự động",
            "Dùng để": "Tự cập nhật dữ liệu, tự phân tích, ghi signal.json cho SonEXEC nếu bật.",
            "Không làm": "Không thay thế quản trị rủi ro của người dùng.",
            "Khuyến nghị": "Chỉ dùng khi đã hiểu hệ thống.",
        },
        {
            "Chế độ": "AI hỗ trợ",
            "Dùng để": "AI giải thích thị trường, logic BUY/SELL, cảnh báo risk và đề xuất kịch bản.",
            "Không làm": "Không tự gửi lệnh nếu Auto Trade chưa bật.",
            "Khuyến nghị": "Người muốn học cách thị trường vận động.",
        },
    ]
    st.dataframe(pd.DataFrame(data), use_container_width=True, hide_index=True)


def render_ai_decision_help() -> None:
    st.subheader("🤖 AI Decision Box là gì?")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.info("**BUY**\n\nAI nghiêng về mua.")
        st.info("**TP**\n\nTake Profit, mục tiêu chốt lời.")
        st.info("**Regime**\n\nTrạng thái thị trường hiện tại.")
    with c2:
        st.warning("**SELL**\n\nAI nghiêng về bán.")
        st.warning("**SL**\n\nStop Loss, mức cắt lỗ.")
        st.warning("**Volatility**\n\nMức biến động thị trường.")
    with c3:
        st.success("**WAIT**\n\nAI chưa thấy điểm vào đủ tốt.")
        st.success("**RR**\n\nRisk Reward Ratio, tỷ lệ lợi nhuận/rủi ro.")
        st.success("**Confidence / Winrate**\n\nĐộ tự tin tương đối của mô hình, không phải cam kết chắc thắng.")
    render_confidence_guide()
    render_wait_explanation()


def render_risk_warning_help() -> None:
    st.subheader("⚠️ Khi nào không nên giao dịch?")
    st.warning(
        "Không nên giao dịch khi có CPI / NFP / FOMC, spread tăng mạnh, volatility quá cao, "
        "AI confidence thấp, RR thấp, các khung thời gian lệch pha mạnh, hoặc bạn không hiểu vì sao AI đưa tín hiệu."
    )
    st.info("Đứng ngoài thị trường cũng là một quyết định giao dịch.")
    with st.container(border=True):
        st.markdown("**Auto Trade là gì?**")
        st.write(
            "Khi bật Auto Trade, SonEXEC có thể tự vào lệnh dựa trên tín hiệu từ SonFED. "
            "Điều này giúp giao dịch nhanh hơn và không cần bấm tay, nhưng AI vẫn có thể phân tích sai, "
            "market có thể đảo chiều mạnh và volatility có thể tăng đột ngột."
        )
        st.write("Không nên bật Auto Trade nếu chưa hiểu risk, chưa test tín hiệu, hoặc chưa biết SonEXEC hoạt động thế nào.")


def render_new_user_timeline() -> None:
    st.subheader("🛡️ Hướng dẫn từng bước")
    steps = [
        ("Bước 1: Chọn timeframe", "1m rất nhanh và nhiễu cao, 5m phù hợp scalping, 15m intraday, 1h swing ngắn, 4h xem xu hướng lớn. Người mới nên dùng 15m hoặc 1h."),
        ("Bước 2: Chọn period", "5 ngày cho ngắn hạn, 1 tháng cho xu hướng gần, 6 tháng cho bối cảnh lớn hơn."),
        ("Bước 3: Bấm Refresh dữ liệu", "App sẽ tải dữ liệu mới nhất và phân tích lại."),
        ("Bước 4: Đọc AI Decision Box", "BUY là ưu tiên mua, SELL là ưu tiên bán, WAIT là đứng ngoài."),
        ("Bước 5: Kiểm tra Winrate", "Winrate là xác suất tương đối theo mô hình AI, không phải đảm bảo chắc thắng."),
        ("Bước 6: Kiểm tra RR", "RR = tỷ lệ lợi nhuận/rủi ro. Ví dụ RR 2.0 nghĩa là lợi nhuận kỳ vọng gấp 2 lần rủi ro."),
        ("Bước 7: Đọc cảnh báo risk", "Nếu volatility quá mạnh, spread cao hoặc có tin tức lớn thì nên hạn chế giao dịch."),
        ("Bước 8: Chỉ bật Auto Trade khi đã sẵn sàng", "Chỉ bật sau khi đã hiểu risk, test tín hiệu và dùng bán tự động đủ lâu."),
    ]
    for title, body in steps:
        with st.container(border=True):
            st.markdown(f"**{title}**")
            st.write(body)


def render_faq_section() -> None:
    st.subheader("FAQ")
    faqs = [
        ("SonFED có chắc thắng không?", "Không. SonFED là hệ thống phân tích xác suất."),
        ("Tại sao AI bảo SELL nhưng giá lại tăng?", "Market luôn có xác suất. Không có hệ thống nào đúng 100%."),
        ("WAIT nghĩa là gì?", "AI chưa thấy điểm vào đủ tốt."),
        ("Người mới nên dùng chế độ nào?", "Hướng dẫn sử dụng → Bán tự động."),
        ("SonEXEC khác SonFED thế nào?", "SonFED phân tích. SonEXEC thực thi."),
    ]
    for question, answer in faqs:
        with st.expander(question):
            st.write(answer)


def render_user_guide_mode(config: dict) -> None:
    st.title("Chào mừng đến với SonFED")
    st.write(
        "SonFED là hệ thống radar vĩ mô và AI hỗ trợ phân tích giao dịch XAU/USD (vàng). "
        "Ứng dụng giúp theo dõi giá vàng, phân tích kỹ thuật, phân tích dữ liệu vĩ mô, "
        "nhận diện trạng thái thị trường, đưa ra tín hiệu BUY / SELL / WAIT và cảnh báo rủi ro giao dịch."
    )
    st.warning("SonFED KHÔNG đảm bảo lợi nhuận. Đây là công cụ hỗ trợ phân tích, không phải máy in tiền tự động.")
    st.info("Khuyến nghị: Người mới nên dùng Hướng dẫn sử dụng → Bán tự động → Tự động. Không nên bật Auto Trade ngay từ đầu.")

    st.subheader("SonFED là gì?")
    cards = st.columns(4)
    card_data = [
        ("📈 Radar thị trường", "SonFED không chỉ nhìn giá vàng. Hệ thống còn theo dõi DXY, lợi suất Mỹ, Nasdaq, dầu, VIX, volatility và momentum để hiểu bối cảnh thị trường."),
        ("🤖 AI phân tích", "AI đọc dữ liệu, phân tích xu hướng, nhận diện regime, đánh giá volatility, đưa ra BUY / SELL / WAIT và giải thích lý do."),
        ("🛡️ Quản lý rủi ro", "SonFED có thể chặn giao dịch khi spread cao, tránh trade gần tin mạnh, tránh volatility nguy hiểm và lọc tín hiệu yếu."),
        ("📊 Kết nối SonEXEC", "SonFED phân tích. SonEXEC thực thi lệnh, trailing stop, break-even, partial close và quản lý risk chi tiết."),
    ]
    for col, (title, body) in zip(cards, card_data):
        with col:
            with st.container(border=True):
                st.markdown(f"**{title}**")
                st.write(body)

    render_dashboard_explanation()
    render_trading_mode_help()
    render_new_user_timeline()
    render_ai_decision_help()
    render_risk_warning_help()
    render_operating_system_help()

    st.subheader("Lộ trình khuyến nghị cho người mới")
    roadmap = [
        "Giai đoạn 1: Chỉ xem phân tích.",
        "Giai đoạn 2: Dùng bán tự động.",
        "Giai đoạn 3: Theo dõi signal.json.",
        "Giai đoạn 4: Cho SonEXEC đọc tín hiệu nhưng chưa auto trade.",
        "Giai đoạn 5: Bật auto trade với lot rất nhỏ.",
        "Giai đoạn 6: Tối ưu risk và execution.",
    ]
    for item in roadmap:
        st.success(item)

    render_faq_section()

    st.subheader("Bắt đầu")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Tôi đã hiểu - Chuyển sang Bán tự động", use_container_width=True):
            config.setdefault("trade", {})["mode"] = "Bán tự động"
            save_config(config)
            st.success("Đã chuyển sang chế độ Bán tự động.")
            st.rerun()
    with c2:
        if st.button("Tiếp tục ở chế độ hướng dẫn", use_container_width=True):
            st.info("Bạn vẫn đang ở chế độ Hướng dẫn sử dụng. Có thể xem dữ liệu thị trường ở các phần bên dưới.")


def render_status_strip(config: dict, refresh_info: dict, signal: dict) -> None:
    mode = normalize_trading_mode(config.get("trade", {}).get("mode", "Hướng dẫn sử dụng"))
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Chế độ hiện tại", mode)
    c2.metric("Auto refresh", "Bật" if refresh_info.get("enabled") else "Tắt")
    c3.metric("Telegram", "Bật" if config.get("telegram", {}).get("enabled") else "Tắt")
    c4.metric("Auto trade", "Bật" if config.get("trade", {}).get("allow_auto_trade") else "Tắt")
    c5.metric("Tín hiệu", signal.get("action", "WAIT"))
    c6.metric("Risk", signal.get("risk_level", "N/A"))
    st.caption(f"Lần cập nhật cuối: {fmt_time(refresh_info.get('state', {}).get('last_update_time'))}")
    st.caption(f"Lần cập nhật tiếp theo: {fmt_time(refresh_info.get('state', {}).get('next_update_time'))}")


def render_auto_refresh_controls(config: dict, mode_config: dict) -> dict:
    st.sidebar.divider()
    st.sidebar.subheader("Tự động cập nhật")
    state = load_state()
    settings = load_sonfed_settings()
    auto = config.setdefault("auto_refresh", {})

    if "auto_refresh_enabled" not in st.session_state:
        st.session_state["auto_refresh_enabled"] = get_auto_refresh_enabled(default=bool(auto.get("enabled", False)))

    interval = int(settings.get("refresh_interval_minutes", auto.get("interval_minutes", 5)))
    if interval not in INTERVAL_OPTIONS:
        interval = 5

    selected_interval = st.sidebar.selectbox(
        "Chu kỳ cập nhật tiếp theo",
        INTERVAL_OPTIONS,
        index=INTERVAL_OPTIONS.index(interval),
        format_func=lambda value: f"{value} phút",
        help="SonFED sẽ tự tải lại dữ liệu và phân tích lại sau mỗi chu kỳ này khi Tự động cập nhật đang bật.",
    )
    if selected_interval != interval:
        settings["refresh_interval_minutes"] = int(selected_interval)
        save_sonfed_settings(settings)
        auto["interval_minutes"] = int(selected_interval)
        save_config(config)
        st.sidebar.success("Đã lưu chu kỳ cập nhật tiếp theo.")

    auto_refresh_enabled = st.sidebar.toggle(
        "Bật tự động cập nhật",
        value=bool(st.session_state["auto_refresh_enabled"]),
        key="auto_refresh_toggle",
        disabled=not mode_config["allow_auto_refresh"],
        help="Khi bật, SonFED tự cập nhật thị trường theo chu kỳ đã chọn. Trạng thái này được lưu lại sau khi app rerun hoặc reload trình duyệt.",
    )
    if auto_refresh_enabled != st.session_state["auto_refresh_enabled"]:
        st.session_state["auto_refresh_enabled"] = bool(auto_refresh_enabled)
        set_auto_refresh_enabled(bool(auto_refresh_enabled))
        settings["auto_refresh_enabled"] = bool(auto_refresh_enabled)
        auto["enabled"] = bool(auto_refresh_enabled)
        save_config(config)
        st.sidebar.success("Đã lưu trạng thái tự động cập nhật.")

    if not mode_config["allow_auto_refresh"]:
        st.sidebar.info("Tự động cập nhật chỉ chạy ở chế độ Tự động. Trạng thái ON/OFF vẫn được lưu và không bị reset.")

    auto["enabled"] = bool(st.session_state["auto_refresh_enabled"]) and mode_config["allow_auto_refresh"]
    auto["interval_minutes"] = int(selected_interval)
    settings["telegram_enabled"] = bool(config.get("telegram", {}).get("enabled", False))
    settings["auto_trade_enabled"] = bool(config.get("trade", {}).get("allow_auto_trade", False))
    settings["refresh_interval_minutes"] = int(selected_interval)
    save_sonfed_settings(settings)

    st.sidebar.caption(f"Tự động cập nhật: {'Bật' if st.session_state['auto_refresh_enabled'] else 'Tắt'}")
    st.sidebar.caption(f"Lần cập nhật cuối: {fmt_time(state.get('last_update_time'))}")
    st.sidebar.caption(f"Lần cập nhật tiếp theo: {fmt_time(state.get('next_update_time'))}")
    return config


def sidebar(config: dict) -> tuple[dict, str, str, bool]:
    st.sidebar.title("SonFED")
    timeframe_options = ["15m", "1h", "4h", "1d"]
    default_timeframe = config.get("app", {}).get("default_timeframe", "15m")
    if default_timeframe not in timeframe_options:
        default_timeframe = "15m"
    timeframe = st.sidebar.selectbox("Timeframe", timeframe_options, index=timeframe_options.index(default_timeframe))
    period_options = ["5d", "1mo", "3mo", "6mo", "1y", "2y"]
    default_period = config.get("app", {}).get("default_period", "5d")
    if default_period not in period_options:
        default_period = "5d"
    period = st.sidebar.selectbox("Period", period_options, index=period_options.index(default_period))
    refresh = st.sidebar.button("Refresh dữ liệu", use_container_width=True)
    if refresh:
        st.cache_data.clear()

    st.sidebar.divider()
    trade = config.setdefault("trade", {})
    current_mode = normalize_trading_mode(trade.get("mode", "Hướng dẫn sử dụng"))
    trade["mode"] = st.sidebar.selectbox("Chế độ giao dịch", TRADING_MODES, index=TRADING_MODES.index(current_mode))
    mode_config = get_trading_mode_config(trade["mode"])

    config = render_auto_refresh_controls(config, mode_config)

    st.sidebar.divider()
    config.setdefault("telegram", {})
    config["telegram"]["enabled"] = st.sidebar.toggle("Bật Telegram", value=bool(config["telegram"].get("enabled", False)))
    settings = load_sonfed_settings()
    settings["telegram_enabled"] = bool(config["telegram"].get("enabled", False))
    save_sonfed_settings(settings)

    with st.sidebar.expander("Cấu hình ticker"):
        for key, value in config["tickers"].items():
            config["tickers"][key] = st.text_input(key, value=value)
        if st.button("Lưu ticker", use_container_width=True):
            save_config(config)
            st.sidebar.success("Đã lưu cấu hình ticker.")

    render_ai_trade_policy(config, mode_config)
    settings = load_sonfed_settings()
    settings["auto_trade_enabled"] = bool(config.get("trade", {}).get("allow_auto_trade", False))
    save_sonfed_settings(settings)
    return config, timeframe, period, refresh


def main() -> None:
    config = load_config()
    config, timeframe, period, manual_refresh = sidebar(config)
    mode_config = get_trading_mode_config(config.get("trade", {}).get("mode", "Hướng dẫn sử dụng"))
    policy = get_ai_trade_policy(config)
    refresh_info = prepare_refresh(config, manual_refresh=manual_refresh)
    if refresh_info["enabled"]:
        components.html(
            f"""
            <script>
            setTimeout(function() {{
                window.parent.location.reload();
            }}, {int(refresh_info["interval_minutes"]) * 60 * 1000});
            </script>
            """,
            height=0,
        )
    if mode_config["allow_auto_refresh"] and refresh_info["due"]:
        st.cache_data.clear()

    shared_dir = resolve_shared_dir(config)

    bundle = fetch_market_bundle(config["tickers"], period, timeframe)
    raw_status = data_status(bundle)
    if raw_status["missing"] and "_last_good_bundle" in st.session_state:
        previous_bundle = st.session_state["_last_good_bundle"]
        for key in raw_status["missing"]:
            if key in previous_bundle and not previous_bundle[key].empty:
                bundle[key] = previous_bundle[key]
        st.warning("Không lấy được dữ liệu mới, đang dùng dữ liệu lần cập nhật trước.")
    if data_status(bundle)["ok"]:
        st.session_state["_last_good_bundle"] = bundle

    gold_df = add_indicators(bundle.get("GOLD", pd.DataFrame()))
    bundle["GOLD"] = gold_df
    gold_analysis = analyze_gold(gold_df)
    macro = pressure_index(bundle)
    bias = gold_bias(bundle)
    fred_payload = fetch_fred_latest()
    events_df = load_events(config.get("paths", {}).get("events", "events.csv"))
    erisk = event_risk(events_df)

    mtf_frames = {
        "M1": add_indicators(fetch_ohlcv(config["tickers"]["GOLD"], "1d", "1m")),
        "M5": add_indicators(fetch_ohlcv(config["tickers"]["GOLD"], "5d", "5m")),
        "M15": add_indicators(fetch_ohlcv(config["tickers"]["GOLD"], "5d", "15m")),
        "H1": add_indicators(fetch_ohlcv(config["tickers"]["GOLD"], "1mo", "1h")),
        "H4": add_indicators(fetch_ohlcv(config["tickers"]["GOLD"], "6mo", "4h")),
    }
    mtf = analyze_mtf(mtf_frames)

    shared = read_shared(shared_dir)
    trade_status = shared.get("trade_status", {})
    performance_status = shared.get("performance_status", {})
    trade_feedback = normalize_trade_status(trade_status)
    risk_fb = normalize_risk_status(shared.get("risk_status", {}))
    position = summarize_position(trade_status, {"pressure": macro["score"], "regime": gold_analysis.get("regime", "")})
    strategies = build_strategies(gold_analysis, macro, mtf, erisk, config)
    ai_decision = build_decision(gold_analysis, macro, mtf, strategies)
    signal = create_signal(strategies, gold_analysis, macro, mtf, config, erisk, trade_status)
    signal["ai_decision"] = ai_decision
    market_state = build_market_state(ai_decision, signal, gold_analysis, erisk, risk_fb, trade_feedback)
    signal, policy_result = apply_trade_policy(signal, policy, market_state)
    signal["ai_decision"] = ai_decision
    ai_state = build_ai_state(signal, gold_analysis, macro, mtf, ai_decision, policy_result)
    write_ai_state(ai_state, shared_dir)
    signal_history = append_signal_history(signal)
    # Chỉ ghi signal khi nội dung thực sự thay đổi (tránh spam file + DB mỗi lần re-render)
    _signal_key = signal_key(signal)
    previous_signal_key = refresh_info.get("state", {}).get("last_signal_key")
    signal_changed = previous_signal_key != _signal_key or st.session_state.get("_last_signal_key") != _signal_key
    if mode_config["write_signal"] and signal_changed:
        wrote_signal = write_signal_if_allowed(signal, policy, shared_dir)
        if wrote_signal and mode_config["log_signal"]:
            log_signal(signal)
        if wrote_signal:
            st.session_state["_last_signal_key"] = _signal_key

    # Tạo đề xuất điều chỉnh và ghi ra file cho SonEXEC
    adjustments_payload = build_position_adjustment_payload(
        trade_feedback,
        gold_df,
        signal,
        previous_signal_key.split("|", 1)[0] if previous_signal_key else None,
        gold_analysis.get("market_regime", {}),
        mtf,
        policy.to_dict(),
    )
    adjustments = adjustments_payload.get("adjustments", [])
    _adj_key = "|".join(f"{a.get('ticket')}:{a.get('action')}:{a.get('confidence')}" for a in adjustments)
    if policy.allow_auto_adjustment and mode_config["write_signal"] and st.session_state.get("_last_adj_key") != _adj_key:
        write_trade_adjustment(adjustments_payload, shared_dir)
        st.session_state["_last_adj_key"] = _adj_key

    base_snapshot = build_snapshot(bundle, macro, gold_analysis, mtf, signal, gold_analysis.get("summary", ""))
    changes = detect_changes(refresh_info.get("state", {}).get("last_snapshot"), base_snapshot, signal, macro, erisk, risk_fb)
    market_summary = build_market_summary(base_snapshot, macro, gold_analysis, mtf, bias, signal, changes)
    base_snapshot["summary"] = market_summary
    process_sonfed_telegram_monitoring(config, trade_status, performance_status, signal, risk_fb, changes)
    if refresh_info["due"]:
        should_send, telegram_key = should_send_telegram(changes, refresh_info.get("state", {}), signal)
        telegram_sent = False
        if should_send and config.get("telegram", {}).get("enabled", False):
            ok, _ = send_telegram_queued(
                market_summary,
                enabled=True,
                event_key=telegram_key,
                cooldown_seconds=int(config.get("telegram", {}).get("cooldown_seconds", 300)),
            )
            telegram_sent = bool(ok)
        refresh_info["state"] = finalize_refresh(
            refresh_info,
            base_snapshot,
            signal,
            changes,
            telegram_key=telegram_key,
            telegram_sent=telegram_sent,
        )

    st.title("SonFED - Radar vĩ mô và giao dịch XAU/USD")
    status = data_status(bundle)
    if status["missing"]:
        st.warning("Một số nguồn dữ liệu chưa tải được: " + ", ".join(status["missing"]))

    tabs = st.tabs([
        "Tổng quan",
        "Thống kê",
        "Phân tích kỹ thuật vàng",
        "Lịch tin quan trọng",
        "Chiến lược SonFED",
        "Tín hiệu giao dịch",
        "Cài đặt",
        "Nhật ký giao dịch",
    ])

    with tabs[0]:
        render_ai_trading_radar_overview(
            config,
            refresh_info,
            signal,
            ai_decision,
            gold_analysis,
            macro,
            bias,
            mtf,
            bundle,
            gold_df,
            events_df,
            erisk,
            policy_result,
            risk_fb,
            signal_history,
            fred_payload,
        )

    with tabs[1]:
        render_statistics_tab(performance_status, trade_status, signal, risk_fb, config)

    with tabs[2]:
        render_ai_gold_technical_tab(gold_df, gold_analysis, signal, mtf, signal_history)

    with tabs[3]:
        st.subheader("Lịch tin quan trọng")
        if erisk["blocked"]:
            st.error(erisk["message"])
        else:
            st.success(erisk["message"])
        st.dataframe(events_df.sort_values("time") if not events_df.empty else events_df, use_container_width=True)

    with tabs[4]:
        st.subheader("Chiến lược SonFED")
        if strategies:
            st.dataframe(pd.DataFrame(strategies), use_container_width=True)
        else:
            st.info("Chưa đủ dữ liệu tạo chiến lược.")

    with tabs[5]:
        st.subheader("Tín hiệu giao dịch")
        st.json(signal)
        st.write("File tín hiệu:", str(shared_dir / "signal.json"))

        st.subheader("Trạng thái từ SonEXEC")
        if trade_feedback["connected"]:
            st.success(trade_feedback["message"])
            acc = trade_feedback.get("account", {})
            if acc:
                c1, c2, c3 = st.columns(3)
                c1.metric("Balance", f"{safe_float(acc.get('balance')):,.2f}")
                c2.metric("Equity", f"{safe_float(acc.get('equity')):,.2f}")
                c3.metric("Drawdown", f"{safe_float(acc.get('drawdown_percent')):.2f}%")
            else:
                st.info("MT5 chưa kết nối hoặc chưa có dữ liệu tài khoản.")
        else:
            st.warning(trade_feedback["message"])
        if risk_fb["connected"]:
            if risk_fb["allow"]:
                st.success(f"Risk OK: {risk_fb['reason']}")
            else:
                st.error(f"Risk bị khóa: {risk_fb['reason']}")

        st.write(position["summary"])
        st.write(ai_trade_summary(trade_status, gold_analysis, macro))
        render_position_management_panel(trade_feedback, adjustments_payload, signal, gold_analysis)

        st.subheader("Đề xuất điều chỉnh lệnh đang mở")
        if adjustments:
            rows = position_table_rows(trade_feedback, adjustments)
            if rows:
                st.dataframe(pd.DataFrame(rows), use_container_width=True)
            for adj in adjustments:
                action = normalize_adjustment_action(adj.get("action", ""))
                if action not in {"HOLD_POSITION", "DISABLE_NEW_ENTRY"}:
                    with st.expander(
                        f"Lệnh #{adj.get('ticket')} → {vi_action(action)} "
                        f"(độ tin cậy AI {adj.get('ai_confidence', adj.get('confidence', 0))}%)"
                    ):
                        st.write(adj.get("reason", ""))
                        st.json(adj)
        else:
            st.info("Chưa có lệnh mở hoặc chưa nhận trạng thái từ SonEXEC.")

        if st.button("Gửi cảnh báo Telegram ngay", use_container_width=True):
            price = safe_float(gold_df["Close"].dropna().iloc[-1]) if not gold_df.empty else 0.0
            ok, msg = send_telegram(build_alert(price, gold_analysis, macro, signal, gold_analysis.get("levels", {})))
            st.success(msg) if ok else st.error(msg)

    with tabs[6]:
        st.subheader("Cài đặt")
        st.caption("Các thay đổi được lưu vào config.json.")
        col1, col2 = st.columns(2)
        with col1:
            st.write("Indicator")
            for key, value in config["features"]["indicators"].items():
                config["features"]["indicators"][key] = st.checkbox(key, value=bool(value))
        with col2:
            st.write("Chiến lược")
            for key, value in config["features"]["strategies"].items():
                config["features"]["strategies"][key] = st.checkbox(key, value=bool(value))
        config["telegram"]["cycle_minutes"] = st.number_input("Chu kỳ gửi Telegram (phút)", 1, 1440, int(config["telegram"].get("cycle_minutes", 30)))
        st.divider()
        st.write("Telegram giám sát bot")
        col1, col2, col3 = st.columns(3)
        config["telegram"]["report_enabled"] = col1.checkbox("Bật báo cáo Telegram", value=bool(config["telegram"].get("report_enabled", True)))
        config["telegram"]["order_alerts_enabled"] = col2.checkbox("Bật cảnh báo lệnh", value=bool(config["telegram"].get("order_alerts_enabled", True)))
        config["telegram"]["performance_report_enabled"] = col3.checkbox("Bật báo cáo hiệu suất", value=bool(config["telegram"].get("performance_report_enabled", True)))
        col1, col2, col3 = st.columns(3)
        config["telegram"]["drawdown_alerts_enabled"] = col1.checkbox("Bật báo cáo drawdown", value=bool(config["telegram"].get("drawdown_alerts_enabled", True)))
        config["telegram"]["ai_bias_alerts_enabled"] = col2.checkbox("Bật cảnh báo AI đổi bias", value=bool(config["telegram"].get("ai_bias_alerts_enabled", True)))
        interval_options = [15, 30, 60]
        current_interval = int(config["telegram"].get("report_interval_minutes", 15))
        config["telegram"]["report_interval_minutes"] = col3.selectbox(
            "Chu kỳ gửi báo cáo",
            interval_options,
            index=interval_options.index(current_interval) if current_interval in interval_options else 0,
        )
        col1, col2, col3, col4 = st.columns(4)
        config["telegram"]["drawdown_alert_percent"] = col1.number_input("Ngưỡng drawdown cảnh báo (%)", min_value=0.1, max_value=100.0, value=safe_float(config["telegram"].get("drawdown_alert_percent"), 5.0), step=0.1)
        config["telegram"]["profit_target_percent"] = col2.number_input("Profit target trên vốn (%)", min_value=0.0, max_value=500.0, value=safe_float(config["telegram"].get("profit_target_percent"), 10.0), step=0.5)
        config["telegram"]["cooldown_seconds"] = col3.number_input("Cooldown Telegram (giây)", min_value=0, max_value=86400, value=int(config["telegram"].get("cooldown_seconds", 300)), step=30)
        config["telegram"]["base_capital"] = col4.number_input("Vốn gốc tính % ($)", min_value=1.0, max_value=100000.0, value=safe_float(config["telegram"].get("base_capital"), 200.0), step=10.0)
        if st.button("Lưu toàn bộ cài đặt"):
            save_config(config)
            st.success("Đã lưu cài đặt.")
        if st.button("Test Telegram"):
            ok, msg = send_telegram("SonFED test bot: kết nối Telegram hoạt động.")
            st.success(msg) if ok else st.error(msg)

    with tabs[7]:
        st.subheader("Nhật ký giao dịch và tín hiệu")
        rows = recent_signals()
        if rows:
            table = pd.DataFrame(rows, columns=["Thời gian", "Mã giao dịch", "Tín hiệu", "Độ tin cậy AI", "Payload"])
            table["Tín hiệu"] = table["Tín hiệu"].map(vi_action)
            st.dataframe(table.drop(columns=["Payload"]), use_container_width=True)
            with st.expander("Payload tín hiệu gần nhất"):
                st.code(rows[0][4], language="json")
        else:
            st.info("Chưa có nhật ký.")
        st.subheader("Bot log")
        st.json(shared.get("bot_log", []))


if __name__ == "__main__":
    main()

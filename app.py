from __future__ import annotations

import json
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
from modules.data_fetcher import data_status, fetch_market_bundle, fetch_ohlcv
from modules.database import log_signal, recent_signals
from modules.events import event_risk, load_events
from modules.fred_client import fetch_fred_latest, fred_to_frame
from modules.gold_analyzer import analyze_gold
from modules.indicators import add_indicators
from modules.market_regime_engine import build_decision
from modules.macro_engine import gold_bias, pressure_index
from modules.mtf_engine import analyze_mtf
from modules.position_manager import ai_trade_summary, summarize_position
from modules.signal_engine import create_signal
from modules.smartmoney_engine import smartmoney_notes
from modules.strategy_engine import build_strategies
from modules.telegram_engine import build_alert, send_telegram
from modules.trade_bridge import read_shared
from modules.utils import ROOT, load_json, resolve_shared_dir, save_json
from modules.adjustment_engine import write_trade_adjustment
from modules.position_feedback import normalize_trade_status, position_table_rows
from modules.risk_feedback import normalize_risk_status

load_dotenv()

TRADING_MODES = ["Hướng dẫn sử dụng", "Bán tự động", "Tự động", "AI hỗ trợ"]
LEGACY_MODE_MAP = {
    "Manual": "Hướng dẫn sử dụng",
    "Semi Auto": "Bán tự động",
    "Auto": "Tự động",
    "AI Assisted": "AI hỗ trợ",
}

st.set_page_config(page_title="SonFED", page_icon="🟡", layout="wide")


def load_config() -> dict:
    return ensure_auto_refresh_config(load_json("config.json", {}))


def save_config(config: dict) -> None:
    save_json("config.json", config)


def metric_value(df: pd.DataFrame) -> str:
    if df.empty:
        return "N/A"
    return f"{df['Close'].dropna().iloc[-1]:.2f}"


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
    policy.default_lot = float(clean["default_lot"])
    policy.allow_buy = True
    policy.allow_sell = True
    policy.max_buy_orders = int(clean["max_buy_orders"])
    policy.max_sell_orders = int(clean["max_sell_orders"])
    policy.max_buy_volume = round(policy.default_lot * policy.max_buy_orders, 2)
    policy.max_sell_volume = round(policy.default_lot * policy.max_sell_orders, 2)
    policy.min_confidence = int(advanced["min_ai_confidence"])
    policy.min_rr = float(advanced["min_rr"])
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

        lot = float(st.session_state.get("default_lot", 0.03))
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
    buy_volume = sum(float(p.get("lot", p.get("volume", 0)) or 0) for p in positions if "BUY" in str(p.get("type", p.get("type_name", ""))).upper())
    sell_volume = sum(float(p.get("lot", p.get("volume", 0)) or 0) for p in positions if "SELL" in str(p.get("type", p.get("type_name", ""))).upper())
    floating_profit = sum(float(p.get("profit", 0) or 0) for p in positions)
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


def compact_reason(text: str, limit: int = 150) -> str:
    clean = " ".join(str(text or "").split())
    if len(clean) <= limit:
        return clean
    return clean[: limit - 3].rstrip() + "..."


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
    if spread is not None and max_spread is not None and float(spread) > float(max_spread):
        alerts.append(f"Spread bất thường: {float(spread):.0f} điểm.")

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
        c1.metric("Balance", f"{account.get('balance', 0):,.2f}")
        c2.metric("Equity", f"{account.get('equity', 0):,.2f}")
        c3.metric("Drawdown", f"{account.get('drawdown_percent', 0):.2f}%")

    if risk_fb.get("connected"):
        if risk_fb.get("allow"):
            st.success("Risk OK: " + compact_reason(risk_fb.get("reason", ""), 120))
        else:
            st.error("Risk khóa: " + compact_reason(risk_fb.get("reason", ""), 120))


def render_position_management_compact(trade_feedback: dict, adjustments_payload: dict, signal: dict, gold_analysis: dict) -> None:
    positions = trade_feedback.get("positions", [])
    buy_volume = sum(float(p.get("lot", p.get("volume", 0)) or 0) for p in positions if "BUY" in str(p.get("type", p.get("type_name", ""))).upper())
    sell_volume = sum(float(p.get("lot", p.get("volume", 0)) or 0) for p in positions if "SELL" in str(p.get("type", p.get("type_name", ""))).upper())
    floating_profit = sum(float(p.get("profit", 0) or 0) for p in positions)

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
    events_df = load_events(config.get("paths", {}).get("events", "events.csv"))
    erisk = event_risk(events_df)

    mtf_frames = {
        "M15": add_indicators(fetch_ohlcv(config["tickers"]["GOLD"], "5d", "15m")),
        "H1": add_indicators(fetch_ohlcv(config["tickers"]["GOLD"], "1mo", "1h")),
        "H4": add_indicators(fetch_ohlcv(config["tickers"]["GOLD"], "6mo", "4h")),
    }
    mtf = analyze_mtf(mtf_frames)

    shared = read_shared(shared_dir)
    trade_status = shared.get("trade_status", {})
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
    if refresh_info["due"]:
        should_send, telegram_key = should_send_telegram(changes, refresh_info.get("state", {}), signal)
        telegram_sent = False
        if should_send and config.get("telegram", {}).get("enabled", False):
            ok, _ = send_telegram(market_summary)
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
        "Phân tích kỹ thuật vàng",
        "Radar vĩ mô SonFED",
        "Lịch tin quan trọng",
        "Chiến lược SonFED",
        "Tín hiệu giao dịch",
        "Cài đặt",
        "Nhật ký giao dịch",
    ])

    with tabs[0]:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Giá vàng", metric_value(gold_df))
        c2.metric("DXY", metric_value(bundle.get("DXY", pd.DataFrame())))
        c3.metric("US10Y", metric_value(bundle.get("US10Y", pd.DataFrame())))
        c4.metric("Pressure Index", f"{macro['score']}/100")
        mode = normalize_trading_mode(config.get("trade", {}).get("mode", "Hướng dẫn sử dụng"))
        ui_args = (
            config,
            refresh_info,
            signal,
            ai_decision,
            gold_analysis,
            macro,
            mtf,
            changes,
            policy,
            policy_result,
            bundle,
            gold_df,
            bias,
            erisk,
            risk_fb,
            trade_feedback,
            adjustments_payload,
        )
        if mode == "Hướng dẫn sử dụng":
            render_guided_mode_ui(*ui_args)
        elif mode == "Bán tự động":
            render_semi_auto_mode_ui(*ui_args)
        elif mode == "Tự động":
            render_auto_mode_ui(*ui_args)
        elif mode == "AI hỗ trợ":
            render_ai_assistant_mode_ui(*ui_args)

    with tabs[1]:
        st.plotly_chart(make_gold_chart(gold_df), use_container_width=True, key="technical_gold_chart")
        st.subheader("Diễn giải kỹ thuật")
        for item in gold_analysis.get("items", []):
            st.write("- " + item)
        st.subheader("Smart Money")
        for note in smartmoney_notes(gold_df):
            st.write("- " + note)
        st.subheader("Đa khung thời gian")
        st.json(mtf["trends"])
        st.write(mtf["summary"])

    with tabs[2]:
        st.metric("SonFED Pressure Index", f"{macro['score']}/100")
        st.write(macro["interpretation"])
        st.write(bias)
        st.write("Chi tiết điểm:")
        for item in macro["details"]:
            st.write("- " + item)
        fred = fetch_fred_latest()
        st.subheader("FRED")
        st.write(fred["message"])
        frame = fred_to_frame(fred)
        if not frame.empty:
            st.dataframe(frame, use_container_width=True)

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
                c1.metric("Balance", f"{acc.get('balance', 0):,.2f}")
                c2.metric("Equity", f"{acc.get('equity', 0):,.2f}")
                c3.metric("Drawdown", f"{acc.get('drawdown_percent', 0):.2f}%")
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
            price = float(gold_df["Close"].dropna().iloc[-1]) if not gold_df.empty else 0.0
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

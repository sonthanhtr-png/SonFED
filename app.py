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

POLICY_PRESETS = {
    "Người mới": {"min_confidence": 75, "min_rr": 1.5, "filter_high_volatility": True, "allow_auto_execution": False},
    "An toàn": {"min_confidence": 80, "min_rr": 2.0, "filter_high_volatility": True, "allow_auto_execution": False},
    "Cân bằng": {"min_confidence": 70, "min_rr": 1.2, "filter_high_volatility": True},
    "Scalping": {"min_confidence": 55, "min_rr": 1.0, "filter_high_volatility": True},
    "Aggressive": {"min_confidence": 45, "min_rr": 1.0, "filter_high_volatility": False},
}


def vi_regime(value: str | None) -> str:
    raw = str(value or "Chưa rõ")
    return REGIME_LABELS.get(raw, raw)


def vi_action(value: str | None) -> str:
    raw = str(value or "WAIT").upper()
    return ACTION_LABELS.get(raw, raw)


def apply_policy_preset(policy: AITradePolicy, preset: str) -> AITradePolicy:
    for key, value in POLICY_PRESETS.get(preset, {}).items():
        setattr(policy, key, value)
    policy.max_buy_volume = round(policy.max_buy_orders * policy.default_lot, 2)
    policy.max_sell_volume = round(policy.max_sell_orders * policy.default_lot, 2)
    return policy


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


def render_ai_trade_policy(config: dict, mode_config: dict) -> AITradePolicy:
    st.sidebar.expander("Hệ thống hoạt động thế nào?").write(
        "SonFED là bộ não phân tích thị trường và tạo tín hiệu. SonEXEC là bộ máy thực thi, "
        "vào lệnh và quản lý lệnh đang mở. Người mới nên để Auto Trade tắt cho đến khi hiểu rõ rủi ro."
    )
    policy = get_ai_trade_policy(config)
    with st.sidebar.expander("Chính sách giao dịch AI"):
        preset = st.selectbox("Cấu hình nhanh", ["Tùy chỉnh", *POLICY_PRESETS.keys()])
        if preset != "Tùy chỉnh" and st.button("Áp dụng cấu hình này", use_container_width=True):
            policy = apply_policy_preset(policy, preset)
            st.success(f"Đã áp dụng preset {preset}.")
        policy.symbol = st.text_input("Mã giao dịch trên MT5", value=policy.symbol or "XAUUSD", help="Ví dụ XAUUSD là vàng giao ngay. SonFED sẽ gửi tín hiệu cho đúng mã này.")
        policy.allow_buy = st.toggle("Cho phép tín hiệu BUY", value=policy.allow_buy, help="Nếu tắt, AI có nghiêng mua cũng sẽ chuyển thành WAIT.")
        policy.allow_sell = st.toggle("Cho phép tín hiệu SELL", value=policy.allow_sell, help="Nếu tắt, AI có nghiêng bán cũng sẽ chuyển thành WAIT.")
        policy.max_buy_orders = st.number_input("Số lệnh BUY tối đa", 0, 20, int(policy.max_buy_orders), help="Giới hạn số lệnh mua để tránh mở quá nhiều lệnh cùng chiều.")
        policy.max_sell_orders = st.number_input("Số lệnh SELL tối đa", 0, 20, int(policy.max_sell_orders), help="Giới hạn số lệnh bán để tránh dồn quá nhiều rủi ro một phía.")
        policy.default_lot = st.number_input(
            "Khối lượng mặc định mỗi lệnh",
            0.01,
            10.0,
            float(policy.default_lot),
            step=0.01,
            help="Ví dụ 0.03 nghĩa là mỗi lệnh SonEXEC mở sẽ dùng 0.03 lot nếu Auto Trade được bật. Người mới nên dùng lot nhỏ.",
        )
        policy.max_buy_volume = st.number_input("Tổng khối lượng BUY tối đa", 0.0, 100.0, float(policy.max_buy_volume), step=0.01, help="Ví dụ mỗi lệnh 0.03 và tối đa 3 lệnh BUY thì tổng BUY có thể lên 0.09 lot.")
        policy.max_sell_volume = st.number_input("Tổng khối lượng SELL tối đa", 0.0, 100.0, float(policy.max_sell_volume), step=0.01, help="Ví dụ mỗi lệnh 0.03 và tối đa 3 lệnh SELL thì tổng SELL có thể lên 0.09 lot.")
        policy.min_confidence = st.slider(
            "Độ tin cậy AI tối thiểu",
            1,
            100,
            int(policy.min_confidence),
            format="%d%%",
            help="AI chỉ vào lệnh khi đủ tự tin. 70-80% phù hợp người mới; 50% nhiều lệnh hơn nhưng dễ nhiễu; 85% rất an toàn nhưng ít cơ hội.",
        )
        policy.min_rr = st.number_input(
            "Tỷ lệ lời/lỗ tối thiểu (RR)",
            0.1,
            10.0,
            float(policy.min_rr),
            step=0.1,
            help="RR là tỷ lệ lời/lỗ kỳ vọng. Nếu có thể lỗ 100 USD nhưng lời 200 USD thì RR = 2.0. RR thấp nghĩa là lợi nhuận không đáng so với rủi ro.",
        )
        policy.max_spread = st.number_input(
            "Spread tối đa cho phép",
            1,
            5000,
            int(policy.max_spread),
            help="Spread là chênh lệch giá mua và bán. Spread cao thường xảy ra gần tin tức, khi biến động mạnh hoặc thanh khoản thấp; AI sẽ tránh vào lệnh.",
        )
        policy.filter_high_volatility = st.toggle(
            "Tránh thị trường biến động quá mạnh",
            value=policy.filter_high_volatility,
            help="Khi bật, AI tránh giao dịch lúc nến quá lớn, Bollinger Bands mở rộng mạnh hoặc ATR tăng mạnh để giảm nguy cơ quét stop loss.",
        )
        policy.filter_important_news = st.toggle(
            "Tránh giao dịch gần tin tức mạnh",
            value=policy.filter_important_news,
            help="Khi bật, AI tránh giao dịch gần CPI, NFP, FOMC hoặc các tin có thể làm giá chạy rất mạnh.",
        )
        policy.allow_sonexec_read_signal = st.toggle(
            "Cho phép SonEXEC đọc tín hiệu",
            value=policy.allow_sonexec_read_signal and mode_config["write_signal"],
            disabled=not mode_config["write_signal"],
            help="Khi bật, SonFED sẽ gửi BUY/SELL/WAIT sang SonEXEC qua signal.json.",
        )
        policy.allow_auto_execution = st.toggle(
            "Tự động vào lệnh bằng SonEXEC",
            value=policy.allow_auto_execution and mode_config["allow_auto_trade_toggle"],
            disabled=not mode_config["allow_auto_trade_toggle"],
            help="Khi bật, SonEXEC có thể tự vào lệnh bằng tiền thật nếu tín hiệu và kiểm tra rủi ro đều đạt.",
        )
        if policy.allow_auto_execution:
            st.error("Chỉ bật tự động vào lệnh khi đã hiểu rõ hệ thống và rủi ro.")
        policy.allow_auto_adjustment = st.toggle(
            "Tự động quản lý lệnh đang mở",
            value=policy.allow_auto_adjustment and mode_config["write_signal"],
            disabled=not mode_config["write_signal"],
            help="Tính năng này không mở lệnh mới. SonEXEC có thể dời stop loss, khóa lợi nhuận, trailing stop hoặc chốt lời một phần cho lệnh đang mở.",
        )
        policy.position_management_strategy = st.selectbox(
            "Chiến lược quản lý lệnh",
            ["Bảo toàn vốn", "Dời SL về hòa vốn", "Bám xu hướng", "AI tự thích nghi"],
            index={"Bảo toàn vốn": 0, "Break-even": 1, "Bám xu hướng": 2, "AI thích nghi": 3}.get(policy.position_management_strategy, 3),
            help="SonFED chỉ chọn hướng quản lý cấp cao. SonEXEC sẽ xử lý chi tiết: dời SL theo giá, dời SL về hòa vốn, chốt lời một phần.",
        )
        display_to_engine = {"Dời SL về hòa vốn": "Break-even", "AI tự thích nghi": "AI thích nghi"}
        policy.position_management_strategy = display_to_engine.get(policy.position_management_strategy, policy.position_management_strategy)
        st.warning(f"Nếu mỗi lệnh {policy.default_lot:.2f} lot và cho phép tối đa {policy.max_sell_orders} lệnh SELL, tổng rủi ro SELL có thể lên đến {policy.max_sell_volume:.2f} lot.")
        with st.expander("Giải thích các cài đặt quan trọng"):
            st.write("Độ tin cậy AI tối thiểu: AI chỉ phát tín hiệu khi đủ tự tin. Người mới nên dùng 70-80%.")
            st.write("Tỷ lệ lời/lỗ tối thiểu: nếu RR thấp, lợi nhuận kỳ vọng không đáng so với rủi ro.")
            st.write("Spread tối đa: spread cao làm lệnh vừa vào đã bất lợi, thường xuất hiện gần tin hoặc khi market chạy mạnh.")
            st.write("Tránh biến động mạnh: giúp hạn chế vào lệnh lúc nến quá lớn, BB mở rộng hoặc ATR tăng mạnh.")
            st.write("Cho SonEXEC đọc tín hiệu: SonFED sẽ ghi signal.json để SonEXEC biết BUY/SELL/WAIT.")
            st.write("Tự động vào lệnh: SonEXEC có thể vào lệnh bằng tiền thật. Chỉ bật khi đã hiểu rủi ro.")
            st.write("Tự động quản lý lệnh: không mở lệnh mới, chỉ dời SL, khóa lợi nhuận, trailing stop hoặc chốt lời một phần.")
        st.caption("Khuyến nghị người mới: preset Người mới hoặc An toàn, Auto Trade tắt, bật bộ lọc biến động mạnh và tin tức.")
        if st.button("Lưu chính sách giao dịch AI", use_container_width=True):
            save_policy_to_config(config, policy)
            save_config(config)
            st.success("Đã lưu Chính sách giao dịch AI.")
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


def render_guide_mode(config: dict) -> None:
    render_user_guide_mode(config)


def render_semi_auto_mode(signal: dict, ai_decision: dict) -> None:
    st.warning("Chế độ bán tự động: tín hiệu chỉ là đề xuất, cần người dùng xác nhận thủ công.")
    cols = st.columns(6)
    cols[0].metric("Tín hiệu", signal.get("action", "WAIT"))
    cols[1].metric("Winrate", f"{ai_decision.get('winrate', signal.get('winrate', 0))}%")
    cols[2].metric("Độ tin cậy AI", f"{signal.get('confidence', 0)}%")
    cols[3].metric("Mức rủi ro", signal.get("risk_level", "N/A"))
    cols[4].metric("TP", ai_decision.get("tp") or "N/A")
    cols[5].metric("SL", ai_decision.get("sl") or "N/A")
    st.metric("RR", ai_decision.get("rr") if ai_decision.get("rr") is not None else "N/A")
    st.write(ai_decision.get("reason", signal.get("reason", "")))
    if st.button("Phân tích lại ngay", key="semi_auto_refresh"):
        st.cache_data.clear()
        st.rerun()


def render_auto_mode(config: dict, refresh_info: dict, signal: dict, changes: list[str]) -> None:
    st.success("Chế độ tự động: SonFED tự cập nhật theo chu kỳ, ghi signal.json và ghi nhật ký khi tín hiệu thay đổi.")
    st.write(f"Chu kỳ auto refresh: {refresh_info.get('interval_minutes', 5)} phút.")
    st.write("SonEXEC được phép đọc signal.json khi Auto Trade bật và risk check cho phép.")
    if config.get("trade", {}).get("allow_auto_trade"):
        st.info("Auto Trade đang bật. Signal chỉ có hiệu lực khi risk check đạt.")
    else:
        st.warning("Auto Trade đang tắt. SonFED vẫn phân tích và ghi signal, nhưng không cho phép vào lệnh tự động.")
    if changes:
        st.write("Thay đổi quan trọng:")
        for item in changes:
            st.write("- " + item)


def render_ai_assist_mode(gold_analysis: dict, macro: dict, mtf: dict, signal: dict, ai_decision: dict) -> None:
    st.info("Chế độ AI hỗ trợ: SonFED đưa ra phân tích và kịch bản, không tự động gửi lệnh nếu Auto Trade chưa bật.")
    regime = gold_analysis.get("market_regime", {})
    levels = gold_analysis.get("levels", {})
    bias = ai_decision.get("action", "WAIT")
    resistance = levels.get("resistance")
    support = levels.get("support")
    cancel_condition = "Chờ thêm dữ liệu."
    if bias == "SELL" and resistance:
        cancel_condition = f"Hủy kịch bản SELL nếu giá break xác nhận trên {resistance:.2f}."
    elif bias == "BUY" and support:
        cancel_condition = f"Hủy kịch bản BUY nếu giá breakdown dưới {support:.2f}."
    elif support and resistance:
        cancel_condition = f"Hủy mọi kịch bản sớm nếu giá phá vỡ vùng {support:.2f} - {resistance:.2f} mà không có retest rõ."

    st.write("Nhận định vĩ mô:")
    st.write(macro.get("interpretation", "Chưa có nhận định vĩ mô."))
    st.write("Nhận định kỹ thuật:")
    st.write(gold_analysis.get("ai_analysis", "Chưa có nhận định kỹ thuật."))
    st.write(f"Trạng thái thị trường: {vi_regime(regime.get('label', gold_analysis.get('regime', 'Chưa rõ')))}.")
    st.write(f"Định hướng AI: {vi_action(bias)}.")
    st.write("Lý do:")
    st.write(ai_decision.get("reason", signal.get("reason", "")))
    st.write("Điều kiện hủy kịch bản:")
    st.write(cancel_condition)
    st.write("Gợi ý quản trị rủi ro:")
    st.write("Giảm khối lượng khi volatility cao, luôn dùng SL, không auto trade nếu risk level là Cao.")


def render_trading_mode_panel(
    config: dict,
    refresh_info: dict,
    signal: dict,
    ai_decision: dict,
    gold_analysis: dict,
    macro: dict,
    mtf: dict,
    changes: list[str],
) -> None:
    st.subheader("Trạng thái vận hành")
    render_status_strip(config, refresh_info, signal)
    mode = normalize_trading_mode(config.get("trade", {}).get("mode", "Hướng dẫn sử dụng"))
    if mode == "Hướng dẫn sử dụng":
        render_guide_mode(config)
    elif mode == "Bán tự động":
        render_semi_auto_mode(signal, ai_decision)
    elif mode == "Tự động":
        render_auto_mode(config, refresh_info, signal, changes)
    elif mode == "AI hỗ trợ":
        render_ai_assist_mode(gold_analysis, macro, mtf, signal, ai_decision)


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
    default_timeframe = config.get("app", {}).get("default_timeframe", "1h")
    if default_timeframe not in timeframe_options:
        default_timeframe = "1h"
    timeframe = st.sidebar.selectbox("Timeframe", timeframe_options, index=timeframe_options.index(default_timeframe))
    period = st.sidebar.selectbox("Period", ["5d", "1mo", "3mo", "6mo", "1y", "2y"], index=3)
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
    trade["allow_auto_trade"] = st.sidebar.toggle(
        "Cho phép auto trade",
        value=bool(trade.get("allow_auto_trade", False)) and mode_config["allow_auto_trade_toggle"],
        disabled=not mode_config["allow_auto_trade_toggle"],
    )
    if not mode_config["allow_auto_trade_toggle"]:
        trade["allow_auto_trade"] = False
    settings = load_sonfed_settings()
    settings["telegram_enabled"] = bool(config["telegram"].get("enabled", False))
    settings["auto_trade_enabled"] = bool(trade.get("allow_auto_trade", False))
    save_sonfed_settings(settings)

    with st.sidebar.expander("Cấu hình ticker"):
        for key, value in config["tickers"].items():
            config["tickers"][key] = st.text_input(key, value=value)
        if st.button("Lưu ticker", use_container_width=True):
            save_config(config)
            st.sidebar.success("Đã lưu cấu hình ticker.")

    render_ai_trade_policy(config, mode_config)
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
        render_trading_mode_panel(config, refresh_info, signal, ai_decision, gold_analysis, macro, mtf, changes)
        st.plotly_chart(make_gold_chart(gold_df), use_container_width=True, key="overview_gold_chart")
        st.subheader("Kết luận nhanh")
        st.write(gold_analysis["summary"])
        st.subheader("AI Decision Box")
        d1, d2, d3, d4, d5 = st.columns(5)
        d1.metric("Quyết định AI", vi_action(signal.get("decision", signal.get("action", "WAIT"))))
        d2.metric("Độ tin cậy AI", f"{ai_decision['winrate']}%")
        d3.metric("Chốt lời dự kiến", ai_decision["tp"] if ai_decision["tp"] is not None else "N/A")
        d4.metric("Cắt lỗ dự kiến", ai_decision["sl"] if ai_decision["sl"] is not None else "N/A")
        d5.metric("Tỷ lệ lời/lỗ", ai_decision["rr"] if ai_decision["rr"] is not None else "N/A")
        render_policy_status(policy)
        render_policy_warning(policy_result)
        render_quick_decision_explanation(signal, ai_decision, gold_analysis, macro, mtf)
        render_confidence_guide()
        render_wait_explanation()
        st.write(ai_decision["reason"])
        st.subheader("Phân tích AI")
        st.write(gold_analysis.get("ai_analysis", "Chưa có phân tích AI."))
        st.write(bias)
        st.info(mtf["summary"])
        st.subheader("Cảnh báo risk")
        for alert in smart_alerts(gold_analysis, macro, mtf, erisk):
            st.warning(alert)
        if ai_decision.get("risk_level") == "Cao":
            st.error("Biến động/rủi ro đang cao. Tránh tự động vào lệnh với khối lượng lớn.")
        st.subheader("Auto refresh log")
        auto_summary = dashboard_summary(refresh_info)
        a1, a2, a3, a4 = st.columns(4)
        a1.metric("Trạng thái", auto_summary["enabled"])
        a2.metric("Chu kỳ", auto_summary["interval"])
        a3.metric("Tín hiệu hiện tại", auto_summary["current_action"])
        a4.metric("Tín hiệu trước đó", auto_summary["previous_action"])
        st.write(f"Lần cập nhật cuối: {auto_summary['last_update']}")
        st.write(f"Lần cập nhật tiếp theo: {auto_summary['next_update']}")
        if auto_summary["changes"]:
            st.write("Thay đổi chính so với lần trước:")
            for item in auto_summary["changes"]:
                st.write("- " + item)
        else:
            st.info("Chưa có thay đổi quan trọng.")
        st.caption("Kết luận chi tiết của lần auto refresh được lưu trong data/market_snapshots.json.")

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

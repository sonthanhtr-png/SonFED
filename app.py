from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

from modules.alerts import smart_alerts
from modules.data_fetcher import data_status, fetch_market_bundle, fetch_ohlcv
from modules.database import log_signal, recent_signals
from modules.events import event_risk, load_events
from modules.fred_client import fetch_fred_latest, fred_to_frame
from modules.gold_analyzer import analyze_gold
from modules.indicators import add_indicators
from modules.macro_engine import gold_bias, pressure_index
from modules.mtf_engine import analyze_mtf
from modules.position_manager import ai_trade_summary, summarize_position
from modules.signal_engine import create_signal, write_signal
from modules.smartmoney_engine import smartmoney_notes
from modules.strategy_engine import build_strategies
from modules.telegram_engine import build_alert, send_telegram
from modules.trade_bridge import read_shared
from modules.utils import ROOT, load_json, resolve_shared_dir, save_json
from modules.adjustment_engine import create_trade_adjustments, write_trade_adjustment
from modules.position_feedback import normalize_trade_status, position_table_rows
from modules.risk_feedback import normalize_risk_status

load_dotenv()

st.set_page_config(page_title="SonFED", page_icon="🟡", layout="wide")


def load_config() -> dict:
    return load_json("config.json", {})


def save_config(config: dict) -> None:
    save_json("config.json", config)


def metric_value(df: pd.DataFrame) -> str:
    if df.empty:
        return "N/A"
    return f"{df['Close'].dropna().iloc[-1]:.2f}"


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


def sidebar(config: dict) -> tuple[dict, str, str, bool]:
    st.sidebar.title("SonFED")
    timeframe = st.sidebar.selectbox("Timeframe", ["15m", "1h", "4h", "1d"], index=["15m", "1h", "4h", "1d"].index(config["app"].get("default_timeframe", "1h")))
    period = st.sidebar.selectbox("Period", ["5d", "1mo", "3mo", "6mo", "1y", "2y"], index=3)
    refresh = st.sidebar.button("Refresh dữ liệu", use_container_width=True)
    if refresh:
        st.cache_data.clear()

    st.sidebar.divider()
    config["telegram"]["enabled"] = st.sidebar.toggle("Bật Telegram", value=bool(config.get("telegram", {}).get("enabled", False)))
    config["trade"]["allow_auto_trade"] = st.sidebar.toggle("Cho phép auto trade", value=bool(config.get("trade", {}).get("allow_auto_trade", False)))
    config["trade"]["mode"] = st.sidebar.selectbox("Chế độ giao dịch", ["Manual", "Semi Auto", "Auto", "AI Assisted"], index=["Manual", "Semi Auto", "Auto", "AI Assisted"].index(config["trade"].get("mode", "Manual")))

    with st.sidebar.expander("Cấu hình ticker"):
        for key, value in config["tickers"].items():
            config["tickers"][key] = st.text_input(key, value=value)
        if st.button("Lưu ticker", use_container_width=True):
            save_config(config)
            st.sidebar.success("Đã lưu cấu hình ticker.")

    with st.sidebar.expander("Cấu hình bot trade"):
        config["trade"]["symbol"] = st.text_input("Symbol MT5", value=config["trade"].get("symbol", "XAUUSD"))
        config["trade"]["min_confidence"] = st.number_input("Độ tin cậy tối thiểu", 1, 100, int(config["trade"].get("min_confidence", 70)))
        config["trade"]["max_spread_points"] = st.number_input("Spread tối đa", 1, 500, int(config["trade"].get("max_spread_points", 35)))
        if st.button("Lưu cấu hình bot", use_container_width=True):
            save_config(config)
            st.sidebar.success("Đã lưu cấu hình bot.")
    return config, timeframe, period, refresh


def main() -> None:
    config = load_config()
    config, timeframe, period, _ = sidebar(config)
    shared_dir = resolve_shared_dir(config)

    bundle = fetch_market_bundle(config["tickers"], period, timeframe)
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
    signal = create_signal(strategies, gold_analysis, macro, mtf, config, erisk, trade_status)
    # Chỉ ghi signal khi nội dung thực sự thay đổi (tránh spam file + DB mỗi lần re-render)
    _signal_key = f"{signal['action']}|{signal.get('strategy')}|{signal.get('confidence')}|{signal.get('entry_zone')}"
    if st.session_state.get("_last_signal_key") != _signal_key:
        write_signal(signal, shared_dir)
        log_signal(signal)
        st.session_state["_last_signal_key"] = _signal_key

    # Tạo đề xuất điều chỉnh và ghi ra file cho SonEXEC
    adjustments_payload = create_trade_adjustments(trade_feedback, gold_df, macro, mtf, config, erisk)
    adjustments = adjustments_payload.get("adjustments", [])
    _adj_key = "|".join(f"{a.get('ticket')}:{a.get('action')}:{a.get('confidence')}" for a in adjustments)
    if st.session_state.get("_last_adj_key") != _adj_key:
        write_trade_adjustment(adjustments_payload, shared_dir)
        st.session_state["_last_adj_key"] = _adj_key

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
        st.plotly_chart(make_gold_chart(gold_df), use_container_width=True)
        st.subheader("Kết luận nhanh")
        st.write(gold_analysis["summary"])
        st.write(bias)
        st.info(mtf["summary"])
        for alert in smart_alerts(gold_analysis, macro, mtf, erisk):
            st.warning(alert)

    with tabs[1]:
        st.plotly_chart(make_gold_chart(gold_df), use_container_width=True)
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

        st.subheader("Đề xuất điều chỉnh lệnh đang mở")
        if adjustments:
            rows = position_table_rows(trade_feedback, adjustments)
            if rows:
                st.dataframe(pd.DataFrame(rows), use_container_width=True)
            for adj in adjustments:
                if adj.get("action") not in {"HOLD_POSITION", "DISABLE_NEW_ENTRY"}:
                    with st.expander(
                        f"Lệnh #{adj.get('ticket')} → {adj.get('action')} "
                        f"(confidence {adj.get('confidence')}%)"
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
            table = pd.DataFrame(rows, columns=["Thời gian", "Symbol", "Action", "Confidence", "Payload"])
            st.dataframe(table.drop(columns=["Payload"]), use_container_width=True)
            with st.expander("Payload tín hiệu gần nhất"):
                st.code(rows[0][4], language="json")
        else:
            st.info("Chưa có nhật ký.")
        st.subheader("Bot log")
        st.json(shared.get("bot_log", []))


if __name__ == "__main__":
    main()

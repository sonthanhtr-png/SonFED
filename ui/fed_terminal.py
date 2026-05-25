from __future__ import annotations

from datetime import datetime
from typing import Any

from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPlainTextEdit,
    QScrollArea,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from services.fed_worker import SonFEDWorker
from ui.components import (
    AiDecisionBoxWidget,
    HeaderWidget,
    RadarFedWidget,
    RadarForexWidget,
    SettingsWidget,
    SidebarWidget,
    SignalTimelineWidget,
    TechnicalChartWidget,
    safe_float,
)
from ui.theme import APP_STYLE, BORDER, BUY, MUTED, PANEL, SELL, TEXT, WAIT, bias_color


class DataTable(QTableWidget):
    def __init__(self, columns: list[str]) -> None:
        super().__init__(0, len(columns))
        self.columns = columns
        self.setHorizontalHeaderLabels(columns)
        self.verticalHeader().setVisible(False)
        self.setAlternatingRowColors(True)

    def fill(self, rows: list[dict[str, Any]]) -> None:
        self.setRowCount(len(rows))
        for row_idx, row in enumerate(rows):
            for col_idx, key in enumerate(self.columns):
                value = row.get(key, "")
                item = QTableWidgetItem(str(value))
                self.setItem(row_idx, col_idx, item)
        self.resizeColumnsToContents()
        self.horizontalHeader().setStretchLastSection(True)


def scrollable(widget: QWidget) -> QScrollArea:
    area = QScrollArea()
    area.setWidgetResizable(True)
    area.setFrameShape(QFrame.NoFrame)
    area.setWidget(widget)
    return area


def panel(title: str, child: QWidget) -> QFrame:
    frame = QFrame()
    frame.setObjectName("Panel")
    layout = QVBoxLayout(frame)
    layout.setContentsMargins(12, 10, 12, 12)
    heading = QLabel(title)
    heading.setObjectName("SectionTitle")
    layout.addWidget(heading)
    layout.addWidget(child)
    return frame


class SonFEDTerminal(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("SonFED Desktop AI Radar")
        self.resize(1380, 860)
        self.worker: SonFEDWorker | None = None
        self.build_ui()
        self.start_worker()

    def build_ui(self) -> None:
        root = QWidget()
        root.setStyleSheet(APP_STYLE)
        self.setCentralWidget(root)
        shell = QHBoxLayout(root)
        shell.setContentsMargins(0, 0, 0, 0)
        shell.setSpacing(0)

        self.sidebar = SidebarWidget()
        self.sidebar.start_btn.clicked.connect(self.start_worker)
        self.sidebar.stop_btn.clicked.connect(self.stop_worker)
        self.sidebar.refresh_btn.clicked.connect(self.restart_worker)
        shell.addWidget(self.sidebar)

        main = QWidget()
        main_layout = QVBoxLayout(main)
        main_layout.setContentsMargins(14, 12, 14, 12)
        main_layout.setSpacing(10)
        self.header = HeaderWidget()
        main_layout.addWidget(self.header)

        self.tabs = QTabWidget()
        self.build_tabs()
        main_layout.addWidget(self.tabs, 1)
        shell.addWidget(main, 1)

    def build_tabs(self) -> None:
        self.decision = AiDecisionBoxWidget()
        self.radar_forex = RadarForexWidget()
        self.radar_fed = RadarFedWidget()
        self.timeline = SignalTimelineWidget()
        self.technical = TechnicalChartWidget()
        self.settings = SettingsWidget()

        overview = QWidget()
        overview_layout = QVBoxLayout(overview)
        overview_layout.setSpacing(10)
        overview_layout.addWidget(self.decision)
        overview_layout.addWidget(self.radar_forex)
        overview_layout.addWidget(self.radar_fed)
        overview_layout.addWidget(self.timeline)
        self.tabs.addTab(scrollable(overview), "Tổng quan")

        self.stats_table = DataTable(["Metric", "Value"])
        self.tabs.addTab(panel("Thống kê", self.stats_table), "Thống kê")

        technical_tab = QWidget()
        tech_layout = QVBoxLayout(technical_tab)
        tech_layout.setSpacing(10)
        tech_layout.addWidget(self.technical)
        self.mtf_table = DataTable(["Khung", "Trạng thái"])
        tech_layout.addWidget(panel("Heatmap đa khung", self.mtf_table))
        self.tech_timeline = SignalTimelineWidget()
        tech_layout.addWidget(self.tech_timeline)
        self.tabs.addTab(scrollable(technical_tab), "Phân tích kỹ thuật vàng")

        self.events_table = DataTable(["time", "currency", "event", "impact", "actual", "forecast"])
        self.tabs.addTab(panel("Lịch tin quan trọng", self.events_table), "Lịch tin quan trọng")

        self.strategy_table = DataTable(["strategy", "action", "probability", "risk", "alert"])
        self.tabs.addTab(panel("Chiến lược SonFED", self.strategy_table), "Chiến lược SonFED")

        signal_tab = QWidget()
        signal_layout = QVBoxLayout(signal_tab)
        self.signal_table = DataTable(["Field", "Value"])
        self.state_table = DataTable(["Field", "Value"])
        split = QSplitter()
        split.addWidget(panel("Tín hiệu giao dịch", self.signal_table))
        split.addWidget(panel("AI State / Shared Bridge", self.state_table))
        signal_layout.addWidget(split)
        self.tabs.addTab(signal_tab, "Tín hiệu giao dịch")

        self.tabs.addTab(self.settings, "Cài đặt")

        journal_tab = QWidget()
        journal_layout = QVBoxLayout(journal_tab)
        self.log_box = QPlainTextEdit()
        self.log_box.setReadOnly(True)
        journal_layout.addWidget(self.log_box)
        self.tabs.addTab(journal_tab, "Nhật ký giao dịch")

    def start_worker(self) -> None:
        if self.worker and self.worker.isRunning():
            return
        self.worker = SonFEDWorker(interval_seconds=15)
        self.worker.snapshot.connect(self.update_snapshot)
        self.worker.log.connect(self.append_log)
        self.worker.start()
        self.sidebar.set_status("AI Radar đang chạy · auto refresh 15s")

    def stop_worker(self) -> None:
        if self.worker:
            self.worker.stop()
            self.sidebar.set_status("Đã dừng auto refresh. Shared data giữ nguyên.")

    def restart_worker(self) -> None:
        self.stop_worker()
        self.start_worker()

    def append_log(self, text: str) -> None:
        self.log_box.appendPlainText(f"{datetime.now().strftime('%H:%M:%S')} | {text}")

    def update_snapshot(self, payload: dict[str, Any]) -> None:
        signal = payload.get("signal", {})
        ai_state = payload.get("ai_state", {})
        market = payload.get("market_state", {})
        macro = payload.get("macro", {})

        self.header.update_snapshot(payload)
        self.settings.update_data(payload)
        self.decision.update_data(signal, ai_state)
        self.radar_forex.update_data(payload.get("radar_forex", []))
        self.radar_fed.update_data(payload.get("radar_fed", []))
        self.timeline.update_data(payload.get("signal_history", []))
        self.tech_timeline.update_data(payload.get("signal_history", []))
        self.technical.update_data(payload.get("gold"), signal, ai_state)
        self.fill_statistics(signal, ai_state, market, macro)
        self.fill_mtf(payload.get("mtf", {}))
        self.fill_events(payload.get("events_table", []))
        self.fill_strategies(payload.get("strategies", []))
        self.fill_signal_tables(signal, ai_state, market)

    def fill_statistics(self, signal: dict[str, Any], ai_state: dict[str, Any], market: dict[str, Any], macro: dict[str, Any]) -> None:
        rows = [
            {"Metric": "Signal", "Value": signal.get("action", "WAIT")},
            {"Metric": "Confidence", "Value": f"{safe_float(signal.get('confidence')):.0f}%"},
            {"Metric": "Gold price", "Value": f"{safe_float(market.get('price')):.2f}"},
            {"Metric": "Pressure Index", "Value": macro.get("score", 50)},
            {"Metric": "Market Regime", "Value": ai_state.get("market_regime", "-")},
            {"Metric": "Volatility", "Value": ai_state.get("volatility", "-")},
            {"Metric": "Momentum", "Value": ai_state.get("momentum", "-")},
            {"Metric": "Execution Mode", "Value": ai_state.get("execution_mode", "-")},
            {"Metric": "Recommended Trailing", "Value": ai_state.get("recommended_trailing", "-")},
            {"Metric": "Shared Update", "Value": market.get("timestamp", "-")},
        ]
        self.stats_table.fill(rows)

    def fill_mtf(self, mtf: dict[str, Any]) -> None:
        regimes = mtf.get("regimes", {}) if isinstance(mtf, dict) else {}
        rows: list[dict[str, Any]] = []
        if isinstance(regimes, dict):
            for key in ("M1", "M5", "M15", "H1", "H4"):
                item = regimes.get(key, {})
                if isinstance(item, dict):
                    rows.append({"Khung": key, "Trạng thái": item.get("label") or item.get("bias") or "WAIT"})
        if not rows:
            rows = [{"Khung": "Summary", "Trạng thái": mtf.get("summary", "Chưa có dữ liệu") if isinstance(mtf, dict) else "Chưa có dữ liệu"}]
        self.mtf_table.fill(rows)

    def fill_events(self, rows: list[dict[str, Any]]) -> None:
        normalized = []
        for row in rows:
            normalized.append(
                {
                    "time": row.get("time", ""),
                    "currency": row.get("currency", ""),
                    "event": row.get("event", row.get("name", "")),
                    "impact": row.get("impact", ""),
                    "actual": row.get("actual", ""),
                    "forecast": row.get("forecast", row.get("expected", "")),
                }
            )
        self.events_table.fill(normalized)

    def fill_strategies(self, strategies: list[dict[str, Any]]) -> None:
        rows = []
        for item in strategies:
            rows.append(
                {
                    "strategy": item.get("strategy", ""),
                    "action": item.get("action", ""),
                    "probability": item.get("probability", ""),
                    "risk": item.get("risk", item.get("risk_level", "")),
                    "alert": item.get("alert", ""),
                }
            )
        self.strategy_table.fill(rows)

    def fill_signal_tables(self, signal: dict[str, Any], ai_state: dict[str, Any], market: dict[str, Any]) -> None:
        signal_rows = [{"Field": key, "Value": value} for key, value in signal.items() if key not in {"risk_check"}]
        state_rows = [{"Field": key, "Value": value} for key, value in {**market, **ai_state}.items()]
        self.signal_table.fill(signal_rows)
        self.state_table.fill(state_rows)

    def closeEvent(self, event) -> None:
        self.stop_worker()
        if self.worker:
            self.worker.wait(2500)
        super().closeEvent(event)

from __future__ import annotations

from datetime import datetime
from typing import Any

from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from shared.file_bus import DEFAULT_SHARED_DIR
from ui.theme import BG, BORDER, BUY, MUTED, PANEL, PANEL_SOFT, SELL, TEXT, WAIT, WARN, bias_color


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def fmt_money(value: Any) -> str:
    number = safe_float(value)
    return "-" if number == 0 else f"{number:.2f}"


class CardFrame(QFrame):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("Card")


class HeaderWidget(QWidget):
    def __init__(self) -> None:
        super().__init__()
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        title_box = QVBoxLayout()
        title = QLabel("SonFED - Radar vĩ mô và giao dịch XAU/USD")
        title.setObjectName("Title")
        subtitle = QLabel("AI Trading Radar · M15 scalp · macro pressure · shared bridge")
        subtitle.setStyleSheet(f"color: {MUTED};")
        title_box.addWidget(title)
        title_box.addWidget(subtitle)
        layout.addLayout(title_box, 1)

        self.refresh_label = QLabel("Auto refresh: chờ")
        self.shared_label = QLabel(f"Shared: {DEFAULT_SHARED_DIR}")
        self.status_label = QLabel("WAIT")
        for label in (self.refresh_label, self.shared_label, self.status_label):
            label.setStyleSheet(f"background: #0b1220; border: 1px solid {BORDER}; border-radius: 6px; padding: 6px 9px; font-weight: 800;")
            layout.addWidget(label)

    def update_snapshot(self, payload: dict[str, Any]) -> None:
        market = payload.get("market_state", {})
        signal = payload.get("signal", {})
        action = str(signal.get("action", "WAIT")).upper()
        color = bias_color(action)
        updated = market.get("timestamp") or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.refresh_label.setText(f"Auto refresh: {updated}")
        self.status_label.setText(f"AI: {action}")
        self.status_label.setStyleSheet(f"background: #0b1220; border: 1px solid {color}; border-radius: 6px; padding: 6px 9px; font-weight: 900; color: {color};")


class SidebarWidget(QFrame):
    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("Sidebar")
        self.setFixedWidth(270)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(12)

        brand = QLabel("SONFED")
        brand.setStyleSheet("font-size: 18px; font-weight: 900;")
        layout.addWidget(brand)
        layout.addWidget(self.caption("AI Radar Dashboard"))

        self.timeframe = self.combo("Timeframe", ["15m", "1h", "4h", "1d"])
        self.period = self.combo("Period", ["5d", "1mo", "3mo", "6mo"])
        self.mode = self.combo("Chế độ giao dịch", ["Hướng dẫn", "Bán tự động", "Tự động", "AI hỗ trợ"])
        layout.addWidget(self.timeframe[0])
        layout.addWidget(self.period[0])
        layout.addWidget(self.mode[0])

        self.auto_refresh = QCheckBox("Auto refresh")
        self.auto_refresh.setChecked(True)
        self.telegram = QCheckBox("Telegram report")
        self.ai_policy = QCheckBox("Chính sách AI")
        self.ai_policy.setChecked(True)
        layout.addWidget(self.auto_refresh)
        layout.addWidget(self.telegram)
        layout.addWidget(self.ai_policy)

        self.start_btn = QPushButton("Start AI Radar")
        self.stop_btn = QPushButton("Stop")
        self.refresh_btn = QPushButton("Refresh now")
        layout.addWidget(self.start_btn)
        layout.addWidget(self.stop_btn)
        layout.addWidget(self.refresh_btn)

        self.status = QLabel("Chưa chạy")
        self.status.setWordWrap(True)
        self.status.setStyleSheet(f"color: {MUTED}; background: #08111f; border: 1px solid {BORDER}; border-radius: 6px; padding: 8px;")
        layout.addWidget(self.status)
        layout.addStretch()

        shared = QLabel(f"Shared folder\n{DEFAULT_SHARED_DIR}")
        shared.setWordWrap(True)
        shared.setStyleSheet(f"color: {MUTED}; font-size: 11px;")
        layout.addWidget(shared)

    def caption(self, text: str) -> QLabel:
        label = QLabel(text)
        label.setObjectName("Caption")
        return label

    def combo(self, caption: str, values: list[str]) -> tuple[QWidget, QComboBox]:
        wrapper = QWidget()
        box = QVBoxLayout(wrapper)
        box.setContentsMargins(0, 0, 0, 0)
        box.setSpacing(4)
        label = self.caption(caption)
        combo = QComboBox()
        combo.addItems(values)
        box.addWidget(label)
        box.addWidget(combo)
        return wrapper, combo

    def set_status(self, text: str) -> None:
        self.status.setText(text)


class AiDecisionBoxWidget(CardFrame):
    def __init__(self) -> None:
        super().__init__()
        layout = QGridLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(10)
        title = QLabel("AI Decision Box")
        title.setObjectName("SectionTitle")
        layout.addWidget(title, 0, 0, 1, 4)

        self.action = QLabel("WAIT")
        self.action.setStyleSheet(f"font-size: 42px; font-weight: 900; color: {WAIT};")
        layout.addWidget(self.action, 1, 0, 2, 1)

        self.items: dict[str, QLabel] = {}
        for idx, key in enumerate(["Confidence", "RR", "TP", "SL", "Risk", "Mode"]):
            label = QLabel(f"{key}\n-")
            label.setStyleSheet(f"background: #0b1220; border: 1px solid {BORDER}; border-radius: 6px; padding: 8px; font-weight: 800;")
            self.items[key] = label
            layout.addWidget(label, 1 + idx // 3, 1 + idx % 3)

    def update_data(self, signal: dict[str, Any], ai_state: dict[str, Any]) -> None:
        action = str(signal.get("action", "WAIT")).upper()
        color = bias_color(action)
        self.action.setText(action)
        self.action.setStyleSheet(f"font-size: 42px; font-weight: 900; color: {color};")
        tp = signal.get("take_profit")
        sl = signal.get("stop_loss")
        entry = safe_float(signal.get("entry"))
        rr = "-"
        if entry and tp and sl:
            risk = abs(entry - safe_float(sl))
            reward = abs(safe_float(tp) - entry)
            rr = f"{reward / risk:.2f}" if risk else "-"
        values = {
            "Confidence": f"{safe_float(signal.get('confidence')):.0f}%",
            "RR": rr,
            "TP": fmt_money(tp),
            "SL": fmt_money(sl),
            "Risk": str(signal.get("risk_level", ai_state.get("risk_level", "-"))),
            "Mode": str(ai_state.get("execution_mode", signal.get("mode", "-"))),
        }
        for key, value in values.items():
            self.items[key].setText(f"{key}\n{value}")


class RadarCard(CardFrame):
    def __init__(self, title: str) -> None:
        super().__init__()
        self.title = QLabel(title)
        self.value = QLabel("-")
        self.bias = QLabel("WAIT")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        top = QHBoxLayout()
        self.title.setStyleSheet("font-weight: 900;")
        help_label = QLabel("?")
        help_label.setAlignment(Qt.AlignCenter)
        help_label.setStyleSheet(f"background: #0b1220; border: 1px solid {BORDER}; border-radius: 9px; min-width: 18px; max-width: 18px;")
        top.addWidget(self.title)
        top.addStretch()
        top.addWidget(help_label)
        self.value.setStyleSheet("font-size: 17px; font-weight: 900;")
        self.bias.setStyleSheet(f"font-weight: 900; color: {WAIT};")
        layout.addLayout(top)
        layout.addWidget(self.value)
        layout.addWidget(self.bias)

    def update_card(self, row: dict[str, Any], value_text: str | None = None) -> None:
        bias = str(row.get("bias", "WAIT")).upper()
        color = bias_color(bias)
        self.value.setText(value_text if value_text is not None else f"{safe_float(row.get('value')):.2f}  {safe_float(row.get('change_pct')):+.2f}%")
        self.bias.setText(bias)
        self.bias.setStyleSheet(f"font-weight: 900; color: {color};")
        self.setToolTip(str(row.get("explanation", "")))
        self.setStyleSheet(f"QFrame#Card {{ background: {PANEL}; border: 1px solid {BORDER}; border-left: 4px solid {color}; border-radius: 8px; }}")


class RadarForexWidget(CardFrame):
    def __init__(self) -> None:
        super().__init__()
        self.cards: dict[str, RadarCard] = {}
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 12)
        title = QLabel("Radar Forex")
        title.setObjectName("SectionTitle")
        layout.addWidget(title)
        grid = QGridLayout()
        grid.setSpacing(8)
        for index, name in enumerate(["Gold", "DXY", "US10Y", "VIX", "Oil", "Nasdaq"]):
            card = RadarCard(name)
            self.cards[name] = card
            grid.addWidget(card, index // 3, index % 3)
        layout.addLayout(grid)

    def update_data(self, rows: list[dict[str, Any]]) -> None:
        for row in rows:
            card = self.cards.get(str(row.get("name")))
            if card:
                card.update_card(row)


class RadarFedWidget(CardFrame):
    def __init__(self) -> None:
        super().__init__()
        self.cards: dict[str, RadarCard] = {}
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 12)
        title = QLabel("Radar FED / Vĩ mô")
        title.setObjectName("SectionTitle")
        layout.addWidget(title)
        grid = QGridLayout()
        grid.setSpacing(8)
        names = ["CPI", "Core CPI", "PCE", "Nonfarm", "FED Rate", "Powell Speech", "GDP", "Unemployment"]
        for index, name in enumerate(names):
            card = RadarCard(name)
            self.cards[name] = card
            grid.addWidget(card, index // 4, index % 4)
        layout.addLayout(grid)

    def update_data(self, rows: list[dict[str, Any]]) -> None:
        for row in rows:
            card = self.cards.get(str(row.get("name")))
            if card:
                card.update_card(row, f"Exp: {row.get('expected', '-')} · Act: {row.get('actual', '-')}")


class CandleChart(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.rows: list[tuple[float, float, float, float]] = []
        self.setMinimumHeight(300)

    def set_rows(self, dataframe: Any) -> None:
        self.rows = []
        if dataframe is not None and not getattr(dataframe, "empty", True):
            for _, row in dataframe.tail(80).iterrows():
                self.rows.append((safe_float(row.get("Open")), safe_float(row.get("High")), safe_float(row.get("Low")), safe_float(row.get("Close"))))
        self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#0b1220"))
        if not self.rows:
            painter.setPen(QColor(MUTED))
            painter.drawText(self.rect(), Qt.AlignCenter, "Đang chờ dữ liệu chart...")
            return
        highs = [item[1] for item in self.rows]
        lows = [item[2] for item in self.rows]
        high = max(highs)
        low = min(lows)
        span = max(high - low, 0.01)
        width = max(self.width() - 34, 1)
        height = max(self.height() - 28, 1)
        step = width / max(len(self.rows), 1)
        candle_w = max(4, min(10, step * 0.58))
        painter.setPen(QPen(QColor("#1f2937"), 1))
        for i in range(5):
            y = 14 + height * i / 4
            painter.drawLine(16, int(y), self.width() - 18, int(y))

        def y_of(price: float) -> float:
            return 14 + (high - price) / span * height

        for index, (open_, high_, low_, close_) in enumerate(self.rows):
            x = 17 + index * step + step / 2
            color = QColor(BUY if close_ >= open_ else SELL)
            painter.setPen(QPen(color, 1))
            painter.drawLine(QPointF(x, y_of(high_)), QPointF(x, y_of(low_)))
            top = min(y_of(open_), y_of(close_))
            bottom = max(y_of(open_), y_of(close_))
            painter.fillRect(int(x - candle_w / 2), int(top), int(candle_w), max(2, int(bottom - top)), color)


class TechnicalChartWidget(CardFrame):
    def __init__(self) -> None:
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 12)
        top = QHBoxLayout()
        title = QLabel("Phân tích kỹ thuật vàng")
        title.setObjectName("SectionTitle")
        self.status = QLabel("WAIT")
        self.status.setStyleSheet(f"font-weight: 900; color: {WAIT};")
        top.addWidget(title)
        top.addStretch()
        top.addWidget(self.status)
        layout.addLayout(top)
        self.chart = CandleChart()
        layout.addWidget(self.chart, 1)
        self.buy_bar = self.bar("BUY strength")
        self.sell_bar = self.bar("SELL strength")
        self.momentum_bar = self.bar("Momentum")
        layout.addWidget(self.buy_bar[0])
        layout.addWidget(self.sell_bar[0])
        layout.addWidget(self.momentum_bar[0])

    def bar(self, label: str) -> tuple[QWidget, QProgressBar]:
        wrapper = QWidget()
        box = QHBoxLayout(wrapper)
        box.setContentsMargins(0, 0, 0, 0)
        caption = QLabel(label)
        caption.setFixedWidth(110)
        progress = QProgressBar()
        progress.setRange(0, 100)
        progress.setValue(0)
        box.addWidget(caption)
        box.addWidget(progress, 1)
        return wrapper, progress

    def update_data(self, dataframe: Any, signal: dict[str, Any], ai_state: dict[str, Any]) -> None:
        self.chart.set_rows(dataframe)
        action = str(signal.get("action", "WAIT")).upper()
        confidence = int(safe_float(signal.get("confidence")))
        buy = confidence if action == "BUY" else max(0, 100 - confidence if action == "SELL" else 50)
        sell = confidence if action == "SELL" else max(0, 100 - confidence if action == "BUY" else 50)
        momentum = int(safe_float(ai_state.get("momentum_score"), 0) * 25)
        momentum = max(0, min(100, momentum))
        self.buy_bar[1].setValue(max(0, min(100, buy)))
        self.sell_bar[1].setValue(max(0, min(100, sell)))
        self.momentum_bar[1].setValue(momentum)
        color = bias_color(action)
        self.status.setText(f"{action} · {confidence}% · {ai_state.get('market_regime', '-')}")
        self.status.setStyleSheet(f"font-weight: 900; color: {color};")


class SignalTimelineWidget(CardFrame):
    def __init__(self) -> None:
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 12)
        title = QLabel("Trade Signal Timeline")
        title.setObjectName("SectionTitle")
        layout.addWidget(title)
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["Thời gian", "Tín hiệu", "Confidence", "Lý do"])
        self.table.verticalHeader().setVisible(False)
        self.table.setAlternatingRowColors(True)
        layout.addWidget(self.table)

    def update_data(self, rows: list[dict[str, Any]], limit: int = 15) -> None:
        rows = list(rows or [])[-limit:][::-1]
        self.table.setRowCount(len(rows))
        for row_idx, row in enumerate(rows):
            values = [row.get("timestamp", ""), row.get("signal", "WAIT"), f"{safe_float(row.get('confidence')):.0f}%", row.get("reason", "")]
            for col_idx, value in enumerate(values):
                item = QTableWidgetItem(str(value))
                if col_idx == 1:
                    item.setForeground(QColor(bias_color(str(value))))
                self.table.setItem(row_idx, col_idx, item)
        self.table.resizeColumnsToContents()
        self.table.horizontalHeader().setStretchLastSection(True)


class SettingsWidget(CardFrame):
    def __init__(self) -> None:
        super().__init__()
        layout = QGridLayout(self)
        layout.setContentsMargins(12, 10, 12, 12)
        self.labels: dict[str, QLabel] = {}
        title = QLabel("Cài đặt / Trạng thái")
        title.setObjectName("SectionTitle")
        layout.addWidget(title, 0, 0, 1, 2)
        for index, key in enumerate(["Shared folder", "Auto refresh", "Telegram", "Trade mode", "Policy", "Last update"]):
            caption = QLabel(key)
            caption.setObjectName("Caption")
            value = QLabel("-")
            value.setStyleSheet(f"background: #0b1220; border: 1px solid {BORDER}; border-radius: 6px; padding: 8px; font-weight: 800;")
            self.labels[key] = value
            layout.addWidget(caption, index + 1, 0)
            layout.addWidget(value, index + 1, 1)

    def update_data(self, payload: dict[str, Any]) -> None:
        market = payload.get("market_state", {})
        signal = payload.get("signal", {})
        self.labels["Shared folder"].setText(str(DEFAULT_SHARED_DIR))
        self.labels["Auto refresh"].setText("ON")
        self.labels["Telegram"].setText("Theo config")
        self.labels["Trade mode"].setText(str(signal.get("mode", "-")))
        self.labels["Policy"].setText("AI policy enabled")
        self.labels["Last update"].setText(str(market.get("timestamp", "")))

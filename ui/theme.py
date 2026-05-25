BUY = "#22c55e"
SELL = "#ef4444"
WAIT = "#9ca3af"
WARN = "#f59e0b"
BLUE = "#38bdf8"
BG = "#08111f"
SIDEBAR = "#0b1220"
PANEL = "#111827"
PANEL_SOFT = "#162033"
BORDER = "#263244"
TEXT = "#e5e7eb"
MUTED = "#9ca3af"


def bias_color(value: str) -> str:
    text = str(value or "WAIT").upper()
    if text == "BUY":
        return BUY
    if text == "SELL":
        return SELL
    if text in {"CAUTION", "WARN"}:
        return WARN
    return WAIT


APP_STYLE = f"""
QWidget {{
    background: {BG};
    color: {TEXT};
    font-family: Segoe UI;
    font-size: 12px;
}}
QFrame#Card, QFrame#Panel {{
    background: {PANEL};
    border: 1px solid {BORDER};
    border-radius: 8px;
}}
QFrame#Sidebar {{
    background: {SIDEBAR};
    border-right: 1px solid {BORDER};
}}
QLabel#Title {{
    font-size: 20px;
    font-weight: 900;
}}
QLabel#SectionTitle {{
    font-size: 14px;
    font-weight: 900;
}}
QLabel#Caption {{
    color: {MUTED};
    font-size: 11px;
}}
QPushButton {{
    background: #1f2937;
    border: 1px solid #374151;
    padding: 7px 10px;
    border-radius: 5px;
    font-weight: 800;
}}
QPushButton:hover {{
    background: #273449;
}}
QComboBox {{
    background: #0f172a;
    border: 1px solid {BORDER};
    padding: 5px;
    border-radius: 5px;
}}
QCheckBox {{
    spacing: 6px;
}}
QTabWidget::pane {{
    border: 1px solid {BORDER};
    border-radius: 8px;
    background: {BG};
}}
QTabBar::tab {{
    background: #111827;
    color: {MUTED};
    padding: 8px 12px;
    border: 1px solid {BORDER};
    border-bottom: 0;
    border-top-left-radius: 6px;
    border-top-right-radius: 6px;
    margin-right: 2px;
}}
QTabBar::tab:selected {{
    color: {TEXT};
    background: #1f2937;
}}
QTableWidget {{
    background: {PANEL};
    alternate-background-color: #0b1220;
    gridline-color: #1f2937;
    border: 1px solid {BORDER};
    border-radius: 6px;
}}
QHeaderView::section {{
    background: #1f2937;
    color: {TEXT};
    padding: 6px;
    border: 0;
    font-weight: 800;
}}
QPlainTextEdit {{
    background: #020617;
    color: #d1d5db;
    border: 1px solid {BORDER};
    border-radius: 6px;
    font-family: Consolas;
}}
QProgressBar {{
    background: #0b1220;
    border: 1px solid {BORDER};
    border-radius: 5px;
    height: 12px;
    text-align: center;
}}
QProgressBar::chunk {{
    background: {BLUE};
    border-radius: 5px;
}}
"""

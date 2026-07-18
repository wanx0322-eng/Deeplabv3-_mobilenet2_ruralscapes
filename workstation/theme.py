"""统一的深色主题样式表"""

ACCENT = "#4f8cff"
ACCENT_HOVER = "#6da2ff"
BG = "#1e2228"
BG_PANEL = "#262b33"
BG_INPUT = "#2e343e"
BORDER = "#3a4150"
TEXT = "#e8eaf0"
TEXT_DIM = "#9aa3b2"

STYLESHEET = f"""
* {{
    font-family: "Microsoft YaHei UI", "Segoe UI", sans-serif;
    font-size: 13px;
    color: {TEXT};
}}
QMainWindow, QDialog {{
    background-color: {BG};
}}
QWidget#sidebar {{
    background-color: #171a1f;
    border-right: 1px solid {BORDER};
}}
QPushButton#navButton {{
    background: transparent;
    border: none;
    border-radius: 6px;
    padding: 12px 16px;
    text-align: left;
    font-size: 14px;
    color: {TEXT_DIM};
}}
QPushButton#navButton:hover {{
    background-color: #232830;
    color: {TEXT};
}}
QPushButton#navButton:checked {{
    background-color: {ACCENT};
    color: white;
    font-weight: bold;
}}
QLabel#appTitle {{
    font-size: 16px;
    font-weight: bold;
    color: {TEXT};
    padding: 18px 16px 8px 16px;
}}
QLabel#appSubtitle {{
    font-size: 11px;
    color: {TEXT_DIM};
    padding: 0 16px 14px 16px;
}}
QLabel#pageTitle {{
    font-size: 18px;
    font-weight: bold;
    padding: 2px 0;
}}
QLabel#dim {{
    color: {TEXT_DIM};
}}
QGroupBox {{
    background-color: {BG_PANEL};
    border: 1px solid {BORDER};
    border-radius: 8px;
    margin-top: 10px;
    padding: 10px 8px 8px 8px;
    font-weight: bold;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 4px;
    color: {ACCENT};
}}
QPushButton {{
    background-color: {BG_INPUT};
    border: 1px solid {BORDER};
    border-radius: 6px;
    padding: 6px 14px;
}}
QPushButton:hover {{
    background-color: #38404c;
}}
QPushButton:pressed {{
    background-color: #2a3039;
}}
QPushButton:disabled {{
    color: #5d6673;
    background-color: #262b33;
}}
QPushButton#primary {{
    background-color: {ACCENT};
    border: none;
    color: white;
    font-weight: bold;
    padding: 8px 18px;
}}
QPushButton#primary:hover {{
    background-color: {ACCENT_HOVER};
}}
QPushButton#primary:disabled {{
    background-color: #33415c;
    color: #7a879c;
}}
QPushButton#danger {{
    background-color: #b33939;
    border: none;
    color: white;
}}
QPushButton#danger:hover {{
    background-color: #cc4444;
}}
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QTextEdit, QPlainTextEdit {{
    background-color: {BG_INPUT};
    border: 1px solid {BORDER};
    border-radius: 5px;
    padding: 4px 6px;
    selection-background-color: {ACCENT};
}}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{
    border-color: {ACCENT};
}}
QComboBox::drop-down {{
    border: none;
    width: 22px;
}}
QComboBox QAbstractItemView {{
    background-color: {BG_INPUT};
    border: 1px solid {BORDER};
    selection-background-color: {ACCENT};
}}
QCheckBox::indicator {{
    width: 16px;
    height: 16px;
    border: 1px solid {BORDER};
    border-radius: 4px;
    background: {BG_INPUT};
}}
QCheckBox::indicator:checked {{
    background-color: {ACCENT};
    border-color: {ACCENT};
}}
QTableWidget, QTableView, QListWidget, QTreeWidget {{
    background-color: {BG_PANEL};
    alternate-background-color: #2a303a;
    border: 1px solid {BORDER};
    border-radius: 6px;
    gridline-color: {BORDER};
}}
QHeaderView::section {{
    background-color: #2e343e;
    border: none;
    border-right: 1px solid {BORDER};
    border-bottom: 1px solid {BORDER};
    padding: 5px 8px;
    font-weight: bold;
}}
QTableWidget::item:selected, QListWidget::item:selected {{
    background-color: {ACCENT};
    color: white;
}}
QProgressBar {{
    background-color: {BG_INPUT};
    border: 1px solid {BORDER};
    border-radius: 6px;
    text-align: center;
    height: 18px;
}}
QProgressBar::chunk {{
    background-color: {ACCENT};
    border-radius: 5px;
}}
QPlainTextEdit#console {{
    background-color: #14161a;
    color: #c9d2e0;
    font-family: "Consolas", "Courier New", monospace;
    font-size: 12px;
}}
QScrollBar:vertical {{
    background: {BG};
    width: 10px;
    margin: 0;
}}
QScrollBar::handle:vertical {{
    background: #3d4552;
    border-radius: 5px;
    min-height: 30px;
}}
QScrollBar::handle:vertical:hover {{
    background: #4a5468;
}}
QScrollBar:horizontal {{
    background: {BG};
    height: 10px;
}}
QScrollBar::handle:horizontal {{
    background: #3d4552;
    border-radius: 5px;
    min-width: 30px;
}}
QScrollBar::add-line, QScrollBar::sub-line {{
    height: 0; width: 0;
}}
QStatusBar {{
    background: #171a1f;
    border-top: 1px solid {BORDER};
    color: {TEXT_DIM};
}}
QTabWidget::pane {{
    border: 1px solid {BORDER};
    border-radius: 6px;
    background: {BG_PANEL};
}}
QTabBar::tab {{
    background: transparent;
    padding: 7px 16px;
    color: {TEXT_DIM};
    border-bottom: 2px solid transparent;
}}
QTabBar::tab:selected {{
    color: {TEXT};
    border-bottom: 2px solid {ACCENT};
}}
QSplitter::handle {{
    background: {BORDER};
}}
QToolTip {{
    background-color: {BG_INPUT};
    color: {TEXT};
    border: 1px solid {BORDER};
}}
"""

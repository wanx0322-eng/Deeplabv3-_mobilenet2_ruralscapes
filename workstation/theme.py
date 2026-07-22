"""Widgets design tokens and the generated application stylesheet."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class ThemeTokens:
    """Semantic theme contract shared by Widgets, charts and icon rendering."""

    SURFACE_BASE: str = "#1e2228"
    SURFACE_SIDEBAR: str = "#171a1f"
    SURFACE_PANEL: str = "#262b33"
    SURFACE_INPUT: str = "#2e343e"
    SURFACE_CANVAS: str = "#14161a"
    SURFACE_HOVER: str = "#38404c"
    SURFACE_PRESSED: str = "#2a3039"
    SURFACE_ALT: str = "#2a303a"
    CONTENT_PRIMARY: str = "#e8eaf0"
    CONTENT_SECONDARY: str = "#9aa3b2"
    CONTENT_DISABLED: str = "#5d6673"
    CONTENT_INVERSE: str = "#ffffff"
    CONTENT_CONSOLE: str = "#c9d2e0"
    ACTION_PRIMARY: str = "#4f8cff"
    ACTION_PRIMARY_HOVER: str = "#6da2ff"
    ACTION_PRIMARY_DISABLED: str = "#33415c"
    ACTION_DANGER: str = "#b33939"
    ACTION_DANGER_HOVER: str = "#cc4444"
    FEEDBACK_SUCCESS: str = "#7bd88f"
    FEEDBACK_WARNING: str = "#ffd75f"
    FEEDBACK_ERROR: str = "#ff6b6b"
    FEEDBACK_ERROR_MUTED: str = "#e0685f"
    FEEDBACK_INFO: str = "#69b7ff"
    BORDER_DEFAULT: str = "#3a4150"
    BORDER_FOCUS: str = "#7aa7ff"
    CHART_TRAIN: str = "#4f8cff"
    CHART_VALIDATION: str = "#ff8c5f"
    CHART_MIOU: str = "#7bd88f"
    CHART_GRID: str = "#2e343e"
    CHART_AXES: str = "#1a1d23"
    FONT_UI: str = '"Microsoft YaHei UI", "Segoe UI", sans-serif'
    FONT_MONO: str = '"Consolas", "Courier New", monospace'
    FONT_SIZE_XS: int = 11
    FONT_SIZE_SM: int = 12
    FONT_SIZE_MD: int = 13
    FONT_SIZE_LG: int = 14
    FONT_SIZE_XL: int = 16
    FONT_SIZE_TITLE: int = 18
    FONT_SIZE_STAT: int = 20
    SPACE_0: int = 0
    SPACE_XXS: int = 2
    SPACE_XS: int = 4
    SPACE_SM: int = 6
    SPACE_MD: int = 8
    SPACE_LG: int = 10
    SPACE_XL: int = 12
    SPACE_XXL: int = 16
    RADIUS_SM: int = 4
    RADIUS_MD: int = 5
    RADIUS_LG: int = 6
    RADIUS_XL: int = 8
    BORDER_WIDTH: int = 1
    FOCUS_WIDTH: int = 2
    ICON_SM: int = 16
    ICON_MD: int = 20
    ICON_LG: int = 24
    CONTROL_HEIGHT: int = 28
    SCROLLBAR_SIZE: int = 10
    SIDEBAR_WIDTH: int = 190

    def registry(self) -> dict[str, Any]:
        return asdict(self)


DARK_TOKENS = ThemeTokens()

COLOR_TOKEN_ALIASES = {
    "CHART_TRAIN": "ACTION_PRIMARY",
    "CHART_MIOU": "FEEDBACK_SUCCESS",
    "CHART_GRID": "SURFACE_INPUT",
}



def build_stylesheet(tokens: ThemeTokens) -> str:
    """Build QSS exclusively from semantic tokens."""
    t = tokens
    return f"""
* {{
    font-family: {t.FONT_UI};
    font-size: {t.FONT_SIZE_MD}px;
    color: {t.CONTENT_PRIMARY};
}}
QMainWindow, QDialog, QWidget#pageRoot {{ background-color: {t.SURFACE_BASE}; }}
QWidget#sidebar {{
    background-color: {t.SURFACE_SIDEBAR};
    border-right: {t.BORDER_WIDTH}px solid {t.BORDER_DEFAULT};
}}
QPushButton#navButton {{
    background: transparent; border: none; border-radius: {t.RADIUS_LG}px;
    padding: {t.SPACE_XL}px {t.SPACE_XXL}px; text-align: left;
    font-size: {t.FONT_SIZE_LG}px; color: {t.CONTENT_SECONDARY};
}}
QPushButton#navButton:hover {{ background-color: {t.SURFACE_PANEL}; color: {t.CONTENT_PRIMARY}; }}
QPushButton#navButton:checked {{ background-color: {t.ACTION_PRIMARY}; color: {t.CONTENT_INVERSE}; font-weight: bold; }}
QPushButton#navButton:focus {{ border: {t.FOCUS_WIDTH}px solid {t.BORDER_FOCUS}; }}
QLabel#appTitle {{
    font-size: {t.FONT_SIZE_XL}px; font-weight: bold; color: {t.CONTENT_PRIMARY};
    padding: {t.FONT_SIZE_TITLE}px {t.SPACE_XXL}px {t.SPACE_MD}px {t.SPACE_XXL}px;
}}
QLabel#appSubtitle {{
    font-size: {t.FONT_SIZE_XS}px; color: {t.CONTENT_SECONDARY};
    padding: {t.SPACE_0}px {t.SPACE_XXL}px {t.FONT_SIZE_LG}px {t.SPACE_XXL}px;
}}
QLabel#pageTitle {{ font-size: {t.FONT_SIZE_TITLE}px; font-weight: bold; padding: {t.SPACE_XXS}px {t.SPACE_0}px; }}
QLabel#dim, QLabel#statCaption {{ color: {t.CONTENT_SECONDARY}; }}
QLabel#dirtyState {{ color: {t.FEEDBACK_WARNING}; font-weight: bold; }}
QLabel#imageViewer {{
QLabel#inlineMessage {{
    background-color: {t.SURFACE_INPUT}; border: {t.BORDER_WIDTH}px solid {t.FEEDBACK_INFO};
    border-radius: {t.RADIUS_MD}px; padding: {t.SPACE_SM}px {t.SPACE_MD}px;
}}
QLabel#inlineMessage[severity="error"] {{
    color: {t.FEEDBACK_ERROR}; border-color: {t.FEEDBACK_ERROR};
}}
QLabel#inlineMessage[severity="success"] {{
    color: {t.FEEDBACK_SUCCESS}; border-color: {t.FEEDBACK_SUCCESS};
}}

    background-color: {t.SURFACE_CANVAS}; border: {t.BORDER_WIDTH}px solid {t.BORDER_DEFAULT};
    border-radius: {t.RADIUS_LG}px; color: {t.CONTENT_DISABLED};
}}
QFrame#separator {{ color: {t.BORDER_DEFAULT}; }}
QFrame#statCard {{
    background: {t.SURFACE_PANEL}; border: {t.BORDER_WIDTH}px solid {t.BORDER_DEFAULT};
    border-radius: {t.RADIUS_XL}px;
}}
QLabel#statValue {{ font-size: {t.FONT_SIZE_STAT}px; font-weight: bold; border: none; }}
QGroupBox {{
    background-color: {t.SURFACE_PANEL}; border: {t.BORDER_WIDTH}px solid {t.BORDER_DEFAULT};
    border-radius: {t.RADIUS_XL}px; margin-top: {t.SPACE_LG}px;
    padding: {t.SPACE_LG}px {t.SPACE_MD}px {t.SPACE_MD}px {t.SPACE_MD}px; font-weight: bold;
}}
QGroupBox::title {{
    subcontrol-origin: margin; left: {t.SPACE_LG}px; padding: {t.SPACE_0}px {t.SPACE_XS}px;
    color: {t.ACTION_PRIMARY};
}}
QPushButton, QToolButton {{
    background-color: {t.SURFACE_INPUT}; border: {t.BORDER_WIDTH}px solid {t.BORDER_DEFAULT};
    border-radius: {t.RADIUS_LG}px; padding: {t.SPACE_SM}px {t.FONT_SIZE_LG}px;
    min-height: {t.CONTROL_HEIGHT}px;
}}
QPushButton:hover, QToolButton:hover {{ background-color: {t.SURFACE_HOVER}; }}
QPushButton:pressed, QToolButton:pressed {{ background-color: {t.SURFACE_PRESSED}; }}
QToolButton:checked {{ background-color: {t.ACTION_PRIMARY}; border-color: {t.ACTION_PRIMARY}; color: {t.CONTENT_INVERSE}; }}
QPushButton:focus, QToolButton:focus {{ border: {t.FOCUS_WIDTH}px solid {t.BORDER_FOCUS}; }}
QPushButton:disabled, QToolButton:disabled {{ color: {t.CONTENT_DISABLED}; background-color: {t.SURFACE_PANEL}; }}
QPushButton#primary {{
    background-color: {t.ACTION_PRIMARY}; border: none; color: {t.CONTENT_INVERSE};
    font-weight: bold; padding: {t.SPACE_MD}px {t.FONT_SIZE_TITLE}px;
}}
QPushButton#primary:hover {{ background-color: {t.ACTION_PRIMARY_HOVER}; }}
QPushButton#primary:disabled {{ background-color: {t.ACTION_PRIMARY_DISABLED}; color: {t.CONTENT_SECONDARY}; }}
QPushButton#danger {{ background-color: {t.ACTION_DANGER}; border: none; color: {t.CONTENT_INVERSE}; }}
QPushButton#danger:hover {{ background-color: {t.ACTION_DANGER_HOVER}; }}
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QTextEdit, QPlainTextEdit {{
    background-color: {t.SURFACE_INPUT}; border: {t.BORDER_WIDTH}px solid {t.BORDER_DEFAULT};
    border-radius: {t.RADIUS_MD}px; padding: {t.SPACE_XS}px {t.SPACE_SM}px;
    selection-background-color: {t.ACTION_PRIMARY};
}}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus,
QTextEdit:focus, QPlainTextEdit:focus {{ border: {t.FOCUS_WIDTH}px solid {t.BORDER_FOCUS}; }}
QComboBox::drop-down {{ border: none; width: {t.ICON_LG}px; }}
QComboBox QAbstractItemView {{
    background-color: {t.SURFACE_INPUT}; border: {t.BORDER_WIDTH}px solid {t.BORDER_DEFAULT};
    selection-background-color: {t.ACTION_PRIMARY};
}}
QCheckBox::indicator {{
    width: {t.ICON_SM}px; height: {t.ICON_SM}px; border: {t.BORDER_WIDTH}px solid {t.BORDER_DEFAULT};
    border-radius: {t.RADIUS_SM}px; background: {t.SURFACE_INPUT};
}}
QCheckBox::indicator:checked {{ background-color: {t.ACTION_PRIMARY}; border-color: {t.ACTION_PRIMARY}; }}
QCheckBox:focus {{ outline: {t.FOCUS_WIDTH}px solid {t.BORDER_FOCUS}; }}
QTableWidget, QTableView, QListWidget, QTreeWidget {{
    background-color: {t.SURFACE_PANEL}; alternate-background-color: {t.SURFACE_ALT};
    border: {t.BORDER_WIDTH}px solid {t.BORDER_DEFAULT}; border-radius: {t.RADIUS_LG}px;
    gridline-color: {t.BORDER_DEFAULT};
}}
QHeaderView::section {{
    background-color: {t.SURFACE_INPUT}; border: none;
    border-right: {t.BORDER_WIDTH}px solid {t.BORDER_DEFAULT};
    border-bottom: {t.BORDER_WIDTH}px solid {t.BORDER_DEFAULT};
    padding: {t.RADIUS_MD}px {t.SPACE_MD}px; font-weight: bold;
}}
QTableWidget::item:selected, QListWidget::item:selected {{ background-color: {t.ACTION_PRIMARY}; color: {t.CONTENT_INVERSE}; }}
QProgressBar {{
    background-color: {t.SURFACE_INPUT}; border: {t.BORDER_WIDTH}px solid {t.BORDER_DEFAULT};
    border-radius: {t.RADIUS_LG}px; text-align: center; height: {t.FONT_SIZE_TITLE}px;
}}
QProgressBar::chunk {{ background-color: {t.ACTION_PRIMARY}; border-radius: {t.RADIUS_MD}px; }}
QPlainTextEdit#console {{
    background-color: {t.SURFACE_CANVAS}; color: {t.CONTENT_CONSOLE}; font-family: {t.FONT_MONO};
    font-size: {t.FONT_SIZE_SM}px;
}}
QAbstractScrollArea, QAbstractScrollArea QWidget#qt_scrollarea_viewport,
QScrollArea, QScrollArea > QWidget > QWidget {{ background-color: {t.SURFACE_BASE}; border: none; }}
QScrollBar:vertical {{ background: {t.SURFACE_BASE}; width: {t.SCROLLBAR_SIZE}px; margin: {t.SPACE_0}px; }}
QScrollBar::handle:vertical {{ background: {t.BORDER_DEFAULT}; border-radius: {t.RADIUS_MD}px; min-height: {t.CONTROL_HEIGHT}px; }}
QScrollBar::handle:vertical:hover {{ background: {t.CONTENT_DISABLED}; }}
QScrollBar:horizontal {{ background: {t.SURFACE_BASE}; height: {t.SCROLLBAR_SIZE}px; }}
QScrollBar::handle:horizontal {{ background: {t.BORDER_DEFAULT}; border-radius: {t.RADIUS_MD}px; min-width: {t.CONTROL_HEIGHT}px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: {t.SPACE_0}px; width: {t.SPACE_0}px; }}
QStatusBar {{ background: {t.SURFACE_SIDEBAR}; border-top: {t.BORDER_WIDTH}px solid {t.BORDER_DEFAULT}; color: {t.CONTENT_SECONDARY}; }}
QTabWidget::pane {{
    border: {t.BORDER_WIDTH}px solid {t.BORDER_DEFAULT}; border-radius: {t.RADIUS_LG}px;
    background: {t.SURFACE_PANEL};
}}
QTabBar::tab {{
    background: transparent; padding: {t.SPACE_MD}px {t.SPACE_XXL}px; color: {t.CONTENT_SECONDARY};
    border-bottom: {t.FOCUS_WIDTH}px solid transparent;
}}
QTabBar::tab:selected {{ color: {t.CONTENT_PRIMARY}; border-bottom: {t.FOCUS_WIDTH}px solid {t.ACTION_PRIMARY}; }}
QSplitter::handle {{ background: {t.BORDER_DEFAULT}; }}
QToolTip {{
    background-color: {t.SURFACE_INPUT}; color: {t.CONTENT_PRIMARY};
    border: {t.BORDER_WIDTH}px solid {t.BORDER_DEFAULT};
}}
"""


STYLESHEET = build_stylesheet(DARK_TOKENS)

# Compatibility exports. Values still originate in the registry.
ACCENT = DARK_TOKENS.ACTION_PRIMARY
ACCENT_HOVER = DARK_TOKENS.ACTION_PRIMARY_HOVER
BG = DARK_TOKENS.SURFACE_BASE
BG_PANEL = DARK_TOKENS.SURFACE_PANEL
BG_INPUT = DARK_TOKENS.SURFACE_INPUT
BORDER = DARK_TOKENS.BORDER_DEFAULT
TEXT = DARK_TOKENS.CONTENT_PRIMARY
TEXT_DIM = DARK_TOKENS.CONTENT_SECONDARY

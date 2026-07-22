"""Shared font resolution for Widgets and QML frontends."""

from __future__ import annotations

import os
from pathlib import Path

from PySide6.QtGui import QFont, QFontDatabase, QFontMetrics, QGuiApplication


HAN_SAMPLE = ord("项")
PREFERRED_FAMILIES = (
    "Inter",
    "Microsoft YaHei UI",
    "Microsoft YaHei",
    "Noto Sans CJK SC",
    "SimHei",
    "Arial Unicode MS",
)
WINDOWS_FONT_FILES = ("msyh.ttc", "msyhbd.ttc", "simhei.ttf", "ARIALUNI.ttf")


def resolve_ui_font(point_size: int = 10) -> QFont:
    families = set(QFontDatabase.families())
    for family in PREFERRED_FAMILIES:
        font = QFont(family, point_size)
        if family in families and QFontMetrics(font).inFontUcs4(HAN_SAMPLE):
            return font

    font_root = Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts"
    for filename in WINDOWS_FONT_FILES:
        candidate = font_root / filename
        if not candidate.is_file():
            continue
        font_id = QFontDatabase.addApplicationFont(str(candidate))
        if font_id < 0:
            continue
        for family in QFontDatabase.applicationFontFamilies(font_id):
            font = QFont(family, point_size)
            if QFontMetrics(font).inFontUcs4(HAN_SAMPLE):
                return font

    fallback = QGuiApplication.font()
    fallback.setPointSize(point_size)
    if not QFontMetrics(fallback).inFontUcs4(HAN_SAMPLE):
        raise RuntimeError("No installed or bundled font can render Chinese UI glyphs")
    return fallback


def configure_application_font(
    application: QGuiApplication, point_size: int = 10
) -> QFont:
    font = resolve_ui_font(point_size)
    application.setFont(font)
    return font

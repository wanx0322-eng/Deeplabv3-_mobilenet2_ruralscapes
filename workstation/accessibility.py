"""Accessibility metadata helpers for Qt Widgets pages."""

from __future__ import annotations

from PySide6.QtWidgets import (
    QAbstractButton,
    QAbstractItemView,
    QAbstractSlider,
    QAbstractSpinBox,
    QComboBox,
    QLineEdit,
    QTextEdit,
    QWidget,
)
from workstation.ui_text import localize_widget_tree



INTERACTIVE_TYPES = (
    QAbstractButton,
    QAbstractItemView,
    QAbstractSlider,
    QAbstractSpinBox,
    QComboBox,
    QLineEdit,
    QTextEdit,
)


def _derived_name(widget: QWidget) -> str:
    if isinstance(widget, QAbstractButton) and widget.text().strip():
        return widget.text().replace("&", "").strip()
    if isinstance(widget, QLineEdit) and widget.placeholderText().strip():
        return widget.placeholderText().strip()
    name = widget.objectName().strip()
    if name:
        return name
    return widget.metaObject().className()


def ensure_accessibility(root: QWidget) -> None:
    localize_widget_tree(root)
    widgets = [root, *root.findChildren(QWidget)]
    for widget in widgets:
        if isinstance(widget, INTERACTIVE_TYPES) and not widget.accessibleName().strip():
            widget.setAccessibleName(_derived_name(widget))
        if isinstance(widget, INTERACTIVE_TYPES) and not widget.accessibleDescription():
            widget.setAccessibleDescription(widget.toolTip() or widget.accessibleName())


def unnamed_controls(root: QWidget) -> list[QWidget]:
    return [
        widget
        for widget in [root, *root.findChildren(QWidget)]
        if isinstance(widget, INTERACTIVE_TYPES) and not widget.accessibleName().strip()
    ]

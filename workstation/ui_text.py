"""Runtime traversal for translating and auditing visible Qt Widgets copy."""

from __future__ import annotations

from collections.abc import Callable, Iterator

from PySide6.QtCore import QCoreApplication, Qt
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QAbstractButton,
    QComboBox,
    QGroupBox,
    QLabel,
    QLineEdit,
    QTabWidget,
    QTableWidget,
    QTextEdit,
    QWidget,
)


TextSlot = tuple[Callable[[], str], Callable[[str], None]]
from workstation.translation_rules import english_for


def contains_han(text: str) -> bool:
    return any("\u3400" <= character <= "\u9fff" for character in text)


def translate_runtime_text(source: str) -> str:
    application = QCoreApplication.instance()
    if (
        application is not None
        and application.property("ruralscapeLanguage") == "en_US"
        and contains_han(source)
    ):
        return english_for(source)
    return source


def _widget_slots(widget: QWidget) -> Iterator[TextSlot]:
    if isinstance(widget, (QLabel, QAbstractButton)):
        yield widget.text, widget.setText
    if isinstance(widget, QGroupBox):
        yield widget.title, widget.setTitle
    if isinstance(widget, (QLineEdit, QTextEdit)):
        yield widget.placeholderText, widget.setPlaceholderText
    if isinstance(widget, QComboBox):
        for index in range(widget.count()):
            yield (
                lambda index=index: widget.itemText(index),
                lambda value, index=index: widget.setItemText(index, value),
            )
    if isinstance(widget, QTabWidget):
        for index in range(widget.count()):
            yield (
                lambda index=index: widget.tabText(index),
                lambda value, index=index: widget.setTabText(index, value),
            )
    if isinstance(widget, QTableWidget):
        for index in range(widget.columnCount()):
            item = widget.horizontalHeaderItem(index)
            if item is not None:
                yield item.text, item.setText
    yield widget.toolTip, widget.setToolTip
    yield widget.accessibleName, widget.setAccessibleName
    yield widget.accessibleDescription, widget.setAccessibleDescription


def _action_slots(action: QAction) -> Iterator[TextSlot]:
    yield action.text, action.setText
    yield action.toolTip, action.setToolTip
    yield action.statusTip, action.setStatusTip
    yield action.accessibleName, action.setAccessibleName


def iter_text_slots(root: QWidget) -> Iterator[TextSlot]:
    for widget in (root, *root.findChildren(QWidget)):
        yield from _widget_slots(widget)
    for action in root.findChildren(QAction):
        yield from _action_slots(action)


def collect_visible_text(root: QWidget) -> set[str]:
    return {value for getter, _setter in iter_text_slots(root) if (value := getter())}


def localize_widget_tree(root: QWidget) -> None:
    application = QCoreApplication.instance()
    if application is None or application.property("ruralscapeLanguage") != "en_US":
        return
    for getter, setter in iter_text_slots(root):
        source = getter()
        if source and contains_han(source):
            setter(QCoreApplication.translate("Ui", source))

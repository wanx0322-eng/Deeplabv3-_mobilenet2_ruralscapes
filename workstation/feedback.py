"""Non-modal feedback primitives and field-addressable validation issues."""

from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import QLabel


@dataclass(frozen=True, slots=True)
class ConfigIssue:
    field: str
    message: str
    severity: str = "error"


class InlineMessage(QLabel):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("inlineMessage")
        self.setWordWrap(True)
        self.setVisible(False)

    def show_issue(self, issue: ConfigIssue) -> None:
        self.setProperty("severity", issue.severity)
        self.setText(f"{issue.field}: {issue.message}")
        self.setVisible(True)

    def show_message(self, message: str, severity: str = "info") -> None:
        self.setProperty("severity", severity)
        self.setText(message)
        self.setVisible(True)
        self.style().unpolish(self)
        self.style().polish(self)

    def clear(self) -> None:
        self.setText("")
        self.setVisible(False)


class ToastManager(QObject):
    messageShown = Signal(str, str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.last_message = ""
        self.last_action = ""

    def show(self, message: str, action: str = "") -> None:
        self.last_message = message
        self.last_action = action
        self.messageShown.emit(message, action)

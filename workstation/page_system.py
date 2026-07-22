"""Page lifecycle, lazy loading and replayable workspace events."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import QLabel, QStackedWidget, QVBoxLayout, QWidget

from workstation.theme import DARK_TOKENS
from workstation.feedback import InlineMessage, ToastManager
from workstation.ui_text import translate_runtime_text


class BasePage(QWidget):
    """Common page shell and lifecycle contract for the official workstation."""

    def __init__(self, title: str = "", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("pageRoot")
        self.page_layout = QVBoxLayout(self)
        self.page_layout.setContentsMargins(
            DARK_TOKENS.SPACE_XXL,
            DARK_TOKENS.SPACE_XL,
            DARK_TOKENS.SPACE_XXL,
            DARK_TOKENS.SPACE_XL,
        )
        self.page_layout.setSpacing(DARK_TOKENS.SPACE_LG)
        self.page_title = QLabel(title, self)
        self.page_title.setObjectName("pageTitle")
        self.page_title.setAccessibleName(title)
        self.page_layout.addWidget(self.page_title)
        self.feedback = InlineMessage(self)
        self.page_layout.addWidget(self.feedback)
        self.toast = ToastManager(self)
        self.toast.messageShown.connect(
            lambda message, _action: self.feedback.show_message(message)
        )

    def on_activated(self) -> None:
        """Called after the page becomes visible."""
        self.refresh()

    def on_deactivated(self) -> None:
        """Called before another page becomes visible."""

    def refresh(self) -> None:
        """Refresh external state. Subclasses override when needed."""

    def bind_workspace_events(self, events: "WorkspaceEvents") -> None:
        """Attach to the shared replayable event bus."""

    def show_message(self, message: str, severity: str = "info") -> None:
        self.toast.show(message)
        message = translate_runtime_text(message)
        self.feedback.show_message(message, severity)

    def has_running_task(self) -> bool:
        return False

    def stop_running_task(self) -> bool:
        return False


@dataclass(frozen=True, slots=True)
class PageSpec:
    id: str
    title: str
    icon: str
    shortcut: str
    factory: Callable[[], QWidget]


class WorkspaceEvents(QObject):
    """Cross-page event bus with last-value replay for lazy pages."""

    datasetChanged = Signal(object)
    labelsChanged = Signal(object)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._latest: dict[str, Any] = {}
        self._subscribers: dict[str, list[Callable[[Any], None]]] = {}

    def publish(self, topic: str, payload: Any = None) -> None:
        self._latest[topic] = payload
        signal = {
            "dataset": self.datasetChanged,
            "labels": self.labelsChanged,
        }.get(topic)
        if signal is not None:
            signal.emit(payload)
        for callback in tuple(self._subscribers.get(topic, ())):
            callback(payload)

    def subscribe(
        self, topic: str, callback: Callable[[Any], None], *, replay: bool = True
    ) -> Callable[[], None]:
        self._subscribers.setdefault(topic, []).append(callback)
        if replay and topic in self._latest:
            callback(self._latest[topic])

        def unsubscribe() -> None:
            callbacks = self._subscribers.get(topic, [])
            if callback in callbacks:
                callbacks.remove(callback)

        return unsubscribe


class PageHost(QStackedWidget):
    """Lazy page factory. Only the first registered page is built at startup."""

    pageLoaded = Signal(str, object)

    def __init__(
        self, specs: tuple[PageSpec, ...] | list[PageSpec], parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        if not specs:
            raise ValueError("PageHost requires at least one PageSpec")
        self._specs = tuple(specs)
        self._by_id = {spec.id: spec for spec in self._specs}
        if len(self._by_id) != len(self._specs):
            raise ValueError("PageSpec ids must be unique")
        self._pages: dict[str, QWidget] = {}
        self._active_id: str | None = None
        self.activate(self._specs[0].id)

    @property
    def loaded_page_count(self) -> int:
        return len(self._pages)

    def page(self, page_id: str) -> QWidget | None:
        return self._pages.get(page_id)

    def activate(self, page: str | int) -> QWidget:
        spec = self._specs[page] if isinstance(page, int) else self._by_id[page]
        previous = self._pages.get(self._active_id or "")
        if previous is not None and isinstance(previous, BasePage):
            previous.on_deactivated()
        widget = self._pages.get(spec.id)
        if widget is None:
            widget = spec.factory()
            self._pages[spec.id] = widget
            self.addWidget(widget)
            self.pageLoaded.emit(spec.id, widget)
        self.setCurrentWidget(widget)
        self._active_id = spec.id
        if isinstance(widget, BasePage):
            widget.on_activated()
        return widget

    def loaded_pages(self) -> tuple[QWidget, ...]:
        return tuple(self._pages.values())

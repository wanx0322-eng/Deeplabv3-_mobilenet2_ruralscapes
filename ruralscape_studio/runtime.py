"""Explicit QML runtime contract for connected and demonstration modes."""

from __future__ import annotations

from PySide6.QtCore import Property, QObject, Signal, Slot


class StudioRuntime(QObject):
    """Single object injected into Main.qml through initial properties."""

    def __init__(
        self,
        controllers: dict[str, QObject],
        backends: dict[str, QObject],
        console: QObject,
        app_settings: QObject,
        *,
        mode: str,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        if mode not in {"connected", "demo"}:
            raise ValueError("StudioRuntime mode must be connected or demo")
        self._controllers = dict(controllers)
        self._backends = dict(backends)
        self._console = console
        self._app_settings = app_settings
        self._mode = mode

    def _controller(self, name: str) -> QObject:
        return self._controllers[name]

    def _backend(self, name: str) -> QObject:
        return self._backends[name]

    @Property(str, constant=True)
    def mode(self) -> str:
        return self._mode

    @Property(bool, constant=True)
    def isDemo(self) -> bool:
        return self._mode == "demo"

    @Property(QObject, constant=True)
    def projectController(self) -> QObject:
        return self._controller("projectController")

    @Property(QObject, constant=True)
    def datasetController(self) -> QObject:
        return self._controller("datasetController")

    @Property(QObject, constant=True)
    def annotationController(self) -> QObject:
        return self._controller("annotationController")

    @Property(QObject, constant=True)
    def trainingController(self) -> QObject:
        return self._controller("trainingController")

    @Property(QObject, constant=True)
    def inferenceController(self) -> QObject:
        return self._controller("inferenceController")

    @Property(QObject, constant=True)
    def evaluationController(self) -> QObject:
        return self._controller("evaluationController")

    @Property(QObject, constant=True)
    def modelController(self) -> QObject:
        return self._controller("modelController")

    @Property(QObject, constant=True)
    def taskManager(self) -> QObject:
        return self._controller("taskManager")

    @Property(QObject, constant=True)
    def datasetBackend(self) -> QObject:
        return self._backend("datasetBackend")

    @Property(QObject, constant=True)
    def trainingBackend(self) -> QObject:
        return self._backend("trainingBackend")

    @Property(QObject, constant=True)
    def evaluationBackend(self) -> QObject:
        return self._backend("evaluationBackend")

    @Property(QObject, constant=True)
    def inferenceBackend(self) -> QObject:
        return self._backend("inferenceBackend")

    @Property(QObject, constant=True)
    def modelBackend(self) -> QObject:
        return self._backend("modelBackend")

    @Property(QObject, constant=True)
    def console(self) -> QObject:
        return self._console

    @Property(QObject, constant=True)
    def appSettings(self) -> QObject:
        return self._app_settings


class RuntimeConsole(QObject):
    """Bounded runtime log exposed to QML."""

    linesChanged = Signal()
    MAX_LINES = 500

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._lines: list[str] = []

    @Property("QStringList", notify=linesChanged)
    def lines(self) -> list[str]:
        return list(self._lines)

    @Property(str, notify=linesChanged)
    def lastLine(self) -> str:
        return self._lines[-1] if self._lines else ""

    @Slot(str)
    def append(self, line: str) -> None:
        self._lines.append(line)
        if len(self._lines) > self.MAX_LINES:
            del self._lines[: len(self._lines) - self.MAX_LINES]
        self.linesChanged.emit()

    @Slot()
    def clear(self) -> None:
        if self._lines:
            self._lines.clear()
            self.linesChanged.emit()


class DemoBackend(QObject):
    """Signature-compatible backend that always provides explicit feedback."""

    feedback = Signal(str)

    def __init__(
        self, capability: str, console: RuntimeConsole, parent: QObject | None = None
    ) -> None:
        super().__init__(parent)
        self._capability = capability
        self._console = console

    def _report(self, action: str) -> None:
        message = f"Demo mode: {self._capability}.{action} is not connected"
        self._console.append(message)
        self.feedback.emit(message)

    @Slot()
    def useProjectDataset(self) -> None:
        self._report("useProjectDataset")

    @Slot()
    def scan(self) -> None:
        self._report("scan")

    @Slot()
    @Slot(str)
    def start(self, detail: str = "") -> None:
        self._report(f"start({detail})" if detail else "start")

    @Slot()
    def requestStop(self) -> None:
        self._report("requestStop")

    @Slot(str)
    def startEvaluation(self, split: str = "") -> None:
        self._report(f"start({split})")

    @Slot()
    def refresh(self) -> None:
        self._report("refresh")


def build_demo_backends(
    console: RuntimeConsole, parent: QObject
) -> dict[str, DemoBackend]:
    return {
        name: DemoBackend(name, console, parent)
        for name in (
            "datasetBackend",
            "trainingBackend",
            "evaluationBackend",
            "inferenceBackend",
            "modelBackend",
        )
    }

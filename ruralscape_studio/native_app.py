"""Native PySide6 bootstrap for the RuralScape Studio desktop shell."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Sequence

from PySide6.QtCore import Property, QObject, QUrl
from PySide6.QtGui import QGuiApplication
from PySide6.QtQml import QQmlApplicationEngine
from PySide6.QtQuickControls2 import QQuickStyle

from .controllers import (
    AnnotationController,
    DatasetController,
    EvaluationController,
    InferenceController,
    ModelController,
    ProjectController,
    TaskManager,
    TrainingController,
)
from .runtime import RuntimeConsole, StudioRuntime, build_demo_backends
from workstation.fonts import configure_application_font
from workstation.i18n import install_translation, resolve_language



class AppSettings(QObject):
    """Read-only runtime preferences consumed by QML."""

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        value = os.environ.get("RURALSCAPE_REDUCE_MOTION", "").strip().lower()
        self._reduced_motion = value in {"1", "true", "yes", "on"}

    @Property(bool, constant=True)
    def reducedMotion(self) -> bool:
        return self._reduced_motion


CONTEXT_FACTORIES = {
    "projectController": ProjectController,
    "datasetController": DatasetController,
    "annotationController": AnnotationController,
    "trainingController": TrainingController,
    "inferenceController": InferenceController,
    "evaluationController": EvaluationController,
    "modelController": ModelController,
    "taskManager": TaskManager,
}


def qml_path() -> Path:
    return Path(__file__).resolve().parent / "qml" / "Main.qml"


def _configure_application_font(application: QGuiApplication) -> None:
    configure_application_font(application)


def create_application(argv: Sequence[str] | None = None) -> QGuiApplication:
    QQuickStyle.setStyle("Basic")
    arguments = list(argv) if argv is not None else sys.argv
    language = resolve_language(arguments)
    existing = QGuiApplication.instance()
    if existing is not None:
        _configure_application_font(existing)
        install_translation(existing, language)
        return existing
    application = QGuiApplication(arguments)
    application.setApplicationName("RuralScape Studio")
    application.setOrganizationName("RuralScape Studio")
    _configure_application_font(application)
    install_translation(application, language)
    return application


def create_engine(*, load: bool = True) -> QQmlApplicationEngine:
    engine = QQmlApplicationEngine()
    context = engine.rootContext()
    context_objects: list[QObject] = []
    controllers: dict[str, QObject] = {}
    for name, factory in CONTEXT_FACTORIES.items():
        controller = factory(engine)
        context.setContextProperty(name, controller)
        context_objects.append(controller)
        controllers[name] = controller
    settings = AppSettings(engine)
    context.setContextProperty("appSettings", settings)
    context_objects.append(settings)
    console = RuntimeConsole(engine)
    backends = build_demo_backends(console, engine)
    for name, backend in backends.items():
        context.setContextProperty(name, backend)
    context.setContextProperty("console", console)
    runtime = StudioRuntime(
        controllers, backends, console, settings, mode="demo", parent=engine
    )
    engine.setInitialProperties({"runtime": runtime})
    engine._studio_runtime = runtime
    engine._studio_objects = (controllers, backends, console, settings, runtime)

    engine.setProperty("contextObjectCount", len(context_objects))
    if load:
        load_main_qml(engine)
    return engine


def load_main_qml(engine: QQmlApplicationEngine) -> None:
    source = qml_path()
    engine.load(QUrl.fromLocalFile(str(source)))
    if not engine.rootObjects():
        raise RuntimeError(f"Unable to load native QML shell: {source}")


def main(argv: Sequence[str] | None = None) -> int:
    application = create_application(argv)
    engine = create_engine(load=True)
    if not engine.rootObjects():
        return 1
    return application.exec()


if __name__ == "__main__":
    raise SystemExit(main())



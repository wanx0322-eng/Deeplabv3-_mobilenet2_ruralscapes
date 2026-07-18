"""Native PySide6 bootstrap for the RuralScape Studio desktop shell."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Sequence

from PySide6.QtCore import Property, QObject, QUrl
from PySide6.QtGui import QFont, QFontDatabase, QFontMetrics, QGuiApplication
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
    sample = ord("项")
    preferred = ("Inter", "Microsoft YaHei UI", "Noto Sans CJK SC", "Arial Unicode MS")
    families = set(QFontDatabase.families())
    selected = next(
        (
            family
            for family in preferred
            if family in families and QFontMetrics(QFont(family)).inFontUcs4(sample)
        ),
        "",
    )
    if not selected:
        font_root = Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts"
        for filename in ("msyh.ttc", "ARIALUNI.ttf", "simhei.ttf"):
            candidate = font_root / filename
            if not candidate.is_file():
                continue
            font_id = QFontDatabase.addApplicationFont(str(candidate))
            if font_id < 0:
                continue
            loaded = QFontDatabase.applicationFontFamilies(font_id)
            selected = next(
                (
                    family
                    for family in loaded
                    if QFontMetrics(QFont(family)).inFontUcs4(sample)
                ),
                "",
            )
            if selected:
                break
    if selected:
        application.setFont(QFont(selected, 10))


def create_application(argv: Sequence[str] | None = None) -> QGuiApplication:
    QQuickStyle.setStyle("Basic")
    existing = QGuiApplication.instance()
    if existing is not None:
        _configure_application_font(existing)
        return existing
    application = QGuiApplication(list(argv) if argv is not None else sys.argv)
    application.setApplicationName("RuralScape Studio")
    application.setOrganizationName("RuralScape Studio")
    _configure_application_font(application)
    return application


def create_engine(*, load: bool = True) -> QQmlApplicationEngine:
    engine = QQmlApplicationEngine()
    context = engine.rootContext()
    context_objects: list[QObject] = []
    for name, factory in CONTEXT_FACTORIES.items():
        controller = factory(engine)
        context.setContextProperty(name, controller)
        context_objects.append(controller)
    settings = AppSettings(engine)
    context.setContextProperty("appSettings", settings)
    context_objects.append(settings)
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



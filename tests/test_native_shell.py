from __future__ import annotations

import os
import subprocess
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QT_QUICK_BACKEND", "software")
os.environ.setdefault("QSG_RHI_BACKEND", "software")

from PySide6.QtCore import QObject
from PySide6.QtTest import QSignalSpy

from ruralscape_studio import controllers


CONTROLLER_NAMES = (
    "ProjectController",
    "DatasetController",
    "AnnotationController",
    "TrainingController",
    "InferenceController",
    "EvaluationController",
    "ModelController",
    "TaskManager",
)


def test_controller_module_does_not_import_torch() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import ruralscape_studio.controllers; "
                "assert 'torch' not in sys.modules"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_all_public_controllers_are_qobjects() -> None:
    for name in CONTROLLER_NAMES:
        controller_type = getattr(controllers, name)
        assert isinstance(controller_type(), QObject)


def test_project_controller_updates_properties_and_emits_signal() -> None:
    controller = controllers.ProjectController()
    spy = QSignalSpy(controller.projectChanged)

    controller.selectProject("村景语义分割", "E:/ruralscape")

    assert controller.projectName == "村景语义分割"
    assert controller.rootPath == "E:/ruralscape"
    assert controller.hasProject is True
    assert spy.count() == 1


def test_dataset_controller_exposes_real_empty_index_state() -> None:
    controller = controllers.DatasetController()
    spy = QSignalSpy(controller.datasetChanged)

    controller.selectDataset("E:/dataset")

    assert controller.datasetPath == "E:/dataset"
    assert controller.imageCount == 0
    assert controller.maskCount == 0
    assert controller.issueCount == 0
    assert controller.indexState == "not_indexed"
    assert spy.count() == 1


def test_workflow_controller_status_and_error_signals_are_observable() -> None:
    controller = controllers.TrainingController()
    status_spy = QSignalSpy(controller.statusChanged)
    error_spy = QSignalSpy(controller.errorMessageChanged)

    controller.begin()
    controller.setError("训练配置尚未完成")
    controller.clearError()

    assert controller.status == "running"
    assert controller.running is True
    assert status_spy.count() == 1
    assert error_spy.count() == 2


def test_task_manager_records_only_enqueued_tasks() -> None:
    manager = controllers.TaskManager()
    spy = QSignalSpy(manager.tasksChanged)

    task_id = manager.enqueueTask("扫描数据集", "dataset")

    assert task_id == 1
    assert manager.runningCount == 0
    assert manager.tasks == [
        {"id": 1, "title": "扫描数据集", "kind": "dataset", "status": "queued"}
    ]
    assert spy.count() == 1

def test_native_engine_registers_all_shell_context_objects() -> None:
    from ruralscape_studio.native_app import create_application, create_engine

    create_application(["ruralscape-studio-test"])
    engine = create_engine(load=False)
    context = engine.rootContext()

    for name in (
        "projectController",
        "datasetController",
        "annotationController",
        "trainingController",
        "inferenceController",
        "evaluationController",
        "modelController",
        "taskManager",
        "appSettings",
    ):
        assert isinstance(context.contextProperty(name), QObject), name


def test_qml_shell_loads_offscreen_without_warnings() -> None:
    from ruralscape_studio.native_app import (
        create_application,
        create_engine,
        load_main_qml,
    )

    application = create_application(["ruralscape-studio-test"])
    engine = create_engine(load=False)
    warnings: list[str] = []
    engine.warnings.connect(
        lambda messages: warnings.extend(message.toString() for message in messages)
    )

    load_main_qml(engine)
    application.processEvents()

    assert len(engine.rootObjects()) == 1
    window = engine.rootObjects()[0]
    assert window.property("minimumWidth") == 1024
    assert window.property("minimumHeight") >= 700
    assert warnings == []
    window.close()

def test_native_application_font_renders_chinese_navigation() -> None:
    from PySide6.QtGui import QFontMetrics
    from ruralscape_studio.native_app import create_application

    application = create_application(["ruralscape-studio-font-test"])
    assert QFontMetrics(application.font()).inFontUcs4(ord("项"))


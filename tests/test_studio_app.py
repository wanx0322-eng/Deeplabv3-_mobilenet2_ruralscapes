"""接了真实后端的 QML 外壳启动测试（离屏）。"""
from __future__ import annotations

import os
import sys

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QT_QUICK_BACKEND", "software")

from PySide6.QtQml import QQmlApplicationEngine

from ruralscape_studio.native_app import create_application
from workstation.config import Config
from workstation.studio_app import ConsoleLog, build_context, create_engine


@pytest.fixture(scope="module")
def gui_app():
    #   必须走 create_application()：它把 QQuickStyle 设成 Basic，
    #   否则原生样式不支持组件自定义，Main.qml 会刷一屏警告。
    yield create_application(sys.argv[:1])


@pytest.fixture
def config(tmp_path):
    return Config(path=str(tmp_path / "workstation_config.json"))


def test_build_context_registers_controllers_and_backends(gui_app, config):
    engine = QQmlApplicationEngine()
    controllers, backends, console = build_context(engine, config)

    assert set(controllers) == {
        "projectController", "datasetController", "annotationController",
        "trainingController", "inferenceController", "evaluationController",
        "modelController", "taskManager",
    }
    assert set(backends) == {
        "datasetBackend", "trainingBackend", "evaluationBackend",
        "inferenceBackend", "modelBackend",
    }
    assert isinstance(console, ConsoleLog)


def test_project_and_dataset_are_preselected(gui_app, config):
    engine = QQmlApplicationEngine()
    controllers, _backends, _console = build_context(engine, config)
    assert controllers["projectController"].hasProject is True
    assert controllers["datasetController"].datasetPath.endswith("VOCdevkit")


def test_backend_logs_reach_the_shared_console(gui_app, config):
    engine = QQmlApplicationEngine()
    _controllers, backends, console = build_context(engine, config)
    backends["datasetBackend"].logLine.emit("扫描完成")
    assert "扫描完成" in console.lines
    assert console.lastLine == "扫描完成"


def test_console_keeps_a_bounded_buffer(gui_app):
    console = ConsoleLog()
    for i in range(ConsoleLog.MAX_LINES + 50):
        console.append("line %d" % i)
    assert len(console.lines) == ConsoleLog.MAX_LINES
    assert console.lines[0] == "line 50"
    console.clear()
    assert console.lines == []


def test_qml_shell_loads_with_backends_and_no_warnings(gui_app, config):
    """真正的集成检查：注入后端后 Main.qml 仍能零警告加载。"""
    engine = QQmlApplicationEngine()
    warnings = []
    engine.warnings.connect(lambda errors: warnings.extend(
        error.toString() for error in errors))

    build_context(engine, config)
    from ruralscape_studio.native_app import load_main_qml

    load_main_qml(engine)

    assert engine.rootObjects(), "Main.qml 未能加载"
    assert warnings == [], "QML 警告：\n" + "\n".join(warnings)


def test_create_engine_end_to_end(gui_app, config):
    engine = create_engine(config, load=True)
    assert engine.rootObjects()
    assert engine.property("studioControllerCount") == 8
    assert engine.property("studioBackendCount") == 5

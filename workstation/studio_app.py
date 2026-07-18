"""QML 外壳的启动入口：在 ruralscape_studio 的 controllers 之上接真实后端。

与 ruralscape_studio/native_app.py 的区别：那边只创建 torch-free 的状态对象，
这里额外注入 workstation.studio_bridge 里的后端，让按钮真的能跑任务。

    python run_studio.py
"""
import sys

from PySide6.QtCore import Property, QObject, Signal, Slot
from PySide6.QtQml import QQmlApplicationEngine

from ruralscape_studio.native_app import (
    CONTEXT_FACTORIES,
    AppSettings,
    create_application,
    load_main_qml,
)
from workstation.config import PROJECT_ROOT, Config
from workstation.studio_bridge import (
    DatasetBackend,
    EvaluationBackend,
    InferenceBackend,
    ModelBackend,
    TrainingBackend,
)


class ConsoleLog(QObject):
    """所有后端的日志汇总到一处，QML 侧只订阅这一个对象。"""

    linesChanged = Signal()

    MAX_LINES = 500

    def __init__(self, parent=None):
        super().__init__(parent)
        self._lines = []

    @Property("QStringList", notify=linesChanged)
    def lines(self):
        return list(self._lines)

    @Property(str, notify=linesChanged)
    def lastLine(self):
        return self._lines[-1] if self._lines else ""

    @Slot(str)
    def append(self, line):
        self._lines.append(line)
        if len(self._lines) > self.MAX_LINES:
            del self._lines[: len(self._lines) - self.MAX_LINES]
        self.linesChanged.emit()

    @Slot()
    def clear(self):
        if self._lines:
            self._lines.clear()
            self.linesChanged.emit()


def build_context(engine, config=None):
    """创建 controllers + backends，全部注册为 QML 上下文属性。

    返回 (controllers, backends, console)，测试里可以直接拿来断言。
    """
    config = config or Config()
    context = engine.rootContext()

    controllers = {}
    for name, factory in CONTEXT_FACTORIES.items():
        controller = factory(engine)
        context.setContextProperty(name, controller)
        controllers[name] = controller

    settings = AppSettings(engine)
    context.setContextProperty("appSettings", settings)

    console = ConsoleLog(engine)
    context.setContextProperty("console", console)

    task_manager = controllers["taskManager"]
    backends = {
        "datasetBackend": DatasetBackend(
            config, controllers["datasetController"], engine),
        "trainingBackend": TrainingBackend(
            config, controllers["trainingController"], task_manager, engine),
        "evaluationBackend": EvaluationBackend(
            config, controllers["evaluationController"], task_manager, engine),
        "inferenceBackend": InferenceBackend(
            config, controllers["inferenceController"], task_manager, engine),
        "modelBackend": ModelBackend(
            config, controllers["modelController"], engine),
    }
    for name, backend in backends.items():
        backend.logLine.connect(console.append)
        context.setContextProperty(name, backend)

    #   项目信息直接来自当前工程目录，不需要用户再选一次
    controllers["projectController"].selectProject(
        "RuralScape", PROJECT_ROOT)
    backends["datasetBackend"].useProjectDataset()

    return controllers, backends, console


def create_engine(config=None, *, load=True):
    engine = QQmlApplicationEngine()
    controllers, backends, console = build_context(engine, config)
    #   防止 Python 侧对象被 GC —— 它们的 parent 是 engine，这里再存一份引用
    engine.setProperty("studioControllerCount", len(controllers))
    engine.setProperty("studioBackendCount", len(backends))
    engine._studio_objects = (controllers, backends, console)
    if load:
        load_main_qml(engine)
    return engine


def main(argv=None):
    application = create_application(argv)
    engine = create_engine(load=True)
    if not engine.rootObjects():
        return 1
    return application.exec()


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

"""Torch-free QObject state contracts for the native RuralScape Studio shell."""

from __future__ import annotations

from PySide6.QtCore import Property, QObject, Signal, Slot


class _WorkflowController(QObject):
    statusChanged = Signal()
    runningChanged = Signal()
    progressChanged = Signal()
    errorMessageChanged = Signal()

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._status = "idle"
        self._running = False
        self._progress = 0.0
        self._error_message = ""

    @Property(str, notify=statusChanged)
    def status(self) -> str:
        return self._status

    @Property(bool, notify=runningChanged)
    def running(self) -> bool:
        return self._running

    @Property(float, notify=progressChanged)
    def progress(self) -> float:
        return self._progress

    @Property(str, notify=errorMessageChanged)
    def errorMessage(self) -> str:
        return self._error_message

    @Slot()
    def begin(self) -> None:
        if self._status != "running":
            self._status = "running"
            self.statusChanged.emit()
        if not self._running:
            self._running = True
            self.runningChanged.emit()

    @Slot(float)
    def updateProgress(self, value: float) -> None:
        progress = max(0.0, min(1.0, float(value)))
        if progress != self._progress:
            self._progress = progress
            self.progressChanged.emit()

    @Slot(str)
    def setError(self, message: str) -> None:
        message = message.strip()
        if message != self._error_message:
            self._error_message = message
            self.errorMessageChanged.emit()

    @Slot()
    def clearError(self) -> None:
        self.setError("")

    @Slot(bool)
    def finish(self, ok: bool = True) -> None:
        """结束一次运行，保留错误信息与最终进度。

        D 项目侧扩展：原有的 reset() 会把 errorMessage 和 progress 一并清空，
        任务失败后界面就看不到失败原因了。finish() 只负责离开 running 状态。
        """

        status = "succeeded" if ok else "failed"
        if self._status != status:
            self._status = status
            self.statusChanged.emit()
        if self._running:
            self._running = False
            self.runningChanged.emit()
        if ok and self._progress != 1.0:
            self._progress = 1.0
            self.progressChanged.emit()

    @Slot()
    def reset(self) -> None:
        status_changed = self._status != "idle"
        running_changed = self._running
        progress_changed = self._progress != 0.0
        self._status = "idle"
        self._running = False
        self._progress = 0.0
        if status_changed:
            self.statusChanged.emit()
        if running_changed:
            self.runningChanged.emit()
        if progress_changed:
            self.progressChanged.emit()
        self.clearError()


class ProjectController(QObject):
    projectChanged = Signal()

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._project_name = ""
        self._root_path = ""

    @Property(str, notify=projectChanged)
    def projectName(self) -> str:
        return self._project_name

    @Property(str, notify=projectChanged)
    def rootPath(self) -> str:
        return self._root_path

    @Property(bool, notify=projectChanged)
    def hasProject(self) -> bool:
        return bool(self._root_path)

    @Slot(str, str)
    def selectProject(self, name: str, root_path: str) -> None:
        name = name.strip()
        root_path = root_path.strip()
        if (name, root_path) != (self._project_name, self._root_path):
            self._project_name = name
            self._root_path = root_path
            self.projectChanged.emit()

    @Slot()
    def clearProject(self) -> None:
        self.selectProject("", "")


class DatasetController(QObject):
    datasetChanged = Signal()

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._dataset_path = ""
        self._image_count = 0
        self._mask_count = 0
        self._issue_count = 0
        self._index_state = "not_indexed"

    @Property(str, notify=datasetChanged)
    def datasetPath(self) -> str:
        return self._dataset_path

    @Property(int, notify=datasetChanged)
    def imageCount(self) -> int:
        return self._image_count

    @Property(int, notify=datasetChanged)
    def maskCount(self) -> int:
        return self._mask_count

    @Property(int, notify=datasetChanged)
    def issueCount(self) -> int:
        return self._issue_count

    @Property(str, notify=datasetChanged)
    def indexState(self) -> str:
        return self._index_state

    @Slot(str)
    def selectDataset(self, path: str) -> None:
        path = path.strip()
        if path != self._dataset_path:
            self._dataset_path = path
            self._image_count = 0
            self._mask_count = 0
            self._issue_count = 0
            self._index_state = "not_indexed"
            self.datasetChanged.emit()

    @Slot(int, int, int)
    def applyIndexSummary(self, images: int, masks: int, issues: int) -> None:
        self._image_count = max(0, images)
        self._mask_count = max(0, masks)
        self._issue_count = max(0, issues)
        self._index_state = "ready"
        self.datasetChanged.emit()


class AnnotationController(QObject):
    documentChanged = Signal()

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._active_image_path = ""
        self._version_count = 0
        self._dirty = False

    @Property(str, notify=documentChanged)
    def activeImagePath(self) -> str:
        return self._active_image_path

    @Property(int, notify=documentChanged)
    def versionCount(self) -> int:
        return self._version_count

    @Property(bool, notify=documentChanged)
    def dirty(self) -> bool:
        return self._dirty

    @Slot(str)
    def selectImage(self, path: str) -> None:
        path = path.strip()
        if path != self._active_image_path:
            self._active_image_path = path
            self._version_count = 0
            self._dirty = False
            self.documentChanged.emit()

    @Slot(bool)
    def markDirty(self, dirty: bool) -> None:
        if dirty != self._dirty:
            self._dirty = dirty
            self.documentChanged.emit()


class TrainingController(_WorkflowController):
    engineChanged = Signal()

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._engine = "deeplab_v3_plus"

    @Property(str, notify=engineChanged)
    def engine(self) -> str:
        return self._engine

    @Slot(str)
    def selectEngine(self, engine: str) -> None:
        if engine in {"deeplab_v3_plus", "segformer_b2"} and engine != self._engine:
            self._engine = engine
            self.engineChanged.emit()


class InferenceController(_WorkflowController):
    pathsChanged = Signal()

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._model_path = ""
        self._input_path = ""
        self._output_path = ""

    @Property(str, notify=pathsChanged)
    def modelPath(self) -> str:
        return self._model_path

    @Property(str, notify=pathsChanged)
    def inputPath(self) -> str:
        return self._input_path

    @Property(str, notify=pathsChanged)
    def outputPath(self) -> str:
        return self._output_path

    @Slot(str)
    def selectModel(self, path: str) -> None:
        self._set_path("_model_path", path)

    @Slot(str)
    def selectInput(self, path: str) -> None:
        self._set_path("_input_path", path)

    @Slot(str)
    def selectOutput(self, path: str) -> None:
        self._set_path("_output_path", path)

    def _set_path(self, name: str, value: str) -> None:
        value = value.strip()
        if getattr(self, name) != value:
            setattr(self, name, value)
            self.pathsChanged.emit()


class EvaluationController(_WorkflowController):
    reportChanged = Signal()

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._split = "val"
        self._report_path = ""

    @Property(str, notify=reportChanged)
    def split(self) -> str:
        return self._split

    @Property(str, notify=reportChanged)
    def reportPath(self) -> str:
        return self._report_path

    @Property(bool, notify=reportChanged)
    def hasReport(self) -> bool:
        return bool(self._report_path)

    @Slot(str)
    def selectSplit(self, split: str) -> None:
        if split in {"train", "val", "test"} and split != self._split:
            self._split = split
            self.reportChanged.emit()

    @Slot(str)
    def setReportPath(self, path: str) -> None:
        path = path.strip()
        if path != self._report_path:
            self._report_path = path
            self.reportChanged.emit()


class ModelController(QObject):
    modelsChanged = Signal()

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._artifact_count = 0
        self._selected_model_path = ""
        self._export_status = "idle"

    @Property(int, notify=modelsChanged)
    def artifactCount(self) -> int:
        return self._artifact_count

    @Property(str, notify=modelsChanged)
    def selectedModelPath(self) -> str:
        return self._selected_model_path

    @Property(str, notify=modelsChanged)
    def exportStatus(self) -> str:
        return self._export_status

    @Slot(str)
    def selectModel(self, path: str) -> None:
        path = path.strip()
        if path != self._selected_model_path:
            self._selected_model_path = path
            self.modelsChanged.emit()

    @Slot(int)
    def setArtifactCount(self, count: int) -> None:
        count = max(0, count)
        if count != self._artifact_count:
            self._artifact_count = count
            self.modelsChanged.emit()


class TaskManager(QObject):
    tasksChanged = Signal()
    runningCountChanged = Signal()

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._tasks: list[dict[str, object]] = []
        self._next_id = 1

    @Property("QVariantList", notify=tasksChanged)
    def tasks(self) -> list[dict[str, object]]:
        return [dict(task) for task in self._tasks]

    @Property(int, notify=runningCountChanged)
    def runningCount(self) -> int:
        return sum(task["status"] == "running" for task in self._tasks)

    @Slot(str, str, result=int)
    def enqueueTask(self, title: str, kind: str) -> int:
        task_id = self._next_id
        self._next_id += 1
        self._tasks.append(
            {
                "id": task_id,
                "title": title.strip(),
                "kind": kind.strip(),
                "status": "queued",
            }
        )
        self.tasksChanged.emit()
        return task_id

    @Slot(int, str)
    def updateTaskStatus(self, task_id: int, status: str) -> None:
        old_running = self.runningCount
        for task in self._tasks:
            if task["id"] == task_id and task["status"] != status:
                task["status"] = status
                self.tasksChanged.emit()
                if old_running != self.runningCount:
                    self.runningCountChanged.emit()
                return


__all__ = [
    "AnnotationController",
    "DatasetController",
    "EvaluationController",
    "InferenceController",
    "ModelController",
    "ProjectController",
    "TaskManager",
    "TrainingController",
]

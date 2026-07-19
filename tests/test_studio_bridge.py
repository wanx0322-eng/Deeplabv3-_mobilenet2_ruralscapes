"""QML 外壳 <-> 真实后端 的桥接层测试。

全程离屏运行，不启动子进程：直接把 workers 的 @@JSON 协议消息喂给 WorkerBridge，
断言 controller 的状态变化。这样协议一旦改动，这里会先红。
"""
from __future__ import annotations

import json
import os
import sys

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QCoreApplication

from ruralscape_studio.controllers import (
    EvaluationController,
    InferenceController,
    TaskManager,
    TrainingController,
)
from workstation.config import DEFAULT_CONFIG, Config
from workstation.studio_bridge import (
    DatasetBackend,
    EvaluationBackend,
    TrainingBackend,
    WorkerBridge,
    _fraction,
)


@pytest.fixture(scope="module")
def qt_app():
    app = QCoreApplication.instance() or QCoreApplication(sys.argv[:1])
    yield app


@pytest.fixture
def config(tmp_path):
    return Config(path=str(tmp_path / "workstation_config.json"))


# ---------------- 进度换算 ----------------

def test_fraction_clamps_and_handles_zero():
    assert _fraction(5, 10) == 0.5
    assert _fraction(0, 0) == 0.0
    assert _fraction(20, 10) == 1.0
    assert _fraction(-5, 10) == 0.0
    assert _fraction("x", 10) == 0.0


# ---------------- 协议翻译 ----------------

def test_step_message_maps_to_global_training_progress(qt_app):
    controller = TrainingController()
    bridge = WorkerBridge(controller)
    #   第 2 个 epoch（共 4 个）跑到一半 -> (1 + 0.5) / 4 = 0.375
    bridge._on_message({"type": "step", "epoch": 2, "total_epoch": 4,
                        "step": 5, "total_step": 10})
    assert controller.progress == pytest.approx(0.375)


def test_epoch_message_maps_to_progress(qt_app):
    controller = TrainingController()
    WorkerBridge(controller)._on_message(
        {"type": "epoch", "epoch": 3, "total_epoch": 6,
         "train_loss": 0.5, "val_loss": 0.6})
    assert controller.progress == pytest.approx(0.5)


def test_progress_message_from_miou_worker(qt_app):
    controller = EvaluationController()
    WorkerBridge(controller)._on_message(
        {"type": "progress", "current": 25, "total": 50})
    assert controller.progress == pytest.approx(0.5)


def test_error_message_surfaces_on_controller(qt_app):
    controller = TrainingController()
    WorkerBridge(controller)._on_message(
        {"type": "error", "message": "权值文件不存在"})
    assert controller.errorMessage == "权值文件不存在"


def test_status_message_captures_log_dir_for_stop_flag(qt_app, tmp_path):
    controller = TrainingController()
    bridge = WorkerBridge(controller)
    bridge._on_message({"type": "status", "message": "日志目录",
                        "log_dir": str(tmp_path)})
    assert bridge.request_stop() is True
    assert (tmp_path / "stop.flag").is_file()


def test_request_stop_without_log_dir_is_a_noop(qt_app):
    assert WorkerBridge(TrainingController()).request_stop() is False


def test_log_lines_are_forwarded(qt_app):
    controller = TrainingController()
    bridge = WorkerBridge(controller)
    seen = []
    bridge.logLine.connect(seen.append)
    bridge._on_message({"type": "status", "message": "torch 2.11"})
    bridge._on_message({"type": "saved", "path": "logs/best.pth", "kind": "best"})
    bridge._on_message({"type": "done", "message": "训练结束"})
    assert any("torch 2.11" in line for line in seen)
    assert any("best.pth" in line for line in seen)
    assert any("训练结束" in line for line in seen)


# ---------------- 结束状态 ----------------

def test_clean_exit_marks_succeeded_and_full_progress(qt_app):
    controller = TrainingController()
    bridge = WorkerBridge(controller)
    controller.begin()
    bridge._on_finished(0)
    assert controller.status == "succeeded"
    assert controller.running is False
    assert controller.progress == 1.0
    assert controller.errorMessage == ""


def test_nonzero_exit_marks_failed_with_message(qt_app):
    controller = TrainingController()
    bridge = WorkerBridge(controller)
    controller.begin()
    bridge._on_finished(1)
    assert controller.status == "failed"
    assert controller.running is False
    assert "exit code 1" in controller.errorMessage


def test_worker_reported_error_survives_finish(qt_app):
    """关键回归：以前 reset() 会把失败原因一起清掉，界面上什么都看不到。"""
    controller = TrainingController()
    bridge = WorkerBridge(controller)
    controller.begin()
    bridge._on_message({"type": "error", "message": "数据集过小"})
    bridge._on_finished(1)
    assert controller.status == "failed"
    assert controller.errorMessage == "数据集过小"


def test_clean_exit_after_error_message_still_fails(qt_app):
    """worker 报了 error 但退出码是 0，仍应判定为失败。"""
    controller = EvaluationController()
    bridge = WorkerBridge(controller)
    controller.begin()
    bridge._on_message({"type": "error", "message": "权值不匹配"})
    bridge._on_finished(0)
    assert controller.status == "failed"


# ---------------- 任务记录联动 ----------------

def test_task_manager_tracks_run_lifecycle(qt_app, tmp_path):
    controller = TrainingController()
    tasks = TaskManager()
    bridge = WorkerBridge(controller, tasks)
    #   不真的起子进程，手工走一遍生命周期
    bridge._task_id = tasks.enqueueTask("训练 mobilenet", "train")
    tasks.updateTaskStatus(bridge._task_id, "running")
    assert tasks.runningCount == 1
    bridge._on_finished(0)
    assert tasks.runningCount == 0
    assert tasks.tasks[0]["status"] == "succeeded"


# ---------------- 后端装配 ----------------

def test_training_backend_refuses_invalid_config(qt_app, config):
    controller = TrainingController()
    backend = TrainingBackend(config, controller)
    config.data["train"]["backbone"] = "resnet50"
    assert backend.start() is False
    assert "backbone" in controller.errorMessage


def test_training_backend_refuses_engine_mismatch(qt_app, config):
    controller = TrainingController()
    backend = TrainingBackend(config, controller)
    controller.selectEngine("segformer_b2")           # 配置里的 backbone 是 mobilenet
    assert backend.start() is False
    assert "SegFormer" in controller.errorMessage


def test_training_backend_refuses_missing_initial_weights(qt_app, config):
    controller = TrainingController()
    backend = TrainingBackend(config, controller)
    config.data["train"]["model_path"] = "model_data/does_not_exist.pth"
    assert backend.start() is False
    assert "不存在" in controller.errorMessage


def test_evaluation_backend_refuses_missing_weights(qt_app, config):
    controller = EvaluationController()
    backend = EvaluationBackend(config, controller)
    assert backend.start("logs/nope.pth") is False
    assert "不存在" in controller.errorMessage


def test_evaluation_config_uses_project_metric_convention(qt_app, config, tmp_path):
    """写给 miou_worker 的配置必须带 remove_classes，否则背景会混进平均。"""
    controller = EvaluationController()
    #   output_dir 指向 tmp：早先这里写死了项目的 miou_out/，
    #   跑一次测试就把真实的 eval_config.json 覆盖成 pytest 临时路径。
    out_dir = tmp_path / "miou_out"
    backend = EvaluationBackend(config, controller, output_dir=str(out_dir))
    weight = tmp_path / "fake.pth"
    weight.write_bytes(b"not a real checkpoint")
    #   不真的启动子进程：只检查配置文件内容
    backend.bridge.worker.start = lambda *a, **k: None
    assert backend.start(str(weight)) is True

    written = json.loads((out_dir / "eval_config.json").read_text(encoding="utf-8"))
    assert written["remove_classes"] == [0]
    assert written["class_names"] == DEFAULT_CONFIG["dataset"]["class_names"]
    assert written["split"] == "val"


def test_evaluation_backend_does_not_touch_project_miou_out(qt_app, config, tmp_path):
    """回归：后端默认写项目目录，但任何测试都不许污染它。"""
    from workstation.config import PROJECT_ROOT

    real = os.path.join(PROJECT_ROOT, "miou_out", "eval_config.json")
    before = open(real, encoding="utf-8").read() if os.path.exists(real) else None

    controller = EvaluationController()
    backend = EvaluationBackend(config, controller,
                                output_dir=str(tmp_path / "miou_out"))
    weight = tmp_path / "fake.pth"
    weight.write_bytes(b"x")
    backend.bridge.worker.start = lambda *a, **k: None
    backend.start(str(weight))

    after = open(real, encoding="utf-8").read() if os.path.exists(real) else None
    assert after == before


def test_dataset_backend_reports_real_profile(qt_app, config):
    from ruralscape_studio.controllers import DatasetController

    controller = DatasetController()
    backend = DatasetBackend(config, controller)
    backend.useProjectDataset()
    assert controller.datasetPath.endswith("VOCdevkit")

    #   直接走同步路径，避免测试里等线程
    from ruralscape_studio.dataset import inspect_dataset

    backend._on_done(inspect_dataset(controller.datasetPath))
    assert controller.imageCount == 313
    assert controller.maskCount == 313
    assert controller.issueCount == 0
    assert controller.indexState == "ready"


def test_inference_backend_refuses_without_input(qt_app, config):
    controller = InferenceController()
    from workstation.studio_bridge import InferenceBackend

    backend = InferenceBackend(config, controller)
    assert backend.start() is False
    assert "输入" in controller.errorMessage


def test_bridge_does_not_import_qtwidgets(qt_app):
    """QML 壳的桥接层不该拖进 QtWidgets。

    历史原因：WorkerProcess 曾在 workstation/widgets.py（顶层 import QtWidgets）、
    PredictThread 曾在 pages/predict_page.py（同样）。两者已移至
    core/qt_workers.py（QtCore-only）—— 这个测试防止有人把 import 改回去。
    """
    import subprocess

    from workstation.config import PROJECT_ROOT

    result = subprocess.run(
        [sys.executable, "-c",
         "import workstation.studio_bridge, sys; "
         "print('PySide6.QtWidgets' in sys.modules)"],
        cwd=PROJECT_ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "False"


def test_bridge_does_not_import_torch(qt_app):
    """桥接层本身必须 torch-free，界面启动才能秒开。"""
    assert "torch" not in sys.modules or True  # 其它测试可能已加载 torch
    import subprocess

    from workstation.config import PROJECT_ROOT

    result = subprocess.run(
        [sys.executable, "-c",
         "import workstation.studio_bridge, sys; "
         "print('torch' in sys.modules)"],
        cwd=PROJECT_ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "False"

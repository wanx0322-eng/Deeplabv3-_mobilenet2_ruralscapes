"""把 QML 外壳（ruralscape_studio）接到本项目的真实后端。

依赖方向：workstation → ruralscape_studio，反向没有依赖。
ruralscape_studio 保持自包含，方便以后整包重新同步。

协议来自 workstation/workers/*.py，统一是 stdout 上的 "@@" + JSON 行：
  train_worker : status / step / epoch / miou / saved / error / done
  miou_worker  : status / progress / result / error / done
WorkerBridge 负责把这些消息翻译成 controller 的属性变化。
"""
import json
import os

from PySide6.QtCore import QObject, QThread, Signal, Slot

from ruralscape_studio.dataset import inspect_dataset
from workstation.config import PROJECT_ROOT
from workstation.widgets import WorkerProcess


def _fraction(current, total):
    """把 (当前, 总数) 压成 0..1，总数非正时返回 0。"""
    try:
        total = float(total)
        if total <= 0:
            return 0.0
        return max(0.0, min(1.0, float(current) / total))
    except (TypeError, ValueError):
        return 0.0


class WorkerBridge(QObject):
    """一个 controller + 一个子进程的绑定。"""

    logLine = Signal(str)
    resultReady = Signal(dict)

    def __init__(self, controller, task_manager=None, parent=None):
        super().__init__(parent)
        self.controller = controller
        self.task_manager = task_manager
        self._task_id = None
        self._failed = False
        self._log_dir = ""
        self.worker = WorkerProcess(self)
        self.worker.message.connect(self._on_message)
        self.worker.log.connect(self.logLine)
        self.worker.finished.connect(self._on_finished)

    # ---------------- 生命周期 ----------------
    def is_running(self):
        return self.worker.is_running()

    def start(self, module, config_path, task_title="", task_kind=""):
        if self.is_running():
            raise RuntimeError("已有任务在运行")
        self._failed = False
        self._log_dir = ""
        self.controller.clearError()
        self.controller.updateProgress(0.0)
        self.controller.begin()
        if self.task_manager is not None and task_title:
            self._task_id = self.task_manager.enqueueTask(task_title, task_kind)
            self.task_manager.updateTaskStatus(self._task_id, "running")
        self.worker.start(module, config_path)

    def request_stop(self):
        """写 stop.flag，让训练在当前 epoch 结束后安全停止。"""
        if self._log_dir:
            with open(os.path.join(self._log_dir, "stop.flag"), "w") as handle:
                handle.write("stop")
            return True
        return False

    def kill(self):
        self.worker.kill()

    # ---------------- 协议翻译 ----------------
    @Slot(dict)
    def _on_message(self, msg):
        msg_type = msg.get("type")

        if msg_type == "status":
            #   train_worker 在这条消息里带出日志目录，stop.flag 要写到那里
            if msg.get("log_dir"):
                self._log_dir = msg["log_dir"]
            self.logLine.emit("[状态] " + str(msg.get("message", "")))

        elif msg_type == "step":
            #   训练总进度 = (已完成 epoch + 当前 epoch 内进度) / 总 epoch
            epoch = float(msg.get("epoch", 0))
            total_epoch = float(msg.get("total_epoch", 0) or 0)
            within = _fraction(msg.get("step", 0), msg.get("total_step", 0))
            if total_epoch > 0:
                self.controller.updateProgress(
                    max(0.0, min(1.0, (epoch - 1.0 + within) / total_epoch)))

        elif msg_type == "epoch":
            self.controller.updateProgress(
                _fraction(msg.get("epoch", 0), msg.get("total_epoch", 0)))
            self.logLine.emit(
                "[epoch %s/%s] train_loss=%s val_loss=%s"
                % (msg.get("epoch"), msg.get("total_epoch"),
                   msg.get("train_loss"), msg.get("val_loss")))

        elif msg_type == "progress":
            self.controller.updateProgress(
                _fraction(msg.get("current", 0), msg.get("total", 0)))

        elif msg_type == "miou":
            self.logLine.emit("[mIoU] epoch %s -> %s"
                              % (msg.get("epoch"), msg.get("miou")))
            self.resultReady.emit(dict(msg))

        elif msg_type == "saved":
            self.logLine.emit("[保存] %s (%s)"
                              % (msg.get("path"), msg.get("kind")))

        elif msg_type == "result":
            self.resultReady.emit(dict(msg))

        elif msg_type == "error":
            self._failed = True
            self.controller.setError(str(msg.get("message", "未知错误")))

        elif msg_type == "done":
            self.logLine.emit("[完成] " + str(msg.get("message", "")))

    @Slot(int)
    def _on_finished(self, code):
        ok = code == 0 and not self._failed
        if not ok and not self.controller.errorMessage:
            self.controller.setError("子进程异常退出（exit code %d）" % code)
        self.controller.finish(ok)
        if self.task_manager is not None and self._task_id is not None:
            self.task_manager.updateTaskStatus(
                self._task_id, "succeeded" if ok else "failed")
            self._task_id = None


class _InspectThread(QThread):
    """数据集扫描放到子线程，避免 313 张图的 IO 卡住界面。"""

    done = Signal(object)
    failed = Signal(str)

    def __init__(self, root, parent=None):
        super().__init__(parent)
        self.root = root

    def run(self):
        try:
            self.done.emit(inspect_dataset(self.root))
        except Exception as exc:                      # noqa: BLE001 - 传给界面显示
            self.failed.emit(str(exc))


class DatasetBackend(QObject):
    """数据集页：选目录 + 扫描。"""

    profileReady = Signal(object)
    logLine = Signal(str)

    def __init__(self, config, controller, parent=None):
        super().__init__(parent)
        self.config = config
        self.controller = controller
        self._thread = None
        self._profile = None

    @property
    def profile(self):
        return self._profile

    @Slot(str)
    def selectDataset(self, path):
        self.controller.selectDataset(path)

    @Slot()
    def useProjectDataset(self):
        self.selectDataset(self.config.abs_path(self.config.dataset["voc_root"]))

    @Slot()
    def scan(self):
        root = self.controller.datasetPath
        if not root or (self._thread is not None and self._thread.isRunning()):
            return
        self._thread = _InspectThread(root, self)
        self._thread.done.connect(self._on_done)
        self._thread.failed.connect(self._on_failed)
        self._thread.start()

    def _on_done(self, profile):
        self._profile = profile
        self.controller.applyIndexSummary(
            profile.total_images, profile.total_masks, len(profile.issues))
        self.profileReady.emit(profile)
        self.logLine.emit(
            "扫描完成：图 %d / 掩膜 %d / 问题 %d，划分 %s，类别值 %s"
            % (profile.total_images, profile.total_masks, len(profile.issues),
               dict(profile.split_counts), list(profile.class_values)))

    def _on_failed(self, message):
        self.logLine.emit("扫描失败：" + message)


class TrainingBackend(QObject):
    """训练中心：校验配置 -> 写 train_config.json -> 拉起 train_worker。"""

    logLine = Signal(str)

    def __init__(self, config, controller, task_manager=None, parent=None):
        super().__init__(parent)
        self.config = config
        self.controller = controller
        self.bridge = WorkerBridge(controller, task_manager, self)
        self.bridge.logLine.connect(self.logLine)

    @Slot(result=bool)
    def start(self):
        if self.bridge.is_running():
            return False
        try:
            self.config.validate_or_raise()
        except ValueError as exc:
            self.controller.setError(str(exc))
            return False

        cfg = dict(self.config.train)
        #   QML 的引擎选择只区分 deeplab / segformer，具体 backbone 仍由配置决定
        if self.controller.engine == "segformer_b2" and not str(
                cfg.get("backbone", "")).startswith("segformer"):
            self.controller.setError(
                "训练中心选择了 SegFormer，但 train.backbone 是 %r。"
                "SegFormer 训练请走 tools/train_segformer.py。" % cfg.get("backbone"))
            return False

        cfg["num_classes"] = self.config.num_classes
        cfg["voc_root"] = self.config.dataset["voc_root"]
        cfg["remove_classes"] = self.config.dataset.get("remove_classes", [0])
        cfg["ema"] = self.config.train.get("ema", True)

        model_path = cfg.get("model_path")
        if model_path and not os.path.exists(self.config.abs_path(model_path)):
            self.controller.setError("初始权值不存在：%s" % model_path)
            return False

        save_dir = self.config.abs_path(cfg["save_dir"])
        os.makedirs(save_dir, exist_ok=True)
        config_path = os.path.join(save_dir, "train_config.json")
        with open(config_path, "w", encoding="utf-8") as handle:
            json.dump(cfg, handle, ensure_ascii=False, indent=2)

        self.bridge.start("workstation.workers.train_worker", config_path,
                          "训练 %s" % cfg.get("backbone", ""), "train")
        return True

    @Slot(result=bool)
    def requestStop(self):
        return self.bridge.request_stop()

    @Slot()
    def kill(self):
        self.bridge.kill()


class EvaluationBackend(QObject):
    """评估报告：写 eval_config.json -> 拉起 miou_worker。"""

    logLine = Signal(str)
    resultReady = Signal(dict)

    def __init__(self, config, controller, task_manager=None, parent=None,
                 output_dir=None):
        super().__init__(parent)
        self.config = config
        self.controller = controller
        #   评估产物目录。默认写进项目的 miou_out/，测试里指向 tmp
        #   —— 否则跑一次测试就把真实的 eval_config.json 覆盖掉了。
        self.output_dir = output_dir or os.path.join(PROJECT_ROOT, "miou_out")
        self.bridge = WorkerBridge(controller, task_manager, self)
        self.bridge.logLine.connect(self.logLine)
        self.bridge.resultReady.connect(self._on_result)

    def _on_result(self, msg):
        if msg.get("type") == "result":
            self.controller.setReportPath(
                os.path.join(self.output_dir, "mIoU.png"))
        self.resultReady.emit(msg)

    @Slot(str, result=bool)
    def start(self, model_rel_path=""):
        if self.bridge.is_running():
            return False
        rel = model_rel_path or self.config.predict.get("model_path", "")
        if not rel:
            self.controller.setError("没有选择权值文件")
            return False
        abs_path = self.config.abs_path(rel)
        if not os.path.exists(abs_path):
            self.controller.setError("权值文件不存在：%s" % rel)
            return False

        cfg = {
            "model_path": abs_path,
            "backbone": self.config.predict["backbone"],
            "num_classes": self.config.num_classes,
            "class_names": self.config.dataset["class_names"],
            "downsample_factor": self.config.predict["downsample_factor"],
            "input_shape": self.config.predict["input_shape"],
            "cuda": self.config.predict.get("cuda", True),
            "voc_root": self.config.dataset["voc_root"],
            "split": self.controller.split,
            "miou_out": self.output_dir,
            #   与 get_miou.py / 训练中评估统一：背景展示但不计入平均
            "remove_classes": self.config.dataset.get("remove_classes", [0]),
        }
        config_path = os.path.join(self.output_dir, "eval_config.json")
        os.makedirs(self.output_dir, exist_ok=True)
        with open(config_path, "w", encoding="utf-8") as handle:
            json.dump(cfg, handle, ensure_ascii=False, indent=2)

        self.bridge.start("workstation.workers.miou_worker", config_path,
                          "评估 %s @ %s" % (rel, cfg["split"]), "eval")
        return True

    @Slot()
    def kill(self):
        self.bridge.kill()


class InferenceBackend(QObject):
    """识别工作台：复用 predict_page 的 PredictThread（纯 QtCore，可在 QML 下用）。"""

    logLine = Signal(str)
    previewReady = Signal(object, object)

    def __init__(self, config, controller, task_manager=None, parent=None):
        super().__init__(parent)
        self.config = config
        self.controller = controller
        self.task_manager = task_manager
        self._engine = None
        self._thread = None
        self._task_id = None

    def _load_params(self):
        predict = self.config.predict
        model_path = self.controller.modelPath or predict["model_path"]
        return {
            "model_path": self.config.abs_path(model_path),
            "backbone": predict["backbone"],
            "num_classes": self.config.num_classes,
            "downsample_factor": predict["downsample_factor"],
            "input_shape": predict["input_shape"],
            "cuda": predict.get("cuda", True),
        }

    @Slot(result=bool)
    def start(self):
        if self._thread is not None and self._thread.isRunning():
            return False
        source = self.controller.inputPath
        if not source:
            self.controller.setError("没有选择输入图像或目录")
            return False
        params = self._load_params()
        if not os.path.exists(params["model_path"]):
            self.controller.setError("权值文件不存在：%s" % params["model_path"])
            return False

        #   torch 只在这里才被导入，界面启动阶段不受影响
        from workstation.core.engine import SegEngine
        from workstation.pages.predict_page import PredictThread

        if self._engine is None:
            self._engine = SegEngine()

        if os.path.isdir(source):
            paths = [os.path.join(source, name) for name in sorted(os.listdir(source))
                     if os.path.splitext(name)[1].lower()
                     in (".png", ".jpg", ".jpeg", ".bmp")]
            out_dir = self.controller.outputPath or self.config.abs_path(
                self.config.predict["save_dir"])
        else:
            paths, out_dir = [source], None
        if not paths:
            self.controller.setError("输入目录里没有可识别的图像")
            return False

        self.controller.clearError()
        self.controller.updateProgress(0.0)
        self.controller.begin()
        if self.task_manager is not None:
            self._task_id = self.task_manager.enqueueTask(
                "识别 %d 张" % len(paths), "predict")
            self.task_manager.updateTaskStatus(self._task_id, "running")

        self._thread = PredictThread(
            self._engine, params, paths, out_dir,
            colors=self.config.dataset["class_colors"],
            mode=self.config.predict.get("mix_type", 0),
            alpha=self.config.predict.get("blend_alpha", 0.7),
            tta=self.config.predict.get("tta", False))
        self._thread.loaded.connect(self.logLine)
        self._thread.failed.connect(self._on_failed)
        self._thread.single_done.connect(self._on_single_done)
        self._thread.batch_progress.connect(self._on_batch_progress)
        self._thread.batch_done.connect(self._on_batch_done)
        self._thread.start()
        return True

    def _finish(self, ok):
        self.controller.finish(ok)
        if self.task_manager is not None and self._task_id is not None:
            self.task_manager.updateTaskStatus(
                self._task_id, "succeeded" if ok else "failed")
            self._task_id = None

    def _on_failed(self, message):
        self.controller.setError(message)
        self._finish(False)

    def _on_single_done(self, image, mask):
        self.previewReady.emit(image, mask)
        self.logLine.emit("单张识别完成")
        self._finish(True)

    def _on_batch_progress(self, current, total, name):
        self.controller.updateProgress(_fraction(current, total))
        self.logLine.emit("[%d/%d] %s" % (current, total, name))

    def _on_batch_done(self, count, out_dir):
        self.logLine.emit("批量识别完成：%d 张 -> %s" % (count, out_dir))
        self._finish(True)


class ModelBackend(QObject):
    """模型与导出：列出可用权值。"""

    logLine = Signal(str)

    def __init__(self, config, controller, parent=None):
        super().__init__(parent)
        self.config = config
        self.controller = controller

    @Slot(result="QVariantList")
    def refresh(self):
        from workstation.core.models import scan_weights

        weights = [dict(item) for item in scan_weights()]
        self.controller.setArtifactCount(len(weights))
        self.logLine.emit("发现 %d 个权值文件" % len(weights))
        return weights


__all__ = [
    "DatasetBackend",
    "EvaluationBackend",
    "InferenceBackend",
    "ModelBackend",
    "TrainingBackend",
    "WorkerBridge",
]

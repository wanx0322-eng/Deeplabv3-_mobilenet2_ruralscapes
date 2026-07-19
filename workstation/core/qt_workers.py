"""QtCore-only 的后台执行组件，供 Widgets 工作站与 QML 壳（studio_bridge）共用。

原先 WorkerProcess 在 workstation/widgets.py、PredictThread 在
workstation/pages/predict_page.py —— 两个模块顶层都 import QtWidgets，
导致 QML 壳仅仅为了跑任务就要加载整个 QtWidgets。这里只依赖 QtCore，
torch 仍然按 engine 的约定惰性导入。widgets.py / predict_page.py 保留
同名再导出，旧的 import 路径不受影响。
"""
import json
import os
import sys

from PIL import Image
from PySide6.QtCore import QObject, QProcess, QThread, Signal

from workstation.config import PROJECT_ROOT
from workstation.core.engine import compose_view


def python_exe():
    """优先使用当前解释器（venv）"""
    return sys.executable


class WorkerProcess(QObject):
    """QProcess 包装：运行 `python -m workstation.workers.xxx --config`，
    解析 @@JSON 行发 message 信号，其余行发 log 信号。"""

    message = Signal(dict)
    log = Signal(str)
    finished = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.proc = None
        self._buffer = ""

    def is_running(self):
        return self.proc is not None and self.proc.state() != QProcess.NotRunning

    def start(self, module, config_path):
        if self.is_running():
            raise RuntimeError("已有任务在运行")
        self.proc = QProcess(self)
        self.proc.setWorkingDirectory(PROJECT_ROOT)
        self.proc.setProcessChannelMode(QProcess.MergedChannels)
        self.proc.readyReadStandardOutput.connect(self._on_output)
        self.proc.finished.connect(lambda code, _status: self.finished.emit(code))
        self.proc.start(python_exe(), ["-u", "-X", "utf8", "-m", module,
                                       "--config", config_path])

    def kill(self):
        if self.is_running():
            self.proc.kill()

    def _on_output(self):
        data = bytes(self.proc.readAllStandardOutput()).decode("utf-8", "replace")
        self._buffer += data
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            line = line.rstrip("\r")
            if not line:
                continue
            if line.startswith("@@"):
                try:
                    self.message.emit(json.loads(line[2:]))
                    continue
                except json.JSONDecodeError:
                    pass
            self.log.emit(line)


class PredictThread(QThread):
    """加载模型（如需要）并对一张或多张图片推理"""
    loaded = Signal(str)
    single_done = Signal(object, object)          # (PIL image, mask)
    batch_progress = Signal(int, int, str)
    batch_done = Signal(int, str)
    failed = Signal(str)

    def __init__(self, engine, load_params, image_paths, out_dir=None,
                 colors=None, mode=0, alpha=0.7, save_raw_mask=False, tta=False):
        super().__init__()
        self.engine = engine
        self.load_params = load_params
        self.image_paths = image_paths
        self.out_dir = out_dir
        self.colors = colors
        self.mode = mode
        self.alpha = alpha
        self.save_raw_mask = save_raw_mask
        self.tta = tta

    def run(self):
        try:
            reloaded = self.engine.load(**self.load_params)
            if reloaded:
                self.loaded.emit(
                    f"模型已加载（{self.load_params['backbone']}, "
                    f"device={self.engine.cfg['device']}）")
        except Exception as exc:
            self.failed.emit(str(exc))
            return

        if self.out_dir is None:
            # 单张
            try:
                image = Image.open(self.image_paths[0])
                mask = self.engine.predict_mask(image, tta=self.tta)
                self.single_done.emit(image.convert("RGB"), mask)
            except Exception as exc:
                self.failed.emit(f"预测失败: {exc}")
            return

        # 批量
        os.makedirs(self.out_dir, exist_ok=True)
        if self.save_raw_mask:
            os.makedirs(os.path.join(self.out_dir, "mask"), exist_ok=True)
        count = 0
        for i, path in enumerate(self.image_paths):
            try:
                image = Image.open(path)
                mask = self.engine.predict_mask(image, tta=self.tta)
                view = compose_view(image, mask, self.colors, self.mode, self.alpha)
                stem = os.path.splitext(os.path.basename(path))[0]
                view.save(os.path.join(self.out_dir, stem + ".png"))
                if self.save_raw_mask:
                    Image.fromarray(mask).save(
                        os.path.join(self.out_dir, "mask", stem + ".png"))
                count += 1
            except Exception as exc:
                self.batch_progress.emit(i + 1, len(self.image_paths),
                                         f"{os.path.basename(path)} 失败: {exc}")
                continue
            self.batch_progress.emit(i + 1, len(self.image_paths),
                                     os.path.basename(path))
        self.batch_done.emit(count, self.out_dir)


__all__ = ["PredictThread", "WorkerProcess", "python_exe"]

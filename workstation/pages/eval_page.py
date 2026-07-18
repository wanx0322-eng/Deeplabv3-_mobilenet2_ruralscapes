"""精度评估页：对验证/测试集计算 mIoU、mPA、Accuracy"""
import json
import os

from PySide6.QtWidgets import (QComboBox, QFormLayout, QGroupBox, QHBoxLayout,
                               QHeaderView, QLabel, QMessageBox,
                               QPlainTextEdit, QProgressBar, QPushButton,
                               QSplitter, QTableWidget, QTableWidgetItem,
                               QVBoxLayout, QWidget)
from PySide6.QtCore import Qt

from workstation.config import PROJECT_ROOT
from workstation.core.models import scan_weights
from workstation.widgets import StatRow, WorkerProcess


class EvalPage(QWidget):
    def __init__(self, config, parent=None):
        super().__init__(parent)
        self.config = config
        self.worker = WorkerProcess(self)
        self.worker.message.connect(self._on_message)
        self.worker.log.connect(self._append_log)
        self.worker.finished.connect(self._on_finished)
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 12, 16, 12)
        layout.setSpacing(10)
        title = QLabel("精度评估 (mIoU)")
        title.setObjectName("pageTitle")
        layout.addWidget(title)

        top = QHBoxLayout()
        form_group = QGroupBox("评估配置")
        form = QFormLayout(form_group)
        weights_row = QHBoxLayout()
        self.weights_combo = QComboBox()
        refresh = QPushButton("刷新")
        refresh.setFixedWidth(48)
        refresh.clicked.connect(self.refresh_weights)
        weights_row.addWidget(self.weights_combo, 1)
        weights_row.addWidget(refresh)
        self.backbone = QComboBox()
        self.backbone.addItems(["mobilenet", "xception",
                                "segformer-b0", "segformer-b1", "segformer-b2"])
        self.backbone.setCurrentText(self.config.predict["backbone"])
        self.downsample = QComboBox()
        self.downsample.addItems(["8", "16"])
        self.downsample.setCurrentText(str(self.config.predict["downsample_factor"]))
        self.input_size = QComboBox()
        self.input_size.setEditable(True)
        self.input_size.addItems(["256", "512"])
        self.input_size.setCurrentText(str(self.config.predict["input_shape"][0]))
        self.split_combo = QComboBox()
        self.split_combo.addItems(["val", "test", "train"])
        form.addRow("权值文件", weights_row)
        form.addRow("主干网络", self.backbone)
        form.addRow("下采样倍数", self.downsample)
        form.addRow("输入尺寸", self.input_size)
        form.addRow("评估划分", self.split_combo)
        run_row = QHBoxLayout()
        self.run_btn = QPushButton("开始评估")
        self.run_btn.setObjectName("primary")
        self.run_btn.clicked.connect(self.start_eval)
        self.stop_btn = QPushButton("终止")
        self.stop_btn.setObjectName("danger")
        self.stop_btn.clicked.connect(self.worker.kill)
        self.stop_btn.setEnabled(False)
        run_row.addWidget(self.run_btn)
        run_row.addWidget(self.stop_btn)
        form.addRow(run_row)
        top.addWidget(form_group, 1)

        summary_box = QVBoxLayout()
        self.stats = StatRow([("miou", "mIoU"), ("mpa", "mPA"),
                              ("acc", "Accuracy"), ("num", "评估图片数")])
        summary_box.addWidget(self.stats)
        self.progress = QProgressBar()
        self.progress.setFormat("生成预测 %v / %m")
        summary_box.addWidget(self.progress)
        summary_box.addStretch()
        top.addLayout(summary_box, 2)
        layout.addLayout(top)

        splitter = QSplitter(Qt.Vertical)
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["类别", "IoU (%)", "Recall/PA (%)", "Precision (%)"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        splitter.addWidget(self.table)
        self.console = QPlainTextEdit()
        self.console.setObjectName("console")
        self.console.setReadOnly(True)
        self.console.setMaximumBlockCount(2000)
        splitter.addWidget(self.console)
        splitter.setSizes([300, 160])
        layout.addWidget(splitter, 1)

        self.refresh_weights()

    def refresh_weights(self):
        current = self.weights_combo.currentData()
        self.weights_combo.clear()
        for w in scan_weights():
            if w["name"].endswith(".onnx"):
                continue
            self.weights_combo.addItem(
                f"{w['rel_path']}  ({w['size_mb']:.1f} MB)", w["rel_path"])
        saved = current or self.config.predict.get("model_path")
        if saved:
            index = self.weights_combo.findData(saved.replace("\\", "/"))
            if index >= 0:
                self.weights_combo.setCurrentIndex(index)

    def start_eval(self):
        if self.worker.is_running():
            return
        rel = self.weights_combo.currentData()
        if not rel:
            QMessageBox.warning(self, "错误", "没有可用的权值文件")
            return
        size = int(self.input_size.currentText())
        cfg = {
            "model_path": self.config.abs_path(rel),
            "backbone": self.backbone.currentText(),
            "num_classes": self.config.num_classes,
            "class_names": self.config.dataset["class_names"],
            "downsample_factor": int(self.downsample.currentText()),
            "input_shape": [size, size],
            "cuda": True,
            "voc_root": self.config.dataset["voc_root"],
            "split": self.split_combo.currentText(),
            "miou_out": "miou_out",
            #   与 get_miou.py / 训练中的评估统一：背景类展示但不计入平均
            "remove_classes": self.config.dataset.get("remove_classes", [0]),
        }
        config_path = os.path.join(PROJECT_ROOT, "miou_out", "eval_config.json")
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)
        self.console.clear()
        self.table.setRowCount(0)
        for key in ("miou", "mpa", "acc", "num"):
            self.stats.set(key, "-")
        self._append_log(f"评估 {rel} @ {cfg['split']} 集")
        self.worker.start("workstation.workers.miou_worker", config_path)
        self.run_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)

    def _on_message(self, msg):
        msg_type = msg.get("type")
        if msg_type == "progress":
            self.progress.setMaximum(msg["total"])
            self.progress.setValue(msg["current"])
        elif msg_type == "status":
            self._append_log("[状态] " + msg.get("message", ""))
        elif msg_type == "result":
            self.stats.set("miou", f"{msg['miou']}%")
            self.stats.set("mpa", f"{msg['mpa']}%")
            self.stats.set("acc", f"{msg['accuracy']}%")
            self.stats.set("num", msg["num_images"])
            rows = msg["classes"]
            self.table.setRowCount(len(rows))
            for i, row in enumerate(rows):
                self.table.setItem(i, 0, QTableWidgetItem(row["name"]))
                for col, key in ((1, "iou"), (2, "recall"), (3, "precision")):
                    value = row[key]
                    self.table.setItem(i, col, QTableWidgetItem(
                        "-" if value is None else f"{value:.2f}"))
            self._append_log(f"结果图与混淆矩阵已保存到 {msg['out_dir']}")
        elif msg_type == "error":
            self._append_log("[错误] " + msg.get("message", ""))
        elif msg_type == "done":
            self._append_log("✓ " + msg.get("message", "完成"))

    def _append_log(self, text):
        self.console.appendPlainText(text)

    def _on_finished(self, code):
        self.run_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self._append_log(f"—— 评估进程结束（退出码 {code}）——")

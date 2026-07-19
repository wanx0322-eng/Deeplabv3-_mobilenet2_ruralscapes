"""模型训练页：参数配置 + 子进程训练 + 实时曲线"""
import json
import os
import time

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox,
                               QFileDialog, QFormLayout, QGroupBox,
                               QHBoxLayout, QLabel, QLineEdit, QMessageBox,
                               QPlainTextEdit, QProgressBar, QPushButton,
                               QScrollArea, QSpinBox, QSplitter, QVBoxLayout,
                               QWidget)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from workstation.config import PROJECT_ROOT
from workstation.widgets import WorkerProcess


class LossCanvas(FigureCanvasQTAgg):
    def __init__(self):
        self.figure = Figure(figsize=(5, 3), facecolor="#262b33")
        super().__init__(self.figure)
        self.ax = self.figure.add_subplot(111)
        self.ax2 = self.ax.twinx()
        self.reset()

    def reset(self):
        self.epochs, self.train_loss, self.val_loss = [], [], []
        self.miou_epochs, self.mious = [], []
        self.redraw()

    def add_epoch(self, epoch, train_loss, val_loss):
        self.epochs.append(epoch)
        self.train_loss.append(train_loss)
        self.val_loss.append(val_loss)
        self.redraw()

    def add_miou(self, epoch, miou):
        self.miou_epochs.append(epoch)
        self.mious.append(miou)
        self.redraw()

    def redraw(self):
        for ax in (self.ax, self.ax2):
            ax.clear()
            ax.set_facecolor("#1a1d23")
            ax.tick_params(colors="#9aa3b2", labelsize=8)
            for spine in ax.spines.values():
                spine.set_color("#3a4150")
        self.ax.set_xlabel("Epoch", color="#9aa3b2", fontsize=9)
        self.ax.set_ylabel("Loss", color="#9aa3b2", fontsize=9)
        self.ax2.set_ylabel("mIoU (%)", color="#7bd88f", fontsize=9)
        if self.epochs:
            self.ax.plot(self.epochs, self.train_loss, color="#4f8cff",
                         linewidth=1.6, label="train loss")
            self.ax.plot(self.epochs, self.val_loss, color="#ff8c5f",
                         linewidth=1.6, label="val loss")
            legend = self.ax.legend(loc="upper right", fontsize=8,
                                    facecolor="#262b33", edgecolor="#3a4150")
            for text in legend.get_texts():
                text.set_color("#e8eaf0")
        if self.mious:
            self.ax2.plot(self.miou_epochs, self.mious, color="#7bd88f",
                          linewidth=1.4, linestyle="--", marker="o",
                          markersize=3, label="val mIoU")
        self.ax.grid(True, color="#2e343e", linewidth=0.6)
        self.figure.tight_layout()
        self.draw_idle()


class TrainPage(QWidget):
    def __init__(self, config, parent=None):
        super().__init__(parent)
        self.config = config
        self.worker = WorkerProcess(self)
        self.worker.message.connect(self._on_message)
        self.worker.log.connect(self._append_log)
        self.worker.finished.connect(self._on_finished)
        self._start_time = None
        self._build_ui()

    # ---------------- UI ----------------
    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 12, 16, 12)
        layout.setSpacing(10)
        title = QLabel("模型训练")
        title.setObjectName("pageTitle")
        layout.addWidget(title)

        splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(splitter, 1)

        # ---- 左：参数表单 ----
        form_widget = QWidget()
        form_layout = QVBoxLayout(form_widget)
        form_layout.setContentsMargins(0, 0, 6, 0)
        t = self.config.train

        model_group = QGroupBox("模型")
        model_form = QFormLayout(model_group)
        self.backbone = QComboBox()
        self.backbone.addItems(["mobilenet", "xception"])
        self.backbone.setCurrentText(t["backbone"])
        self.downsample = QComboBox()
        self.downsample.addItems(["8", "16"])
        self.downsample.setCurrentText(str(t["downsample_factor"]))
        self.input_size = QComboBox()
        self.input_size.setEditable(True)
        self.input_size.addItems(["256", "512"])
        self.input_size.setCurrentText(str(t["input_shape"][0]))
        path_row = QHBoxLayout()
        self.model_path = QLineEdit(t["model_path"])
        self.model_path.setPlaceholderText("留空=不加载整模型权值")
        browse = QPushButton("…")
        browse.setFixedWidth(28)
        browse.clicked.connect(self._browse_weights)
        path_row.addWidget(self.model_path)
        path_row.addWidget(browse)
        self.pretrained = QCheckBox("使用主干 ImageNet 预训练（model_path 为空时生效，需联网）")
        self.pretrained.setChecked(bool(t["pretrained"]))
        model_form.addRow("主干网络", self.backbone)
        model_form.addRow("下采样倍数", self.downsample)
        model_form.addRow("输入尺寸", self.input_size)
        model_form.addRow("初始权值", path_row)
        model_form.addRow(self.pretrained)
        form_layout.addWidget(model_group)

        epoch_group = QGroupBox("训练轮次")
        epoch_form = QFormLayout(epoch_group)
        self.init_epoch = QSpinBox(); self.init_epoch.setRange(0, 1000)
        self.init_epoch.setValue(t["init_epoch"])
        self.freeze_epoch = QSpinBox(); self.freeze_epoch.setRange(0, 1000)
        self.freeze_epoch.setValue(t["freeze_epoch"])
        self.unfreeze_epoch = QSpinBox(); self.unfreeze_epoch.setRange(1, 1000)
        self.unfreeze_epoch.setValue(t["unfreeze_epoch"])
        self.freeze_batch = QSpinBox(); self.freeze_batch.setRange(2, 128)
        self.freeze_batch.setValue(t["freeze_batch_size"])
        self.unfreeze_batch = QSpinBox(); self.unfreeze_batch.setRange(2, 128)
        self.unfreeze_batch.setValue(t["unfreeze_batch_size"])
        self.freeze_train = QCheckBox("冻结主干训练（先冻结后解冻）")
        self.freeze_train.setChecked(bool(t["freeze_train"]))
        epoch_form.addRow("起始 Epoch", self.init_epoch)
        epoch_form.addRow("冻结 Epoch", self.freeze_epoch)
        epoch_form.addRow("总 Epoch", self.unfreeze_epoch)
        epoch_form.addRow("冻结 batch_size", self.freeze_batch)
        epoch_form.addRow("解冻 batch_size", self.unfreeze_batch)
        epoch_form.addRow(self.freeze_train)
        form_layout.addWidget(epoch_group)

        optim_group = QGroupBox("优化器与学习率")
        optim_form = QFormLayout(optim_group)
        self.optimizer = QComboBox()
        self.optimizer.addItems(["sgd", "adam"])
        self.optimizer.setCurrentText(t["optimizer_type"])
        self.optimizer.currentTextChanged.connect(self._suggest_lr)
        self.init_lr = QDoubleSpinBox()
        self.init_lr.setDecimals(6); self.init_lr.setRange(1e-6, 1.0)
        self.init_lr.setSingleStep(1e-4); self.init_lr.setValue(t["init_lr"])
        self.momentum = QDoubleSpinBox()
        self.momentum.setDecimals(3); self.momentum.setRange(0, 1)
        self.momentum.setValue(t["momentum"])
        self.weight_decay = QDoubleSpinBox()
        self.weight_decay.setDecimals(6); self.weight_decay.setRange(0, 1)
        self.weight_decay.setValue(t["weight_decay"])
        self.lr_decay = QComboBox()
        self.lr_decay.addItems(["cos", "step"])
        self.lr_decay.setCurrentText(t["lr_decay_type"])
        optim_form.addRow("优化器", self.optimizer)
        optim_form.addRow("初始学习率", self.init_lr)
        optim_form.addRow("momentum", self.momentum)
        optim_form.addRow("weight_decay", self.weight_decay)
        optim_form.addRow("学习率衰减", self.lr_decay)
        form_layout.addWidget(optim_group)

        misc_group = QGroupBox("损失与其它")
        misc_form = QFormLayout(misc_group)
        self.dice = QCheckBox("Dice Loss（类别少时建议开启）")
        self.dice.setChecked(bool(t["dice_loss"]))
        self.focal = QCheckBox("Focal Loss（正负样本不平衡时）")
        self.focal.setChecked(bool(t["focal_loss"]))
        self.fp16 = QCheckBox("混合精度 fp16（省显存）")
        self.fp16.setChecked(bool(t["fp16"]))
        self.cuda = QCheckBox("使用 GPU (CUDA)")
        self.cuda.setChecked(bool(t["cuda"]))
        self.save_period = QSpinBox(); self.save_period.setRange(1, 100)
        self.save_period.setValue(t["save_period"])
        self.eval_flag = QCheckBox("训练中周期评估 mIoU")
        self.eval_flag.setChecked(bool(t["eval_flag"]))
        self.eval_period = QSpinBox(); self.eval_period.setRange(1, 100)
        self.eval_period.setValue(t["eval_period"])
        self.num_workers = QSpinBox(); self.num_workers.setRange(0, 16)
        self.num_workers.setValue(t["num_workers"])
        misc_form.addRow(self.dice)
        misc_form.addRow(self.focal)
        misc_form.addRow(self.fp16)
        misc_form.addRow(self.cuda)
        misc_form.addRow("保存周期(epoch)", self.save_period)
        misc_form.addRow(self.eval_flag)
        misc_form.addRow("评估周期(epoch)", self.eval_period)
        misc_form.addRow("数据加载线程", self.num_workers)
        form_layout.addWidget(misc_group)
        form_layout.addStretch()

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(form_widget)
        scroll.setStyleSheet("QScrollArea { border: none; }")
        splitter.addWidget(scroll)

        # ---- 右：监控 ----
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(6, 0, 0, 0)

        control_row = QHBoxLayout()
        self.start_btn = QPushButton("开始训练")
        self.start_btn.setObjectName("primary")
        self.start_btn.clicked.connect(self.start_training)
        self.stop_btn = QPushButton("停止（epoch 结束后）")
        self.stop_btn.clicked.connect(self._request_stop)
        self.stop_btn.setEnabled(False)
        self.kill_btn = QPushButton("强制终止")
        self.kill_btn.setObjectName("danger")
        self.kill_btn.clicked.connect(self._kill)
        self.kill_btn.setEnabled(False)
        self.status_label = QLabel("就绪")
        self.status_label.setObjectName("dim")
        control_row.addWidget(self.start_btn)
        control_row.addWidget(self.stop_btn)
        control_row.addWidget(self.kill_btn)
        control_row.addWidget(self.status_label, 1)
        right_layout.addLayout(control_row)

        self.epoch_bar = QProgressBar()
        self.epoch_bar.setFormat("Epoch %v / %m")
        self.step_bar = QProgressBar()
        self.step_bar.setFormat("Step %v / %m")
        right_layout.addWidget(self.epoch_bar)
        right_layout.addWidget(self.step_bar)

        self.canvas = LossCanvas()
        right_layout.addWidget(self.canvas, 2)

        self.console = QPlainTextEdit()
        self.console.setObjectName("console")
        self.console.setReadOnly(True)
        self.console.setMaximumBlockCount(3000)
        right_layout.addWidget(self.console, 1)

        splitter.addWidget(right)
        splitter.setSizes([380, 760])

    def _browse_weights(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择初始权值", PROJECT_ROOT,
                                              "权值 (*.pth *.pt)")
        if path:
            rel = os.path.relpath(path, PROJECT_ROOT)
            self.model_path.setText(rel if not rel.startswith("..") else path)

    def _suggest_lr(self, opt):
        self.init_lr.setValue(5e-4 if opt == "adam" else 7e-3)
        self.weight_decay.setValue(0.0 if opt == "adam" else 1e-4)

    # ---------------- 配置收集 ----------------
    def collect_config(self):
        size = int(self.input_size.currentText())
        cfg = {
            "cuda": self.cuda.isChecked(),
            "seed": self.config.train.get("seed", 11),
            "fp16": self.fp16.isChecked(),
            "backbone": self.backbone.currentText(),
            "pretrained": self.pretrained.isChecked(),
            "model_path": self.model_path.text().strip(),
            "downsample_factor": int(self.downsample.currentText()),
            "input_shape": [size, size],
            "init_epoch": self.init_epoch.value(),
            "freeze_epoch": self.freeze_epoch.value(),
            "freeze_batch_size": self.freeze_batch.value(),
            "unfreeze_epoch": self.unfreeze_epoch.value(),
            "unfreeze_batch_size": self.unfreeze_batch.value(),
            "freeze_train": self.freeze_train.isChecked(),
            "init_lr": self.init_lr.value(),
            "min_lr_ratio": 0.01,
            "optimizer_type": self.optimizer.currentText(),
            "momentum": self.momentum.value(),
            "weight_decay": self.weight_decay.value(),
            "lr_decay_type": self.lr_decay.currentText(),
            "save_period": self.save_period.value(),
            "save_dir": self.config.train["save_dir"],
            "eval_flag": self.eval_flag.isChecked(),
            "eval_period": self.eval_period.value(),
            "dice_loss": self.dice.isChecked(),
            "focal_loss": self.focal.isChecked(),
            "num_workers": self.num_workers.value(),
        }
        return cfg

    def _persist(self, cfg):
        self.config.train.update(cfg)
        self.config.save()

    # ---------------- 运行控制 ----------------
    def start_training(self):
        if self.worker.is_running():
            return
        cfg = self.collect_config()
        self._persist(cfg)

        #-----------------------------------------------------------------#
        #   在拉起训练子进程之前先校验配置。以前配错（比如 cls_weights 长度
        #   与类别数不一致）要等子进程跑到 one-hot 那一步才崩，日志还在另一
        #   个进程里，很难定位。
        #-----------------------------------------------------------------#
        try:
            self.config.validate_or_raise()
        except ValueError as exc:
            QMessageBox.warning(self, "配置有误", str(exc))
            return

        worker_cfg = dict(cfg)
        worker_cfg["num_classes"] = self.config.num_classes
        worker_cfg["voc_root"] = self.config.dataset["voc_root"]
        #   mIoU 评估口径（默认排除背景）与类别损失权重，
        #   界面上没有对应控件，直接从配置里带给训练子进程
        worker_cfg["remove_classes"] = self.config.dataset.get("remove_classes", [0])
        if self.config.train.get("cls_weights"):
            worker_cfg["cls_weights"] = self.config.train["cls_weights"]
        worker_cfg["ema"] = self.config.train.get("ema", True)
        worker_cfg["deterministic"] = self.config.train.get("deterministic", True)

        if cfg["model_path"] and not os.path.exists(
                self.config.abs_path(cfg["model_path"])):
            QMessageBox.warning(self, "错误", f"初始权值不存在：{cfg['model_path']}")
            return

        save_dir = self.config.abs_path(cfg["save_dir"])
        os.makedirs(save_dir, exist_ok=True)
        config_path = os.path.join(save_dir, "train_config.json")
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(worker_cfg, f, ensure_ascii=False, indent=2)

        self.console.clear()
        self.canvas.reset()
        self.epoch_bar.setMaximum(cfg["unfreeze_epoch"])
        self.epoch_bar.setValue(cfg["init_epoch"])
        self.step_bar.setValue(0)
        self._start_time = time.time()
        self._append_log(f"启动训练：num_classes={worker_cfg['num_classes']} "
                         f"backbone={cfg['backbone']} epochs={cfg['unfreeze_epoch']}")
        self.worker.start("workstation.workers.train_worker", config_path)
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.kill_btn.setEnabled(True)
        self.status_label.setText("训练中…")

    def _request_stop(self):
        save_dir = self.config.abs_path(self.config.train["save_dir"])
        with open(os.path.join(save_dir, "stop.flag"), "w") as f:
            f.write("stop")
        self._append_log("已请求停止，将在当前 epoch 结束后保存并退出…")
        self.stop_btn.setEnabled(False)

    def _kill(self):
        self.worker.kill()
        self._append_log("已强制终止训练进程")

    # ---------------- 消息处理 ----------------
    def _on_message(self, msg):
        msg_type = msg.get("type")
        if msg_type == "step":
            self.step_bar.setMaximum(msg["total_step"])
            self.step_bar.setValue(msg["step"])
            self.status_label.setText(
                f"Epoch {msg['epoch']}/{msg['total_epoch']}  "
                f"loss={msg['loss']:.4f}  f_score={msg['f_score']:.4f}  "
                f"lr={msg['lr']:.6f}")
        elif msg_type == "epoch":
            self.epoch_bar.setMaximum(msg["total_epoch"])
            self.epoch_bar.setValue(msg["epoch"])
            self.canvas.add_epoch(msg["epoch"], msg["train_loss"], msg["val_loss"])
            elapsed = time.time() - self._start_time if self._start_time else 0
            self._append_log(
                f"[Epoch {msg['epoch']}/{msg['total_epoch']}] "
                f"train_loss={msg['train_loss']:.4f} val_loss={msg['val_loss']:.4f} "
                f"lr={msg['lr']:.6f} 已用时 {elapsed/60:.1f} 分钟")
        elif msg_type == "miou":
            self.canvas.add_miou(msg["epoch"], msg["miou"])
            self._append_log(f"[评估] Epoch {msg['epoch']} 验证集 mIoU = {msg['miou']}%")
        elif msg_type == "saved":
            kind = {"best": "最优", "period": "周期"}.get(msg.get("kind"), "")
            self._append_log(f"[保存] {kind}权值 → {msg['path']}")
        elif msg_type == "status":
            self._append_log("[状态] " + msg.get("message", ""))
        elif msg_type == "error":
            self._append_log("[错误] " + msg.get("message", ""))
            if msg.get("traceback"):
                self._append_log(msg["traceback"])
        elif msg_type == "done":
            self._append_log("✓ " + msg.get("message", "完成"))

    def _append_log(self, text):
        self.console.appendPlainText(text)

    def _on_finished(self, code):
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.kill_btn.setEnabled(False)
        if code == 0:
            self.status_label.setText("训练结束")
        else:
            self.status_label.setText(f"训练进程退出（代码 {code}）")
        self._append_log(f"—— 进程结束（退出码 {code}）——")

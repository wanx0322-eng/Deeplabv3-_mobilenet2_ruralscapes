"""图像预测页：单张 / 批量语义分割"""
import os

from PIL import Image
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox,
                               QFileDialog, QFormLayout, QGroupBox,
                               QHBoxLayout, QHeaderView, QLabel, QMessageBox,
                               QProgressBar, QPushButton, QSlider, QSplitter,
                               QTableWidget, QTableWidgetItem, QVBoxLayout,
                               QWidget)

from workstation.config import PROJECT_ROOT
from workstation.core.engine import SegEngine, compose_view, mask_statistics
from workstation.core.qt_workers import PredictThread  # noqa: F401  移至 core，此处再导出兼容旧引用
from workstation.page_system import BasePage
from workstation.widgets import TitledViewer, WeightPicker, pil_to_qpixmap

IMAGE_FILTER = "图片 (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)"
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


class PredictPage(BasePage):
    def __init__(self, config, parent=None):
        super().__init__("图像预测", parent)
        self.config = config
        self.engine = SegEngine()
        self.thread = None
        self.current_image = None
        self.current_mask = None
        self.current_path = None
        self._build_ui()

    # ---------------- UI ----------------
    def _build_ui(self):
        layout = self.page_layout

        splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(splitter, 1)

        # ---- 左：参数 ----
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 6, 0)
        p = self.config.predict

        model_group = QGroupBox("模型")
        model_form = QFormLayout(model_group)
        self.weight_picker = WeightPicker()
        self.weights_combo = self.weight_picker.combo
        self.backbone = QComboBox()
        self.backbone.addItems(["mobilenet", "xception",
                                "segformer-b0", "segformer-b1", "segformer-b2"])
        self.backbone.setCurrentText(p["backbone"])
        self.backbone.setToolTip(
            "mobilenet / xception 对应 DeepLabV3+ 权重（logs_v2_* 等），\n"
            "segformer-* 对应 tools/train_segformer.py 训练的权重（logs_segformer_*）。\n"
            "主干必须与权重文件匹配。")
        self.downsample = QComboBox()
        self.downsample.addItems(["8", "16"])
        self.downsample.setCurrentText(str(p["downsample_factor"]))
        self.input_size = QComboBox()
        self.input_size.setEditable(True)
        self.input_size.addItems(["256", "512"])
        self.input_size.setCurrentText(str(p["input_shape"][0]))
        self.device_combo = QComboBox()
        self.device_combo.addItems(["GPU (CUDA)", "CPU"])
        self.device_combo.setCurrentIndex(0 if p["cuda"] else 1)
        self.tta_check = QCheckBox("高质量模式 (TTA)")
        self.tta_check.setChecked(bool(p.get("tta", False)))
        self.tta_check.setToolTip(
            "测试时增强：水平翻转 × 多尺度概率平均。\n"
            "边界更平滑、细小类别更稳，但单张耗时约 6 倍。\n"
            "批量大图或视频建议关闭。")
        model_form.addRow("权值文件", self.weight_picker)
        model_form.addRow("主干网络", self.backbone)
        model_form.addRow("下采样倍数", self.downsample)
        model_form.addRow("输入尺寸", self.input_size)
        model_form.addRow("运行设备", self.device_combo)
        model_form.addRow("", self.tta_check)
        left_layout.addWidget(model_group)

        view_group = QGroupBox("可视化")
        view_form = QFormLayout(view_group)
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["叠加混合", "仅分割图", "扣除背景"])
        self.mode_combo.setCurrentIndex(p["mix_type"])
        self.mode_combo.currentIndexChanged.connect(self._recompose)
        self.alpha_slider = QSlider(Qt.Horizontal)
        self.alpha_slider.setRange(0, 100)
        self.alpha_slider.setValue(int(p["blend_alpha"] * 100))
        self.alpha_slider.valueChanged.connect(self._recompose)
        view_form.addRow("显示方式", self.mode_combo)
        view_form.addRow("混合透明度", self.alpha_slider)
        left_layout.addWidget(view_group)

        single_group = QGroupBox("单张预测")
        single_layout = QVBoxLayout(single_group)
        open_btn = QPushButton("打开图片并预测…")
        open_btn.setObjectName("primary")
        open_btn.clicked.connect(self._open_and_predict)
        self.save_btn = QPushButton("保存当前结果…")
        self.save_btn.clicked.connect(self._save_result)
        self.save_btn.setEnabled(False)
        single_layout.addWidget(open_btn)
        single_layout.addWidget(self.save_btn)
        left_layout.addWidget(single_group)

        batch_group = QGroupBox("批量预测")
        batch_layout = QVBoxLayout(batch_group)
        batch_btn = QPushButton("选择文件夹批量预测…")
        batch_btn.clicked.connect(self._batch_predict)
        self.batch_bar = QProgressBar()
        self.batch_bar.setVisible(False)
        self.save_raw_mask = QCheckBox("批量同时保存类别索引 mask")
        self.save_raw_mask.setChecked(True)
        batch_layout.addWidget(batch_btn)
        batch_layout.addWidget(self.save_raw_mask)
        batch_layout.addWidget(self.batch_bar)
        left_layout.addWidget(batch_group)

        self.status_label = QLabel("模型将在首次预测时加载")
        self.status_label.setObjectName("dim")
        self.status_label.setWordWrap(True)
        left_layout.addWidget(self.status_label)
        left_layout.addStretch()
        splitter.addWidget(left)

        # ---- 右：结果 ----
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(6, 0, 0, 0)
        views = QHBoxLayout()
        self.view_orig = TitledViewer("原图")
        self.view_result = TitledViewer("分割结果")
        views.addWidget(self.view_orig)
        views.addWidget(self.view_result)
        right_layout.addLayout(views, 3)

        stats_group = QGroupBox("像素统计")
        stats_layout = QVBoxLayout(stats_group)
        self.stats_table = QTableWidget(0, 4)
        self.stats_table.setHorizontalHeaderLabels(["颜色", "类别", "像素数", "占比"])
        self.stats_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.stats_table.setEditTriggers(QTableWidget.NoEditTriggers)
        stats_layout.addWidget(self.stats_table)
        right_layout.addWidget(stats_group, 2)

        splitter.addWidget(right)
        splitter.setSizes([340, 800])
        self.refresh_weights()

    # ---------------- 权值扫描 ----------------
    def refresh_weights(self):
        current = self.weights_combo.currentData()
        self.weight_picker.refresh()
        saved = current or self.config.predict.get("model_path")
        if saved:
            self.weight_picker.set_current_path(saved.replace("\\", "/"))

    def refresh(self):
        self.refresh_weights()

    def has_running_task(self):
        return self._busy()

    def stop_running_task(self):
        if not self._busy():
            return False
        self.thread.requestInterruption()
        return True


    def _load_params(self):
        rel = self.weights_combo.currentData()
        if not rel:
            raise ValueError("没有可用的权值文件，请先训练或导入权值")
        size = int(self.input_size.currentText())
        return {
            "model_path": self.config.abs_path(rel),
            "backbone": self.backbone.currentText(),
            "num_classes": self.config.num_classes,
            "downsample_factor": int(self.downsample.currentText()),
            "input_shape": [size, size],
            "cuda": self.device_combo.currentIndex() == 0,
        }

    def _persist(self):
        p = self.config.predict
        p["model_path"] = self.weights_combo.currentData() or p["model_path"]
        p["backbone"] = self.backbone.currentText()
        p["downsample_factor"] = int(self.downsample.currentText())
        size = int(self.input_size.currentText())
        p["input_shape"] = [size, size]
        p["cuda"] = self.device_combo.currentIndex() == 0
        p["mix_type"] = self.mode_combo.currentIndex()
        p["blend_alpha"] = self.alpha_slider.value() / 100.0
        p["tta"] = self.tta_check.isChecked()
        self.config.save()

    def _busy(self):
        return self.thread is not None and self.thread.isRunning()

    # ---------------- 单张 ----------------
    def _open_and_predict(self):
        if self._busy():
            self.show_message("正在推理中，请稍候", "info")
            return
        path, _ = QFileDialog.getOpenFileName(self, "选择图片", PROJECT_ROOT,
                                              IMAGE_FILTER)
        if not path:
            return
        try:
            params = self._load_params()
        except ValueError as exc:
            self.show_message(str(exc), "error")
            return
        self._persist()
        self.current_path = path
        self.status_label.setText("正在加载模型…")
        self.batch_bar.setRange(0, 0)
        self.batch_bar.setVisible(True)
        self.thread = PredictThread(self.engine, params, [path],
                                    tta=self.tta_check.isChecked())
        self.thread.loaded.connect(self.status_label.setText)
        self.thread.single_done.connect(self._on_single_done)
        self.thread.failed.connect(self._on_failed)
        self.thread.start()

    def _on_single_done(self, image, mask):
        self.batch_bar.setVisible(False)
        self.batch_bar.setRange(0, 100)
        self.current_image = image
        self.current_mask = mask
        self.view_orig.viewer.set_image(pil_to_qpixmap(image))
        self._recompose()
        self._fill_stats(mask)
        self.save_btn.setEnabled(True)
        self.status_label.setText(
            f"完成：{os.path.basename(self.current_path)} "
            f"({image.width}×{image.height})")

    def _recompose(self):
        if self.current_image is None or self.current_mask is None:
            return
        view = compose_view(self.current_image, self.current_mask,
                            self.config.dataset["class_colors"],
                            self.mode_combo.currentIndex(),
                            self.alpha_slider.value() / 100.0)
        self.view_result.viewer.set_image(pil_to_qpixmap(view))

    def _fill_stats(self, mask):
        rows = mask_statistics(mask, self.config.dataset["class_names"])
        colors = self.config.dataset["class_colors"]
        self.stats_table.setRowCount(len(rows))
        for i, (name, pixels, ratio) in enumerate(rows):
            color_item = QTableWidgetItem("")
            if i < len(colors):
                color_item.setBackground(QColor(*colors[i]))
            self.stats_table.setItem(i, 0, color_item)
            self.stats_table.setItem(i, 1, QTableWidgetItem(name))
            self.stats_table.setItem(i, 2, QTableWidgetItem(f"{pixels:,}"))
            self.stats_table.setItem(i, 3, QTableWidgetItem(f"{ratio:.2f}%"))

    def _save_result(self):
        if self.current_image is None:
            return
        stem = os.path.splitext(os.path.basename(self.current_path))[0]
        default = os.path.join(
            self.config.abs_path(self.config.predict["save_dir"]),
            stem + "_seg.png")
        path, _ = QFileDialog.getSaveFileName(self, "保存结果", default,
                                              "PNG (*.png);;JPEG (*.jpg)")
        if not path:
            return
        view = compose_view(self.current_image, self.current_mask,
                            self.config.dataset["class_colors"],
                            self.mode_combo.currentIndex(),
                            self.alpha_slider.value() / 100.0)
        view.save(path)
        self.status_label.setText(f"已保存 → {path}")

    # ---------------- 批量 ----------------
    def _batch_predict(self):
        if self._busy():
            self.show_message("正在推理中，请稍候", "info")
            return
        src = QFileDialog.getExistingDirectory(self, "选择输入文件夹", PROJECT_ROOT)
        if not src:
            return
        images = [os.path.join(src, fn) for fn in sorted(os.listdir(src))
                  if os.path.splitext(fn)[1].lower() in IMAGE_EXTS]
        if not images:
            self.show_message("该文件夹中没有图片", "error")
            return
        out = QFileDialog.getExistingDirectory(
            self, "选择输出文件夹",
            self.config.abs_path(self.config.predict["save_dir"]))
        if not out:
            return
        save_mask = self.save_raw_mask.isChecked()
        try:
            params = self._load_params()
        except ValueError as exc:
            self.show_message(str(exc), "error")
            return
        self._persist()
        self.batch_bar.setVisible(True)
        self.batch_bar.setMaximum(len(images))
        self.batch_bar.setValue(0)
        self.status_label.setText(f"批量预测 {len(images)} 张…")
        self.thread = PredictThread(
            self.engine, params, images, out_dir=out,
            colors=self.config.dataset["class_colors"],
            mode=self.mode_combo.currentIndex(),
            alpha=self.alpha_slider.value() / 100.0,
            save_raw_mask=save_mask,
            tta=self.tta_check.isChecked())
        self.thread.loaded.connect(self.status_label.setText)
        self.thread.batch_progress.connect(self._on_batch_progress)
        self.thread.batch_done.connect(self._on_batch_done)
        self.thread.failed.connect(self._on_failed)
        self.thread.start()

    def _on_batch_progress(self, current, total, name):
        self.batch_bar.setValue(current)
        self.status_label.setText(f"[{current}/{total}] {name}")

    def _on_batch_done(self, count, out_dir):
        self.batch_bar.setVisible(False)
        self.status_label.setText(f"批量完成：成功 {count} 张 → {out_dir}")
        self.show_message(f"成功处理 {count} 张，输出目录：{out_dir}", "success")

    def _on_failed(self, message):
        self.batch_bar.setVisible(False)
        self.status_label.setText("失败")
        QMessageBox.critical(self, "预测失败", message)

"""Annotation automation threads, export dialog and runner boundary."""

from __future__ import annotations

from collections.abc import Callable, Iterable
import os
from typing import Any

import numpy as np
from PIL import Image
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QCheckBox, QDialog, QDialogButtonBox, QFileDialog, QHBoxLayout,
    QLabel, QLineEdit, QMessageBox, QPushButton, QRadioButton, QVBoxLayout,
    QProgressDialog,
)

from workstation.core.export import FORMATS, export_one, save_palette_png
from workstation.theme import DARK_TOKENS


class AnnotationExporter:
    def __init__(self, runner: Callable[..., Any] = export_one) -> None:
        self.runner = runner

    def export_jobs(
        self, jobs: Iterable[tuple], formats: Iterable[str], output_dir: str, **options
    ) -> list[Any]:
        return [self.runner(*job, output_dir, formats, **options) for job in jobs]


class AutoLabelThread(QThread):
    done = Signal(object)
    failed = Signal(str)
    loaded = Signal(str)
    status = Signal(str)

    def __init__(self, engine, load_params, image, sam_refine=False):
        super().__init__()
        self.engine = engine
        self.load_params = load_params
        self.image = image
        self.sam_refine = sam_refine

    def run(self):
        try:
            if self.engine.load(**self.load_params):
                self.loaded.emit("模型已加载")
            mask = self.engine.predict_mask(self.image)
            if self.sam_refine:
                #   语义模型出类别，SAM2 出边界。首次使用需下载
                #   sam2.1-hiera-tiny（约 40MB），之后走本地缓存。
                self.status.emit("SAM2 边界精修中…（首次使用需下载模型）")
                from workstation.core.sam_refine import refine_mask
                mask, used = refine_mask(self.image, mask)
                self.status.emit(f"SAM2 精修完成，采用 {used} 个区域提案")
            self.done.emit(mask)
        except Exception as exc:
            self.failed.emit(str(exc))


# ======================================================================
#  导出对话框
# ======================================================================
class ExportDialog(QDialog):
    def __init__(self, has_current, parent=None):
        super().__init__(parent)
        self.setWindowTitle("导出标签")
        self.setMinimumWidth(460)
        layout = QVBoxLayout(self)

        layout.addWidget(QLabel("导出范围："))
        self.scope_current = QRadioButton("当前图片（含未保存的修改）")
        self.scope_all = QRadioButton("数据集中所有已保存标签的图片")
        self.scope_current.setChecked(has_current)
        self.scope_all.setChecked(not has_current)
        self.scope_current.setEnabled(has_current)
        layout.addWidget(self.scope_current)
        layout.addWidget(self.scope_all)

        layout.addWidget(QLabel("导出格式（可多选）："))
        self.format_checks = {}
        for key, text in FORMATS:
            check = QCheckBox(text)
            check.setChecked(key == "palette_png")
            self.format_checks[key] = check
            layout.addWidget(check)

        dir_row = QHBoxLayout()
        self.dir_edit = QLineEdit()
        self.dir_edit.setPlaceholderText("选择输出目录…")
        browse = QPushButton("浏览…")
        browse.clicked.connect(self._browse)
        dir_row.addWidget(self.dir_edit, 1)
        dir_row.addWidget(browse)
        layout.addLayout(dir_row)
        self.validation_message = QLabel()
        self.validation_message.setObjectName("inlineMessage")
        self.validation_message.setVisible(False)
        layout.addWidget(self.validation_message)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._validate)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _browse(self):
        path = QFileDialog.getExistingDirectory(self, "选择输出目录")
        if path:
            self.dir_edit.setText(path)

    def _validate(self):
        if not self.selected_formats():
            self.validation_message.setText("请至少选择一种导出格式")
            self.validation_message.setVisible(True)
            return
        if not self.dir_edit.text().strip():
            self.validation_message.setText("请选择输出目录")
            self.validation_message.setVisible(True)
            return
        self.accept()

    def selected_formats(self):
        return [k for k, c in self.format_checks.items() if c.isChecked()]


class ExportThread(QThread):
    progress = Signal(int, int)
    done = Signal(int, str)
    failed = Signal(str)

    def __init__(self, jobs, out_dir, formats, class_names, class_colors):
        super().__init__()
        self.jobs = jobs          # [(stem, mask_or_None, mask_path, image_path)]
        self.out_dir = out_dir
        self.formats = formats
        self.class_names = class_names
        self.class_colors = class_colors

    def run(self):
        count = 0
        try:
            for i, (stem, mask, mask_path, image_path) in enumerate(self.jobs):
                if mask is None:
                    mask = np.array(Image.open(mask_path), np.uint8)
                    if mask.ndim == 3:
                        mask = mask[..., 0]
                export_one(stem, mask, image_path, self.out_dir, self.formats,
                           self.class_names, self.class_colors)
                count += 1
                self.progress.emit(i + 1, len(self.jobs))
            self.done.emit(count, self.out_dir)
        except Exception as exc:
            self.failed.emit(str(exc))


class ExportActionsMixin:
    """Save/export actions shared by the modular annotation view."""

    def save_label(self):
        if self.current_entry is None or self.canvas.mask is None:
            return False
        self.manager.ensure_dirs()
        out_path = os.path.join(
            self.manager.label_dir, self.current_entry.stem + ".png"
        )
        save_palette_png(
            self.canvas.mask, self.config.dataset["class_colors"], out_path
        )
        self.dirty = False
        self.dirty_label.setText("")
        self.info_label.setText(f"已保存标签 → {out_path}")
        row = self.list_widget.currentRow()
        if 0 <= row < len(self.entries):
            self.entries[row].label_path = out_path
            self.list_widget.item(row).setText(self.tr("已标注 ") + self.current_entry.stem)
            self.list_widget.item(row).setForeground(
                QColor(DARK_TOKENS.FEEDBACK_SUCCESS)
            )
        self.labels_changed.emit()
        return True

    def _export(self):
        has_current = self.current_entry is not None and self.canvas.mask is not None
        dialog = ExportDialog(has_current, self)
        if dialog.exec() != QDialog.Accepted:
            return
        formats = dialog.selected_formats()
        out_dir = dialog.dir_edit.text().strip()
        jobs = []
        if dialog.scope_current.isChecked():
            entry = self.current_entry
            jobs.append((entry.stem, self.canvas.get_mask(), None, entry.image_path))
        else:
            for entry in self.manager.list_entries():
                if entry.label_path:
                    jobs.append((entry.stem, None, entry.label_path, entry.image_path))
        if not jobs:
            self.show_message("没有可导出的标签", "error")
            return
        progress = QProgressDialog("正在导出…", None, 0, len(jobs), self)
        progress.setWindowTitle("导出标签")
        progress.setWindowModality(Qt.WindowModal)
        progress.show()
        self._thread = ExportThread(
            jobs,
            out_dir,
            formats,
            self.config.dataset["class_names"],
            self.config.dataset["class_colors"],
        )
        self._thread.progress.connect(lambda current, total: progress.setValue(current))
        self._thread.done.connect(
            lambda count, path: (
                progress.close(),
                self.show_message(
                    f"已导出 {count} 张标签（{len(formats)} 种格式），输出目录：{path}",
                    "success",
                ),
            )
        )
        self._thread.failed.connect(
            lambda message: (
                progress.close(),
                QMessageBox.critical(self, "导出失败", message),
            )
        )
        self._thread.start()

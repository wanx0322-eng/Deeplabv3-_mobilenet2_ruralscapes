"""Official modular annotation page."""

from __future__ import annotations

import os

import numpy as np
from PIL import Image
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QIcon, QKeySequence, QPixmap, QShortcut
from PySide6.QtWidgets import (
    QButtonGroup, QCheckBox, QComboBox, QHBoxLayout, QLabel, QLineEdit,
    QListWidget, QListWidgetItem, QMessageBox, QPushButton, QSlider,
    QSpinBox, QToolButton, QVBoxLayout, QWidget,
)

from workstation.core.dataset import DatasetManager
from workstation.core.engine import SegEngine
from workstation.core.models import scan_weights
from workstation.page_system import BasePage
from workstation.icons import ICONS
from workstation.theme import DARK_TOKENS

from .canvas import AnnotationCanvas
from .controller import AnnotationController
from .export import AutoLabelThread, ExportActionsMixin

class AnnotatePage(ExportActionsMixin, BasePage):
    labels_changed = Signal()

    def __init__(self, config, parent=None):
        super().__init__("图像标注", parent)
        self.config = config
        self.engine = SegEngine()
        self.entries = []
        self.current_entry = None
        self.dirty = False
        self._thread = None
        self._build_ui()

    @property
    def manager(self):
        return DatasetManager(self.config.voc2007_dir())

    def bind_workspace_events(self, events):
        self.labels_changed.connect(lambda: events.publish("labels"))
        events.subscribe("dataset", lambda _payload: self.refresh())

    def has_running_task(self):
        return self._thread is not None and self._thread.isRunning()

    def stop_running_task(self):
        if not self.has_running_task():
            return False
        self._thread.requestInterruption()
        self.annotation_controller.cancel_auto_annotation()
        return True


    # ---------------- UI ----------------
    def _build_ui(self):
        layout = self.page_layout

        header = QHBoxLayout()
        self.dirty_label = QLabel("")
        self.dirty_label.setObjectName("dirtyState")
        header.addWidget(self.dirty_label)
        header.addStretch()
        hint = QLabel("左键绘制 · Shift/中键拖动平移 · 滚轮缩放 · 多边形右键/双击闭合 · Esc 取消")
        hint.setObjectName("dim")
        header.addWidget(hint)
        layout.addLayout(header)

        # ---- 工具栏 ----
        toolbar = QHBoxLayout()
        toolbar.setSpacing(6)
        self.tool_group = QButtonGroup(self)
        self.tool_buttons = {}
        for key, text, tip, icon_name in (
                ("brush", "画笔", "B：按住左键涂抹当前类别", "brush"),
                ("eraser", "橡皮", "E：擦除为背景", "eraser"),
                ("polygon", "多边形", "P：单击加点，右键/双击闭合填充", "polygon"),
                ("fill", "区域填充", "F：将点击处的连通区域改为当前类别", "fill")):
            button = QToolButton()
            button.setText(text)
            button.setToolTip(tip)
            button.setCheckable(True)
            button.setIcon(ICONS.icon(icon_name))
            button.setAccessibleName(text)
            self.tool_group.addButton(button)
            self.tool_buttons[key] = button
            toolbar.addWidget(button)
            button.clicked.connect(lambda checked=False, k=key: self._set_tool(k))
        self.tool_buttons["brush"].setChecked(True)

        toolbar.addSpacing(10)
        toolbar.addWidget(QLabel("笔刷"))
        self.brush_spin = QSpinBox()
        self.brush_spin.setRange(1, 200)
        self.brush_spin.setValue(12)
        self.brush_spin.valueChanged.connect(self._on_brush_changed)
        toolbar.addWidget(self.brush_spin)

        toolbar.addSpacing(10)
        toolbar.addWidget(QLabel("类别"))
        self.class_combo = QComboBox()
        self.class_combo.setMinimumWidth(140)
        self.class_combo.currentIndexChanged.connect(self._on_class_changed)
        toolbar.addWidget(self.class_combo)

        toolbar.addSpacing(10)
        toolbar.addWidget(QLabel("不透明度"))
        self.alpha_slider = QSlider(Qt.Horizontal)
        self.alpha_slider.setRange(10, 100)
        self.alpha_slider.setValue(55)
        self.alpha_slider.setFixedWidth(90)
        self.alpha_slider.valueChanged.connect(
            lambda v: self.canvas.set_alpha(v / 100.0))
        toolbar.addWidget(self.alpha_slider)

        toolbar.addStretch()
        undo_btn = QPushButton("撤销")
        undo_btn.setIcon(ICONS.icon("arrow-counter-clockwise"))
        undo_btn.clicked.connect(lambda: self.canvas.undo())
        redo_btn = QPushButton("重做")
        redo_btn.setIcon(ICONS.icon("arrow-clockwise"))
        redo_btn.clicked.connect(lambda: self.canvas.redo())
        fit_btn = QPushButton("适应窗口")
        fit_btn.clicked.connect(lambda: self.canvas.fit_view())
        toolbar.addWidget(undo_btn)
        toolbar.addWidget(redo_btn)
        toolbar.addWidget(fit_btn)
        layout.addLayout(toolbar)

        # ---- 主体 ----
        body = QHBoxLayout()
        body.setSpacing(8)

        left = QVBoxLayout()
        self.filter_edit = QLineEdit()
        self.filter_edit.setPlaceholderText("过滤图片…")
        self.filter_edit.textChanged.connect(self._apply_filter)
        left.addWidget(self.filter_edit)
        self.list_widget = QListWidget()
        self.list_widget.currentRowChanged.connect(self._on_row_changed)
        left.addWidget(self.list_widget, 1)

        self.auto_btn = QPushButton("AI 预标注")
        self.auto_btn.setIcon(ICONS.icon("robot"))
        self.auto_btn.setToolTip("用当前模型自动生成标签，再手动修正（模型设置沿用「图像预测」页）")
        self.auto_btn.clicked.connect(self._auto_label)
        left.addWidget(self.auto_btn)
        self.sam_check = QCheckBox("SAM2 边界精修")
        self.sam_check.setChecked(True)
        self.sam_check.setToolTip(
            "预标注后用 SAM2 修整区域边界（语义模型出类别，SAM2 出边界）。\n"
            "高分辨率照片上效果明显；首次使用需联网下载模型（约 40MB）。")
        left.addWidget(self.sam_check)
        clear_btn = QPushButton("清空标签")
        clear_btn.clicked.connect(lambda: self.canvas.clear_mask())
        left.addWidget(clear_btn)
        self.save_btn = QPushButton("保存标签 (Ctrl+S)")
        self.save_btn.setIcon(ICONS.icon("floppy-disk"))
        self.save_btn.setObjectName("primary")
        self.save_btn.clicked.connect(self.save_label)
        left.addWidget(self.save_btn)
        export_btn = QPushButton("导出标签…")
        export_btn.setIcon(ICONS.icon("export"))
        export_btn.clicked.connect(self._export)
        left.addWidget(export_btn)

        left_box = QWidget()
        left_box.setLayout(left)
        left_box.setFixedWidth(230)
        body.addWidget(left_box)

        self.canvas = AnnotationCanvas()
        self.canvas.mask_edited.connect(self._mark_dirty)
        self.canvas.cursor_info.connect(self._show_cursor_info)
        body.addWidget(self.canvas, 1)
        layout.addLayout(body, 1)

        self.info_label = QLabel("就绪")
        self.info_label.setObjectName("dim")
        layout.addWidget(self.info_label)

        # 快捷键
        QShortcut(QKeySequence("Ctrl+S"), self, self.save_label)
        QShortcut(QKeySequence("Ctrl+Z"), self, self.canvas.undo)
        QShortcut(QKeySequence("Ctrl+Y"), self, self.canvas.redo)
        self.canvas_shortcuts = []
        callbacks = [
            ("B", lambda: self._select_tool_button("brush")),
            ("E", lambda: self._select_tool_button("eraser")),
            ("P", lambda: self._select_tool_button("polygon")),
            ("F", lambda: self._select_tool_button("fill")),
            ("[", lambda: self.brush_spin.setValue(
                max(1, self.brush_spin.value() - 4))),
            ("]", lambda: self.brush_spin.setValue(
                self.brush_spin.value() + 4)),
        ]
        for key, callback in callbacks:
            shortcut = QShortcut(QKeySequence(key), self.canvas)
            shortcut.setContext(Qt.WidgetWithChildrenShortcut)
            shortcut.activated.connect(callback)
            self.canvas_shortcuts.append(shortcut)

        self.refresh()

    # ---------------- 列表 ----------------
    def refresh(self):
        self._reload_classes()
        self.canvas.set_palette(self.config.dataset["class_colors"])
        selected = self.current_entry.stem if self.current_entry else None
        self.entries = self.manager.list_entries()
        self.list_widget.blockSignals(True)
        self.list_widget.clear()
        for e in self.entries:
            item = QListWidgetItem((self.tr("已标注 ") if e.label_path else self.tr("未标注 ")) + e.stem)
            item.setForeground(QColor(DARK_TOKENS.FEEDBACK_SUCCESS) if e.label_path
                               else QColor(DARK_TOKENS.CONTENT_SECONDARY))
            self.list_widget.addItem(item)
        self.list_widget.blockSignals(False)
        if selected:
            for i, e in enumerate(self.entries):
                if e.stem == selected:
                    self.list_widget.setCurrentRow(i)
                    break
        self._apply_filter()

    def _apply_filter(self):
        text = self.filter_edit.text().lower()
        for i in range(self.list_widget.count()):
            self.list_widget.item(i).setHidden(
                text not in self.entries[i].stem.lower())

    def _reload_classes(self):
        names = self.config.dataset["class_names"]
        colors = self.config.dataset["class_colors"]
        current = self.class_combo.currentIndex()
        self.class_combo.blockSignals(True)
        self.class_combo.clear()
        for i, name in enumerate(names):
            pix = QPixmap(14, 14)
            pix.fill(QColor(*colors[i]) if i < len(colors) else QColor("gray"))
            self.class_combo.addItem(QIcon(pix), f"{i}  {name}")
        self.class_combo.blockSignals(False)
        if 0 <= current < len(names):
            self.class_combo.setCurrentIndex(current)
        elif len(names) > 1:
            self.class_combo.setCurrentIndex(1)
        self.canvas.active_class = max(0, self.class_combo.currentIndex())

    # ---------------- 图片切换 ----------------
    def _confirm_discard(self):
        if not self.dirty:
            return True
        answer = QMessageBox.question(
            self, "未保存的修改", "当前标注尚未保存，是否保存？",
            QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel)
        if answer == QMessageBox.Save:
            return self.save_label()
        return answer == QMessageBox.Discard

    def _on_row_changed(self, row):
        if row < 0 or row >= len(self.entries):
            return
        entry = self.entries[row]
        if self.current_entry is not None and entry.stem == self.current_entry.stem:
            return
        if not self._confirm_discard():
            # 恢复选择
            self.list_widget.blockSignals(True)
            for i, e in enumerate(self.entries):
                if self.current_entry and e.stem == self.current_entry.stem:
                    self.list_widget.setCurrentRow(i)
                    break
            self.list_widget.blockSignals(False)
            return
        self._load_entry(entry)

    def _load_entry(self, entry):
        try:
            image = Image.open(entry.image_path)
        except Exception as exc:
            self.show_message(f"无法打开图片：{exc}", "error")
            return
        mask = None
        if entry.label_path and os.path.exists(entry.label_path):
            try:
                mask = np.array(Image.open(entry.label_path), np.uint8)
                if mask.ndim == 3:
                    mask = mask[..., 0]
            except Exception:
                mask = None
        self.current_entry = entry
        self.canvas.set_palette(self.config.dataset["class_colors"])
        self.canvas.load(image, mask)
        self.dirty = False
        self.dirty_label.setText("")
        self.info_label.setText(
            f"{entry.stem}  ({image.size[0]}×{image.size[1]})"
            + ("  已有标签" if mask is not None else "  无标签"))
        self._update_canvas_accessibility()

    # ---------------- 工具 ----------------
    def _set_tool(self, tool):
        self.canvas.tool = tool
        self.canvas.polygon_points.clear()
        self.canvas.update()
        self._update_canvas_accessibility()

    def _select_tool_button(self, tool):
        self.tool_buttons[tool].setChecked(True)
        self._set_tool(tool)

    def _on_brush_changed(self, value):
        self.canvas.brush_size = value
        self.canvas.update()

    def _on_class_changed(self, index):
        self.canvas.active_class = max(0, index)
        self._update_canvas_accessibility()

    def _mark_dirty(self):
        self.dirty = True
        self.dirty_label.setText(self.tr("● 未保存"))
        self._update_canvas_accessibility()

    def _update_canvas_accessibility(self):
        image_name = self.current_entry.stem if self.current_entry else self.tr("未选择图像")
        class_name = self.class_combo.currentText() or self.tr("未选择类别")
        dirty_state = self.tr("有未保存修改") if self.dirty else self.tr("已保存")
        self.canvas.setAccessibleDescription(
            self.tr("{image}；工具 {tool}；类别 {class_name}；{state}").format(
                image=image_name, tool=self.canvas.tool, class_name=class_name, state=dirty_state
            )
        )


    def _show_cursor_info(self, text):
        if self.current_entry:
            self.info_label.setText(f"{self.current_entry.stem}   {text}")

    # ---------------- AI 预标注 ----------------
    def _auto_label(self):
        if self.current_entry is None:
            self.show_message("请先选择一张图片", "error")
            return
        if self._thread is not None and self._thread.isRunning():
            return
        p = self.config.predict
        weights = scan_weights()
        model_rel = p.get("model_path", "")
        model_abs = self.config.abs_path(model_rel) if model_rel else ""
        if not model_abs or not os.path.exists(model_abs):
            if weights:
                model_abs = weights[0]["abs_path"]
            else:
                self.show_message(
                    "找不到可用权值，请先在图像预测页选择模型", "error"
                )
                return
        params = {
            "model_path": model_abs,
            "backbone": p["backbone"],
            "num_classes": self.config.num_classes,
            "downsample_factor": p["downsample_factor"],
            "input_shape": p["input_shape"],
            "cuda": p["cuda"],
        }
        self.auto_btn.setEnabled(False)
        self.info_label.setText("AI 预标注推理中…")
        image = Image.open(self.current_entry.image_path)
        request_id = self.annotation_controller.begin_auto_annotation()
        self._thread = AutoLabelThread(self.engine, params, image,
                                       sam_refine=self.sam_check.isChecked())
        self._thread.status.connect(self.info_label.setText)
        self._thread.done.connect(
            lambda mask, rid=request_id: self._on_auto_done(rid, mask))
        self._thread.failed.connect(
            lambda message, rid=request_id: self._on_auto_failed(rid, message))
        self._thread.start()

    def _on_auto_done(self, request_id, mask):
        if not self.annotation_controller.accept_auto_result(request_id, mask):
            return
        self.auto_btn.setEnabled(True)
        self.canvas.set_mask(self.annotation_controller.state.mask)
        self.info_label.setText("AI 预标注完成，可在此基础上手动修正")

    def _on_auto_failed(self, request_id, message):
        if not self.annotation_controller.fail_auto_annotation(request_id, message):
            return
        self.auto_btn.setEnabled(True)
        self.info_label.setText("AI 预标注失败")
        QMessageBox.critical(self, "AI 预标注失败", message)

"""图像标注页：手动画笔/多边形/区域填充 + AI 预标注 + 多格式导出"""
import os

import cv2
import numpy as np
from PIL import Image
from PySide6.QtCore import QPointF, QRectF, Qt, QThread, Signal
from PySide6.QtGui import (QColor, QIcon, QImage, QKeySequence, QPainter,
                           QPen, QPixmap, QShortcut)
from PySide6.QtWidgets import (QButtonGroup, QCheckBox, QComboBox, QDialog,
                               QDialogButtonBox, QFileDialog, QHBoxLayout,
                               QLabel, QLineEdit, QListWidget,
                               QListWidgetItem, QMessageBox, QProgressDialog,
                               QPushButton, QRadioButton, QSlider, QSpinBox,
                               QToolButton, QVBoxLayout, QWidget)

from workstation.core.dataset import DatasetManager
from workstation.core.engine import SegEngine
from workstation.core.export import FORMATS, export_one, make_palette, save_palette_png
from workstation.core.models import scan_weights
from workstation.widgets import pil_to_qpixmap

MAX_UNDO = 20

from workstation.page_system import BasePage
from workstation.theme import DARK_TOKENS

# ======================================================================
#  画布
# ======================================================================
class AnnotationCanvas(QWidget):
    mask_edited = Signal()
    cursor_info = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.ClickFocus)
        self.setCursor(Qt.CrossCursor)
        self.setObjectName("annotationCanvas")

        self.image = None            # PIL RGB
        self.pixmap = None           # 底图 QPixmap
        self.mask = None             # np.uint8 (H, W)
        self.palette = np.zeros((256, 3), np.uint8)
        self.overlay = None          # QImage RGBA
        self.alpha = 0.55

        self.scale = 1.0
        self.offset = QPointF(0, 0)

        self.tool = "brush"          # brush / eraser / polygon / fill
        self.brush_size = 12
        self.active_class = 1

        self.undo_stack = []
        self.redo_stack = []

        self._drawing = False
        self._panning = False
        self._pan_start = QPointF()
        self._last_img_pos = None
        self._cursor_pos = None      # widget 坐标
        self.polygon_points = []     # 图像坐标

    # ---------------- 数据 ----------------
    def load(self, image, mask=None):
        self.image = image.convert("RGB")
        self.pixmap = pil_to_qpixmap(self.image)
        w, h = self.image.size
        if mask is None:
            self.mask = np.zeros((h, w), np.uint8)
        else:
            if mask.shape != (h, w):
                mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
            self.mask = mask.astype(np.uint8)
        self.undo_stack.clear()
        self.redo_stack.clear()
        self.polygon_points.clear()
        self._rebuild_overlay()
        self.fit_view()

    def set_palette(self, class_colors):
        self.palette = make_palette(class_colors)
        if self.mask is not None:
            self._rebuild_overlay()
            self.update()

    def set_alpha(self, alpha):
        self.alpha = alpha
        if self.mask is not None:
            self._rebuild_overlay()
            self.update()

    def get_mask(self):
        return None if self.mask is None else self.mask.copy()

    def set_mask(self, mask, push_undo=True):
        if self.image is None:
            return
        w, h = self.image.size
        if mask.shape != (h, w):
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        if push_undo:
            self._push_undo()
        self.mask = mask.astype(np.uint8)
        self._rebuild_overlay()
        self.update()
        self.mask_edited.emit()

    def clear_mask(self):
        if self.mask is None:
            return
        self._push_undo()
        self.mask[:] = 0
        self._rebuild_overlay()
        self.update()
        self.mask_edited.emit()

    # ---------------- 撤销 ----------------
    def _push_undo(self):
        if self.mask is None:
            return
        self.undo_stack.append(self.mask.copy())
        if len(self.undo_stack) > MAX_UNDO:
            self.undo_stack.pop(0)
        self.redo_stack.clear()

    def undo(self):
        if not self.undo_stack:
            return
        self.redo_stack.append(self.mask.copy())
        self.mask = self.undo_stack.pop()
        self._rebuild_overlay()
        self.update()
        self.mask_edited.emit()

    def redo(self):
        if not self.redo_stack:
            return
        self.undo_stack.append(self.mask.copy())
        self.mask = self.redo_stack.pop()
        self._rebuild_overlay()
        self.update()
        self.mask_edited.emit()

    # ---------------- 视图 ----------------
    def fit_view(self):
        if self.pixmap is None:
            self.update()
            return
        pw, ph = self.pixmap.width(), self.pixmap.height()
        if pw == 0 or ph == 0 or self.width() < 10:
            return
        margin = 16
        self.scale = min((self.width() - margin) / pw,
                         (self.height() - margin) / ph)
        self.scale = max(self.scale, 0.05)
        self.offset = QPointF((self.width() - pw * self.scale) / 2,
                              (self.height() - ph * self.scale) / 2)
        self.update()

    def widget_to_image(self, pos):
        x = (pos.x() - self.offset.x()) / self.scale
        y = (pos.y() - self.offset.y()) / self.scale
        return x, y

    def _in_image(self, x, y):
        if self.image is None:
            return False
        w, h = self.image.size
        return 0 <= x < w and 0 <= y < h

    # ---------------- 覆盖层 ----------------
    def _rebuild_overlay(self):
        if self.mask is None:
            self.overlay = None
            return
        h, w = self.mask.shape
        rgba = np.zeros((h, w, 4), np.uint8)
        fg = self.mask > 0
        rgba[..., :3] = self.palette[self.mask]
        rgba[fg, 3] = int(self.alpha * 255)
        self.overlay = QImage(rgba.data, w, h, w * 4,
                              QImage.Format_RGBA8888).copy()

    # ---------------- 绘制操作 ----------------
    def _paint_stroke(self, x0, y0, x1, y1):
        value = 0 if self.tool == "eraser" else self.active_class
        thickness = max(1, self.brush_size)
        cv2.line(self.mask, (int(round(x0)), int(round(y0))),
                 (int(round(x1)), int(round(y1))), int(value),
                 thickness=thickness, lineType=cv2.LINE_8)
        # 端点补圆，笔触更顺滑
        cv2.circle(self.mask, (int(round(x1)), int(round(y1))),
                   max(1, thickness // 2), int(value), -1)
        self._rebuild_overlay()
        self.update()

    def _fill_region(self, x, y):
        self._push_undo()
        seed = (int(x), int(y))
        cv2.floodFill(self.mask, None, seed, int(self.active_class),
                      loDiff=0, upDiff=0)
        self._rebuild_overlay()
        self.update()
        self.mask_edited.emit()

    def _close_polygon(self):
        if len(self.polygon_points) >= 3:
            self._push_undo()
            pts = np.array([[int(round(x)), int(round(y))]
                            for x, y in self.polygon_points], np.int32)
            cv2.fillPoly(self.mask, [pts], int(self.active_class))
            self._rebuild_overlay()
            self.mask_edited.emit()
        self.polygon_points.clear()
        self.update()

    # ---------------- 事件 ----------------
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(DARK_TOKENS.SURFACE_CANVAS))
        if self.pixmap is None:
            painter.setPen(QColor(DARK_TOKENS.CONTENT_DISABLED))
            painter.drawText(self.rect(), Qt.AlignCenter,
                             "从左侧列表选择图片开始标注")
            return
        target = QRectF(self.offset.x(), self.offset.y(),
                        self.pixmap.width() * self.scale,
                        self.pixmap.height() * self.scale)
        painter.drawPixmap(target, self.pixmap,
                           QRectF(self.pixmap.rect()))
        if self.overlay is not None:
            painter.drawImage(target, self.overlay,
                              QRectF(self.overlay.rect()))

        # 进行中的多边形
        if self.polygon_points:
            pen = QPen(QColor(DARK_TOKENS.FEEDBACK_WARNING), 2)
            painter.setPen(pen)
            pts = [QPointF(x * self.scale + self.offset.x(),
                           y * self.scale + self.offset.y())
                   for x, y in self.polygon_points]
            for i in range(len(pts) - 1):
                painter.drawLine(pts[i], pts[i + 1])
            for p in pts:
                painter.drawEllipse(p, 3, 3)
            if self._cursor_pos is not None:
                painter.setPen(QPen(QColor(DARK_TOKENS.FEEDBACK_WARNING), 1, Qt.DashLine))
                painter.drawLine(pts[-1], QPointF(self._cursor_pos))

        # 画笔光标
        if (self.tool in ("brush", "eraser") and self._cursor_pos is not None
                and not self._panning):
            radius = self.brush_size * self.scale / 2
            color = QColor(DARK_TOKENS.FEEDBACK_ERROR) if self.tool == "eraser" else \
                QColor(*self.palette[self.active_class])
            painter.setPen(QPen(color, 1.5))
            painter.setBrush(Qt.NoBrush)
            painter.drawEllipse(QPointF(self._cursor_pos), radius, radius)

    def mousePressEvent(self, event):
        if self.image is None:
            return
        if event.button() == Qt.MiddleButton or (
                event.button() == Qt.LeftButton
                and event.modifiers() & Qt.ShiftModifier):
            self._panning = True
            self._pan_start = event.position() - self.offset
            self.setCursor(Qt.ClosedHandCursor)
            return
        x, y = self.widget_to_image(event.position())
        if event.button() == Qt.LeftButton:
            if self.tool in ("brush", "eraser"):
                if not self._in_image(x, y):
                    return
                self._push_undo()
                self._drawing = True
                self._last_img_pos = (x, y)
                self._paint_stroke(x, y, x, y)
            elif self.tool == "polygon":
                if self._in_image(x, y):
                    self.polygon_points.append((x, y))
                    self.update()
            elif self.tool == "fill":
                if self._in_image(x, y):
                    self._fill_region(x, y)
        elif event.button() == Qt.RightButton and self.tool == "polygon":
            self._close_polygon()

    def mouseMoveEvent(self, event):
        self._cursor_pos = event.position()
        if self._panning:
            self.offset = event.position() - self._pan_start
            self.update()
            return
        x, y = self.widget_to_image(event.position())
        if self._in_image(x, y) and self.mask is not None:
            value = self.mask[int(y), int(x)]
            self.cursor_info.emit(
                f"({int(x)}, {int(y)})  类别 {value}  缩放 {self.scale:.2f}×")
        if self._drawing and self._last_img_pos is not None:
            x0, y0 = self._last_img_pos
            xc = float(np.clip(x, 0, self.image.size[0] - 1))
            yc = float(np.clip(y, 0, self.image.size[1] - 1))
            self._paint_stroke(x0, y0, xc, yc)
            self._last_img_pos = (xc, yc)
        else:
            self.update()

    def mouseReleaseEvent(self, event):
        if self._panning and event.button() in (Qt.MiddleButton, Qt.LeftButton):
            self._panning = False
            self.setCursor(Qt.CrossCursor)
            return
        if self._drawing and event.button() == Qt.LeftButton:
            self._drawing = False
            self._last_img_pos = None
            self.mask_edited.emit()

    def mouseDoubleClickEvent(self, event):
        if self.tool == "polygon" and event.button() == Qt.LeftButton:
            self._close_polygon()

    def wheelEvent(self, event):
        if self.image is None:
            return
        factor = 1.25 if event.angleDelta().y() > 0 else 0.8
        new_scale = float(np.clip(self.scale * factor, 0.05, 40.0))
        mouse = event.position()
        ix, iy = self.widget_to_image(mouse)
        self.scale = new_scale
        self.offset = QPointF(mouse.x() - ix * self.scale,
                              mouse.y() - iy * self.scale)
        self.update()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape and self.polygon_points:
            self.polygon_points.clear()
            self.update()
        else:
            super().keyPressEvent(event)

    def resizeEvent(self, event):
        super().resizeEvent(event)


# ======================================================================
#  AI 预标注线程
# ======================================================================
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
            QMessageBox.warning(self, "提示", "请至少选择一种导出格式")
            return
        if not self.dir_edit.text().strip():
            QMessageBox.warning(self, "提示", "请选择输出目录")
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


# ======================================================================
#  标注页
# ======================================================================
class AnnotatePage(BasePage):
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
        for key, text, tip in (
                ("brush", "🖌 画笔", "B：按住左键涂抹当前类别"),
                ("eraser", "⌫ 橡皮", "E：擦除为背景"),
                ("polygon", "⬠ 多边形", "P：单击加点，右键/双击闭合填充"),
                ("fill", "▧ 区域填充", "F：将点击处的连通区域改为当前类别")):
            button = QToolButton()
            button.setText(text)
            button.setToolTip(tip)
            button.setCheckable(True)
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
        undo_btn = QPushButton("↩ 撤销")
        undo_btn.clicked.connect(lambda: self.canvas.undo())
        redo_btn = QPushButton("↪ 重做")
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

        self.auto_btn = QPushButton("🤖 AI 预标注")
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
        self.save_btn = QPushButton("💾 保存标签 (Ctrl+S)")
        self.save_btn.setObjectName("primary")
        self.save_btn.clicked.connect(self.save_label)
        left.addWidget(self.save_btn)
        export_btn = QPushButton("📤 导出标签…")
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
        for key, tool in (("B", "brush"), ("E", "eraser"),
                          ("P", "polygon"), ("F", "fill")):
            QShortcut(QKeySequence(key), self,
                      lambda t=tool: self._select_tool_button(t))
        QShortcut(QKeySequence("["), self,
                  lambda: self.brush_spin.setValue(max(1, self.brush_spin.value() - 4)))
        QShortcut(QKeySequence("]"), self,
                  lambda: self.brush_spin.setValue(self.brush_spin.value() + 4))

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
            item = QListWidgetItem(("✓ " if e.label_path else "○ ") + e.stem)
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
            QMessageBox.warning(self, "错误", f"无法打开图片：{exc}")
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

    # ---------------- 工具 ----------------
    def _set_tool(self, tool):
        self.canvas.tool = tool
        self.canvas.polygon_points.clear()
        self.canvas.update()

    def _select_tool_button(self, tool):
        self.tool_buttons[tool].setChecked(True)
        self._set_tool(tool)

    def _on_brush_changed(self, value):
        self.canvas.brush_size = value
        self.canvas.update()

    def _on_class_changed(self, index):
        self.canvas.active_class = max(0, index)

    def _mark_dirty(self):
        self.dirty = True
        self.dirty_label.setText("● 未保存")

    def _show_cursor_info(self, text):
        if self.current_entry:
            self.info_label.setText(f"{self.current_entry.stem}   {text}")

    # ---------------- AI 预标注 ----------------
    def _auto_label(self):
        if self.current_entry is None:
            QMessageBox.information(self, "提示", "请先选择一张图片")
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
                QMessageBox.warning(self, "错误",
                                    "找不到可用权值，请先在「图像预测」页选择模型")
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
        self._thread = AutoLabelThread(self.engine, params, image,
                                       sam_refine=self.sam_check.isChecked())
        self._thread.status.connect(self.info_label.setText)
        self._thread.done.connect(self._on_auto_done)
        self._thread.failed.connect(self._on_auto_failed)
        self._thread.start()

    def _on_auto_done(self, mask):
        self.auto_btn.setEnabled(True)
        self.canvas.set_mask(mask)
        self.info_label.setText("AI 预标注完成，可在此基础上手动修正")

    def _on_auto_failed(self, message):
        self.auto_btn.setEnabled(True)
        self.info_label.setText("AI 预标注失败")
        QMessageBox.critical(self, "AI 预标注失败", message)

    # ---------------- 保存 / 导出 ----------------
    def save_label(self):
        if self.current_entry is None or self.canvas.mask is None:
            return False
        self.manager.ensure_dirs()
        out_path = os.path.join(self.manager.label_dir,
                                self.current_entry.stem + ".png")
        save_palette_png(self.canvas.mask,
                         self.config.dataset["class_colors"], out_path)
        self.dirty = False
        self.dirty_label.setText("")
        self.info_label.setText(f"已保存标签 → {out_path}")
        # 更新列表状态标记
        row = self.list_widget.currentRow()
        if 0 <= row < len(self.entries):
            self.entries[row].label_path = out_path
            self.list_widget.item(row).setText("✓ " + self.current_entry.stem)
            self.list_widget.item(row).setForeground(QColor(DARK_TOKENS.FEEDBACK_SUCCESS))
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
            e = self.current_entry
            jobs.append((e.stem, self.canvas.get_mask(), None, e.image_path))
        else:
            for e in self.manager.list_entries():
                if e.label_path:
                    jobs.append((e.stem, None, e.label_path, e.image_path))
        if not jobs:
            QMessageBox.information(self, "提示", "没有可导出的标签")
            return

        progress = QProgressDialog("正在导出…", None, 0, len(jobs), self)
        progress.setWindowTitle("导出标签")
        progress.setWindowModality(Qt.WindowModal)
        progress.show()
        self._thread = ExportThread(jobs, out_dir, formats,
                                    self.config.dataset["class_names"],
                                    self.config.dataset["class_colors"])
        self._thread.progress.connect(
            lambda c, t: progress.setValue(c))
        self._thread.done.connect(
            lambda count, path: (progress.close(), QMessageBox.information(
                self, "导出完成",
                f"已导出 {count} 张标签（{len(formats)} 种格式）\n输出目录：{path}")))
        self._thread.failed.connect(
            lambda message: (progress.close(),
                             QMessageBox.critical(self, "导出失败", message)))
        self._thread.start()

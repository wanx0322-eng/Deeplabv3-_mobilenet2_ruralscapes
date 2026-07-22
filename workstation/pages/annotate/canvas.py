"""Interactive annotation canvas."""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QWidget

from workstation.theme import DARK_TOKENS
from workstation.core.export import make_palette
from workstation.widgets import pil_to_qpixmap

MAX_UNDO = 20

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
        self.setAccessibleName(self.tr("标注画布"))
        self.setAccessibleDescription(self.tr("当前图像的语义分割标注画布"))

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

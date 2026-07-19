"""通用控件与工具（QtWidgets 侧）。

WorkerProcess / python_exe 已移至 workstation/core/qt_workers.py（QtCore-only，
QML 壳也要用，不能连带 QtWidgets），这里再导出以保持旧 import 路径可用。
"""
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (QFrame, QHBoxLayout, QLabel, QSizePolicy,
                               QVBoxLayout, QWidget)

from workstation.core.qt_workers import WorkerProcess, python_exe  # noqa: F401


def pil_to_qpixmap(pil_image):
    if pil_image.mode != "RGB":
        pil_image = pil_image.convert("RGB")
    data = pil_image.tobytes("raw", "RGB")
    qimage = QImage(data, pil_image.width, pil_image.height,
                    pil_image.width * 3, QImage.Format_RGB888)
    return QPixmap.fromImage(qimage.copy())


class ImageViewer(QLabel):
    """保持纵横比缩放显示的图片框"""

    def __init__(self, placeholder="无图像", parent=None):
        super().__init__(parent)
        self._pixmap = None
        self._placeholder = placeholder
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(160, 120)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setStyleSheet(
            "background-color: #14161a; border: 1px solid #3a4150; "
            "border-radius: 6px; color: #5d6673;")
        self.setText(placeholder)

    def set_image(self, pixmap):
        self._pixmap = pixmap
        self._update_scaled()

    def clear_image(self):
        self._pixmap = None
        self.setText(self._placeholder)

    def _update_scaled(self):
        if self._pixmap is None:
            return
        self.setPixmap(self._pixmap.scaled(
            self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_scaled()


class TitledViewer(QWidget):
    """带标题的图片框"""

    def __init__(self, title, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        label = QLabel(title)
        label.setObjectName("dim")
        label.setAlignment(Qt.AlignCenter)
        self.viewer = ImageViewer()
        layout.addWidget(label)
        layout.addWidget(self.viewer, 1)


def hline():
    line = QFrame()
    line.setFrameShape(QFrame.HLine)
    line.setStyleSheet("color: #3a4150;")
    return line


class StatRow(QWidget):
    """一行统计小卡片"""

    def __init__(self, items, parent=None):
        super().__init__(parent)
        self.labels = {}
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        for key, title in items:
            card = QFrame()
            card.setStyleSheet(
                "QFrame { background: #262b33; border: 1px solid #3a4150; "
                "border-radius: 8px; }")
            v = QVBoxLayout(card)
            v.setContentsMargins(12, 8, 12, 8)
            v.setSpacing(2)
            value = QLabel("-")
            value.setStyleSheet("font-size: 20px; font-weight: bold; border: none;")
            caption = QLabel(title)
            caption.setStyleSheet("color: #9aa3b2; border: none;")
            v.addWidget(value)
            v.addWidget(caption)
            layout.addWidget(card)
            self.labels[key] = value

    def set(self, key, text):
        self.labels[key].setText(str(text))

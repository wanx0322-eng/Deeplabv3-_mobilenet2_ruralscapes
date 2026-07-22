"""Shared model-weight selector used by prediction and evaluation pages."""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QComboBox, QHBoxLayout, QPushButton, QWidget

from workstation.core.models import scan_weights


class WeightPicker(QWidget):
    weightsChanged = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.combo = QComboBox(self)
        self.combo.setMinimumWidth(180)
        self.combo.setAccessibleName(self.tr("Weight file"))
        self.refresh_button = QPushButton(self.tr("Refresh"), self)
        self.refresh_button.setAccessibleName(self.tr("Refresh weight list"))
        self.refresh_button.clicked.connect(self.refresh)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.combo, 1)
        layout.addWidget(self.refresh_button)
        self._weights: list[dict] = []

    def refresh(self) -> None:
        current = self.current_path()
        self._weights = list(scan_weights())
        self.combo.clear()
        for weight in self._weights:
            if weight["name"].endswith(".onnx"):
                continue
            value = weight.get("rel_path") or weight["abs_path"]
            label = weight.get("rel_path", weight["name"])
            if "size_mb" in weight:
                label = f"{label}  ({weight['size_mb']:.1f} MB)"
            self.combo.addItem(label, value)
        if current:
            index = self.combo.findData(current)
            if index >= 0:
                self.combo.setCurrentIndex(index)
        self.weightsChanged.emit()

    def current_path(self) -> str:
        return str(self.combo.currentData() or "")

    def set_current_path(self, path: str) -> None:
        index = self.combo.findData(path)
        if index >= 0:
            self.combo.setCurrentIndex(index)

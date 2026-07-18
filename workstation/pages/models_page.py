"""模型管理页：权值文件的查看 / 导入 / 重命名 / 删除"""
import os

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QFileDialog, QHBoxLayout, QHeaderView,
                               QInputDialog, QLabel, QMessageBox, QPushButton,
                               QTableWidget, QTableWidgetItem, QVBoxLayout,
                               QWidget)

from workstation.config import PROJECT_ROOT
from workstation.core.models import import_weight, scan_weights


class ModelsPage(QWidget):
    def __init__(self, config, parent=None):
        super().__init__(parent)
        self.config = config
        self.weights = []
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 12, 16, 12)
        layout.setSpacing(10)
        title = QLabel("模型管理")
        title.setObjectName("pageTitle")
        layout.addWidget(title)

        hint = QLabel("扫描 model_data/ 与 logs/ 目录下的权值文件。"
                      "训练产生的权值保存在 logs/，预训练与导入的权值放在 model_data/。")
        hint.setObjectName("dim")
        hint.setWordWrap(True)
        layout.addWidget(hint)

        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["文件", "位置", "大小", "修改时间"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        layout.addWidget(self.table, 1)

        buttons = QHBoxLayout()
        import_btn = QPushButton("导入权值…")
        import_btn.setObjectName("primary")
        import_btn.clicked.connect(self._import)
        rename_btn = QPushButton("重命名")
        rename_btn.clicked.connect(self._rename)
        delete_btn = QPushButton("删除")
        delete_btn.setObjectName("danger")
        delete_btn.clicked.connect(self._delete)
        open_btn = QPushButton("打开所在文件夹")
        open_btn.clicked.connect(self._open_folder)
        refresh_btn = QPushButton("刷新")
        refresh_btn.clicked.connect(self.refresh)
        buttons.addWidget(import_btn)
        buttons.addWidget(rename_btn)
        buttons.addWidget(delete_btn)
        buttons.addWidget(open_btn)
        buttons.addStretch()
        buttons.addWidget(refresh_btn)
        layout.addLayout(buttons)

        self.refresh()

    def refresh(self):
        self.weights = scan_weights()
        self.table.setRowCount(len(self.weights))
        for i, w in enumerate(self.weights):
            self.table.setItem(i, 0, QTableWidgetItem(w["name"]))
            self.table.setItem(i, 1, QTableWidgetItem(os.path.dirname(w["rel_path"]) or "."))
            size_item = QTableWidgetItem(f"{w['size_mb']:.1f} MB")
            size_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self.table.setItem(i, 2, size_item)
            self.table.setItem(i, 3, QTableWidgetItem(w["mtime_str"]))

    def _selected(self):
        rows = sorted({item.row() for item in self.table.selectedItems()})
        return [self.weights[r] for r in rows]

    def _import(self):
        files, _ = QFileDialog.getOpenFileNames(self, "选择权值文件", "",
                                                "权值 (*.pth *.pt *.onnx)")
        for f in files:
            import_weight(f)
        if files:
            self.refresh()

    def _rename(self):
        selected = self._selected()
        if len(selected) != 1:
            QMessageBox.information(self, "提示", "请选择一个文件")
            return
        w = selected[0]
        new_name, ok = QInputDialog.getText(self, "重命名", "新文件名：",
                                            text=w["name"])
        if not ok or not new_name.strip() or new_name == w["name"]:
            return
        new_path = os.path.join(os.path.dirname(w["abs_path"]), new_name.strip())
        if os.path.exists(new_path):
            QMessageBox.warning(self, "错误", "目标文件已存在")
            return
        os.rename(w["abs_path"], new_path)
        self.refresh()

    def _delete(self):
        selected = self._selected()
        if not selected:
            QMessageBox.information(self, "提示", "请先选择文件")
            return
        names = "\n".join("  " + w["rel_path"] for w in selected)
        answer = QMessageBox.question(
            self, "确认删除", f"确定删除以下权值文件？此操作不可恢复：\n{names}",
            QMessageBox.Yes | QMessageBox.No)
        if answer != QMessageBox.Yes:
            return
        for w in selected:
            try:
                os.remove(w["abs_path"])
            except OSError as exc:
                QMessageBox.warning(self, "删除失败", f"{w['name']}: {exc}")
        self.refresh()

    def _open_folder(self):
        selected = self._selected()
        target = (os.path.dirname(selected[0]["abs_path"]) if selected
                  else os.path.join(PROJECT_ROOT, "model_data"))
        os.startfile(target)

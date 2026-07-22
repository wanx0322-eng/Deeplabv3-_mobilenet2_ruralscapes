"""模型管理页：权值文件的查看 / 导入 / 重命名 / 删除"""
import os
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QFileDialog, QHBoxLayout, QHeaderView,
                               QInputDialog, QLabel, QMessageBox, QPushButton,
                               QTableWidget, QTableWidgetItem, QVBoxLayout,
                               QWidget)

from workstation.config import PROJECT_ROOT
from workstation.core.models import import_weight, scan_weights
from workstation.page_system import BasePage
from workstation.feedback import InlineMessage, ToastManager
from workstation.trash import TrashManager



class ModelsPage(BasePage):
    def __init__(self, config, parent=None):
        super().__init__("模型管理", parent)
        self.config = config
        self.weights = []
        self.trash_manager = TrashManager(
            PROJECT_ROOT, trash_root=Path(PROJECT_ROOT) / "models" / "_trash"
        )
        self.toast = ToastManager(self)
        self._last_batch = None
        self._build_ui()

    def _build_ui(self):
        layout = self.page_layout

        hint = QLabel("扫描 model_data/ 与 logs/ 目录下的权值文件。"
                      "训练产生的权值保存在 logs/，预训练与导入的权值放在 model_data/。")
        hint.setObjectName("dim")
        hint.setWordWrap(True)
        self.message = InlineMessage()
        layout.addWidget(self.message)

        layout.addWidget(hint)

        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["文件", "位置", "大小", "修改时间"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        layout.addWidget(self.table, 1)

        buttons = QHBoxLayout()
        self.undo_btn = QPushButton("撤销最近删除")
        self.undo_btn.setEnabled(False)
        self.undo_btn.clicked.connect(self._undo_delete)
        import_btn = QPushButton("导入权值…")
        import_btn.setObjectName("primary")
        import_btn.clicked.connect(self._import)
        rename_btn = QPushButton("重命名")
        rename_btn.clicked.connect(self._rename)
        delete_btn = QPushButton("删除")
        delete_btn.setObjectName("danger")
        buttons.addWidget(self.undo_btn)
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
            self.show_message("请选择一个文件", "error")
            return
        w = selected[0]
        new_name, ok = QInputDialog.getText(self, "重命名", "新文件名：",
                                            text=w["name"])
        if not ok or not new_name.strip() or new_name == w["name"]:
            return
        new_path = os.path.join(os.path.dirname(w["abs_path"]), new_name.strip())
        if os.path.exists(new_path):
            self.show_message("目标文件已存在", "error")
            return
        os.rename(w["abs_path"], new_path)
        self.refresh()

    def _delete(self):
        selected = self._selected()
        if not selected:
            self.message.setText("请先选择文件")
            self.message.setVisible(True)
            return
        names = "\n".join("  " + weight["rel_path"] for weight in selected)
        answer = QMessageBox.question(
            self,
            "移入回收站",
            f"将以下权值移入 models/_trash？完成后可撤销：\n{names}",
            QMessageBox.Yes | QMessageBox.No,
        )
        if answer != QMessageBox.Yes:
            return
        try:
            self._last_batch = self.trash_manager.move_batch(
                [weight["abs_path"] for weight in selected]
            )
        except OSError as error:
            QMessageBox.critical(self, "移动失败", str(error))
            return
        self.undo_btn.setEnabled(True)
        self.message.setText(f"已移入回收站：{len(selected)} 个文件")
        self.message.setVisible(True)
        self.toast.show(self.message.text(), "撤销")
        self.refresh()

    def _undo_delete(self):
        try:
            restored = self.trash_manager.undo_latest()
        except OSError as error:
            QMessageBox.critical(self, "恢复失败", str(error))
            return
        if not restored:
            return
        self.undo_btn.setEnabled(False)
        self.message.setText("已恢复最近一次删除")
        self.toast.show(self.message.text())
        self.refresh()

    def _open_folder(self):
        selected = self._selected()
        target = (os.path.dirname(selected[0]["abs_path"]) if selected
                  else os.path.join(PROJECT_ROOT, "model_data"))
        os.startfile(target)

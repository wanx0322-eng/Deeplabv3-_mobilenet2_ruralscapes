"""数据管理页：浏览 / 导入 / 删除 / 划分 / 类别 / 校验"""
import os

import numpy as np
from PIL import Image
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (QAbstractItemView, QColorDialog, QComboBox,
                               QDialog, QDialogButtonBox, QDoubleSpinBox,
                               QFileDialog, QFormLayout, QGroupBox,
                               QHBoxLayout, QHeaderView, QInputDialog, QLabel,
                               QLineEdit, QMenu, QMessageBox, QPlainTextEdit,
                               QProgressDialog, QPushButton, QSpinBox,
                               QSplitter, QTableWidget, QTableWidgetItem,
                               QVBoxLayout, QWidget)

from workstation.core.dataset import DatasetManager
from workstation.widgets import StatRow, TitledViewer, pil_to_qpixmap


class LabelCheckThread(QThread):
    progress = Signal(int, int)
    done = Signal(object, object)

    def __init__(self, manager, num_classes):
        super().__init__()
        self.manager = manager
        self.num_classes = num_classes

    def run(self):
        counts, problems = self.manager.check_labels(
            self.num_classes, progress_cb=lambda c, t: self.progress.emit(c, t))
        self.done.emit(counts, problems)


class SplitDialog(QDialog):
    def __init__(self, dataset_cfg, parent=None):
        super().__init__(parent)
        self.setWindowTitle("随机划分数据集")
        form = QFormLayout(self)
        self.trainval = QDoubleSpinBox()
        self.trainval.setRange(0.0, 1.0)
        self.trainval.setSingleStep(0.05)
        self.trainval.setValue(dataset_cfg.get("trainval_percent", 1.0))
        self.train = QDoubleSpinBox()
        self.train.setRange(0.0, 1.0)
        self.train.setSingleStep(0.05)
        self.train.setValue(dataset_cfg.get("train_percent", 0.7))
        self.seed = QSpinBox()
        self.seed.setRange(0, 99999)
        self.seed.setValue(dataset_cfg.get("split_seed", 0))
        form.addRow("训练+验证 占比（其余为测试集）", self.trainval)
        form.addRow("训练集在(训练+验证)中的占比", self.train)
        form.addRow("随机种子", self.seed)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)


class DatasetPage(QWidget):
    dataset_changed = Signal()

    def __init__(self, config, parent=None):
        super().__init__(parent)
        self.config = config
        self.entries = []
        self._build_ui()

    @property
    def manager(self):
        return DatasetManager(self.config.voc2007_dir())

    # ---------------- UI ----------------
    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 12, 16, 12)
        layout.setSpacing(10)

        title = QLabel("数据管理")
        title.setObjectName("pageTitle")
        layout.addWidget(title)

        self.stats = StatRow([("total", "全部图片"), ("train", "训练集"),
                              ("val", "验证集"), ("test", "测试集"),
                              ("no_label", "缺少标签")])
        layout.addWidget(self.stats)

        splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(splitter, 1)

        # 左侧：列表与操作
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)

        toolbar = QHBoxLayout()
        self.filter_edit = QLineEdit()
        self.filter_edit.setPlaceholderText("按名称过滤…")
        self.filter_edit.textChanged.connect(self._apply_filter)
        self.split_filter = QComboBox()
        self.split_filter.addItems(["全部", "train", "val", "test", "未划分"])
        self.split_filter.currentTextChanged.connect(self._apply_filter)
        toolbar.addWidget(self.filter_edit, 1)
        toolbar.addWidget(self.split_filter)
        left_layout.addLayout(toolbar)

        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["名称", "划分", "标签"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        self.table.itemSelectionChanged.connect(self._on_select)
        self.table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self._context_menu)
        left_layout.addWidget(self.table, 1)

        btn_row1 = QHBoxLayout()
        import_btn = QPushButton("导入图片…")
        import_btn.clicked.connect(self._import_images)
        import_label_btn = QPushButton("导入标签…")
        import_label_btn.clicked.connect(self._import_labels)
        delete_btn = QPushButton("删除选中")
        delete_btn.setObjectName("danger")
        delete_btn.clicked.connect(self._delete_selected)
        btn_row1.addWidget(import_btn)
        btn_row1.addWidget(import_label_btn)
        btn_row1.addWidget(delete_btn)
        left_layout.addLayout(btn_row1)

        btn_row2 = QHBoxLayout()
        split_btn = QPushButton("随机划分…")
        split_btn.setObjectName("primary")
        split_btn.clicked.connect(self._random_split)
        check_btn = QPushButton("检查数据集")
        check_btn.clicked.connect(self._check_dataset)
        refresh_btn = QPushButton("刷新")
        refresh_btn.clicked.connect(self.refresh)
        btn_row2.addWidget(split_btn)
        btn_row2.addWidget(check_btn)
        btn_row2.addWidget(refresh_btn)
        left_layout.addLayout(btn_row2)

        splitter.addWidget(left)

        # 右侧：预览 + 类别
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)

        preview_group = QGroupBox("预览")
        preview_layout = QHBoxLayout(preview_group)
        self.view_image = TitledViewer("原图")
        self.view_label = TitledViewer("标签（着色）")
        self.view_overlay = TitledViewer("叠加")
        preview_layout.addWidget(self.view_image)
        preview_layout.addWidget(self.view_label)
        preview_layout.addWidget(self.view_overlay)
        right_layout.addWidget(preview_group, 2)

        class_group = QGroupBox("类别定义（第 0 类为背景，训练/预测共用）")
        class_layout = QVBoxLayout(class_group)
        self.class_table = QTableWidget(0, 2)
        self.class_table.setHorizontalHeaderLabels(["类别名", "颜色"])
        self.class_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.class_table.verticalHeader().setVisible(True)
        self.class_table.cellDoubleClicked.connect(self._edit_class_color)
        class_layout.addWidget(self.class_table)
        class_btns = QHBoxLayout()
        add_class = QPushButton("添加类别")
        add_class.clicked.connect(self._add_class)
        remove_class = QPushButton("删除末尾类别")
        remove_class.clicked.connect(self._remove_class)
        save_class = QPushButton("保存类别设置")
        save_class.setObjectName("primary")
        save_class.clicked.connect(self._save_classes)
        class_btns.addWidget(add_class)
        class_btns.addWidget(remove_class)
        class_btns.addWidget(save_class)
        class_layout.addLayout(class_btns)
        right_layout.addWidget(class_group, 1)

        splitter.addWidget(right)
        splitter.setSizes([420, 700])

        self.refresh()

    # ---------------- 数据加载 ----------------
    def refresh(self):
        self.entries = self.manager.list_entries()
        counts = {"total": len(self.entries), "train": 0, "val": 0, "test": 0,
                  "未划分": 0, "no_label": 0}
        for e in self.entries:
            counts[e.split] = counts.get(e.split, 0) + 1
            if e.label_path is None:
                counts["no_label"] += 1
        for key in ("total", "train", "val", "test", "no_label"):
            self.stats.set(key, counts.get(key, 0))

        self.table.setRowCount(len(self.entries))
        for row, e in enumerate(self.entries):
            self.table.setItem(row, 0, QTableWidgetItem(e.stem))
            self.table.setItem(row, 1, QTableWidgetItem(e.split))
            label_item = QTableWidgetItem("✓" if e.label_path else "✗ 缺失")
            if not e.label_path:
                label_item.setForeground(QColor("#e0685f"))
            self.table.setItem(row, 2, label_item)
        self._apply_filter()
        self._load_class_table()

    def _apply_filter(self):
        text = self.filter_edit.text().lower()
        split = self.split_filter.currentText()
        for row in range(self.table.rowCount()):
            name = self.table.item(row, 0).text().lower()
            row_split = self.table.item(row, 1).text()
            visible = (text in name) and (split == "全部" or row_split == split)
            self.table.setRowHidden(row, not visible)

    # ---------------- 预览 ----------------
    def _palette(self):
        colors = self.config.dataset["class_colors"]
        palette = np.zeros((256, 3), np.uint8)
        for i, c in enumerate(colors[:256]):
            palette[i] = c
        return palette

    def _on_select(self):
        rows = sorted({item.row() for item in self.table.selectedItems()})
        if not rows:
            return
        entry = self.entries[rows[0]]
        try:
            image = Image.open(entry.image_path).convert("RGB")
            self.view_image.viewer.set_image(pil_to_qpixmap(image))
        except Exception:
            self.view_image.viewer.clear_image()
            return
        if entry.label_path and os.path.exists(entry.label_path):
            try:
                mask = np.array(Image.open(entry.label_path), np.uint8)
                if mask.ndim == 3:
                    mask = mask[..., 0]
                seg = Image.fromarray(self._palette()[mask])
                self.view_label.viewer.set_image(pil_to_qpixmap(seg))
                overlay = Image.blend(image, seg.resize(image.size, Image.NEAREST), 0.6)
                self.view_overlay.viewer.set_image(pil_to_qpixmap(overlay))
            except Exception:
                self.view_label.viewer.clear_image()
                self.view_overlay.viewer.clear_image()
        else:
            self.view_label.viewer.clear_image()
            self.view_overlay.viewer.clear_image()

    def _selected_stems(self):
        rows = sorted({item.row() for item in self.table.selectedItems()})
        return [self.entries[r].stem for r in rows]

    # ---------------- 操作 ----------------
    def _import_images(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "选择要导入的图片", "",
            "图片 (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)")
        if not files:
            return
        answer = QMessageBox.question(
            self, "匹配标签", "是否同时从某个文件夹按同名匹配导入标签 png？",
            QMessageBox.Yes | QMessageBox.No)
        label_dir = None
        if answer == QMessageBox.Yes:
            label_dir = QFileDialog.getExistingDirectory(self, "选择标签文件夹")
            if not label_dir:
                label_dir = None
        imported, labeled, skipped = self.manager.import_pairs(files, label_dir)
        msg = f"导入图片 {imported} 张"
        if label_dir:
            msg += f"，匹配到标签 {labeled} 张"
        if skipped:
            msg += f"\n跳过 {len(skipped)} 个：\n" + "\n".join(
                f"  {os.path.basename(s)}: {r}" for s, r in skipped[:10])
        QMessageBox.information(self, "导入完成", msg)
        self.refresh()
        self.dataset_changed.emit()

    def _import_labels(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "选择标签 png（需与图片同名）", "", "标签 (*.png)")
        if not files:
            return
        imported, skipped = self.manager.import_labels(files)
        msg = f"导入标签 {imported} 张"
        if skipped:
            msg += f"\n跳过 {len(skipped)} 个"
        QMessageBox.information(self, "导入完成", msg)
        self.refresh()
        self.dataset_changed.emit()

    def _delete_selected(self):
        stems = self._selected_stems()
        if not stems:
            QMessageBox.information(self, "提示", "请先在列表中选择要删除的条目")
            return
        answer = QMessageBox.question(
            self, "确认删除",
            f"将把选中的 {len(stems)} 个条目（图片+标签）移入数据集回收目录并从划分中移除，继续？",
            QMessageBox.Yes | QMessageBox.No)
        if answer != QMessageBox.Yes:
            return
        trash = self.manager.delete_entries(stems)
        QMessageBox.information(self, "已删除",
                                f"已移入回收目录：\n{trash}\n如需恢复可手动移回。")
        self.refresh()
        self.dataset_changed.emit()

    def _random_split(self):
        dialog = SplitDialog(self.config.dataset, self)
        if dialog.exec() != QDialog.Accepted:
            return
        cfg = self.config.dataset
        cfg["trainval_percent"] = dialog.trainval.value()
        cfg["train_percent"] = dialog.train.value()
        cfg["split_seed"] = dialog.seed.value()
        self.config.save()
        n_train, n_val, n_test = self.manager.random_split(
            cfg["trainval_percent"], cfg["train_percent"], cfg["split_seed"])
        QMessageBox.information(
            self, "划分完成",
            f"训练集 {n_train} 张 / 验证集 {n_val} 张 / 测试集 {n_test} 张")
        self.refresh()
        self.dataset_changed.emit()

    def _context_menu(self, pos):
        stems = self._selected_stems()
        if not stems:
            return
        menu = QMenu(self)
        for split, text in (("train", "设为训练集"), ("val", "设为验证集"),
                            ("test", "设为测试集")):
            action = menu.addAction(text)
            action.triggered.connect(
                lambda checked=False, s=split: self._assign(stems, s))
        menu.exec(self.table.viewport().mapToGlobal(pos))

    def _assign(self, stems, split):
        self.manager.assign_split(stems, split)
        self.refresh()
        self.dataset_changed.emit()

    def _check_dataset(self):
        progress = QProgressDialog("正在统计标签像素…", None, 0, 100, self)
        progress.setWindowTitle("检查数据集")
        progress.setWindowModality(Qt.WindowModal)
        progress.show()
        self._check_thread = LabelCheckThread(self.manager, self.config.num_classes)
        self._check_thread.progress.connect(
            lambda c, t: (progress.setMaximum(t), progress.setValue(c)))
        self._check_thread.done.connect(
            lambda counts, problems: (progress.close(),
                                      self._show_check_result(counts, problems)))
        self._check_thread.start()

    def _show_check_result(self, counts, problems):
        names = self.config.dataset["class_names"]
        lines = ["像素值分布：", "-" * 40]
        for value in np.nonzero(counts)[0]:
            name = names[value] if value < len(names) else ("忽略(255)" if value == 255 else "越界!")
            lines.append(f"  值 {value:>3} ({name:<14}): {counts[value]:,} 像素")
        lines.append("-" * 40)
        if problems:
            lines.append("发现问题：")
            lines.extend("  ⚠ " + p for p in problems)
        else:
            lines.append("✓ 未发现格式问题")
        dialog = QDialog(self)
        dialog.setWindowTitle("数据集检查结果")
        dialog.resize(560, 420)
        v = QVBoxLayout(dialog)
        text = QPlainTextEdit("\n".join(lines))
        text.setReadOnly(True)
        text.setObjectName("console")
        v.addWidget(text)
        dialog.exec()

    # ---------------- 类别编辑 ----------------
    def _load_class_table(self):
        names = self.config.dataset["class_names"]
        colors = self.config.dataset["class_colors"]
        self.class_table.setRowCount(len(names))
        for i, name in enumerate(names):
            self.class_table.setItem(i, 0, QTableWidgetItem(name))
            color_item = QTableWidgetItem("")
            if i < len(colors):
                color_item.setBackground(QColor(*colors[i]))
            color_item.setFlags(Qt.ItemIsEnabled)
            self.class_table.setItem(i, 1, color_item)

    def _edit_class_color(self, row, col):
        if col != 1:
            return
        colors = self.config.dataset["class_colors"]
        initial = QColor(*colors[row]) if row < len(colors) else QColor("white")
        color = QColorDialog.getColor(initial, self, "选择类别颜色")
        if color.isValid():
            self.class_table.item(row, 1).setBackground(color)

    def _add_class(self):
        name, ok = QInputDialog.getText(self, "添加类别", "类别名：")
        if not ok or not name.strip():
            return
        row = self.class_table.rowCount()
        self.class_table.insertRow(row)
        self.class_table.setItem(row, 0, QTableWidgetItem(name.strip()))
        item = QTableWidgetItem("")
        item.setBackground(QColor(128, 128, 128))
        item.setFlags(Qt.ItemIsEnabled)
        self.class_table.setItem(row, 1, item)

    def _remove_class(self):
        if self.class_table.rowCount() <= 2:
            QMessageBox.warning(self, "提示", "至少保留背景 + 1 个前景类别")
            return
        self.class_table.removeRow(self.class_table.rowCount() - 1)

    def _save_classes(self):
        names, colors = [], []
        for row in range(self.class_table.rowCount()):
            name = self.class_table.item(row, 0).text().strip()
            if not name:
                QMessageBox.warning(self, "错误", f"第 {row} 行类别名为空")
                return
            names.append(name)
            qcolor = self.class_table.item(row, 1).background().color()
            colors.append([qcolor.red(), qcolor.green(), qcolor.blue()])
        self.config.dataset["class_names"] = names
        self.config.dataset["class_colors"] = colors
        self.config.save()
        QMessageBox.information(
            self, "已保存",
            f"共 {len(names)} 个类别（num_classes={len(names)}）。\n"
            "注意：修改类别数后需要重新训练，且预测时的权值要与类别数匹配。")
        self.dataset_changed.emit()

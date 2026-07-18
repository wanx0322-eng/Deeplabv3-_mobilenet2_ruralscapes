"""主窗口与应用入口"""
import os
import sys

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (QApplication, QButtonGroup, QHBoxLayout,
                               QLabel, QMainWindow, QPushButton,
                               QStackedWidget, QVBoxLayout, QWidget)

from workstation.config import Config, PROJECT_ROOT
from workstation.theme import STYLESHEET


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.config = Config()
        self.setWindowTitle("DeepLabV3+ 语义分割工作站")
        self.resize(1280, 800)

        central = QWidget()
        root = QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        self.setCentralWidget(central)

        # ---- 侧边栏 ----
        sidebar = QWidget()
        sidebar.setObjectName("sidebar")
        sidebar.setFixedWidth(190)
        side_layout = QVBoxLayout(sidebar)
        side_layout.setContentsMargins(8, 0, 8, 12)
        side_layout.setSpacing(4)

        app_title = QLabel("DeepLabV3+")
        app_title.setObjectName("appTitle")
        subtitle = QLabel("语义分割研究工作站")
        subtitle.setObjectName("appSubtitle")
        side_layout.addWidget(app_title)
        side_layout.addWidget(subtitle)

        self.stack = QStackedWidget()
        self.nav_group = QButtonGroup(self)
        self.nav_group.setExclusive(True)

        # 延迟导入页面（页面模块较重）
        from workstation.pages.dataset_page import DatasetPage
        from workstation.pages.annotate_page import AnnotatePage
        from workstation.pages.train_page import TrainPage
        from workstation.pages.predict_page import PredictPage
        from workstation.pages.eval_page import EvalPage
        from workstation.pages.models_page import ModelsPage

        self.dataset_page = DatasetPage(self.config)
        self.annotate_page = AnnotatePage(self.config)
        self.train_page = TrainPage(self.config)
        self.predict_page = PredictPage(self.config)
        self.eval_page = EvalPage(self.config)
        self.models_page = ModelsPage(self.config)
        # 标注保存/数据集变动后互相刷新
        self.annotate_page.labels_changed.connect(self.dataset_page.refresh)
        self.dataset_page.dataset_changed.connect(self.annotate_page.refresh)

        pages = [
            ("📁  数据管理", self.dataset_page),
            ("✏️  图像标注", self.annotate_page),
            ("🎯  模型训练", self.train_page),
            ("🖼️  图像预测", self.predict_page),
            ("📊  精度评估", self.eval_page),
            ("💾  模型管理", self.models_page),
        ]
        for index, (text, page) in enumerate(pages):
            button = QPushButton(text)
            button.setObjectName("navButton")
            button.setCheckable(True)
            button.setCursor(Qt.PointingHandCursor)
            self.nav_group.addButton(button, index)
            side_layout.addWidget(button)
            self.stack.addWidget(page)
        self.nav_group.idClicked.connect(self._switch_page)
        self.nav_group.button(0).setChecked(True)
        side_layout.addStretch()

        version = QLabel("RTX · PyTorch · PySide6")
        version.setObjectName("appSubtitle")
        side_layout.addWidget(version)

        root.addWidget(sidebar)
        root.addWidget(self.stack, 1)

        # 数据集变化后，训练/预测页读取 config 即可；权值列表在切页时刷新
        self.statusBar().showMessage(f"项目目录：{PROJECT_ROOT}")
        self._show_device_info()

    def _switch_page(self, index):
        self.stack.setCurrentIndex(index)
        page = self.stack.widget(index)
        # 切换到用权值的页面时自动刷新权值列表 / 数据列表
        if hasattr(page, "refresh_weights"):
            page.refresh_weights()
        if page in (self.dataset_page, self.models_page, self.annotate_page):
            page.refresh()

    def _show_device_info(self):
        """启动时不导入 torch，用 nvidia-smi 轻量获取 GPU 名称"""
        try:
            import subprocess
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total",
                 "--format=csv,noheader"],
                capture_output=True, text=True, timeout=5,
                creationflags=subprocess.CREATE_NO_WINDOW)
            if out.returncode == 0 and out.stdout.strip():
                name, mem = out.stdout.strip().splitlines()[0].split(",")
                self.statusBar().showMessage(
                    f"GPU: {name.strip()} ({mem.strip()})   |   项目目录: {PROJECT_ROOT}")
        except Exception:
            pass

    def closeEvent(self, event):
        if self.train_page.worker.is_running():
            from PySide6.QtWidgets import QMessageBox
            answer = QMessageBox.question(
                self, "训练进行中",
                "训练任务仍在运行，退出将终止训练。确定退出？",
                QMessageBox.Yes | QMessageBox.No)
            if answer != QMessageBox.Yes:
                event.ignore()
                return
            self.train_page.worker.kill()
        if self.eval_page.worker.is_running():
            self.eval_page.worker.kill()
        self.config.save()
        event.accept()


def main():
    os.chdir(PROJECT_ROOT)
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    app = QApplication(sys.argv)
    app.setStyleSheet(STYLESHEET)
    app.setFont(QFont("Microsoft YaHei UI", 9))
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

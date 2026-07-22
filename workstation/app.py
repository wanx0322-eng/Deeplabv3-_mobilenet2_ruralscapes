"""Official Qt Widgets workstation window and application entry point."""

from __future__ import annotations

import importlib
import os
import subprocess
import sys

from PySide6.QtCore import QEvent, QSize, Qt
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from workstation.config import Config, PROJECT_ROOT
from workstation.accessibility import ensure_accessibility
from workstation.fonts import configure_application_font
from workstation.icons import ICONS
from workstation.i18n import install_translation, resolve_language
from workstation.page_system import BasePage, PageHost, PageSpec, WorkspaceEvents
from workstation.theme import DARK_TOKENS, STYLESHEET


PAGE_DEFINITIONS = (
    ("dataset", "数据管理", "folder", "Ctrl+1", "dataset_page", "DatasetPage"),
    ("annotate", "图像标注", "pencil", "Ctrl+2", "annotate_page", "AnnotatePage"),
    ("train", "模型训练", "target", "Ctrl+3", "train_page", "TrainPage"),
    ("predict", "图像预测", "image", "Ctrl+4", "predict_page", "PredictPage"),
    ("evaluate", "精度评估", "chart-bar", "Ctrl+5", "eval_page", "EvalPage"),
    ("models", "模型管理", "floppy-disk", "Ctrl+6", "models_page", "ModelsPage"),
)


class MainWindow(QMainWindow):
    def __init__(self, config: Config | None = None) -> None:
        super().__init__()
        self.config = config or Config()
        self.events = WorkspaceEvents(self)
        self.setWindowTitle(self.tr("DeepLabV3+ 语义分割工作站（正式版）"))
        self.resize(1280, 800)

        central = QWidget()
        root = QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        self.setCentralWidget(central)

        self.sidebar = QWidget()
        self.sidebar.setObjectName("sidebar")
        self.sidebar.setFixedWidth(DARK_TOKENS.SIDEBAR_WIDTH)
        side_layout = QVBoxLayout(self.sidebar)
        side_layout.setContentsMargins(
            DARK_TOKENS.SPACE_MD,
            DARK_TOKENS.SPACE_0,
            DARK_TOKENS.SPACE_MD,
            DARK_TOKENS.SPACE_XL,
        )
        side_layout.setSpacing(DARK_TOKENS.SPACE_XS)

        app_title = QLabel("DeepLabV3+")
        app_title.setObjectName("appTitle")
        subtitle = QLabel(self.tr("语义分割研究工作站 · 正式入口"))
        subtitle.setObjectName("appSubtitle")
        side_layout.addWidget(app_title)
        side_layout.addWidget(subtitle)

        self.nav_group = QButtonGroup(self)
        self.nav_group.setExclusive(True)
        self.page_specs = tuple(self._make_page_spec(item) for item in PAGE_DEFINITIONS)
        self.page_host = PageHost(self.page_specs)
        self.stack = self.page_host

        for index, spec in enumerate(self.page_specs):
            button = QPushButton(spec.title)
            button.setObjectName("navButton")
            button.setCheckable(True)
            button.setCursor(Qt.PointingHandCursor)
            button.setFocusPolicy(Qt.StrongFocus)
            button.setAccessibleName(spec.title)
            button.setAccessibleDescription(
                self.tr("切换到 {title}，快捷键 {shortcut}").format(
                    title=spec.title, shortcut=spec.shortcut
                )
            )
            button.setProperty("iconName", spec.icon)
            button.setIcon(ICONS.icon(spec.icon))
            button.setIconSize(QSize(DARK_TOKENS.ICON_MD, DARK_TOKENS.ICON_MD))
            button.installEventFilter(self)
            self.nav_group.addButton(button, index)
            side_layout.addWidget(button)
            QShortcut(
                QKeySequence(spec.shortcut),
                self,
                lambda page_index=index: self._switch_page(page_index),
            )
        self.nav_group.idClicked.connect(self._switch_page)
        self.nav_group.button(0).setChecked(True)
        side_layout.addStretch()

        version = QLabel("RTX · PyTorch · PySide6")
        version.setObjectName("appSubtitle")
        side_layout.addWidget(version)
        self._update_nav_icons(0)

        root.addWidget(self.sidebar)
        root.addWidget(self.page_host, 1)

        QShortcut(QKeySequence("F6"), self, self._toggle_focus_region)
        self.statusBar().showMessage(self.tr("项目目录：{path}").format(path=PROJECT_ROOT))
        self._show_device_info()
        ensure_accessibility(self)

    @property
    def loaded_page_count(self) -> int:
        return self.page_host.loaded_page_count

    def _make_page_spec(self, definition) -> PageSpec:
        page_id, title, icon, shortcut, module_name, class_name = definition

        def factory() -> BasePage:
            module = importlib.import_module(f"workstation.pages.{module_name}")
            page_type = getattr(module, class_name)
            page = page_type(self.config)
            page.bind_workspace_events(self.events)
            ensure_accessibility(page)
            return page

        return PageSpec(page_id, self.tr(title), icon, shortcut, factory)

    def _switch_page(self, index: int) -> None:
        self.page_host.activate(index)
        button = self.nav_group.button(index)
        if button is not None:
            button.setChecked(True)
        self._update_nav_icons(index)

    def _toggle_focus_region(self) -> None:
        focused = QApplication.focusWidget()
        nav_buttons = tuple(self.nav_group.buttons())
        if focused in nav_buttons:
            page = self.page_host.currentWidget()
            target = page
            for _ in range(256):
                target = target.nextInFocusChain()
                if target is page:
                    break
                if (
                    target.focusPolicy() != Qt.NoFocus
                    and target.isEnabled()
                    and target.isVisibleTo(page)
                ):
                    target.setFocus(Qt.ShortcutFocusReason)
                    break
        else:
            checked = self.nav_group.checkedButton() or self.nav_group.button(0)
            checked.setFocus(Qt.ShortcutFocusReason)

    def _update_nav_icons(self, active_index: int) -> None:
        for index, button in enumerate(self.nav_group.buttons()):
            color = (
                DARK_TOKENS.CONTENT_INVERSE
                if index == active_index
                else DARK_TOKENS.CONTENT_SECONDARY
            )
            button.setIcon(ICONS.icon(button.property("iconName"), color=color))

    def eventFilter(self, watched, event):
        if (
            event.type() == QEvent.KeyPress
            and watched in self.nav_group.buttons()
            and event.key() in (Qt.Key_Up, Qt.Key_Down)
        ):
            buttons = self.nav_group.buttons()
            current = buttons.index(watched)
            step = -1 if event.key() == Qt.Key_Up else 1
            target = (current + step) % len(buttons)
            buttons[target].setFocus(Qt.TabFocusReason)
            self._switch_page(target)
            return True
        return super().eventFilter(watched, event)

    def _show_device_info(self) -> None:
        """Get the GPU label without importing torch during application startup."""
        try:
            output = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=name,memory.total",
                    "--format=csv,noheader",
                ],
                capture_output=True,
                text=True,
                timeout=5,
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
            if output.returncode == 0 and output.stdout.strip():
                name, memory = output.stdout.strip().splitlines()[0].split(",")
                self.statusBar().showMessage(
                    self.tr("GPU: {name} ({memory})   |   项目目录: {path}").format(
                        name=name.strip(), memory=memory.strip(), path=PROJECT_ROOT
                    )
                )
        except Exception:
            pass

    def closeEvent(self, event) -> None:
        running_pages = [
            page
            for page in self.page_host.loaded_pages()
            if isinstance(page, BasePage) and page.has_running_task()
        ]
        if running_pages:
            answer = QMessageBox.question(
                self,
                self.tr("任务进行中"),
                self.tr("仍有任务在运行，退出将终止任务。确定退出？"),
                QMessageBox.Yes | QMessageBox.No,
            )
            if answer != QMessageBox.Yes:
                event.ignore()
                return
            for page in running_pages:
                page.stop_running_task()
        self.config.save()
        event.accept()


def main(argv=None) -> None:
    os.chdir(PROJECT_ROOT)
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )
    arguments = list(argv) if argv is not None else sys.argv
    language = resolve_language(arguments)
    application = QApplication(arguments)
    install_translation(application, language)
    application.setStyleSheet(STYLESHEET)
    window = MainWindow()
    window.show()
    raise SystemExit(application.exec())


if __name__ == "__main__":
    main()

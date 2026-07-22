from __future__ import annotations

import os

import pytest


def test_f6_and_shortcuts_reach_training_preflight_without_mouse(qapp, tmp_path) -> None:
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest
    from workstation.app import MainWindow
    from workstation.config import Config

    config = Config(path=str(tmp_path / "workstation_config.json"))
    config.train["model_path"] = "definitely-missing.pth"
    window = MainWindow(config)
    window.show()
    qapp.processEvents()

    first_nav = window.nav_group.button(0)
    first_nav.setFocus()
    QTest.keyClick(first_nav, Qt.Key_F6)
    qapp.processEvents()
    assert qapp.focusWidget() not in window.nav_group.buttons()
    assert window.page_host.currentWidget().isAncestorOf(qapp.focusWidget())

    QTest.keyClick(window, Qt.Key_3, Qt.ControlModifier)
    qapp.processEvents()
    training = window.page_host.currentWidget()
    training.start_btn.setFocus()
    QTest.keyClick(training.start_btn, Qt.Key_Space)
    qapp.processEvents()
    assert training.config_message.isVisibleTo(training)
    assert "model_path" in training.config_message.text()
    assert not training.worker.is_running()
    window.close()


@pytest.fixture
def qapp():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])

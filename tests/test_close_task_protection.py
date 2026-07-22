from __future__ import annotations

import os

import pytest


def test_close_event_refuses_then_stops_running_pages(qapp, tmp_path, monkeypatch) -> None:
    from PySide6.QtGui import QCloseEvent
    from PySide6.QtWidgets import QMessageBox
    from workstation.app import MainWindow
    from workstation.config import Config

    window = MainWindow(Config(path=str(tmp_path / "workstation_config.json")))
    page = window.page_host.currentWidget()
    stopped: list[bool] = []
    monkeypatch.setattr(page, "has_running_task", lambda: True)
    monkeypatch.setattr(page, "stop_running_task", lambda: stopped.append(True) or True)

    monkeypatch.setattr(QMessageBox, "question", lambda *_args, **_kwargs: QMessageBox.No)
    refused = QCloseEvent()
    window.closeEvent(refused)
    assert refused.isAccepted() is False
    assert stopped == []

    monkeypatch.setattr(QMessageBox, "question", lambda *_args, **_kwargs: QMessageBox.Yes)
    accepted = QCloseEvent()
    window.closeEvent(accepted)
    assert accepted.isAccepted() is True
    assert stopped == [True]


@pytest.fixture
def qapp():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])

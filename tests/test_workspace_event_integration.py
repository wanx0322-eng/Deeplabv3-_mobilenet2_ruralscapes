from __future__ import annotations

import os

import pytest


def test_dataset_and_label_events_cross_lazy_page_boundary(qapp, tmp_path) -> None:
    from workstation.app import MainWindow
    from workstation.config import Config

    window = MainWindow(Config(path=str(tmp_path / "workstation_config.json")))
    dataset = window.page_host.currentWidget()
    dataset.dataset_changed.emit()
    assert window.events._latest["dataset"] is None

    annotation = window.page_host.activate("annotate")
    annotation_refreshes: list[bool] = []
    annotation.refresh = lambda: annotation_refreshes.append(True)
    dataset.dataset_changed.emit()
    assert annotation_refreshes == [True]

    dataset_refreshes: list[bool] = []
    dataset.refresh = lambda: dataset_refreshes.append(True)
    annotation.labels_changed.emit()
    assert dataset_refreshes == [True]
    window.close()


@pytest.fixture
def qapp():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])

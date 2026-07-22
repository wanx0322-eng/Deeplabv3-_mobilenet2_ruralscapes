from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def qapp():
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def test_bounded_image_cache_evicts_lru_and_request_gate_rejects_stale() -> None:
    from workstation.async_images import BoundedImageCache, ImageRequestGate

    cache = BoundedImageCache(max_bytes=5)
    cache.put("a", "A", cost=3)
    cache.put("b", "B", cost=3)
    assert cache.get("a") is None
    assert cache.get("b") == "B"
    assert cache.total_bytes <= 5

    gate = ImageRequestGate()
    first = gate.begin()
    second = gate.begin()
    assert gate.accept(first) is False
    assert gate.accept(second) is True


def test_trash_manager_moves_batch_and_restores_every_file(tmp_path: Path) -> None:
    from workstation.trash import TrashManager

    models = tmp_path / "models"
    models.mkdir()
    first = models / "a.pth"
    second = models / "b.pth"
    first.write_bytes(b"a")
    second.write_bytes(b"b")

    manager = TrashManager(models, timestamp_factory=lambda: "20260721-120000")
    batch = manager.move_batch([first, second])
    assert not first.exists() and not second.exists()
    assert len(batch.entries) == 2
    assert manager.undo_latest() is True
    assert first.read_bytes() == b"a"
    assert second.read_bytes() == b"b"


def test_feedback_types_are_non_modal_and_field_addressable(qapp) -> None:
    from workstation.feedback import ConfigIssue, InlineMessage, ToastManager

    issue = ConfigIssue("batch_size", "must be at least 2")
    assert issue.field == "batch_size"
    message = InlineMessage()
    message.show_issue(issue)
    assert "batch_size" in message.text()
    toast = ToastManager()
    toast.show("Saved")
    assert toast.last_message == "Saved"


def test_accessibility_pass_assigns_names_to_interactive_controls(qapp) -> None:
    from PySide6.QtWidgets import QComboBox, QLineEdit, QPushButton, QVBoxLayout, QWidget
    from workstation.accessibility import ensure_accessibility, unnamed_controls

    root = QWidget()
    layout = QVBoxLayout(root)
    layout.addWidget(QPushButton("Run"))
    edit = QLineEdit()
    edit.setPlaceholderText("Dataset path")
    layout.addWidget(edit)
    combo = QComboBox()
    combo.addItems(["A", "B"])
    combo.setObjectName("engineChoice")
    layout.addWidget(combo)
    ensure_accessibility(root)
    assert unnamed_controls(root) == []

def test_main_window_keyboard_navigation_and_accessibility(qapp, tmp_path) -> None:
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest
    from workstation.accessibility import unnamed_controls
    from workstation.app import MainWindow
    from workstation.config import Config

    window = MainWindow(Config(path=str(tmp_path / "workstation_config.json")))
    window.show()
    qapp.processEvents()
    for index in range(6):
        page = window.page_host.activate(index)
        assert unnamed_controls(page) == []

    QTest.keyClick(window, Qt.Key_3, Qt.ControlModifier)
    qapp.processEvents()
    assert window.nav_group.checkedId() == 2

    first = window.nav_group.button(0)
    first.setFocus()
    QTest.keyClick(first, Qt.Key_Down)
    qapp.processEvents()
    assert window.nav_group.checkedId() == 1
    window.close()

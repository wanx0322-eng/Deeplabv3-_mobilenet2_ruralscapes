from __future__ import annotations

import inspect
import re
from dataclasses import fields
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def test_theme_tokens_drive_the_compatibility_stylesheet() -> None:
    from workstation.theme import DARK_TOKENS, STYLESHEET, ThemeTokens, build_stylesheet

    assert isinstance(DARK_TOKENS, ThemeTokens)
    assert STYLESHEET == build_stylesheet(DARK_TOKENS)
    assert all(getattr(DARK_TOKENS, field.name) is not None for field in fields(ThemeTokens))
    assert "#4f8cff" in STYLESHEET
    assert "QToolButton:checked" in STYLESHEET
    assert "QScrollArea" in STYLESHEET

    implementation = inspect.getsource(build_stylesheet)
    assert re.search(r"#[0-9A-Fa-f]{3,8}", implementation) is None


def test_page_host_is_lazy_and_workspace_events_replay_latest_value(qapp) -> None:
    from PySide6.QtWidgets import QWidget

    from workstation.page_system import PageHost, PageSpec, WorkspaceEvents

    created: list[str] = []

    def factory(page_id: str):
        def create() -> QWidget:
            created.append(page_id)
            return QWidget()

        return create

    specs = (
        PageSpec("dataset", "Dataset", "folder", "Ctrl+1", factory("dataset")),
        PageSpec("train", "Train", "target", "Ctrl+2", factory("train")),
    )
    host = PageHost(specs)
    assert host.loaded_page_count == 1
    assert created == ["dataset"]

    host.activate("train")
    assert host.loaded_page_count == 2
    assert created == ["dataset", "train"]

    events = WorkspaceEvents()
    events.publish("dataset", {"revision": 2})
    received: list[object] = []
    events.subscribe("dataset", received.append)
    assert received == [{"revision": 2}]


def test_base_page_exposes_the_lifecycle_contract(qapp) -> None:
    from workstation.page_system import BasePage

    page = BasePage("Contract page")
    assert page.page_title.text() == "Contract page"
    assert page.has_running_task() is False
    assert page.stop_running_task() is False
    assert page.refresh() is None
    assert page.on_activated() is None
    assert page.on_deactivated() is None


def test_weight_picker_is_the_single_weight_scanner(qapp, monkeypatch, tmp_path) -> None:
    from workstation.widgets import WeightPicker

    weights = [{"name": "best.pth", "abs_path": str(tmp_path / "best.pth")}]
    monkeypatch.setattr("workstation.weight_picker.scan_weights", lambda: weights)
    picker = WeightPicker()
    picker.refresh()
    assert picker.current_path() == weights[0]["abs_path"]
    assert picker.combo.count() == 1


def test_qml_uses_an_explicit_runtime_contract_without_typeof_guards() -> None:
    from ruralscape_studio.runtime import StudioRuntime

    assert StudioRuntime
    source = (ROOT / "ruralscape_studio" / "qml" / "Main.qml").read_text(
        encoding="utf-8"
    )
    assert "required property var runtime" in source
    assert "typeof" not in source
    assert "runtime.datasetBackend" in source

def test_main_window_loads_only_dataset_then_builds_each_page_on_demand(qapp, tmp_path):
    from workstation.app import MainWindow
    from workstation.config import Config
    from workstation.page_system import BasePage

    window = MainWindow(Config(path=str(tmp_path / "workstation_config.json")))
    assert window.loaded_page_count == 1
    for index in range(1, 6):
        page = window.page_host.activate(index)
        assert isinstance(page, BasePage)
    assert window.loaded_page_count == 6
    window.close()



@pytest.fixture
def qapp():
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])

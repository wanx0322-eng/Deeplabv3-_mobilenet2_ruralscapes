from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_routine_page_feedback_never_uses_information_or_warning_dialogs() -> None:
    offenders: list[str] = []
    for path in (ROOT / "workstation" / "pages").rglob("*.py"):
        if path.name == "annotate_page_legacy.py":
            continue
        source = path.read_text(encoding="utf-8")
        if "QMessageBox.information" in source or "QMessageBox.warning" in source:
            offenders.append(str(path.relative_to(ROOT)))
    assert offenders == []


def test_remaining_message_boxes_are_dangerous_confirmations_or_fatal_errors() -> None:
    source = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (ROOT / "workstation" / "pages").rglob("*.py")
        if path.name != "annotate_page_legacy.py"
    )
    assert source.count("QMessageBox.question") == 3
    assert "QMessageBox.critical" in source
    assert "save_raw_mask.isChecked()" in source


def test_base_page_exposes_visible_inline_feedback(qapp) -> None:
    from workstation.page_system import BasePage

    page = BasePage("Feedback")
    page.show_message("Saved", "success")
    assert page.feedback.isVisibleTo(page)
    assert page.feedback.text() == "Saved"
    assert page.feedback.property("severity") == "success"
    assert page.toast.last_message == "Saved"


import os
import pytest


@pytest.fixture
def qapp():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])

"""Capture deterministic offscreen screenshots for all six Widgets pages."""

from __future__ import annotations

import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QSize
from PySide6.QtWidgets import QApplication

from workstation.app import MainWindow
from workstation.fonts import configure_application_font
from workstation.i18n import install_translation
from workstation.theme import STYLESHEET


OUTPUT = ROOT / "tests" / "ui_baseline"


def main() -> int:
    application = QApplication.instance() or QApplication([])
    install_translation(application, "zh_CN")
    configure_application_font(application)
    application.setStyleSheet(STYLESHEET)
    window = MainWindow()
    window.resize(QSize(1280, 800))
    window.show()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for index, spec in enumerate(window.page_specs):
        window._switch_page(index)
        for _ in range(3):
            application.processEvents()
        target = OUTPUT / f"{index + 1:02d}-{spec.id}.png"
        if not window.grab().save(str(target), "PNG"):
            raise RuntimeError(f"Unable to save {target}")
        print(target.relative_to(ROOT))
    window.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

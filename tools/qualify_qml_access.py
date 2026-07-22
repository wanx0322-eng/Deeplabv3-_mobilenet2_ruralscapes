"""Qualify page-level QML properties used by nested controls.

This codemod is intentionally narrow: it only touches the eight page files whose
root item is named ``root`` and only qualifies the declared ``controller``
property.  It is safe to rerun.
"""

from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PAGE_DIR = ROOT / "ruralscape_studio" / "qml" / "pages"
PAGE_FILES = (
    "AnnotationWorkbench.qml",
    "DatasetPage.qml",
    "EvaluationReport.qml",
    "InferenceWorkbench.qml",
    "ModelsExport.qml",
    "ProjectOverview.qml",
    "TaskHistory.qml",
    "TrainingCenter.qml",
)


def qualify(text: str) -> str:
    text = text.replace("property var root.controller", "property var controller")
    declaration = "property var controller"
    marker = "property var __CONTROLLER_DECLARATION__"
    text = text.replace(declaration, marker)
    text = re.sub(r"(?<![\w.])controller\b", "root.controller", text)
    return text.replace(marker, declaration)


def main() -> int:
    changed = 0
    for name in PAGE_FILES:
        path = PAGE_DIR / name
        before = path.read_text(encoding="utf-8")
        after = qualify(before)
        if after != before:
            path.write_text(after, encoding="utf-8", newline="\n")
            changed += 1
    print(f"qualified controller access in {changed} QML files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

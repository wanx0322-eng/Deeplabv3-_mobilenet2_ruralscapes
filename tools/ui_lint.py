"""Repository-specific front-end architecture lint."""

from __future__ import annotations

import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HEX_COLOR = re.compile(r"#[0-9a-fA-F]{3,8}\b")
EMOJI = re.compile("[\U0001F300-\U0001FAFF\U00002600-\U000027BF]")


def _production_python() -> list[Path]:
    return [
        path
        for path in (ROOT / "workstation").rglob("*.py")
        if path.name != "annotate_page_legacy.py"
    ]


def main() -> int:
    errors: list[str] = []
    for path in _production_python():
        relative = path.relative_to(ROOT)
        text = path.read_text(encoding="utf-8")
        if path.name != "theme.py" and HEX_COLOR.search(text):
            errors.append(f"{relative}: raw color must be declared in ThemeTokens")
        if EMOJI.search(text):
            errors.append(f"{relative}: emoji is forbidden in production UI")
        if "setStyleSheet(" in text and path.name != "app.py":
            errors.append(f"{relative}: page-level setStyleSheet is forbidden")
    for path in sorted((ROOT / "ruralscape_studio" / "qml").rglob("*.qml")):
        text = path.read_text(encoding="utf-8")
        if re.search(r"typeof[^\n]*Backend", text):
            errors.append(f"{path.relative_to(ROOT)}: typeof Backend guard is forbidden")
    annotation_dir = ROOT / "workstation" / "pages" / "annotate"
    for path in sorted(annotation_dir.glob("*.py")):
        line_count = len(path.read_text(encoding="utf-8").splitlines())
        if line_count > 400:
            errors.append(f"{path.relative_to(ROOT)}: {line_count} lines exceeds 400")
    for path in sorted((ROOT / "workstation" / "pages").glob("*_page.py")):
        if path.name == "annotate_page_legacy.py":
            continue
        text = path.read_text(encoding="utf-8")
        if "pageTitle =" in text or 'setObjectName("pageTitle")' in text:
            errors.append(f"{path.relative_to(ROOT)}: duplicate page title boilerplate")
    if errors:
        print("\n".join(errors))
        return 1
    print("UI lint passed: tokens, styles, icons, runtime contract, and page limits")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

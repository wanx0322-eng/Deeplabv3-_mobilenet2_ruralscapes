"""Fail on untranslated visible UI copy in QML or Qt Widgets."""

from __future__ import annotations

import os
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from workstation.ui_text import collect_visible_text, contains_han


VISIBLE_QML_PROPERTY = re.compile(
    r"\b(?:text|title|label|subtitle|description|primaryText|secondaryText|trailingText|placeholderText)\s*:"
)
QSTR = re.compile(r'qsTr\("((?:[^"\\]|\\.)*)"\)')


def _catalog_messages(path: Path) -> dict[tuple[str, str], str]:
    root = ET.parse(path).getroot()
    messages: dict[tuple[str, str], str] = {}
    for context in root.findall("context"):
        name = context.findtext("name") or ""
        for message in context.findall("message"):
            source = message.findtext("source") or ""
            translation = message.find("translation")
            value = "" if translation is None else "".join(translation.itertext())
            if translation is not None and translation.get("type") == "unfinished":
                value = ""
            messages[(name, source)] = value
    return messages


def scan_qml(catalog: dict[tuple[str, str], str]) -> list[str]:
    errors: list[str] = []
    for path in sorted((ROOT / "ruralscape_studio" / "qml").rglob("*.qml")):
        text = path.read_text(encoding="utf-8")
        for number, line in enumerate(text.splitlines(), 1):
            if VISIBLE_QML_PROPERTY.search(line) and contains_han(line) and "qsTr(" not in line:
                errors.append(f"{path.relative_to(ROOT)}:{number}: visible Han text must use qsTr")
        context = path.stem
        for match in QSTR.finditer(text):
            source = bytes(match.group(1), "utf-8").decode("unicode_escape") if "\\" in match.group(1) else match.group(1)
            translation = catalog.get((context, source), "")
            if not translation:
                errors.append(f"{path.relative_to(ROOT)}: missing en_US translation for {source!r}")
            elif contains_han(translation):
                errors.append(f"{path.relative_to(ROOT)}: Han leaked into en_US translation for {source!r}")
    return errors


def scan_widgets() -> list[str]:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    from workstation.app import MainWindow
    from workstation.i18n import install_translation

    application = QApplication.instance() or QApplication([])
    install_translation(application, "en_US")
    window = MainWindow()
    for index in range(len(window.page_specs)):
        window._switch_page(index)
    errors = [
        f"Widgets en_US contains Han: {text!r}"
        for text in sorted(collect_visible_text(window))
        if contains_han(text)
    ]
    window.close()
    return errors


def main() -> int:
    catalog_path = ROOT / "workstation" / "i18n" / "en_US.ts"
    catalog = _catalog_messages(catalog_path)
    errors = scan_qml(catalog)
    errors.extend(
        f"{context}: generated fallback remains for {source!r}"
        for (context, source), translation in catalog.items()
        if "[[UNTRANSLATED]]" in translation
    )
    errors.extend(scan_widgets())
    if errors:
        print("\n".join(errors))
        return 1
    print("i18n scan passed: QML catalogs complete and Widgets en_US contains no Han")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

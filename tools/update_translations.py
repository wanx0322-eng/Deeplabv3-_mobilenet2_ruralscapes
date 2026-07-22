"""Regenerate complete zh_CN/en_US Qt catalogs from source and live Widgets."""

from __future__ import annotations

import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from workstation.translation_rules import english_for
from workstation.translation_overrides import ENGLISH_OVERRIDES
from workstation.ui_text import collect_visible_text, contains_han


CATALOG_DIR = ROOT / "workstation" / "i18n"
LANGUAGES = ("zh_CN", "en_US")


def _qt_tool(name: str) -> Path:
    executable = Path(sys.executable).with_name(f"pyside6-{name}.exe")
    if not executable.exists():
        raise FileNotFoundError(f"Missing Qt tool: {executable}")
    return executable


def _collect_widget_sources() -> set[str]:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    from workstation.app import MainWindow

    application = QApplication.instance() or QApplication([])
    application.setProperty("ruralscapeLanguage", "zh_CN")
    window = MainWindow()
    for index in range(len(window.page_specs)):
        window._switch_page(index)
    sources = {text for text in collect_visible_text(window) if contains_han(text)}
    window.close()
    return sources


def _ui_context(root: ET.Element) -> ET.Element:
    for context in root.findall("context"):
        if context.findtext("name") == "Ui":
            return context
    context = ET.SubElement(root, "context")
    ET.SubElement(context, "name").text = "Ui"
    return context


def _add_ui_sources(root: ET.Element, sources: set[str]) -> None:
    context = _ui_context(root)
    existing = {message.findtext("source") for message in context.findall("message")}
    for source in sorted(sources - existing):
        message = ET.SubElement(context, "message")
        ET.SubElement(message, "source").text = source
        ET.SubElement(message, "translation")

def _add_named_sources(root: ET.Element, name: str, sources: set[str]) -> None:
    context = None
    for item in root.findall("context"):
        if item.findtext("name") == name:
            context = item
            break
    if context is None:
        context = ET.SubElement(root, "context")
        ET.SubElement(context, "name").text = name
    existing = {message.findtext("source") for message in context.findall("message")}
    for source in sorted(sources - existing):
        message = ET.SubElement(context, "message")
        ET.SubElement(message, "source").text = source
        ET.SubElement(message, "translation")


def _finish_catalog(path: Path, language: str, widget_sources: set[str]) -> None:
    tree = ET.parse(path)
    root = tree.getroot()
    root.set("language", language)
    root.set("sourcelanguage", "zh_CN")
    override_sources = set(ENGLISH_OVERRIDES)
    _add_ui_sources(root, widget_sources | override_sources)
    for context_name in (
        "MainWindow", "DatasetPage", "AnnotatePage", "TrainPage",
        "PredictPage", "EvalPage", "ModelsPage",
    ):
        _add_named_sources(root, context_name, widget_sources | override_sources)
    for message in root.findall("./context/message"):
        source = message.findtext("source") or ""
        translation = message.find("translation")
        if translation is None:
            translation = ET.SubElement(message, "translation")
        translation.attrib.pop("type", None)
        translation.text = source if language == "zh_CN" else english_for(source)
    ET.indent(tree, space="  ")
    tree.write(path, encoding="utf-8", xml_declaration=True)


def main() -> int:
    widget_sources = _collect_widget_sources()
    catalogs = [CATALOG_DIR / f"{language}.ts" for language in LANGUAGES]
    subprocess.run(
        [
            str(_qt_tool("lupdate")),
            str(ROOT / "workstation"),
            str(ROOT / "ruralscape_studio" / "qml"),
            "-no-obsolete",
            "-ts",
            *(str(path) for path in catalogs),
        ],
        check=True,
    )
    for language, catalog in zip(LANGUAGES, catalogs, strict=True):
        _finish_catalog(catalog, language, widget_sources)
        subprocess.run(
            [str(_qt_tool("lrelease")), str(catalog), "-qm", str(catalog.with_suffix(".qm"))],
            check=True,
        )
    print(f"updated {len(catalogs)} catalogs with {len(widget_sources)} Widgets strings")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

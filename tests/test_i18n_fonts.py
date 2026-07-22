from __future__ import annotations

import os
from pathlib import Path
import pytest



ROOT = Path(__file__).resolve().parents[1]


def test_language_precedence_cli_then_environment_then_default(monkeypatch) -> None:
    from workstation.i18n import resolve_language

    monkeypatch.setenv("RURALSCAPE_LANG", "en_US")
    assert resolve_language(["app", "--lang", "zh_CN"]) == "zh_CN"
    assert resolve_language(["app"]) == "en_US"
    monkeypatch.delenv("RURALSCAPE_LANG")
    assert resolve_language(["app"]) == "zh_CN"


def test_invalid_language_is_rejected() -> None:
    from workstation.i18n import resolve_language

    try:
        resolve_language(["app", "--lang", "fr_FR"])
    except ValueError as error:
        assert "zh_CN" in str(error) and "en_US" in str(error)
    else:
        raise AssertionError("unsupported language must fail")


def test_shared_font_resolver_provides_han_glyphs(qapp) -> None:
    from PySide6.QtGui import QFontMetrics
    from workstation.fonts import configure_application_font

    font = configure_application_font(qapp)
    assert QFontMetrics(font).inFontUcs4(ord("项"))


def test_translation_catalog_sources_and_compiled_catalogs_exist() -> None:
    locale = ROOT / "workstation" / "i18n"
    for language in ("zh_CN", "en_US"):
        assert (locale / f"{language}.ts").is_file()
        assert (locale / f"{language}.qm").is_file()


def test_main_qml_marks_visible_navigation_for_translation() -> None:
    source = (ROOT / "ruralscape_studio" / "qml" / "Main.qml").read_text(
        encoding="utf-8"
    )
    assert 'qsTr("实验预览版 · DEMO")' in source
    assert 'qsTr("项目概览")' in source


def test_language_environment_name_is_stable() -> None:
    assert "RURALSCAPE_LANG" not in os.environ or os.environ["RURALSCAPE_LANG"] in {
        "zh_CN",
        "en_US",
    }


@pytest.fixture
def qapp():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])

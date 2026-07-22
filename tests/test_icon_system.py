from __future__ import annotations

import re
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
EMOJI = re.compile("[\U0001F300-\U0001FAFF]")


@pytest.fixture
def qapp():
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def test_phosphor_icon_provider_renders_and_caches_token_tinted_icons(qapp) -> None:
    from workstation.icons import IconProvider
    from workstation.theme import DARK_TOKENS, ThemeTokens

    provider = IconProvider()
    dark = provider.icon("folder", size=20, color=DARK_TOKENS.CONTENT_SECONDARY)
    assert dark.pixmap(20, 20).isNull() is False
    assert provider.cache_size == 1
    provider.icon("folder", size=20, color=DARK_TOKENS.CONTENT_SECONDARY)
    assert provider.cache_size == 1

    light = ThemeTokens(CONTENT_SECONDARY="#222222")
    provider.icon("folder", size=20, color=light.CONTENT_SECONDARY)
    assert provider.cache_size == 2


def test_required_navigation_icons_are_vendored() -> None:
    icon_root = ROOT / "workstation" / "assets" / "icons"
    for name in ("folder", "pencil", "target", "image", "chart-bar", "floppy-disk"):
        assert (icon_root / f"{name}.svg").is_file(), name


def test_production_widgets_sources_do_not_use_emoji() -> None:
    files = list((ROOT / "workstation").rglob("*.py"))
    files = [path for path in files if path.name != "annotate_page_legacy.py"]
    hits = {
        str(path.relative_to(ROOT)): EMOJI.findall(path.read_text(encoding="utf-8"))
        for path in files
        if EMOJI.search(path.read_text(encoding="utf-8"))
    }
    assert hits == {}

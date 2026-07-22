from __future__ import annotations

from collections import defaultdict
from dataclasses import fields
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_color_duplicates_are_explicit_semantic_aliases() -> None:
    from workstation.theme import COLOR_TOKEN_ALIASES, DARK_TOKENS, ThemeTokens

    grouped: dict[str, list[str]] = defaultdict(list)
    for field in fields(ThemeTokens):
        value = getattr(DARK_TOKENS, field.name)
        if isinstance(value, str) and value.startswith("#"):
            grouped[value.lower()].append(field.name)
    for names in grouped.values():
        if len(names) < 2:
            continue
        canonical = next((name for name in names if name not in COLOR_TOKEN_ALIASES), None)
        assert canonical is not None
        assert all(
            name == canonical or COLOR_TOKEN_ALIASES.get(name) == canonical for name in names
        )


def test_every_theme_token_is_consumed_outside_its_declaration() -> None:
    from workstation.theme import ThemeTokens

    sources = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (ROOT / "workstation").rglob("*.py")
        if path.name != "design_tokens.py"
    )
    unconsumed = [field.name for field in fields(ThemeTokens) if sources.count(field.name) < 2]
    assert unconsumed == []


def test_required_semantic_token_families_exist() -> None:
    from workstation.theme import ThemeTokens

    names = {field.name for field in fields(ThemeTokens)}
    for prefix in ("SURFACE_", "CONTENT_", "ACTION_", "FEEDBACK_", "CHART_"):
        assert any(name.startswith(prefix) for name in names)
    assert {"BORDER_FOCUS", "FOCUS_WIDTH", "ICON_SM", "ICON_MD", "ICON_LG"} <= names

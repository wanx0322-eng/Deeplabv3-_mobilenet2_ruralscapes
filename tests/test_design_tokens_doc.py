from __future__ import annotations

from dataclasses import fields
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_design_token_document_is_generated_from_runtime_registry() -> None:
    from workstation.design_tokens import render_design_tokens
    from workstation.theme import DARK_TOKENS, ThemeTokens

    document = ROOT / "docs" / "design_tokens.md"
    assert document.is_file()
    content = document.read_text(encoding="utf-8")
    assert content == render_design_tokens(DARK_TOKENS)
    for field in fields(ThemeTokens):
        assert field.name in content
    assert "Widgets 正式工作站" in content
    assert "QML 实验预览版" in content

"""Regenerate docs/design_tokens.md from the runtime token registry."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from workstation.design_tokens import render_design_tokens  # noqa: E402
from workstation.theme import DARK_TOKENS  # noqa: E402

TARGET = ROOT / "docs" / "design_tokens.md"


def main() -> None:
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    TARGET.write_text(render_design_tokens(DARK_TOKENS), encoding="utf-8")
    print(TARGET)


if __name__ == "__main__":
    main()

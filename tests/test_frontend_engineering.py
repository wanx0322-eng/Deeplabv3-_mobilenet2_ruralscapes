from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("tool", ["ui_lint.py", "qml_lint.py", "i18n_scan.py"])
def test_frontend_quality_gate_passes(tool: str) -> None:
    environment = dict(os.environ, QT_QPA_PLATFORM="offscreen", PYTHONUTF8="1")
    result = subprocess.run(
        [sys.executable, str(ROOT / "tools" / tool)],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_ui_dependencies_are_locked() -> None:
    requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8")
    ui_requirements = (ROOT / "requirements-ui.txt").read_text(encoding="utf-8")
    for content in (requirements, ui_requirements):
        assert "PySide6==6.11.1" in content
        assert "shiboken6==6.11.1" in content


def test_annotation_modules_respect_size_budget() -> None:
    modules = (ROOT / "workstation" / "pages" / "annotate").glob("*.py")
    assert all(len(path.read_text(encoding="utf-8").splitlines()) <= 400 for path in modules)


def test_six_visual_baselines_are_versioned() -> None:
    screenshots = sorted((ROOT / "tests" / "ui_baseline").glob("*.png"))
    assert [path.name for path in screenshots] == [
        "01-dataset.png",
        "02-annotate.png",
        "03-train.png",
        "04-predict.png",
        "05-evaluate.png",
        "06-models.png",
    ]
    assert all(path.stat().st_size > 10_000 for path in screenshots)


def test_windows_ci_uses_isolated_pytest_temp() -> None:
    runner = (ROOT / "tools" / "run_ui_tests.ps1").read_text(encoding="utf-8")
    workflow = (ROOT / ".github" / "workflows" / "ui-ci.yml").read_text(encoding="utf-8")
    assert "--basetemp" in runner
    assert "-p no:cacheprovider" in runner
    assert "windows-latest" in workflow
    assert "tools/qml_lint.py" in workflow
    assert "tools/i18n_scan.py" in workflow

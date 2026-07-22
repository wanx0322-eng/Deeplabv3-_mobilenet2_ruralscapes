"""Run qmllint across every QML file and fail on any warning."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ANSI = re.compile(r"\x1b\[[0-9;]*m")


def main() -> int:
    executable = Path(sys.executable).with_name("pyside6-qmllint.exe")
    warnings: list[str] = []
    failures: list[str] = []
    files = sorted((ROOT / "ruralscape_studio" / "qml").rglob("*.qml"))
    for path in files:
        result = subprocess.run(
            [str(executable), str(path)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        output = ANSI.sub("", result.stdout + result.stderr)
        warnings.extend(line for line in output.splitlines() if line.startswith("Warning:"))
        if result.returncode:
            failures.append(f"{path.relative_to(ROOT)}: qmllint exit {result.returncode}")
    errors = failures + warnings
    if errors:
        print("\n".join(errors))
        return 1
    print(f"QML lint passed: {len(files)} files, 0 warnings")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

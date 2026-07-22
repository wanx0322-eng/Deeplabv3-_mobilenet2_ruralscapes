"""Compare lazy one-page startup with the former eager six-page assembly."""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import statistics
import subprocess
import sys
import time
from ctypes import wintypes
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _working_set() -> int:
    class ProcessMemoryCounters(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    counters = ProcessMemoryCounters()
    counters.cb = ctypes.sizeof(counters)
    get_process = ctypes.windll.kernel32.GetCurrentProcess
    get_process.restype = ctypes.c_void_p
    get_memory = ctypes.windll.psapi.GetProcessMemoryInfo
    get_memory.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(ProcessMemoryCounters), wintypes.DWORD
    ]
    get_memory.restype = wintypes.BOOL
    if not get_memory(get_process(), ctypes.byref(counters), counters.cb):
        raise ctypes.WinError()
    return int(counters.WorkingSetSize)


def _probe(mode: str, started: float) -> int:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    sys.path.insert(0, str(ROOT))
    from PySide6.QtWidgets import QApplication
    from workstation.app import MainWindow

    application = QApplication([])
    window = MainWindow()
    if mode == "eager":
        for index in range(len(window.page_specs)):
            window._switch_page(index)
    application.processEvents()
    print(
        json.dumps(
            {
                "seconds": time.perf_counter() - started,
                "working_set": _working_set(),
                "loaded_pages": window.loaded_page_count,
            }
        )
    )
    return 0


def _samples(mode: str, count: int) -> list[dict[str, float]]:
    environment = dict(os.environ, QT_QPA_PLATFORM="offscreen", PYTHONUTF8="1")
    values = []
    for _ in range(count):
        result = subprocess.run(
            [sys.executable, __file__, "--probe", mode],
            cwd=ROOT,
            env=environment,
            capture_output=True,
            text=True,
            check=True,
        )
        values.append(json.loads(result.stdout.strip().splitlines()[-1]))
    return values


def main() -> int:
    started = time.perf_counter()
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", choices=("lazy", "eager"))
    parser.add_argument("--runs", type=int, default=5)
    options = parser.parse_args()
    if options.probe:
        return _probe(options.probe, started)
    lazy = _samples("lazy", options.runs)
    eager = _samples("eager", options.runs)
    result = {
        "runs": options.runs,
        "lazy_seconds_median": statistics.median(item["seconds"] for item in lazy),
        "eager_seconds_median": statistics.median(item["seconds"] for item in eager),
        "lazy_working_set_median": statistics.median(item["working_set"] for item in lazy),
        "eager_working_set_median": statistics.median(item["working_set"] for item in eager),
    }
    result["time_reduction"] = 1 - result["lazy_seconds_median"] / result["eager_seconds_median"]
    result["memory_reduction"] = 1 - result["lazy_working_set_median"] / result["eager_working_set_median"]
    print(json.dumps(result, indent=2))
    return 0 if result["time_reduction"] >= 0.20 and result["memory_reduction"] >= 0.20 else 1


if __name__ == "__main__":
    raise SystemExit(main())

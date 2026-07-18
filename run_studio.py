"""QML 外壳入口：python run_studio.py

与 run_workstation.py（PySide6 Widgets 工作站）并存，两者共用同一套后端。
标注工作台的 SAM2 交互目前仍只在 Widgets 工作站里，见 工作站使用说明.md。
"""
import os
import sys
import traceback


def _report_crash(exc):
    log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "studio_error.log")
    with open(log_path, "a", encoding="utf-8") as f:
        import datetime
        f.write(f"\n===== {datetime.datetime.now()} =====\n")
        f.write(traceback.format_exc())
    try:
        import ctypes
        ctypes.windll.user32.MessageBoxW(
            0,
            f"RuralScape Studio 启动失败：\n{exc}\n\n详细信息见 studio_error.log",
            "RuralScape Studio", 0x10)
    except Exception:
        pass


if __name__ == "__main__":
    try:
        from workstation.studio_app import main
        sys.exit(main(sys.argv))
    except SystemExit:
        raise
    except Exception as exc:
        _report_crash(exc)
        sys.exit(1)

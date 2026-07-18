"""桌面工作站入口：python run_workstation.py

用 pythonw 无窗口启动时异常不可见，这里把启动失败写入
workstation_error.log 并弹出系统提示框。
"""
import os
import sys
import traceback


def _report_crash(exc):
    log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "workstation_error.log")
    with open(log_path, "a", encoding="utf-8") as f:
        import datetime
        f.write(f"\n===== {datetime.datetime.now()} =====\n")
        f.write(traceback.format_exc())
    try:
        import ctypes
        ctypes.windll.user32.MessageBoxW(
            0,
            f"工作站启动失败：\n{exc}\n\n详细信息见 workstation_error.log",
            "DeepLabV3+ 工作站", 0x10)
    except Exception:
        pass


if __name__ == "__main__":
    try:
        from workstation.app import main
        main()
    except SystemExit:
        raise
    except Exception as exc:
        _report_crash(exc)
        sys.exit(1)

# Widgets 启动性能基线

2026-07-21 在本项目 Windows 环境执行：

```powershell
.venv\Scripts\python.exe tools\benchmark_ui_startup.py --runs 5
```

五次独立进程冷启动的中位数：

| 模式 | 启动时间 | Working Set |
| --- | ---: | ---: |
| PageHost 懒加载（仅数据页） | 0.315 s | 84.4 MiB |
| 兼容对照（强制创建六页） | 0.600 s | 137.1 MiB |

懒加载减少约 47.6% 启动时间和 38.4% Working Set，超过 20% 验收线。脚本每次使用独立进程，并验证懒加载与六页强制装配两种路径。

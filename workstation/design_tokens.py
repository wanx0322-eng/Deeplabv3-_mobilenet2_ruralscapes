"""Deterministic design-token documentation renderer."""

from __future__ import annotations

from dataclasses import fields

from workstation.theme import ThemeTokens


PURPOSES = {
    "SURFACE": "背景、容器与画布表面",
    "CONTENT": "文本与前景内容",
    "ACTION": "主要操作与危险操作",
    "FEEDBACK": "成功、警告、错误和信息反馈",
    "BORDER": "边框与焦点可视化",
    "CHART": "训练曲线与坐标轴",
    "FONT": "字体族与字阶",
    "SPACE": "布局间距",
    "RADIUS": "组件圆角",
    "FOCUS": "键盘焦点环",
    "ICON": "SVG 图标尺寸",
    "CONTROL": "交互控件尺寸",
    "SCROLLBAR": "滚动条尺寸",
    "SIDEBAR": "侧栏尺寸",
}


def _purpose(name: str) -> str:
    prefix = name.split("_", 1)[0]
    return PURPOSES.get(prefix, "组件基础尺寸")


def render_design_tokens(tokens: ThemeTokens) -> str:
    lines = [
        "# RuralScapes Design Tokens",
        "",
        "本文件由 ThemeTokens registry 自动生成，请勿手工复制或修改 Token 值。",
        "",
        "## 定位",
        "",
        "- Widgets 正式工作站：唯一正式入口，使用本表全部语义 Token。",
        "- QML 实验预览版：保持独立实验主题，仅做语义对照，不承诺视觉完全一致。",
        "",
        "## Token registry",
        "",
        "| Token | 值 | 用途 |",
        "| --- | --- | --- |",
    ]
    for field in fields(ThemeTokens):
        name = field.name
        value = getattr(tokens, name)
        suffix = " px" if isinstance(value, int) else ""
        lines.append(f"| {name} | {value}{suffix} | {_purpose(name)} |")
    lines.extend(
        [
            "",
            "## 组件状态",
            "",
            "按钮和工具按钮统一覆盖 normal、hover、pressed、checked、focus、disabled；",
            "输入控件与导航共享 BORDER_FOCUS 和 FOCUS_WIDTH。",
            "",
            "## 截图基线",
            "",
            "确定性截图由 tools/capture_ui_baselines.py 生成：dataset、annotate、train、",
            "predict、evaluate、models 六页。截图仅用于离屏回归，最终仍需 Windows 桌面走查。",
            "",
        ]
    )
    return "\n".join(lines)

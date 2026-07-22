# RuralScapes Design Tokens

本文件由 ThemeTokens registry 自动生成，请勿手工复制或修改 Token 值。

## 定位

- Widgets 正式工作站：唯一正式入口，使用本表全部语义 Token。
- QML 实验预览版：保持独立实验主题，仅做语义对照，不承诺视觉完全一致。

## Token registry

| Token | 值 | 用途 |
| --- | --- | --- |
| SURFACE_BASE | #1e2228 | 背景、容器与画布表面 |
| SURFACE_SIDEBAR | #171a1f | 背景、容器与画布表面 |
| SURFACE_PANEL | #262b33 | 背景、容器与画布表面 |
| SURFACE_INPUT | #2e343e | 背景、容器与画布表面 |
| SURFACE_CANVAS | #14161a | 背景、容器与画布表面 |
| SURFACE_HOVER | #38404c | 背景、容器与画布表面 |
| SURFACE_PRESSED | #2a3039 | 背景、容器与画布表面 |
| SURFACE_ALT | #2a303a | 背景、容器与画布表面 |
| CONTENT_PRIMARY | #e8eaf0 | 文本与前景内容 |
| CONTENT_SECONDARY | #9aa3b2 | 文本与前景内容 |
| CONTENT_DISABLED | #5d6673 | 文本与前景内容 |
| CONTENT_INVERSE | #ffffff | 文本与前景内容 |
| CONTENT_CONSOLE | #c9d2e0 | 文本与前景内容 |
| ACTION_PRIMARY | #4f8cff | 主要操作与危险操作 |
| ACTION_PRIMARY_HOVER | #6da2ff | 主要操作与危险操作 |
| ACTION_PRIMARY_DISABLED | #33415c | 主要操作与危险操作 |
| ACTION_DANGER | #b33939 | 主要操作与危险操作 |
| ACTION_DANGER_HOVER | #cc4444 | 主要操作与危险操作 |
| FEEDBACK_SUCCESS | #7bd88f | 成功、警告、错误和信息反馈 |
| FEEDBACK_WARNING | #ffd75f | 成功、警告、错误和信息反馈 |
| FEEDBACK_ERROR | #ff6b6b | 成功、警告、错误和信息反馈 |
| FEEDBACK_ERROR_MUTED | #e0685f | 成功、警告、错误和信息反馈 |
| FEEDBACK_INFO | #69b7ff | 成功、警告、错误和信息反馈 |
| BORDER_DEFAULT | #3a4150 | 边框与焦点可视化 |
| BORDER_FOCUS | #7aa7ff | 边框与焦点可视化 |
| CHART_TRAIN | #4f8cff | 训练曲线与坐标轴 |
| CHART_VALIDATION | #ff8c5f | 训练曲线与坐标轴 |
| CHART_MIOU | #7bd88f | 训练曲线与坐标轴 |
| CHART_GRID | #2e343e | 训练曲线与坐标轴 |
| CHART_AXES | #1a1d23 | 训练曲线与坐标轴 |
| FONT_UI | "Microsoft YaHei UI", "Segoe UI", sans-serif | 字体族与字阶 |
| FONT_MONO | "Consolas", "Courier New", monospace | 字体族与字阶 |
| FONT_SIZE_XS | 11 px | 字体族与字阶 |
| FONT_SIZE_SM | 12 px | 字体族与字阶 |
| FONT_SIZE_MD | 13 px | 字体族与字阶 |
| FONT_SIZE_LG | 14 px | 字体族与字阶 |
| FONT_SIZE_XL | 16 px | 字体族与字阶 |
| FONT_SIZE_TITLE | 18 px | 字体族与字阶 |
| FONT_SIZE_STAT | 20 px | 字体族与字阶 |
| SPACE_0 | 0 px | 布局间距 |
| SPACE_XXS | 2 px | 布局间距 |
| SPACE_XS | 4 px | 布局间距 |
| SPACE_SM | 6 px | 布局间距 |
| SPACE_MD | 8 px | 布局间距 |
| SPACE_LG | 10 px | 布局间距 |
| SPACE_XL | 12 px | 布局间距 |
| SPACE_XXL | 16 px | 布局间距 |
| RADIUS_SM | 4 px | 组件圆角 |
| RADIUS_MD | 5 px | 组件圆角 |
| RADIUS_LG | 6 px | 组件圆角 |
| RADIUS_XL | 8 px | 组件圆角 |
| BORDER_WIDTH | 1 px | 边框与焦点可视化 |
| FOCUS_WIDTH | 2 px | 键盘焦点环 |
| ICON_SM | 16 px | SVG 图标尺寸 |
| ICON_MD | 20 px | SVG 图标尺寸 |
| ICON_LG | 24 px | SVG 图标尺寸 |
| CONTROL_HEIGHT | 28 px | 交互控件尺寸 |
| SCROLLBAR_SIZE | 10 px | 滚动条尺寸 |
| SIDEBAR_WIDTH | 190 px | 侧栏尺寸 |

## 组件状态

按钮和工具按钮统一覆盖 normal、hover、pressed、checked、focus、disabled；
输入控件与导航共享 BORDER_FOCUS 和 FOCUS_WIDTH。

## 截图基线

确定性截图由 tools/capture_ui_baselines.py 生成：dataset、annotate、train、
predict、evaluate、models 六页。截图仅用于离屏回归，最终仍需 Windows 桌面走查。

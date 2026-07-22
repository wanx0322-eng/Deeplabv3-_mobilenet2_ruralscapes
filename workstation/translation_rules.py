"""Canonical English copy used to build both Qt translation catalogs."""

from __future__ import annotations

import re


_ROWS = r"""
项目概览	Project overview
数据集	Dataset
标注工作台	Annotation workspace
训练中心	Training center
识别工作台	Inference workspace
评估报告	Evaluation report
模型与导出	Models and export
任务记录	Task history
实验预览版 · DEMO	Experimental preview · DEMO
数据管理	Data management
图像标注	Image annotation
模型训练	Model training
图像预测	Image prediction
精度评估	Evaluation
模型管理	Model management
从数据集页面选择目录	Select a directory on the Dataset page
任务	Task
任务\s	Task\s
任务状态	Task status
任务筛选	Task filter
保存版本	Save version
停止训练	Stop training
全部状态	All statuses
关闭	Close
关闭窗口	Close window
参数确认后才能创建训练任务。	Confirm the parameters before creating a training task.
图像	Images
图像已选择，等待加载标注内容。	Image selected; waiting for annotation data.
在原图与掩膜之间切换，保存可追溯的标注版本。	Switch between the image and mask and save traceable annotation versions.
在指定数据划分上执行评估，并从实际结果生成报告。	Evaluate the selected split and generate a report from actual results.
处理中	Processing
失败	Failed
完成训练或导入已有检查点后，模型会出现在这里。	Models appear here after training completes or a checkpoint is imported.
完成评估后，这里显示实际计算结果；当前不展示示例指标。	Actual metrics appear here after evaluation; placeholder metrics are never shown.
导出	Export
尚无评估报告	No evaluation report yet
尚未启动训练任务	No training task has started
尚未打开项目	No project is open
尚未索引	Not indexed
尚未选择图像	No image selected
尚未选择数据集目录	No dataset directory selected
尚未选择模型	No model selected
尚未选择模型文件	No model file selected
尚未选择输入	No input selected
尚未选择输出目录	No output directory selected
工具与版本	Tools and versions
已保存版本：	Saved versions:\s
已保存版本：0	Saved versions: 0
已匹配的标注掩膜	Matched annotation masks
已完成	Completed
已打开	Open
已识别的原始图像	Detected source images
已选模型	Selected model
建立项目工作目录，并确认当前分割引擎与数据位置。	Create a project workspace and confirm the segmentation engine and data location.
开始工作	Get started
开始训练	Start training
开始评估	Start evaluation
开始识别	Start inference
当前没有运行中的任务	No task is running
当前项目	Current project
当前项目已登记的模型数量	Models registered in the current project
当前项目没有模型产物	The current project has no model artifacts
待启动	Ready to start
待处理问题	Issues to resolve
打开项目	Open project
打开项目后，工作台会显示真实的目录状态与任务记录。	Open a project to view its actual directory status and task history.
执行数据扫描、训练、识别或评估后，真实任务记录会显示在这里。	Actual scan, training, inference, and evaluation tasks appear here.
扫描数据集	Scan dataset
批大小	Batch size
报告内容	Report contents
报告将显示真实的类别指标、混淆信息与导出路径。	The report will contain real class metrics, confusion data, and export paths.
报告已生成	Report generated
掩膜	Masks
数据划分	Data split
数据来源	Data source
数据检查	Data checks
数据集目录	Dataset directory
新建项目	New project
无未保存修改	No unsaved changes
最大化或还原窗口	Maximize or restore window
最小化窗口	Minimize window
有任务正在运行	A task is running
有未保存修改	Unsaved changes
未开始	Not started
未打开项目	Project not open
未选择	Not selected
本地工作台	Local workstation
本地识别任务正在运行	A local inference task is running
查看	View
查看本次工作期间实际创建的扫描、训练、识别与评估任务。	Review the scan, training, inference, and evaluation tasks created in this session.
查看类别值和数据划分后再进入训练。	Review class values and data splits before training.
查看项目模型产物，选择检查点并导出可交付文件。	Review model artifacts, select a checkpoint, and export deliverables.
标注画布	Annotation canvas
模型产物	Model artifacts
模型产物已登记	Model artifact registered
模型列表	Model list
模型和图像均保留在本地工作目录。	Models and images remain in the local workspace.
模型引擎	Model engine
模型文件	Model file
测试集	Test set
清空筛选	Clear filter
画笔尺寸	Brush size
空	Empty
空闲	Idle
等待中	Waiting
等待任务	Waiting for tasks
等待扫描数据集	Waiting for a dataset scan
等待模型与输入	Waiting for a model and input
等待评估	Waiting for evaluation
类别数量	Number of classes
索引完成	Index complete
索引结果可供检查	Index results are ready for review
继续前往数据集，核对图像、掩膜与划分。	Continue to Dataset to verify images, masks, and splits.
缺失、重复或无法读取	Missing, duplicate, or unreadable
至少为 2	Must be at least 2
训练中	Training
训练任务正在运行	A training task is running
训练配置	Training configuration
训练集	Training set
评估中	Evaluating
评估设置	Evaluation settings
识别状态	Inference status
识别输入	Inference input
输入像素大小	Input size in pixels
输入图像或目录	Input image or directory
输入类别数量	Number of input classes
输出目录	Output directory
运行中	Running
运行中\s	Running\s
运行中 0	Running 0
选择	Select
选择分割引擎并填写训练参数；启动前不加载模型运行时。	Select a segmentation engine and enter training parameters; the model runtime is not loaded before start.
选择图像	Select image
选择图像后，在这里显示原图与掩膜。	Select an image to display the source image and mask here.
选择实际模型文件以查看和导出。	Select an actual model file to inspect and export.
选择已有项目，或创建新的项目目录。	Select an existing project or create a new project directory.
选择数据划分后创建评估任务。	Select a data split before creating an evaluation task.
选择数据集目录	Select dataset directory
选择模型	Select model
选择模型与输入图像，确认输出目录后创建本地识别任务。	Select a model and input image, confirm the output directory, then create a local inference task.
选择目录后执行扫描，这里将显示真实检查结果。	Select a directory and scan it to display actual validation results.
选择语义分割数据目录，扫描图像、掩膜、类别值与数据划分。	Select a semantic-segmentation dataset and scan its images, masks, class values, and splits.
选择输入	Select input
选择输出目录	Select output directory
选择项目后开始工作	Select a project to begin
重置状态	Reset status
项目就绪后，可从左侧依次检查数据集、标注、训练与评估。	When the project is ready, use the navigation to review data, annotations, training, and evaluation.
项目已就绪	Project ready
项目操作	Project actions
项目数据保存在所选工作目录内。	Project data is stored in the selected workspace.
验证集	Validation set
AI 预标注	AI pre-annotation
B：按住左键涂抹当前类别	B: hold the left mouse button to paint the current class
Dice Loss（类别少时建议开启）	Dice Loss (recommended for few classes)
E：擦除为背景	E: erase to background
Focal Loss（正负样本不平衡时）	Focal Loss (for class imbalance)
F：将点击处的连通区域改为当前类别	F: fill the connected region with the current class
P：单击加点，右键/双击闭合填充	P: click to add points; right-click or double-click to close
SAM2 边界精修	SAM2 boundary refinement
mobilenet / xception 对应 DeepLabV3+ 权重（logs_v2_* 等），\nsegformer-* 对应 tools/train_segformer.py 训练的权重（logs_segformer_*）。\n主干必须与权重文件匹配。	mobilenet/xception require DeepLabV3+ weights (such as logs_v2_*).\nsegformer-* requires weights trained by tools/train_segformer.py (logs_segformer_*).\nThe backbone must match the weight file.
下采样倍数	Downsampling factor
不透明度	Opacity
主干网络	Backbone
仅分割图	Mask only
优化器	Optimizer
优化器与学习率	Optimizer and learning rate
使用 GPU (CUDA)	Use GPU (CUDA)
使用主干 ImageNet 预训练（model_path 为空时生效，需联网）	Use ImageNet-pretrained backbone (requires network when model_path is empty)
保存周期(epoch)	Save interval (epochs)
保存当前结果…	Save current result…
保存标签 (Ctrl+S)	Save label (Ctrl+S)
保存类别设置	Save class settings
停止（epoch 结束后）	Stop after current epoch
像素统计	Pixel statistics
全部	All
全部图片	All images
冻结 Epoch	Freeze epoch
冻结 batch_size	Frozen batch size
冻结主干训练（先冻结后解冻）	Freeze backbone first, then unfreeze
分割结果	Segmentation result
切换到 图像标注，快捷键 Ctrl+2	Switch to Image annotation, shortcut Ctrl+2
切换到 图像预测，快捷键 Ctrl+4	Switch to Image prediction, shortcut Ctrl+4
切换到 数据管理，快捷键 Ctrl+1	Switch to Data management, shortcut Ctrl+1
切换到 模型管理，快捷键 Ctrl+6	Switch to Model management, shortcut Ctrl+6
切换到 模型训练，快捷键 Ctrl+3	Switch to Model training, shortcut Ctrl+3
切换到 精度评估，快捷键 Ctrl+5	Switch to Evaluation, shortcut Ctrl+5
初始学习率	Initial learning rate
初始权值	Initial weights
删除	Delete
删除末尾类别	Remove last class
删除选中	Delete selected
刷新	Refresh
区域填充	Region fill
单张预测	Single-image prediction
原图	Source image
叠加	Overlay
叠加混合	Overlay blend
可视化	Visualization
多边形	Polygon
学习率衰减	Learning-rate decay
导入图片…	Import images…
导入权值…	Import weights…
导入标签…	Import labels…
导出标签…	Export labels…
就绪	Ready
左键绘制 · Shift/中键拖动平移 · 滚轮缩放 · 多边形右键/双击闭合 · Esc 取消	Draw with left mouse · Pan with Shift/middle mouse · Zoom with wheel · Close polygon with right-click/double-click · Esc to cancel
强制终止	Force stop
总 Epoch	Total epochs
打开图片并预测…	Open image and predict…
打开所在文件夹	Open containing folder
扣除背景	Exclude background
扫描 models/、backbones/、runs/ 下的权值文件。训练产物落在 runs/，挑好的成品放进 models/，预训练主干与导入的权值放在 backbones/。	Scan weight files under models/, backbones/, and runs/. Training artifacts go to runs/, selected deliverables to models/, and pretrained or imported weights to backbones/.
批量预测	Batch prediction
按名称过滤…	Filter by name…
损失与其它	Losses and other settings
撤销	Undo
撤销最近删除	Undo latest deletion
数据加载线程	Data-loader workers
无图像	No image
显示方式	Display mode
未划分	Unsplit
未选择图像；工具 brush；类别 1  building；已保存	No image selected; tool brush; class 1 building; saved
权值文件	Weight file
标签（着色）	Label (colored)
检查数据集	Validate dataset
模型	Model
模型将在首次预测时加载	The model will load on the first prediction
橡皮	Eraser
测试时增强：水平翻转 × 多尺度概率平均。\n边界更平滑、细小类别更稳，但单张耗时约 6 倍。\n批量大图或视频建议关闭。	Test-time augmentation: horizontal flip × multi-scale probability averaging.\nIt improves boundaries and small classes but takes about six times longer per image.\nDisable it for large batches or video.
混合精度 fp16（省显存）	Mixed precision fp16 (saves VRAM)
混合透明度	Overlay opacity
添加类别	Add class
清空标签	Clear labels
用当前模型自动生成标签，再手动修正（模型设置沿用「图像预测」页）	Generate labels with the current model, then correct them manually (uses Image prediction settings)
画笔	Brush
留空=不加载整模型权值	Leave blank to avoid loading full model weights
笔刷	Brush
类别	Class
类别定义（第 0 类为背景，训练/预测共用）	Class definitions (class 0 is background; shared by training and prediction)
精度评估 (mIoU)	Evaluation (mIoU)
终止	Stop
缺少标签	Missing label
解冻 batch_size	Unfrozen batch size
训练中周期评估 mIoU	Periodic training mIoU evaluation
训练轮次	Training epochs
评估划分	Evaluation split
评估周期(epoch)	Evaluation interval (epochs)
评估图片数	Number of evaluation images
评估配置	Evaluation configuration
语义分割研究工作站 · 正式入口	Semantic Segmentation Research Workstation · Official
起始 Epoch	Starting epoch
输入尺寸	Input size
过滤图片…	Filter images…
运行设备	Device
适应窗口	Fit to window
选择文件夹批量预测…	Select folder for batch prediction…
重做	Redo
重命名	Rename
随机划分…	Random split…
预标注后用 SAM2 修整区域边界（语义模型出类别，SAM2 出边界）。\n高分辨率照片上效果明显；首次使用需联网下载模型（约 40MB）。	Refine pre-annotation boundaries with SAM2 (the semantic model predicts classes; SAM2 predicts boundaries).\nThis is most useful for high-resolution photos. First use downloads about 40 MB.
预览	Preview
高质量模式 (TTA)	High-quality mode (TTA)
"""


ENGLISH_UI: dict[str, str] = {}
for _row in _ROWS.strip().splitlines():
    _source, _translation = _row.split("\t", 1)
    _source = _source.replace(r"\n", "\n").replace(r"\s", " ")
    _translation = _translation.replace(r"\n", "\n").replace(r"\s", " ")
    ENGLISH_UI[_source] = _translation

from workstation.translation_overrides import ENGLISH_OVERRIDES

ENGLISH_UI.update(ENGLISH_OVERRIDES)


def english_for(source: str) -> str:
    """Return deterministic English copy without leaking Han characters."""
    if source in ENGLISH_UI:
        return ENGLISH_UI[source]
    translated = source
    for phrase, replacement in sorted(
        ENGLISH_UI.items(), key=lambda item: len(item[0]), reverse=True
    ):
        if phrase and phrase in translated:
            translated = translated.replace(phrase, replacement)
    translated = translated.translate(str.maketrans("，。；：（）「」", ",.;:()\"\""))
    translated = re.sub(r"[\u3400-\u9fff]+", "[[UNTRANSLATED]]", translated)
    return translated

"""SAM2 边界精修：用 SAM2 的类别无关掩码修整语义分割的边界。

思路（Semantic-Segment-Anything 式的多数投票）：
  1. SAM2 自动生成一组"物体/区域"掩码提案（不带类别，但边界精准）；
  2. 每个提案区域取语义 mask 中的多数类别，纯度达标就整块涂成该类；
  3. 提案没覆盖到的像素保留语义模型的原始预测。

语义模型负责"是什么"，SAM2 负责"边界在哪"——两者互补。
模型首次调用时懒加载（facebook/sam2.1-hiera-tiny，约 40MB，需联网下载一次）。
"""
import numpy as np

_generator = None          # 惰性单例：mask-generation pipeline


def _get_generator():
    global _generator
    if _generator is None:
        import torch
        from transformers import pipeline
        device = 0 if torch.cuda.is_available() else -1
        _generator = pipeline("mask-generation",
                              model="facebook/sam2.1-hiera-tiny",
                              device=device)
    return _generator


def refine_mask(image, sem_mask, purity=0.55, min_area=64, max_cover=0.8,
                points_per_batch=64, max_side=2048):
    """用 SAM2 提案修整语义 mask 的边界。

    image     : PIL Image（原图）
    sem_mask  : (H, W) uint8 类别索引（语义模型输出）
    purity    : 提案区域内多数类占比达到该值才涂色（低于视为跨类提案，跳过）
    min_area  : 忽略过小的提案（噪声）
    max_cover : 忽略覆盖比例过大的提案（会整幅推平）
    max_side  : 长边超过此值时在缩小副本上精修再放回（SAM2 内部按 1024 处理，
                原尺寸提案毫无额外收益，只多耗显存/时间）
    返回 (refined_mask, num_used)：精修后的 mask 与实际采用的提案数
    """
    from PIL import Image as _Image

    orig_shape = sem_mask.shape
    if max(image.size) > max_side:
        scale = max_side / max(image.size)
        small = image.convert("RGB").resize(
            (int(image.size[0] * scale), int(image.size[1] * scale)), _Image.BICUBIC)
        sem_small = np.asarray(_Image.fromarray(sem_mask).resize(small.size,
                                                                 _Image.NEAREST))
        refined, used = refine_mask(small, sem_small, purity, min_area,
                                    max_cover, points_per_batch, max_side)
        back = _Image.fromarray(refined).resize((orig_shape[1], orig_shape[0]),
                                                _Image.NEAREST)
        return np.asarray(back), used

    gen = _get_generator()
    out = gen(image.convert("RGB"), points_per_batch=points_per_batch)
    proposals = [np.asarray(m, bool) for m in out["masks"]]

    h, w = sem_mask.shape
    total = h * w
    refined = sem_mask.copy()

    #   大区域先涂、小区域后涂：小提案（细节）覆盖大提案（背景块）
    proposals.sort(key=lambda m: int(m.sum()), reverse=True)

    used = 0
    for prop in proposals:
        if prop.shape != sem_mask.shape:
            continue
        area = int(prop.sum())
        if area < min_area or area > total * max_cover:
            continue
        labels, counts = np.unique(sem_mask[prop], return_counts=True)
        top = int(counts.argmax())
        if counts[top] / area < purity:
            continue                      # 跨类提案（如 SAM 把两栋房子连成一块）
        refined[prop] = labels[top]
        used += 1
    return refined, used

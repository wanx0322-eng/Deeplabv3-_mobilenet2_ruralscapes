from __future__ import annotations

import os
from pathlib import Path

from PIL import Image

from ruralscape_studio.dataset import inspect_dataset


def make_voc(root: Path) -> Path:
    voc = root / "VOCdevkit" / "VOC2007"
    (voc / "JPEGImages").mkdir(parents=True)
    (voc / "SegmentationClass").mkdir()
    (voc / "ImageSets" / "Segmentation").mkdir(parents=True)
    return voc


def save_image(path: Path, size: tuple[int, int] = (4, 3)) -> None:
    Image.new("RGB", size, (10, 20, 30)).save(path)


def save_mask(
    path: Path,
    size: tuple[int, int] = (4, 3),
    values: tuple[int, ...] = (0, 1),
) -> None:
    mask = Image.new("L", size, values[0])
    pixels = mask.load()
    for index, value in enumerate(values):
        pixels[index % size[0], index // size[0]] = value
    mask.save(path)


def issue_codes(profile) -> list[str]:
    return [issue.code for issue in profile.issues]


def test_pairs_images_and_masks_by_case_insensitive_stem_and_extension(tmp_path) -> None:
    voc = make_voc(tmp_path)
    save_image(voc / "JPEGImages" / "Scene.JPEG")
    save_image(voc / "JPEGImages" / "second.BMP")
    save_mask(voc / "SegmentationClass" / "scene.PNG", values=(0, 2, 7))
    save_mask(voc / "SegmentationClass" / "SECOND.png", values=(0, 1))

    profile = inspect_dataset(tmp_path)

    assert profile.root_path == str(voc.resolve())
    assert profile.total_images == 2
    assert profile.total_masks == 2
    assert profile.class_values == (0, 1, 2, 7)
    assert not {"missing-image", "missing-mask"} & set(issue_codes(profile))


def test_reports_duplicate_stems_and_missing_pairs_deterministically(tmp_path) -> None:
    voc = make_voc(tmp_path)
    save_image(voc / "JPEGImages" / "DUP.jpg")
    save_image(voc / "JPEGImages" / "dup.PNG")
    save_image(voc / "JPEGImages" / "image-only.jpg")
    save_mask(voc / "SegmentationClass" / "dup.png")
    save_mask(voc / "SegmentationClass" / "mask-only.BMP")
    save_mask(voc / "SegmentationClass" / "MASK-ONLY.png")

    profile = inspect_dataset(voc)

    issues = [(issue.code, issue.stem) for issue in profile.issues]
    assert profile.total_images == 3
    assert profile.total_masks == 3
    assert issues == sorted(issues)
    assert ("duplicate-image-stem", "dup") in issues
    assert ("duplicate-mask-stem", "mask-only") in issues
    assert ("missing-mask", "image-only") in issues
    assert ("missing-image", "mask-only") in issues


def test_reports_rgb_masks_shape_mismatch_and_collects_label_values(tmp_path) -> None:
    voc = make_voc(tmp_path)
    save_image(voc / "JPEGImages" / "rgb.png")
    Image.new("RGB", (4, 3), (1, 2, 3)).save(voc / "SegmentationClass" / "rgb.png")
    save_image(voc / "JPEGImages" / "wrong-size.png", size=(4, 3))
    save_mask(
        voc / "SegmentationClass" / "wrong-size.png",
        size=(2, 2),
        values=(0, 9),
    )

    profile = inspect_dataset(voc)

    assert ("rgb-mask", "rgb") in [
        (issue.code, issue.stem) for issue in profile.issues
    ]
    assert ("shape-mismatch", "wrong-size") in [
        (issue.code, issue.stem) for issue in profile.issues
    ]
    assert profile.class_values == (0, 9)


def test_split_membership_is_case_insensitive_unique_and_tracks_unassigned(tmp_path) -> None:
    voc = make_voc(tmp_path)
    for stem in ("Alpha", "beta", "gamma", "unused"):
        save_image(voc / "JPEGImages" / f"{stem}.png")
        save_mask(voc / "SegmentationClass" / f"{stem}.png")
    split_dir = voc / "ImageSets" / "Segmentation"
    (split_dir / "train.txt").write_text("alpha\nBETA.png\nalpha\n", encoding="utf-8")
    (split_dir / "val.txt").write_text("Gamma\n", encoding="utf-8")
    (split_dir / "test.txt").write_text("missing\n", encoding="utf-8")

    profile = inspect_dataset(voc)

    assert dict(profile.split_counts) == {"train": 2, "val": 1, "test": 0}
    assert profile.assigned_count == 3
    assert profile.unassigned_count == 1


def test_inspection_does_not_write_to_dataset(tmp_path) -> None:
    voc = make_voc(tmp_path)
    save_image(voc / "JPEGImages" / "one.jpg")
    save_mask(voc / "SegmentationClass" / "one.png")
    before = {
        path.relative_to(voc): (path.stat().st_mtime_ns, path.stat().st_size)
        for path in voc.rglob("*")
        if path.is_file()
    }

    inspect_dataset(voc)

    after = {
        path.relative_to(voc): (path.stat().st_mtime_ns, path.stat().st_size)
        for path in voc.rglob("*")
        if path.is_file()
    }
    assert before == after


def test_current_repository_dataset_inventory_read_only() -> None:
    repo_root = Path(__file__).resolve().parents[1]

    profile = inspect_dataset(repo_root)

    assert profile.total_images == 313
    assert profile.total_masks == 313
    #   划分由 tools/rebuild_splits.py 生成：验证集 50 张固定不动，
    #   其余可用图进训练集。剩下的 18 张是与验证集像素重复、被剔除的泄漏图。
    assert profile.train_count == 245
    assert profile.val_count == 50
    assert profile.test_count == 0
    assert profile.unassigned_count == 18
    #   148 张 RGB 标签已由 tools/fix_rgb_masks.py 转成索引图，
    #   这里锁死"零问题"状态：再出现 RGB 标签 / 跨划分泄漏 / 图掩码不配对都会红。
    assert profile.issues == ()
    assert profile.class_values == (0, 1, 2, 3, 4)

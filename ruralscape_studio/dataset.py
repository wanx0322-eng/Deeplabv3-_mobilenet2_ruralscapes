"""Read-only inspection of Pascal VOC semantic-segmentation datasets."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from .domain import DatasetIssue, DatasetProfile


IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".bmp"})
_MASK_SCAN_CHUNK_PIXELS = 65_536


def _is_within(candidate: Path, root: Path) -> bool:
    try:
        candidate.relative_to(root)
    except ValueError:
        return False
    return True


def _resolve_contained(path: Path, root: Path) -> Path:
    resolved = path.resolve()
    if not _is_within(resolved, root):
        raise ValueError(f"dataset path escapes dataset root: {path}")
    return resolved


def _candidate_directory_state(candidate: Path) -> tuple[bool, bool]:
    candidate_root = candidate.resolve()
    contained_directories: list[bool] = []
    for name in ("JPEGImages", "SegmentationClass"):
        directory = candidate / name
        contained_directories.append(
            directory.is_dir() and _is_within(directory.resolve(), candidate_root)
        )
    return contained_directories[0], contained_directories[1]


def _voc_root(path: Path) -> Path:
    """Choose the first complete VOC root, then the first partial root, deterministically."""

    candidates = (path, path / "VOC2007", path / "VOCdevkit" / "VOC2007")
    states = [_candidate_directory_state(candidate) for candidate in candidates]
    for candidate, state in zip(candidates, states):
        if all(state):
            return candidate
    for candidate, state in zip(candidates, states):
        if any(state):
            return candidate
    return path


def _group_images(directory: Path, dataset_root: Path) -> dict[str, list[Path]]:
    grouped: dict[str, list[Path]] = defaultdict(list)
    safe_directory = _resolve_contained(directory, dataset_root)
    if not safe_directory.is_dir():
        return {}
    files: list[Path] = []
    for path in safe_directory.iterdir():
        resolved = _resolve_contained(path, dataset_root)
        if resolved.is_file() and path.suffix.casefold() in IMAGE_EXTENSIONS:
            files.append(path)
    files.sort(key=lambda path: (path.name.casefold(), path.name))
    for path in files:
        grouped[path.stem.casefold()].append(path)
    return dict(grouped)


def _split_stems(path: Path, dataset_root: Path) -> set[str]:
    safe_path = _resolve_contained(path, dataset_root)
    if not safe_path.is_file():
        return set()
    stems: set[str] = set()
    for raw_line in safe_path.read_text(encoding="utf-8-sig").splitlines():
        value = raw_line.strip()
        if value and not value.startswith("#"):
            stems.add(Path(value).stem.casefold())
    return stems


def _iter_mask_value_chunks(
    mask: Any, *, max_pixels: int = _MASK_SCAN_CHUNK_PIXELS
) -> Iterator[tuple[int, ...]]:
    """Yield class values from spatial tiles bounded by ``max_pixels``.

    Each yielded tuple contains the distinct values in one tile. Histograms keep
    the temporary representation fixed-size for byte-indexed masks, avoiding a
    second full-image pixel tuple.
    """

    if max_pixels < 1:
        raise ValueError("max_pixels must be positive")
    width, height = mask.size
    tile_width = min(width, max_pixels)
    tile_height = max(1, max_pixels // max(1, tile_width))
    for top in range(0, height, tile_height):
        bottom = min(height, top + tile_height)
        for left in range(0, width, tile_width):
            right = min(width, left + tile_width)
            tile = mask.crop((left, top, right, bottom))
            tile.load()
            if tile.mode in {"1", "L", "P"}:
                histogram = tile.histogram()
                yield tuple(index for index, count in enumerate(histogram) if count)
            else:
                values = {
                    int(tile.getpixel((x, y)))
                    for y in range(tile.height)
                    for x in range(tile.width)
                }
                yield tuple(sorted(values))


def inspect_dataset(dataset_root: str | Path) -> DatasetProfile:
    """Inventory a VOC dataset and report pairing, split, and mask issues.

    Pillow's configured ``Image.MAX_IMAGE_PIXELS`` ceiling remains in force.
    Files exceeding its hard limit are reported as unreadable per-file issues so
    one decompression bomb does not abort inspection of the remaining dataset.
    """

    # Pillow is deliberately imported only when an inspection is requested.
    from PIL import Image, UnidentifiedImageError

    requested_root = Path(dataset_root).expanduser().resolve()
    root = _voc_root(requested_root).resolve()
    image_groups = _group_images(root / "JPEGImages", root)
    mask_groups = _group_images(root / "SegmentationClass", root)
    issues: list[DatasetIssue] = []

    for stem, paths in image_groups.items():
        if len(paths) > 1:
            issues.append(
                DatasetIssue(
                    code="duplicate-image-stem",
                    message=f"Multiple images use the case-insensitive stem '{stem}'.",
                    stem=stem,
                    image_path=str(paths[0]),
                )
            )
    for stem, paths in mask_groups.items():
        if len(paths) > 1:
            issues.append(
                DatasetIssue(
                    code="duplicate-mask-stem",
                    message=f"Multiple masks use the case-insensitive stem '{stem}'.",
                    stem=stem,
                    mask_path=str(paths[0]),
                )
            )

    image_stems = set(image_groups)
    mask_stems = set(mask_groups)
    for stem in image_stems - mask_stems:
        issues.append(
            DatasetIssue(
                code="missing-mask",
                message=f"No mask matches image stem '{stem}'.",
                stem=stem,
                image_path=str(image_groups[stem][0]),
            )
        )
    for stem in mask_stems - image_stems:
        issues.append(
            DatasetIssue(
                code="missing-image",
                message=f"No image matches mask stem '{stem}'.",
                stem=stem,
                mask_path=str(mask_groups[stem][0]),
            )
        )

    pillow_read_errors = (
        OSError,
        UnidentifiedImageError,
        Image.DecompressionBombError,
    )
    class_values: set[int] = set()
    for stem in sorted(image_stems & mask_stems):
        image_path = image_groups[stem][0]
        mask_path = mask_groups[stem][0]
        try:
            safe_image_path = _resolve_contained(image_path, root)
            with Image.open(safe_image_path) as image:
                image_size = image.size
                image.load()
        except pillow_read_errors as error:
            issues.append(
                DatasetIssue(
                    code="unreadable-image",
                    message=f"Image cannot be read: {error}",
                    severity="error",
                    stem=stem,
                    image_path=str(image_path),
                    mask_path=str(mask_path),
                )
            )
            continue

        try:
            safe_mask_path = _resolve_contained(mask_path, root)
            mask_values: set[int] = set()
            with Image.open(safe_mask_path) as mask:
                mask_size = mask.size
                if len(mask.getbands()) > 1:
                    mask.load()
                    issues.append(
                        DatasetIssue(
                            code="rgb-mask",
                            message="Mask has multiple color channels instead of class indices.",
                            severity="error",
                            stem=stem,
                            image_path=str(image_path),
                            mask_path=str(mask_path),
                        )
                    )
                else:
                    for values in _iter_mask_value_chunks(mask):
                        mask_values.update(values)
            class_values.update(mask_values)
        except pillow_read_errors as error:
            issues.append(
                DatasetIssue(
                    code="unreadable-mask",
                    message=f"Mask cannot be read: {error}",
                    severity="error",
                    stem=stem,
                    image_path=str(image_path),
                    mask_path=str(mask_path),
                )
            )
            continue

        if image_size != mask_size:
            issues.append(
                DatasetIssue(
                    code="shape-mismatch",
                    message=f"Image size {image_size} differs from mask size {mask_size}.",
                    severity="error",
                    stem=stem,
                    image_path=str(image_path),
                    mask_path=str(mask_path),
                )
            )

    split_dir = root / "ImageSets" / "Segmentation"
    split_members = {
        split: _split_stems(split_dir / f"{split}.txt", root)
        for split in ("train", "val", "test")
    }
    split_counts = {
        split: len(members & image_stems)
        for split, members in split_members.items()
    }
    assigned_stems = set().union(*split_members.values()) & image_stems

    membership: dict[str, list[str]] = defaultdict(list)
    for split, members in split_members.items():
        for stem in sorted(members - image_stems):
            issues.append(
                DatasetIssue(
                    code="unknown-split-member",
                    message=f"Split '{split}' references unknown image stem '{stem}'.",
                    severity="error",
                    stem=stem,
                )
            )
        for stem in members & image_stems:
            membership[stem].append(split)
    for stem, splits in membership.items():
        if len(splits) > 1:
            issues.append(
                DatasetIssue(
                    code="cross-split-leakage",
                    message=f"Image stem '{stem}' appears in splits: {', '.join(sorted(splits))}.",
                    severity="error",
                    stem=stem,
                )
            )

    issues.sort(
        key=lambda issue: (
            issue.code,
            issue.stem or "",
            issue.image_path or "",
            issue.mask_path or "",
            issue.message,
        )
    )
    return DatasetProfile(
        root_path=str(root),
        total_images=sum(len(paths) for paths in image_groups.values()),
        total_masks=sum(len(paths) for paths in mask_groups.values()),
        assigned_count=len(assigned_stems),
        unassigned_count=len(image_stems - assigned_stems),
        split_counts=split_counts,
        class_values=tuple(sorted(class_values)),
        issues=tuple(issues),
    )


scan_dataset = inspect_dataset

__all__ = ["IMAGE_EXTENSIONS", "inspect_dataset", "scan_dataset"]

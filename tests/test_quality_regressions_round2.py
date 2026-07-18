from __future__ import annotations

import os
import re
from pathlib import Path

import pytest
from PIL import Image

import ruralscape_studio as studio
from ruralscape_studio import discovery
from ruralscape_studio.storage import StudioRepository


def _make_voc(root: Path) -> Path:
    voc = root / "VOCdevkit" / "VOC2007"
    (voc / "JPEGImages").mkdir(parents=True)
    (voc / "SegmentationClass").mkdir()
    (voc / "ImageSets" / "Segmentation").mkdir(parents=True)
    return voc


def _save_pair(voc: Path, stem: str) -> None:
    Image.new("RGB", (4, 3), (10, 20, 30)).save(voc / "JPEGImages" / f"{stem}.png")
    Image.new("L", (4, 3), 0).save(voc / "SegmentationClass" / f"{stem}.png")


def test_repository_explicitly_rejects_process_local_memory_database() -> None:
    with pytest.raises(ValueError, match=r":memory:.*not supported"):
        StudioRepository(":memory:")


def test_project_id_is_a_safe_windows_path_component_for_very_long_slugs() -> None:
    project_id = discovery._project_id(Path("x" * 400))

    assert len(project_id) <= 255
    assert re.fullmatch(r"x+-[0-9a-f]{12}", project_id)


def test_posix_case_distinct_root_identities_hash_differently(monkeypatch) -> None:
    monkeypatch.setattr(os.path, "normcase", lambda value: value)

    upper = discovery._project_id(Path("parent") / "CaseRoot")
    lower = discovery._project_id(Path("parent") / "caseroot")

    assert upper != lower


def test_voc_root_prefers_complete_nested_candidate_over_partial_parent(tmp_path) -> None:
    parent = tmp_path / "dataset"
    (parent / "JPEGImages").mkdir(parents=True)
    nested = parent / "VOC2007"
    (nested / "JPEGImages").mkdir(parents=True)
    (nested / "SegmentationClass").mkdir()
    (nested / "ImageSets" / "Segmentation").mkdir(parents=True)
    _save_pair(nested, "scene")

    profile = studio.inspect_dataset(parent)

    assert profile.root_path == str(nested.resolve())
    assert profile.total_images == 1
    assert profile.total_masks == 1


@pytest.mark.parametrize(
    ("bomb_directory", "expected_issue"),
    [
        ("JPEGImages", "unreadable-image"),
        ("SegmentationClass", "unreadable-mask"),
    ],
)
def test_decompression_bombs_become_per_file_issues(
    tmp_path, monkeypatch, bomb_directory: str, expected_issue: str
) -> None:
    voc = _make_voc(tmp_path)
    _save_pair(voc, "scene")
    original_open = Image.open

    def bomb_one_file(path, *args, **kwargs):
        if Path(path).parent.name == bomb_directory:
            raise Image.DecompressionBombError("configured pixel ceiling exceeded")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Image, "open", bomb_one_file)

    profile = studio.inspect_dataset(voc)

    assert (expected_issue, "scene") in {
        (issue.code, issue.stem) for issue in profile.issues
    }


def test_dataset_scan_documents_pillow_pixel_ceiling_policy() -> None:
    assert "MAX_IMAGE_PIXELS" in (studio.inspect_dataset.__doc__ or "")


def test_dataset_rejects_image_directory_symlink_escape(tmp_path) -> None:
    voc = tmp_path / "VOCdevkit" / "VOC2007"
    outside = tmp_path / "outside-images"
    outside.mkdir()
    Image.new("RGB", (2, 2)).save(outside / "scene.png")
    voc.mkdir(parents=True)
    (voc / "SegmentationClass").mkdir()
    try:
        (voc / "JPEGImages").symlink_to(outside, target_is_directory=True)
    except OSError as error:
        pytest.skip(f"directory symlinks unavailable: {error}")

    with pytest.raises(ValueError, match="escapes dataset root"):
        studio.inspect_dataset(voc)


def test_dataset_rejects_individual_image_symlink_escape(tmp_path) -> None:
    voc = _make_voc(tmp_path)
    outside = tmp_path / "outside.png"
    Image.new("RGB", (2, 2)).save(outside)
    Image.new("L", (2, 2), 0).save(voc / "SegmentationClass" / "scene.png")
    try:
        (voc / "JPEGImages" / "scene.png").symlink_to(outside)
    except OSError as error:
        pytest.skip(f"file symlinks unavailable: {error}")

    with pytest.raises(ValueError, match="escapes dataset root"):
        studio.inspect_dataset(voc)

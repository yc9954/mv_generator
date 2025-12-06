"""Unit tests for colab_helpers module."""

import pytest
import tempfile
import json
import os
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from colab_helpers import (
    ensure_dirs,
    sha256,
    log_and_print,
    save_manifest,
    load_manifest,
    save_runs_meta,
    list_artifacts,
    validate_input_video
)


@pytest.fixture
def temp_root():
    """Create a temporary root directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


def test_ensure_dirs_creates_structure(temp_root):
    """Test that ensure_dirs creates all required subdirectories."""
    paths = ensure_dirs(temp_root)

    # Check that all expected directories were created
    expected_dirs = [
        "input", "frames", "masks", "masks_alpha", "masks_coarse",
        "poses", "gs_shim", "gs", "outputs", "logs", "checkpoints", "actor_rgba"
    ]

    for dir_name in expected_dirs:
        assert dir_name in paths
        assert os.path.exists(paths[dir_name])
        assert os.path.isdir(paths[dir_name])

    # Check root is also in paths
    assert "root" in paths
    assert paths["root"] == temp_root


def test_ensure_dirs_idempotent(temp_root):
    """Test that ensure_dirs can be called multiple times safely."""
    paths1 = ensure_dirs(temp_root)
    paths2 = ensure_dirs(temp_root)

    assert paths1 == paths2


def test_sha256_valid_file(temp_root):
    """Test SHA256 hash computation for a valid file."""
    test_file = Path(temp_root) / "test.txt"
    test_content = b"Hello, World!"
    test_file.write_bytes(test_content)

    hash_result = sha256(str(test_file))

    # Known SHA256 hash of "Hello, World!"
    expected_hash = "dffd6021bb2bd5b0af676290809ec3a53191dd81c7f70a4b28688a362182986f"
    assert hash_result == expected_hash


def test_sha256_deterministic(temp_root):
    """Test that SHA256 returns same hash for same content."""
    test_file = Path(temp_root) / "test.txt"
    test_file.write_text("test content")

    hash1 = sha256(str(test_file))
    hash2 = sha256(str(test_file))

    assert hash1 == hash2


def test_sha256_nonexistent_file():
    """Test SHA256 behavior with non-existent file."""
    hash_result = sha256("/nonexistent/file.txt")
    assert hash_result == "file_not_found"


def test_log_and_print_no_file(capsys):
    """Test log_and_print prints to stdout."""
    log_and_print("Test message")

    captured = capsys.readouterr()
    assert "Test message" in captured.out
    assert "[INFO]" in captured.out


def test_log_and_print_with_file(temp_root, capsys):
    """Test log_and_print writes to file."""
    log_file = Path(temp_root) / "test.log"

    log_and_print("Test message", str(log_file), level="WARNING")

    # Check file was created and contains message
    assert log_file.exists()
    content = log_file.read_text()
    assert "Test message" in content
    assert "[WARNING]" in content

    # Check it also printed to stdout
    captured = capsys.readouterr()
    assert "Test message" in captured.out


def test_save_manifest_creates_file(temp_root):
    """Test save_manifest creates manifest files."""
    details = {
        "frames_count": 100,
        "artifacts": {
            "frames_dir": "/path/to/frames",
            "masks_dir": "/path/to/masks"
        }
    }

    manifest_path = save_manifest(temp_root, "frames_extracted", details)

    # Check manifest was created
    assert os.path.exists(manifest_path)

    # Check content
    with open(manifest_path) as f:
        manifest = json.load(f)

    assert manifest["job_id"] == "colab-demo"
    assert manifest["stage"] == "frames_extracted"
    assert manifest["frames_count"] == 100
    assert "timestamp" in manifest
    assert manifest["artifacts"]["frames_dir"] == "/path/to/frames"

    # Check latest manifest was also created
    latest_path = Path(temp_root) / "checkpoints" / "manifest_latest.json"
    assert latest_path.exists()


def test_load_manifest_existing(temp_root):
    """Test load_manifest retrieves saved manifest."""
    # Save a manifest first
    details = {"frames_count": 42}
    save_manifest(temp_root, "test_stage", details)

    # Load it back
    manifest = load_manifest(temp_root, "test_stage")

    assert manifest is not None
    assert manifest["stage"] == "test_stage"
    assert manifest["frames_count"] == 42


def test_load_manifest_nonexistent(temp_root):
    """Test load_manifest returns None for non-existent manifest."""
    manifest = load_manifest(temp_root, "nonexistent_stage")
    assert manifest is None


def test_load_manifest_latest(temp_root):
    """Test load_manifest can retrieve latest manifest."""
    # Save multiple manifests
    save_manifest(temp_root, "stage1", {"count": 1})
    save_manifest(temp_root, "stage2", {"count": 2})

    # Load latest (should be stage2)
    manifest = load_manifest(temp_root)

    assert manifest is not None
    assert manifest["stage"] == "stage2"
    assert manifest["count"] == 2


def test_save_runs_meta(temp_root):
    """Test save_runs_meta creates metadata file."""
    meta_path = save_runs_meta(
        temp_root,
        progress_percent=50.0,
        last_stage="frames_extracted",
        is_a100=True,
        additional_meta={"custom_field": "value"}
    )

    assert os.path.exists(meta_path)

    with open(meta_path) as f:
        meta = json.load(f)

    assert meta["progress_percent"] == 50.0
    assert meta["last_stage"] == "frames_extracted"
    assert meta["is_a100"] is True
    assert meta["custom_field"] == "value"
    assert "timestamp" in meta
    assert "timestamp_human" in meta


def test_list_artifacts_empty(temp_root):
    """Test list_artifacts on empty directory."""
    artifacts = list_artifacts(temp_root)
    assert artifacts == []


def test_list_artifacts_with_files(temp_root):
    """Test list_artifacts finds files."""
    # Create some test files
    test_dir = Path(temp_root) / "test"
    test_dir.mkdir()
    (test_dir / "file1.txt").touch()
    (test_dir / "file2.png").touch()

    # List all files
    artifacts = list_artifacts(temp_root)
    assert len(artifacts) >= 2

    # List specific pattern
    png_artifacts = list_artifacts(temp_root, "*.png")
    assert len(png_artifacts) >= 1
    assert all(p.endswith(".png") for p in png_artifacts)


def test_validate_input_video_nonexistent():
    """Test validate_input_video with non-existent file."""
    result = validate_input_video("/nonexistent/video.mp4")

    assert result["valid"] is False
    assert result["exists"] is False
    assert "does not exist" in result["error"]


def test_validate_input_video_empty(temp_root):
    """Test validate_input_video with empty file."""
    empty_file = Path(temp_root) / "empty.mp4"
    empty_file.touch()

    result = validate_input_video(str(empty_file))

    assert result["valid"] is False
    assert result["exists"] is True
    assert result["size_mb"] == 0
    assert "empty" in result["error"]


def test_validate_input_video_valid(temp_root):
    """Test validate_input_video with valid file."""
    video_file = Path(temp_root) / "video.mp4"
    video_file.write_bytes(b"fake video content" * 1000)

    result = validate_input_video(str(video_file))

    assert result["valid"] is True
    assert result["exists"] is True
    assert result["size_mb"] > 0
    assert result["error"] is None

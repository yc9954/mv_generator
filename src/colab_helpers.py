"""
Colab helper utilities for 4DGS MV pipeline.

Provides functions for:
- Directory management
- Manifest and checkpoint saving
- File hashing
- Logging and error handling
"""

import os
import json
import hashlib
import time
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime


def ensure_dirs(root: str) -> Dict[str, str]:
    """
    Create all required subdirectories under ROOT.

    Args:
        root: Root directory path (typically on Google Drive)

    Returns:
        Dictionary mapping directory names to their full paths
    """
    subdirs = [
        "input",
        "frames",
        "masks",
        "masks/alpha",
        "masks/coarse",
        "poses",
        "gs_shim",
        "gs",
        "outputs",
        "logs",
        "checkpoints",
        "actor_rgba"
    ]

    paths = {}
    root_path = Path(root)
    root_path.mkdir(parents=True, exist_ok=True)

    for subdir in subdirs:
        dir_path = root_path / subdir
        dir_path.mkdir(parents=True, exist_ok=True)
        paths[subdir.replace("/", "_")] = str(dir_path)

    # Also save the root
    paths["root"] = str(root_path)

    return paths


def sha256(file_path: str) -> str:
    """
    Compute SHA256 hash of a file.

    Args:
        file_path: Path to file to hash

    Returns:
        Hexadecimal SHA256 hash string
    """
    hash_sha256 = hashlib.sha256()

    try:
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    except FileNotFoundError:
        return "file_not_found"
    except Exception as e:
        return f"error_{str(e)[:20]}"


def log_and_print(msg: str, log_path: Optional[str] = None, level: str = "INFO") -> None:
    """
    Print message to console and optionally append to log file.

    Args:
        msg: Message to log
        log_path: Optional path to log file
        level: Log level (INFO, WARNING, ERROR)
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted_msg = f"[{timestamp}] [{level}] {msg}"

    print(formatted_msg)

    if log_path:
        try:
            Path(log_path).parent.mkdir(parents=True, exist_ok=True)
            with open(log_path, "a") as f:
                f.write(formatted_msg + "\n")
        except Exception as e:
            print(f"Failed to write to log file {log_path}: {e}")


def save_manifest(
    root: str,
    stage: str,
    details: Dict[str, Any],
    job_id: str = "colab-demo"
) -> str:
    """
    Save pipeline manifest with stage progress and artifact paths.

    Args:
        root: Root directory path
        stage: Current pipeline stage name
        details: Dictionary with stage-specific details (frames_count, artifacts, etc.)
        job_id: Unique job identifier

    Returns:
        Path to saved manifest file
    """
    manifest_path = Path(root) / "checkpoints" / f"manifest_{stage}.json"

    manifest = {
        "job_id": job_id,
        "stage": stage,
        "timestamp": int(time.time()),
        **details
    }

    try:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

        # Also update the latest manifest
        latest_path = Path(root) / "checkpoints" / "manifest_latest.json"
        with open(latest_path, "w") as f:
            json.dump(manifest, f, indent=2)

        return str(manifest_path)
    except Exception as e:
        log_and_print(
            f"Failed to save manifest: {e}",
            Path(root) / "logs" / "manifest_err.log",
            level="ERROR"
        )
        raise


def load_manifest(root: str, stage: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    Load pipeline manifest from checkpoint.

    Args:
        root: Root directory path
        stage: Specific stage name, or None to load latest

    Returns:
        Manifest dictionary or None if not found
    """
    if stage:
        manifest_path = Path(root) / "checkpoints" / f"manifest_{stage}.json"
    else:
        manifest_path = Path(root) / "checkpoints" / "manifest_latest.json"

    try:
        if manifest_path.exists():
            with open(manifest_path, "r") as f:
                return json.load(f)
        return None
    except Exception as e:
        log_and_print(
            f"Failed to load manifest: {e}",
            Path(root) / "logs" / "manifest_err.log",
            level="ERROR"
        )
        return None


def save_runs_meta(
    root: str,
    progress_percent: float,
    last_stage: str,
    is_a100: bool = False,
    additional_meta: Optional[Dict[str, Any]] = None
) -> str:
    """
    Save run metadata including progress and GPU info.

    Args:
        root: Root directory path
        progress_percent: Overall pipeline progress (0-100)
        last_stage: Last completed stage name
        is_a100: Whether running on A100 GPU
        additional_meta: Optional additional metadata

    Returns:
        Path to saved metadata file
    """
    meta_path = Path(root) / "runs_meta.json"

    meta = {
        "progress_percent": progress_percent,
        "last_stage": last_stage,
        "is_a100": is_a100,
        "timestamp": int(time.time()),
        "timestamp_human": datetime.now().isoformat()
    }

    if additional_meta:
        meta.update(additional_meta)

    try:
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        return str(meta_path)
    except Exception as e:
        log_and_print(
            f"Failed to save runs metadata: {e}",
            Path(root) / "logs" / "meta_err.log",
            level="ERROR"
        )
        raise


def list_artifacts(root: str, pattern: str = "*") -> List[str]:
    """
    List artifact files matching pattern.

    Args:
        root: Root directory path
        pattern: Glob pattern for matching files

    Returns:
        List of matching file paths
    """
    root_path = Path(root)

    try:
        matches = list(root_path.glob(f"**/{pattern}"))
        return [str(m) for m in matches if m.is_file()]
    except Exception as e:
        log_and_print(
            f"Failed to list artifacts: {e}",
            Path(root) / "logs" / "artifacts_err.log",
            level="ERROR"
        )
        return []


def validate_input_video(video_path: str) -> Dict[str, Any]:
    """
    Validate input video file and extract basic metadata.

    Args:
        video_path: Path to input video file

    Returns:
        Dictionary with validation results and metadata
    """
    result = {
        "valid": False,
        "exists": False,
        "size_mb": 0,
        "error": None
    }

    try:
        path = Path(video_path)
        result["exists"] = path.exists()

        if not result["exists"]:
            result["error"] = "File does not exist"
            return result

        size_bytes = path.stat().st_size
        result["size_mb"] = size_bytes / (1024 * 1024)

        # Basic validation: file should be > 0 bytes
        if size_bytes == 0:
            result["error"] = "File is empty"
            return result

        result["valid"] = True

    except Exception as e:
        result["error"] = str(e)

    return result

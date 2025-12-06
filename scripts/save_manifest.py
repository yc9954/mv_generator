#!/usr/bin/env python3
"""
CLI tool to collect artifact paths and save manifest JSON.

Usage:
    python save_manifest.py --root /path/to/root --stage frames_extracted \\
        --frames-count 100 --artifact frames=/path/to/frames
"""

import argparse
import json
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Save pipeline manifest with artifact paths"
    )

    parser.add_argument(
        "--root",
        required=True,
        help="Root directory path (Google Drive workspace)"
    )

    parser.add_argument(
        "--stage",
        required=True,
        help="Current pipeline stage name"
    )

    parser.add_argument(
        "--job-id",
        default="colab-demo",
        help="Job identifier (default: colab-demo)"
    )

    parser.add_argument(
        "--frames-count",
        type=int,
        help="Number of frames extracted"
    )

    parser.add_argument(
        "--artifact",
        action="append",
        help="Artifact in format key=path (can be specified multiple times)"
    )

    parser.add_argument(
        "--metadata",
        action="append",
        help="Additional metadata in format key=value (can be specified multiple times)"
    )

    args = parser.parse_args()

    # Build artifacts dictionary
    artifacts = {}
    if args.artifact:
        for artifact in args.artifact:
            if "=" not in artifact:
                print(f"Error: Artifact must be in format key=path, got: {artifact}", file=sys.stderr)
                sys.exit(1)

            key, path = artifact.split("=", 1)
            artifacts[key] = path

    # Build metadata dictionary
    metadata = {}
    if args.metadata:
        for meta in args.metadata:
            if "=" not in meta:
                print(f"Error: Metadata must be in format key=value, got: {meta}", file=sys.stderr)
                sys.exit(1)

            key, value = meta.split("=", 1)
            metadata[key] = value

    # Build manifest
    manifest = {
        "job_id": args.job_id,
        "stage": args.stage,
    }

    if args.frames_count is not None:
        manifest["frames_count"] = args.frames_count

    if artifacts:
        manifest["artifacts"] = artifacts

    if metadata:
        manifest.update(metadata)

    # Add timestamp
    import time
    manifest["timestamp"] = int(time.time())

    # Save manifest
    root_path = Path(args.root)
    checkpoints_dir = root_path / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = checkpoints_dir / f"manifest_{args.stage}.json"
    latest_path = checkpoints_dir / "manifest_latest.json"

    try:
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

        with open(latest_path, "w") as f:
            json.dump(manifest, f, indent=2)

        print(f"Manifest saved to: {manifest_path}")
        print(f"Latest manifest updated: {latest_path}")

    except Exception as e:
        print(f"Error saving manifest: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Run the placeholder pipeline on samples/tiny_sample.mp4 locally (CPU, no Colab, no
4DGS build) and render the figures used in README.md into docs/.

It calls the same stage functions as run_pipeline.py (extract_frames, generate_masks,
estimate_poses, run_4dgs --debug_shim, export_green_screen) and the Colab-only stages
(Farneback temporal smoothing, gs_shim, bicubic SR) with the notebook's parameters.

    pip install opencv-python-headless numpy matplotlib open3d tqdm
    python scripts/render_docs_figures.py --root /tmp/mv_demo
"""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

import run_pipeline as rp  # noqa: E402
from colab_helpers import ensure_dirs, save_manifest, validate_input_video  # noqa: E402
from gs_shim import create_simple_scene, generate_bg_plane, load_and_preview_ply  # noqa: E402


def checkerboard(h, w, cell=24):
    yy, xx = np.mgrid[0:h, 0:w]
    board = (((yy // cell) + (xx // cell)) % 2).astype(np.uint8)
    return np.where(board[..., None] == 1, 200, 150).astype(np.uint8).repeat(3, axis=2)


def over(rgba, background):
    a = rgba[:, :, 3:4].astype(np.float32) / 255.0
    return (rgba[:, :, :3] * a + background * (1 - a)).astype(np.uint8)


def rgb(img_bgr):
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/tmp/mv_demo")
    ap.add_argument("--video", default=str(REPO / "samples" / "tiny_sample.mp4"))
    ap.add_argument("--fps", type=int, default=5)
    ap.add_argument("--docs", default=str(REPO / "docs"))
    args = ap.parse_args()

    root = Path(args.root)
    if root.exists():
        shutil.rmtree(root)
    dirs = ensure_dirs(str(root))
    docs = Path(args.docs)
    docs.mkdir(parents=True, exist_ok=True)

    # --- stages, exactly as run_pipeline.py orders them ---------------------------------
    inp = Path(dirs["input"]) / "input.mp4"
    shutil.copy(args.video, inp)
    val = validate_input_video(str(inp))
    assert val["valid"], val
    n = rp.extract_frames(inp, Path(dirs["frames"]), args.fps)
    save_manifest(str(root), "frames_extracted", {"frames_count": n})
    # the notebook's threshold (100) and blur (15x15); run_pipeline.py's default of 30 keeps the sample's
    # #333 background (gray 51) as foreground, see README "Project status"
    rp.generate_masks(Path(dirs["frames"]), Path(dirs["masks_coarse"]), Path(dirs["masks_alpha"]), threshold=100, blur=(15, 15))
    save_manifest(str(root), "masks_generated", {"method": "threshold + gaussian blur (placeholder)"})

    # Colab cell 9: Farneback temporal smoothing (keep the un-smoothed mattes for the figure)
    alpha_paths = sorted(Path(dirs["masks_alpha"]).glob("*_alpha.png"))
    raw_alpha = [cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) for p in alpha_paths]
    flows = [None]
    for i in range(1, len(raw_alpha)):
        flow = cv2.calcOpticalFlowFarneback(raw_alpha[i - 1], raw_alpha[i], None, 0.5, 3, 15, 3, 5, 1.2, 0)
        flows.append(flow)
        smoothed = cv2.addWeighted(raw_alpha[i], 0.7, raw_alpha[i - 1], 0.3, 0)
        cv2.imwrite(str(alpha_paths[i]), smoothed)
    save_manifest(str(root), "temporal_smoothed", {"method": "farneback + 0.7/0.3 blend"})

    rp.estimate_poses(str(root), Path(dirs["frames"]))
    rp.run_4dgs(str(root), "/nonexistent/4dgs_repo", Path(dirs["frames"]), debug_shim=True)
    ply = generate_bg_plane(str(root), grid_size=(600, 300), z_depth=4.0)
    scene = create_simple_scene(str(root))
    stats = load_and_preview_ply(ply)
    save_manifest(str(root), "gs_shim", {"ply": ply, **stats})
    rp.export_green_screen(str(root), args.fps)

    # Colab cell 15: bicubic 2x "super-resolution" placeholder
    first = sorted(Path(dirs["frames"]).glob("*.png"))[0]
    img = cv2.imread(str(first))
    h, w = img.shape[:2]
    cv2.imwrite(str(Path(dirs["outputs"]) / "sr_sample.png"), cv2.resize(img, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC))
    save_manifest(str(root), "complete", {"artifacts": sorted(str(p.relative_to(root)) for p in root.rglob("*") if p.is_file())})

    # --- figure 1: the stage strip ------------------------------------------------------
    frames = sorted(Path(dirs["frames"]).glob("*.png"))
    coarse = sorted(Path(dirs["masks_coarse"]).glob("*_mask.png"))
    rgba = sorted(Path(dirs["actor_rgba"]).glob("*.png"))
    pick = [0, len(frames) - 1]
    cols = ["input frame", "coarse mask (threshold)", "alpha matte (blur + Farneback)", "actor RGBA", "green screen"]
    fig, axes = plt.subplots(len(pick), len(cols), figsize=(15, 2.6 * len(pick) + 0.6))
    for r, i in enumerate(pick):
        f = cv2.imread(str(frames[i]))
        m = cv2.imread(str(coarse[i]), cv2.IMREAD_GRAYSCALE)
        a = cv2.imread(str(alpha_paths[i]), cv2.IMREAD_GRAYSCALE)
        ra = cv2.imread(str(rgba[i]), cv2.IMREAD_UNCHANGED)
        hh, ww = f.shape[:2]
        panels = [rgb(f), m, a, rgb(over(ra, checkerboard(hh, ww))), rgb(over(ra, np.full((hh, ww, 3), (0, 255, 0), np.uint8)))]
        for c, p in enumerate(panels):
            ax = axes[r, c]
            ax.imshow(p, cmap="gray" if p.ndim == 2 else None, vmin=0, vmax=255)
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(cols[c], fontsize=11)
            if c == 0:
                ax.set_ylabel(f"frame {i:06d}", fontsize=11)
    fig.suptitle("run_pipeline.py stages on samples/tiny_sample.mp4 (placeholder masks at the notebook's threshold 100, DEBUG_SHIM)", fontsize=12)
    fig.tight_layout()
    fig.savefig(docs / "stages.png", dpi=110)
    plt.close(fig)

    # --- figure 2: temporal smoothing --------------------------------------------------
    i = len(raw_alpha) - 1
    flow = flows[i]
    mag = np.linalg.norm(flow, axis=2)
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.4))
    axes[0].imshow(raw_alpha[i - 1], cmap="gray", vmin=0, vmax=255); axes[0].set_title(f"alpha t-1 (frame {i - 1:06d})")
    axes[1].imshow(raw_alpha[i], cmap="gray", vmin=0, vmax=255); axes[1].set_title(f"alpha t (frame {i:06d})")
    im = axes[2].imshow(mag, cmap="magma"); axes[2].set_title("Farneback flow magnitude, px")
    fig.colorbar(im, ax=axes[2], fraction=0.046)
    axes[3].imshow(cv2.imread(str(alpha_paths[i]), cv2.IMREAD_GRAYSCALE), cmap="gray", vmin=0, vmax=255)
    axes[3].set_title("smoothed: 0.7 * t + 0.3 * (t-1)")
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Temporal smoothing stage (Colab cell 9): the flow is computed but only the weighted blend is written back", fontsize=11)
    fig.tight_layout()
    fig.savefig(docs / "temporal_smoothing.png", dpi=110)
    plt.close(fig)

    # --- figure 3: the Open3D shim scenes ----------------------------------------------
    import open3d as o3d
    fig = plt.figure(figsize=(13, 5))
    for k, (path, title) in enumerate([(ply, "gs_shim/bg_plane.ply (600 x 300 gradient plane at z = 4)"),
                                       (scene, "gs_shim/simple_scene.ply (floor y = -1, back wall z = 5)")]):
        pcd = o3d.io.read_point_cloud(path)
        pts = np.asarray(pcd.points); col = np.asarray(pcd.colors)
        step = max(1, len(pts) // 20000)
        ax = fig.add_subplot(1, 2, k + 1, projection="3d")
        ax.scatter(pts[::step, 0], pts[::step, 2], pts[::step, 1], c=col[::step], s=1.5, depthshade=False)
        ax.set_xlabel("x"); ax.set_ylabel("z (depth)"); ax.set_zlabel("y")
        ax.set_title(f"{title}\n{len(pts):,} points", fontsize=10)
        ax.view_init(elev=22, azim=-58)
    fig.suptitle("DEBUG_SHIM stand-ins for a trained 4DGS scene, written by src/gs_shim.py with Open3D", fontsize=11)
    fig.tight_layout()
    fig.savefig(docs / "gs_shim.png", dpi=110)
    plt.close(fig)

    # --- figure 4: overview hero (frame -> matte -> green screen, plus manifest) --------
    i = 0  # frame 0 is the only matte the temporal blend leaves untouched
    f = cv2.imread(str(frames[i])); a = cv2.imread(str(alpha_paths[i]), cv2.IMREAD_GRAYSCALE)
    ra = cv2.imread(str(rgba[i]), cv2.IMREAD_UNCHANGED)
    hh, ww = f.shape[:2]
    fig = plt.figure(figsize=(16, 4.2))
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 1.15])
    for c, (p, t) in enumerate([(rgb(f), "1. frames/ (ffmpeg)"), (a, "2. masks/alpha/ (threshold, blur, flow)"),
                                (rgb(over(ra, np.full((hh, ww, 3), (0, 255, 0), np.uint8))), "3. outputs/green_screen.mp4")]):
        ax = fig.add_subplot(gs[0, c]); ax.imshow(p, cmap="gray" if p.ndim == 2 else None, vmin=0, vmax=255)
        ax.set_title(t, fontsize=11); ax.set_xticks([]); ax.set_yticks([])
    ax = fig.add_subplot(gs[0, 3]); ax.axis("off")
    latest = json.load(open(root / "checkpoints" / "manifest_latest.json"))
    lines = ["checkpoints/manifest_latest.json", ""] + [f"{k}: {v if not isinstance(v, list) else str(len(v)) + ' files'}"
                                                         for k, v in latest.items()]
    tree = ["", "workspace layout:"] + [f"  {d}/" for d in ["input", "frames", "masks/coarse", "masks/alpha", "poses",
                                                             "gs_shim", "gs", "actor_rgba", "outputs", "logs", "checkpoints"]]
    ax.text(0.0, 1.0, "\n".join(lines + tree), va="top", ha="left", family="monospace", fontsize=9.5, transform=ax.transAxes)
    ax.set_title("4. resumable manifests", fontsize=11)
    fig.suptitle("mv_generator: single video in, masked actor + manifests out (DEBUG_SHIM run on the bundled sample)", fontsize=12)
    fig.tight_layout()
    fig.savefig(docs / "overview.png", dpi=110)
    plt.close(fig)
    print("wrote", sorted(p.name for p in docs.glob("*.png")))


if __name__ == "__main__":
    main()

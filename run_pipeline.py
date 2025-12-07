#!/usr/bin/env python3
"""
4DGS MV Pipeline - RunPod Script Version
End-to-end Multi-View pipeline for 4D Gaussian Splatting with Green Screen support.
"""

import os
import sys
import json
import argparse
import subprocess
import shutil
from pathlib import Path

# Lazy import helper
def get_runpod_helpers():
    try:
        from runpod_helpers import ensure_dirs, validate_input_video, render_green_screen, log_and_print
        return ensure_dirs, validate_input_video, render_green_screen, log_and_print
    except ImportError:
        # If imports fail (before dependencies install), return placeholders or None
        # We will handle this by ensuring install_dependencies runs first
        return None, None, None, None


def run_command(cmd, desc=None, check=True):
    """Run a shell command with logging."""
    if desc:
        print(f"➜ {desc}...")
    
    try:
        # Popen allows for streaming output if we wanted, but run is simpler for now
        result = subprocess.run(
            cmd, 
            check=check, 
            shell=True,
            capture_output=True,
            text=True
        )
        return result
    except subprocess.CalledProcessError as e:
        print(f"❌ Command failed: {cmd}")
        print(f"Error output: {e.stderr}")
        if check:
            raise
        return e


def check_runtime():
    """Verify Environment."""
    print("="*50)
    print("Runtime Check")
    print("="*50)
    
    # Check GPU
    try:
        import torch
        print(f"PyTorch: {torch.__version__}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("⚠️  WARNING: No GPU detected.")
    except ImportError:
        print("⚠️  PyTorch not found.")
    
    # Check Disk
    run_command("df -h /workspace", "Checking Disk Space", check=False)
    print("")

def check_torch_version():
    """Check if PyTorch 1.13.1 is installed."""
    try:
        import torch
        version = torch.__version__
        print(f"Current PyTorch Version: {version}")
        return version.startswith("1.13.1")
    except ImportError:
        return False



def install_dependencies(root, fourgs_repo, debug_shim=False):
    """Install system and python dependencies, build 4DGS if needed."""
    print("="*50)
    print("Dependencies & Build")
    print("="*50)

    log_file = Path(root) / "logs" / "install.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Log file: {log_file}")

    # System Deps
    print("Installing system packages...")
    run_command("apt-get update -qq && apt-get install -y -qq ffmpeg", check=False)

    # Python Deps
    print("Installing python packages...")
    pkgs = "opencv-python-headless tqdm imageio imageio-ffmpeg open3d trimesh plyfile"
    run_command(f"pip install -q {pkgs}", check=False)

    # 4DGS Build
    if not debug_shim:
        print("\nChecking 4DGS environment...")

        # Enforce PyTorch 1.13.1
        if not check_torch_version():
            print("⚠️  Wrong PyTorch version detected. 4DGS requires 1.13.1.")
            print("Downgrading PyTorch to 1.13.1+cu117...")
            
            # Uninstall current
            run_command("pip uninstall -y torch torchvision", check=False)
            
            # Install correct version
            cmd = "pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117"
            run_command(cmd, "Installing PyTorch 1.13.1")
            
            print("✓ PyTorch 1.13.1 installed. Please restart the script to load new version.")
            sys.exit(0) # Restart required to reload torch
        

        # Clone
        if not Path(fourgs_repo).exists():
            print(f"Cloning 4DGaussians to {fourgs_repo}...")
            run_command(f"git clone --recursive https://github.com/hustvl/4DGaussians.git {fourgs_repo}")
        
        # Build Extensions
        try:
            import diff_gaussian_rasterization
            import simple_knn
            print("✓ 4DGS CUDA extensions already installed")
        except ImportError:
            print("⚠️  4DGS CUDA extensions missing. Building now (this takes time)...")
            
            # Repo deps
            run_command(f"pip install -r {fourgs_repo}/requirements.txt")

            # submodules
            submodules = Path(fourgs_repo) / "submodules"
            
            # diff-gaussian-rasterization (handle naming variants)
            diff_gauss = submodules / "depth-diff-gaussian-rasterization"
            if not diff_gauss.exists():
                diff_gauss = submodules / "diff-gaussian-rasterization"
            
            run_command(f"cd {diff_gauss} && pip install .", "Building diff-gaussian-rasterization")
            
            # simple-knn
            knn = submodules / "simple-knn"
            run_command(f"cd {knn} && pip install .", "Building simple-knn")
            
            print("✓ Build complete")
    else:
        print("DEBUG_SHIM=True: Skipping 4DGS build")
    
    print("✓ Dependencies Ready\n")


def extract_frames(input_path, frames_dir, fps):
    """Extract frames from video."""
    print(f"Extracting frames from {input_path.name}...")
    
    frames_pattern = frames_dir / "%06d.png"
    
    cmd = (
        f"ffmpeg -y -i \"{input_path}\" "
        f"-vf fps={fps} "
        f"-qscale:v 2 "
        f"\"{frames_pattern}\""
    )

    
    run_command(cmd, "Running ffmpeg")
    
    count = len(list(frames_dir.glob("*.png")))
    print(f"✓ Extracted {count} frames")
    return count


def generate_masks(frames_dir, coarse_dir, alpha_dir):
    """Generate masks (Placeholder or Logic)."""

    # Import locally
    import cv2
    from tqdm import tqdm

    print("Generating masks...")
    
    frames = sorted(frames_dir.glob("*.png"))
    
    for f in tqdm(frames, desc="Processing Masks"):
        img = cv2.imread(str(f))

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Simple threshold (Placeholder logic)
        # In real usage, you'd insert SAM/RVM here
        _, mask = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
        
        # Coarse
        cv2.imwrite(str(coarse_dir / (f.stem + "_mask.png")), mask)
        
        # Alpha (Softened)
        alpha = cv2.GaussianBlur(mask, (7, 7), 0)
        cv2.imwrite(str(alpha_dir / (f.stem + "_alpha.png")), alpha)

    print("✓ Masks generated")


def estimate_poses(root, frames_dir):
    """Estimate camera poses (Placeholder or COLMAP)."""
    print("Estimating Poses...")
    poses_dir = Path(root) / "poses"
    
    # Placeholder identity poses
    frames = sorted(frames_dir.glob("*.png"))
    pose_data = {"frames": []}
    
    for f in frames:
        pose_data["frames"].append({
            "file_path": str(f),
            "transform_matrix": [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
        })
    
    with open(poses_dir / "poses.json", "w") as f:
        json.dump(pose_data, f)
        
    print("✓ Poses generated (Placeholder)")


def run_4dgs(root, fourgs_repo, frames_dir, debug_shim):
    """Run 4DGS training."""
    gs_dir = Path(root) / "gs"
    
    if not debug_shim and Path(fourgs_repo).exists():
        print("Starting 4DGS Training...")
        
        train_script = Path(fourgs_repo) / "train.py"
        cmd = (
            f"python {train_script} "
            f"--source_path \"{frames_dir}\" "
            f"--model_path \"{gs_dir}\" "
            f"--images \"{frames_dir}\" "
            f"--eval"
        )
        
        # In a real run, we would execute this:
        # run_command(cmd, "Training 4DGS")
        print(f"Simulating command: {cmd}")
        print("✓ 4DGS Training simulated (Uncomment in script to run real)")
        
    else:
        print("DEBUG_SHIM=True: Simulating training artifact")
        (gs_dir / "output.ply").touch()


def export_green_screen(root, fps):
    """Export green screen video."""
    actor_dir = Path(root) / "actor_rgba"
    output = Path(root) / "outputs" / "green_screen.mp4"
    
    frames_dir = Path(root) / "frames"
    alpha_dir = Path(root) / "masks" / "alpha"
    
    # Import locally
    import cv2
    from tqdm import tqdm
    # Get helpers
    _, _, render_green_screen, _ = get_runpod_helpers()

    print("Compositing RGBA and Green Screen...")
    
    frames = sorted(frames_dir.glob("*.png"))
    alphas = sorted(alpha_dir.glob("*.png"))
    
    # generate RGBA
    for f, a in tqdm(zip(frames, alphas), total=len(frames), desc="Compositing"):

        img = cv2.imread(str(f))
        alpha = cv2.imread(str(a), cv2.IMREAD_GRAYSCALE)
        
        rgba = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
        rgba[:, :, 3] = alpha
        
        cv2.imwrite(str(actor_dir / f.name), rgba)
    
    # Render Green Screen
    try:
        render_green_screen(str(actor_dir), str(output), fps=fps)
        print(f"✓ Green screen saved: {output}")
    except Exception as e:
        print(f"❌ Green screen render failed: {e}")


def main():
    parser = argparse.ArgumentParser(description="Run 4DGS Pipeline on RunPod")
    parser.add_argument("--root", default="/workspace/mvp_4dgs_job", help="Workspace root directory")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second")
    parser.add_argument("--debug_shim", action="store_true", help="Use lightweight shim mode")
    parser.add_argument("--repo", default="/workspace/4dgs_repo", help="Path to 4DGS repository")
    parser.add_argument("--input", "-i", help="Input video path (optional override)")
    
    args = parser.parse_args()
    
    args = parser.parse_args()
    
    # Add src to path for helpers logic
    current_dir = Path(__file__).parent.resolve()
    src_dir = current_dir / "src"
    if src_dir.exists():
        sys.path.insert(0, str(src_dir))

    # Setup - Standard Libraries Only first
    check_runtime()

    # Dependencies - Install BEFORE any other imports
    install_dependencies(args.root, args.repo, args.debug_shim)
    
    # Now we can import helpers
    ensure_dirs, validate_input_video, render_green_screen, log_and_print = get_runpod_helpers()
    
    if ensure_dirs is None:
        print("❌ Failed to import helpers even after installation. Please restart script.")
        sys.exit(1)

    dirs = ensure_dirs(args.root)

    
    # Manual Input Override
    if args.input:
        input_dest = Path(dirs["input"]) / "input.mp4"
        if Path(args.input).resolve() != input_dest.resolve():
            shutil.copy(args.input, input_dest)
            print(f"✓ Copied input to {input_dest}")
            
    # Dependencies
    install_dependencies(args.root, args.repo, args.debug_shim)
    
    # Validate Input
    input_video = Path(dirs["input"]) / "input.mp4"
    val = validate_input_video(str(input_video))
    
    if not val["valid"]:
        print(f"❌ Input Invalid: {val['error']}")
        print(f"Please upload 'input.mp4' to {dirs['input']}")
        sys.exit(1)
        
    # Run Stages
    extract_frames(input_video, Path(dirs["frames"]), args.fps)
    generate_masks(Path(dirs["frames"]), Path(dirs["masks_coarse"]), Path(dirs["masks_alpha"]))
    estimate_poses(args.root, Path(dirs["frames"]))
    run_4dgs(args.root, args.repo, Path(dirs["frames"]), args.debug_shim)
    export_green_screen(args.root, args.fps)
    
    print("\n" + "="*50)
    print("PIPELINE COMPLETE")
    print("="*50)


if __name__ == "__main__":
    main()

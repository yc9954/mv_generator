# RunPod GPU Cloud Setup Guide

This guide explains how to run the 4DGS MV Pipeline on RunPod GPU cloud instead of Google Colab.

## Why RunPod?

**RunPod advantages over Colab:**
- ✅ **No session timeouts** - dedicated GPU instances
- ✅ **Persistent storage** - /workspace survives restarts
- ✅ **Stable CUDA environment** - better 4DGS build compatibility
- ✅ **More GPU options** - RTX 3090/4090 and A100 available
- ✅ **Better performance** - dedicated resources, no sharing

**Recommended for:**
- Production 4DGS builds (DEBUG_SHIM=False)
- Long-running training sessions (30+ minutes)
- Large videos with many frames (>100 frames)
- Users experiencing Colab build failures

## Quick Start

### 1. Launch RunPod Instance

1. Go to [RunPod.io](https://runpod.io) and sign up/login
2. Click **"Deploy"** → **"GPU Instances"**
3. Select a GPU:
   - **Recommended**: RTX 3090/4090 (24GB VRAM) - ~$0.30-0.50/hr
   - **Premium**: A100 (40GB VRAM) - ~$1.00-1.50/hr
   - **Minimum**: RTX 3080 (10GB VRAM) - DEBUG_SHIM mode only

4. Choose a template:
   - **Recommended**: "PyTorch 2.0" or "RunPod PyTorch"
   - Ensure CUDA 11.8 is included

5. Storage: Set at least **40GB volume storage**

6. Click **"Deploy"** and wait for pod to start

### 2. Access JupyterLab

Once pod is running:

1. Click **"Connect"** → **"Start Jupyter Lab"**
2. JupyterLab opens in browser
3. Navigate to file browser on left

### 3. Upload Notebook

**Option A: Direct Upload**
1. Click upload button in JupyterLab
2. Upload `runpod_pipeline.ipynb` from this repository

**Option B: Git Clone**
```bash
# Open Terminal in JupyterLab (File → New → Terminal)
cd /workspace
git clone https://github.com/YOUR_USERNAME/mvp-4dgs-mv-colab.git
cd mvp-4dgs-mv-colab/notebooks
# Open runpod_pipeline.ipynb
```

### 4. Run Pipeline

1. Open `runpod_pipeline.ipynb`
2. Run cells in order (Shift+Enter)
3. First-time setup takes 15-25 minutes:
   - Dependencies: 3-5 min
   - 4DGS build: 10-20 min (cached for future runs)

4. Upload your video when prompted (Cell 6)
5. Pipeline completes in 20-60 minutes depending on video length

## Configuration

### Production Mode (Default)

```python
DEBUG_SHIM = False  # Real 4DGS with CUDA extensions
```

**Best for:**
- High-quality 4DGS output
- Production rendering
- Research/experimentation

**Requirements:**
- 24GB+ VRAM (RTX 3090/4090 or A100)
- 10-20 minute build time (first run only)

### Lightweight Mode

```python
DEBUG_SHIM = True  # Python shims, no 4DGS build
```

**Best for:**
- Quick testing
- Pipeline debugging
- Lower VRAM GPUs (<24GB)

## Storage Management

### Workspace Structure

All data stored in `/workspace/mvp_4dgs_job/`:

```
/workspace/mvp_4dgs_job/
├── input/              # Input video
├── frames/             # Extracted frames (~500MB per 100 frames)
├── masks/              # Segmentation masks
├── actor_rgba/         # Actor RGBA PNGs
├── outputs/            # Final outputs
├── gs/                 # 4DGS training output (if DEBUG_SHIM=False)
├── checkpoints/        # Pipeline checkpoints
└── logs/               # Build and error logs
```

### Check Disk Space

```bash
!df -h /workspace
```

### Clean Up Between Runs

```python
# Clear frames from previous run
import shutil
shutil.rmtree("/workspace/mvp_4dgs_job/frames")
Path("/workspace/mvp_4dgs_job/frames").mkdir(parents=True, exist_ok=True)
```

## Video Upload Options

The RunPod notebook supports 3 upload methods:

### Option 1: Download from URL (Recommended)

```python
# When prompted, choose 'w' and enter URL
wget -O /workspace/mvp_4dgs_job/input/input.mp4 "https://example.com/video.mp4"
```

### Option 2: JupyterLab Upload

1. Drag video file into JupyterLab file browser
2. Move to `/workspace/mvp_4dgs_job/input/input.mp4`

### Option 3: Direct Path

```python
# If video already on instance
shutil.copy("/path/to/your/video.mp4", "/workspace/mvp_4dgs_job/input/input.mp4")
```

## GPU Recommendations

### RTX 3090 (24GB VRAM) - $0.30-0.40/hr
- ✅ Production mode (DEBUG_SHIM=False)
- ✅ Up to 200 frames
- ✅ Full 4DGS training
- **Best value for money**

### RTX 4090 (24GB VRAM) - $0.40-0.60/hr
- ✅ Faster training than 3090
- ✅ Better CUDA performance
- ✅ Up to 300 frames

### A100 (40GB VRAM) - $1.00-1.50/hr
- ✅ Highest performance
- ✅ 500+ frames
- ✅ Batch processing
- **Best for large videos**

### RTX 3080 (10GB VRAM) - $0.20-0.30/hr
- ⚠️ DEBUG_SHIM=True only
- ⚠️ Limited to ~50 frames
- **Budget testing only**

## Troubleshooting

### Build Failures

**Error**: "CUDA extension build failed"

**Solutions**:
1. Check CUDA version: `!nvcc --version` (should be 11.8)
2. Verify submodules: `!ls /workspace/4dgs_repo/submodules/`
3. Check build log: `cat /workspace/mvp_4dgs_job/logs/4dgs_build.log`
4. Try rebuilding:
   ```bash
   rm -rf /workspace/4dgs_repo
   # Re-run Cell 5 (Build 4DGaussians)
   ```

### Out of Memory

**Error**: "CUDA out of memory"

**Solutions**:
1. Reduce video length to 10-30 seconds
2. Lower FPS (change `FPS = 15` instead of 30)
3. Use DEBUG_SHIM=True mode
4. Upgrade to GPU with more VRAM

### Video Not Found

**Error**: "INPUT_VIDEO does not exist"

**Solutions**:
1. Verify upload completed: `!ls -lh /workspace/mvp_4dgs_job/input/`
2. Check file permissions: `!chmod 644 /workspace/mvp_4dgs_job/input/input.mp4`
3. Try re-uploading with different method

### Slow Performance

**Check GPU utilization**:
```bash
!nvidia-smi -l 1  # Monitor GPU every 1 second
```

**If GPU usage is low (<30%)**:
- Video I/O bottleneck (use faster storage)
- Insufficient parallel processing
- Check logs for errors

## Cost Estimates

### Example: 30-second video @ 30fps (900 frames)

**RTX 3090 (~$0.35/hr)**:
- Setup: 20 min ($0.12)
- Frame extraction: 2 min ($0.01)
- Segmentation: 10 min ($0.06)
- 4DGS training: 45 min ($0.26)
- **Total: ~77 min = $0.45**

**A100 (~$1.25/hr)**:
- Setup: 15 min ($0.31)
- Frame extraction: 1 min ($0.02)
- Segmentation: 5 min ($0.10)
- 4DGS training: 25 min ($0.52)
- **Total: ~46 min = $0.95**

**Tips to reduce costs**:
- Use **RTX 3090** for best value
- Set `DEBUG_SHIM=True` for testing (skip expensive build)
- Process shorter clips (10-20 seconds)
- Terminate pod immediately after completion

## Downloading Results

### Method 1: JupyterLab File Browser

1. Navigate to `/workspace/mvp_4dgs_job/outputs/`
2. Right-click `preview.mp4` → Download

### Method 2: Python Code

```python
from IPython.display import FileLink
FileLink("/workspace/mvp_4dgs_job/outputs/preview.mp4")
```

### Method 3: rsync/scp

```bash
# From your local machine
scp -P <port> root@<pod-ip>:/workspace/mvp_4dgs_job/outputs/preview.mp4 ./
```

## Persistent Storage

**Important**: RunPod's `/workspace` is persistent ONLY if you:
1. Created a **volume** when launching pod
2. Attached volume to `/workspace`

**Without volume**: All data lost when pod terminates!

**Best practice**:
- Always create 40GB+ volume
- Mount to `/workspace`
- Backup important results to cloud storage (Google Drive, S3, etc.)

## Comparison: RunPod vs Colab

| Feature | RunPod | Colab Free | Colab Pro |
|---------|--------|------------|-----------|
| **Session timeout** | None | ~12 hours | ~24 hours |
| **Persistent storage** | ✅ /workspace | ❌ Drive only | ❌ Drive only |
| **GPU availability** | Guaranteed | Limited | Better |
| **CUDA stability** | Excellent | Poor | Good |
| **4DGS build success** | ~95% | ~20% | ~60% |
| **Cost** | $0.30-1.50/hr | Free | $10/mo + compute |
| **Best for** | Production | Quick tests | Prototyping |

## Support

**Issues with RunPod setup?**
- Check RunPod docs: https://docs.runpod.io
- GitHub Issues: https://github.com/YOUR_USERNAME/mvp-4dgs-mv-colab/issues
- RunPod Discord: https://discord.gg/runpod

**Issues with 4DGS pipeline?**
- Check logs: `/workspace/mvp_4dgs_job/logs/`
- GitHub Issues: Link to your repository
- Original 4DGaussians: https://github.com/hustvl/4DGaussians

---

**Ready to start?** → Launch your RunPod instance and open `runpod_pipeline.ipynb`!

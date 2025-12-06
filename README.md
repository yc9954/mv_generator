# MVP 4DGS MV Colab

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/mvp-4dgs-mv-colab/blob/main/notebooks/colab_pipeline.ipynb)

**Production-ready Google Colab notebook for end-to-end 4D Gaussian Splatting Multi-View (MV) pipeline**

This repository provides a complete, debuggable implementation of a Multi-View pipeline for 4D Gaussian Splatting that runs entirely on Google Colab with A100 GPU support.

## Features

- 🚀 **One-Click Colab**: Copy notebook to Colab and run all cells
- 🏭 **Production Mode**: Real 4DGS binary with automatic build (default)
- 🔄 **DEBUG_SHIM Mode**: Lightweight Python-based shims for rapid testing
- 💾 **Checkpoint Support**: Resume from any pipeline stage
- 📊 **Progress Tracking**: JSON manifests for every stage
- 🛡️ **Robust Fallbacks**: Graceful degradation when heavy dependencies missing
- 📁 **Drive Integration**: All data persists on Google Drive
- ⚡ **Smart Caching**: Skip 4DGS build on subsequent runs

## Quick Start

### 1. Open in Colab

Click the badge above or use this direct link:

```
https://colab.research.google.com/github/YOUR_USERNAME/mvp-4dgs-mv-colab/blob/main/notebooks/colab_pipeline.ipynb
```

### 2. Mount Google Drive

Run the first cell to mount your Google Drive. This creates a workspace at:

```
/content/drive/MyDrive/mvp_4dgs_job
```

### 3. Run Pipeline

Execute cells in order. The default `DEBUG_SHIM=False` mode automatically builds and uses real 4DGS for production-quality results.

**Note**: First-time setup includes 10-20 minutes for 4DGS build. Subsequent runs skip this step.

## Pipeline Stages

The notebook implements 13 pipeline stages:

| Stage | Description | Runtime (DEBUG_SHIM=True) |
|-------|-------------|---------------------------|
| 1. Runtime Check | Detect GPU, check for A100 | <5s |
| 2. Drive Mount | Mount Drive, create workspace | ~10s |
| 3. Configuration | Set DEBUG_SHIM and parameters | <1s |
| 4. Dependencies | Install packages | 2-5min (first time) |
| 5. Input Upload | Upload or select video | Manual |
| 6. Frame Extraction | ffmpeg extract frames | 10-60s |
| 7. Coarse Segmentation | SAM or placeholder masks | <10s (placeholder) |
| 8. Alpha Matting | RVM or placeholder mattes | <10s (placeholder) |
| 9. Temporal Smoothing | Optical flow smoothing | 30-60s |
| 10. COLMAP Poses | Camera pose estimation | 30-60s (fallback) |
| 11. GS Shim | Background plane generation | <5s |
| 12. Actor RGBA | Export actor with alpha | 10-30s |
| 13. Composite Preview | Generate preview video | 10-20s |

**Total runtime (DEBUG_SHIM=True)**: ~5-10 minutes

## Repository Structure

```
mvp-4dgs-mv-colab/
├── README.md                          # This file
├── LICENSE                            # MIT License
├── requirements.txt                   # Python dependencies
├── env.template                       # Environment variables template
│
├── notebooks/
│   └── colab_pipeline.ipynb          # Main Colab notebook ⭐
│
├── scripts/
│   ├── save_manifest.py              # CLI tool for manifest generation
│   ├── build_4dgs_shim.sh           # Build instructions for real 4DGS
│   └── run_colab_headless.sh        # Papermill automation script
│
├── src/
│   ├── colab_helpers.py              # Drive path, checkpoint, logging utils
│   └── gs_shim.py                    # Lightweight GS shim (Open3D)
│
├── samples/
│   ├── tiny_sample.mp4               # Tiny test video
│   ├── synthetic_test_frames/        # Test PNG frames
│   └── README.md                     # Sample data documentation
│
└── tests/
    ├── test_colab_helpers.py         # Unit tests for helpers
    └── test_gs_shim.py               # Unit tests for GS shim
```

## Installation

### For Colab (Recommended)

No installation needed! Just open the notebook in Colab.

### For Local Development

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/mvp-4dgs-mv-colab.git
cd mvp-4dgs-mv-colab

# Install dependencies
pip install -r requirements.txt

# Run tests
pytest tests/
```

## Configuration

### Production Mode (Default)

**New default mode** uses real 4DGS for production-quality results:

```python
DEBUG_SHIM = False  # Production mode (default)
```

**4DGS is built automatically** during dependency installation (Cell 4.5). No manual setup required!

### DEBUG_SHIM Mode (Lightweight)

For quick testing without building 4DGS:

```python
DEBUG_SHIM = True  # Lightweight demo mode
```

This mode uses:
- Placeholder segmentation (threshold-based)
- Placeholder matting (Gaussian smoothing)
- OpenCV SIFT+PnP fallback for camera poses
- Open3D background plane instead of 4DGS

### Optional Checkpoints

```python
SAM_CHECKPOINT = "/path/to/sam_vit_h_4b8939.pth"  # For real SAM
RVM_CHECKPOINT = "/path/to/rvm_mobilenetv3.pth"   # For real RVM
```

## Building Real 4DGS

**4DGS is now built automatically!**

When `DEBUG_SHIM=False` (default), the notebook automatically:
1. Clones 4DGaussians repository with submodules
2. Installs PyTorch 2.0.1 (Colab-compatible)
3. Builds CUDA extensions
4. Verifies installation

**First-time build**: 10-20 minutes
**Subsequent runs**: Skipped if already built

### Manual Rebuild

If you need to rebuild 4DGS:

1. Delete existing installation:
```python
!rm -rf /content/4dgs_repo
```

2. Re-run Cell 4.5 (Build 4DGaussians)

### Alternative: Manual Build

```bash
# 1. Clone 4DGaussians
git clone https://github.com/hustvl/4DGaussians.git /content/4dgs_repo
cd /content/4dgs_repo

# 2. Install dependencies
pip install -r requirements.txt

# 3. Install PyTorch with CUDA 11.8
pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu118

# 4. Build CUDA extensions
cd submodules/diff-gaussian-rasterization
python setup.py install
cd ../simple-knn
python setup.py install
```

### Expected Output

- Binary: `/content/4dgs_repo/train.py`
- Time: 10-20 minutes
- Disk: ~2GB

### Running Real 4DGS

Once built, set `DEBUG_SHIM=False` and re-run the notebook. Cell 12 will execute:

```bash
python /content/4dgs_repo/train.py \
  --source_path /content/drive/MyDrive/mvp_4dgs_job/frames \
  --model_path /content/drive/MyDrive/mvp_4dgs_job/gs \
  --images /content/drive/MyDrive/mvp_4dgs_job/frames \
  --eval
```

## PyTorch and CUDA

The notebook requires PyTorch with CUDA support. Install with:

```bash
pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu118
```

For Colab, this is typically pre-installed. Verify with:

```python
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
```

## GPU Requirements

### Recommended

- **GPU**: NVIDIA A100 (40GB VRAM)
- **Runtime**: Colab Pro or Pro+ for A100 access
- **VRAM**: 40GB for full pipeline with real 4DGS

### Minimum

- **GPU**: Any CUDA-capable GPU
- **VRAM**: 8GB for DEBUG_SHIM mode
- **Runtime**: Free Colab tier (with limitations)

### Check GPU

```python
!nvidia-smi
```

Look for "A100" in the output. The notebook auto-detects and saves this to metadata.

## Output Artifacts

All outputs are saved to Google Drive under `ROOT` directory:

```
/content/drive/MyDrive/mvp_4dgs_job/
├── input/
│   └── input.mp4                     # Input video
├── frames/
│   └── *.png                         # Extracted frames
├── masks/
│   ├── coarse/*.png                  # Coarse masks
│   └── alpha/*.png                   # Alpha mattes
├── poses/
│   └── poses.json                    # Camera poses
├── gs_shim/
│   └── bg_plane.ply                  # Background plane (DEBUG_SHIM)
├── gs/                               # 4DGS outputs (production mode)
├── actor_rgba/
│   ├── *.png                         # Actor RGBA frames
│   └── actor_meta.json               # Actor metadata
├── outputs/
│   ├── preview.mp4                   # Composite preview
│   └── sr_sample.png                 # Super-resolution sample
├── logs/
│   └── *.log                         # All logs
├── checkpoints/
│   ├── manifest_*.json               # Stage manifests
│   └── manifest_latest.json          # Latest checkpoint
└── runs_meta.json                    # Run metadata
```

## Manifest Format

All checkpoints follow this schema:

```json
{
  "job_id": "colab-demo",
  "stage": "frames_extracted",
  "frames_count": 123,
  "artifacts": {
    "frames_dir": "/content/drive/MyDrive/mvp_4dgs_job/frames",
    "masks_dir": "/content/drive/MyDrive/mvp_4dgs_job/masks",
    "poses": "/content/drive/MyDrive/mvp_4dgs_job/poses/poses.json",
    "bg_shim": "/content/drive/MyDrive/mvp_4dgs_job/gs_shim/bg_plane.ply"
  },
  "timestamp": 1700000000
}
```

## Resuming from Checkpoints

The notebook automatically saves checkpoints after each major stage. To resume:

1. Re-run Configuration cell
2. Skip to the cell after your last checkpoint
3. Continue execution

Checkpoints are stored in `ROOT/checkpoints/manifest_*.json`

## Session Timeout Handling

Google Colab free tier has session timeouts (~12 hours). To handle this:

### Strategy 1: Manual Resume

1. Checkpoints persist on Google Drive
2. Re-open notebook after timeout
3. Re-run from last successful stage

### Strategy 2: Papermill Automation

```bash
bash scripts/run_colab_headless.sh notebooks/colab_pipeline.ipynb output.ipynb
```

### Strategy 3: Colab Pro

Colab Pro/Pro+ offers:
- Longer timeouts
- Background execution
- Priority GPU access

### Strategy 4: Cloud GPU Rental

For uninterrupted long runs:

- **Lambda Labs**: A100 rentals ($1.10/hr)
- **RunPod**: On-demand GPUs
- **Vast.ai**: Spot instances

## Testing

Run unit tests locally:

```bash
# Install test dependencies
pip install pytest pytest-cov

# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test file
pytest tests/test_colab_helpers.py -v
```

### Test Requirements

Some tests require `open3d`. Install with:

```bash
pip install open3d
```

Tests will skip if dependencies are missing.

## Troubleshooting

### Common Issues

**1. "No GPU detected"**

- Ensure GPU runtime: Runtime → Change runtime type → GPU
- Free tier has limited GPU availability

**2. "Drive mount failed"**

- Re-run mount cell
- Check browser popup for authorization

**3. "ffmpeg not found"**

- Run dependencies installation cell
- Colab usually has ffmpeg pre-installed

**4. "Out of memory"**

- Reduce frame count
- Use DEBUG_SHIM mode
- Request A100 in Colab Pro

**5. "COLMAP installation failed"**

- Expected in free tier
- Notebook uses OpenCV fallback automatically

### Debug Logs

Check logs in `ROOT/logs/`:

```
install.log          # Dependency installation
frames_err.log       # Frame extraction errors
4dgs_err.log         # 4DGS build/run errors
manifest_err.log     # Manifest save errors
```

## Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup

```bash
# Clone with submodules
git clone --recursive https://github.com/YOUR_USERNAME/mvp-4dgs-mv-colab.git

# Install dev dependencies
pip install -r requirements.txt
pip install pytest pytest-cov black flake8

# Run tests before committing
pytest tests/
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this work in your research, please cite:

```bibtex
@software{mvp_4dgs_mv_colab,
  title={MVP 4DGS MV Colab: Production-Ready Multi-View Pipeline},
  author={Your Name},
  year={2025},
  url={https://github.com/YOUR_USERNAME/mvp-4dgs-mv-colab}
}
```

## Acknowledgments

- [4DGaussians](https://github.com/hustvl/4DGaussians) - Original 4D Gaussian Splatting implementation
- [Segment Anything (SAM)](https://github.com/facebookresearch/segment-anything) - Segmentation
- [Robust Video Matting (RVM)](https://github.com/PeterL1n/RobustVideoMatting) - Video matting
- [COLMAP](https://colmap.github.io/) - Structure from Motion
- [Open3D](http://www.open3d.org/) - 3D geometry processing
- [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) - Super-resolution

## Support

- **Issues**: [GitHub Issues](https://github.com/YOUR_USERNAME/mvp-4dgs-mv-colab/issues)
- **Discussions**: [GitHub Discussions](https://github.com/YOUR_USERNAME/mvp-4dgs-mv-colab/discussions)

## Roadmap

- [ ] Real SAM2 integration
- [ ] Real RVM integration
- [ ] RAFT optical flow support
- [ ] Multi-GPU support
- [ ] Automatic checkpoint recovery
- [ ] Web UI for parameter tuning
- [ ] Batch processing support

---

**Made with ❤️ for the 4DGS community**

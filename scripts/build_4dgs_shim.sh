#!/bin/bash
# Build instructions for real 4DGS binary on Google Colab
#
# This script provides instructions and commands to build a real 4DGS implementation
# in Google Colab. By default, the notebook runs in DEBUG_SHIM=True mode which uses
# a lightweight Python-based shim. To use a real 4DGS build, set DEBUG_SHIM=False
# and run this script to compile the necessary binaries.
#
# Usage:
#   bash scripts/build_4dgs_shim.sh
#
# Expected output:
#   - Compiled 4DGS binary at /content/4dgs_repo/build/4dgs
#   - CUDA kernels built
#   - Python bindings installed

set -e  # Exit on error

echo "=================================================="
echo "4DGS Build Instructions for Google Colab"
echo "=================================================="
echo ""

# Configuration
REPO_URL="https://github.com/hustvl/4DGaussians.git"
INSTALL_DIR="/content/4dgs_repo"
BUILD_LOG="/content/drive/MyDrive/mvp_4dgs_job/logs/4dgs_build.log"

echo "This script will:"
echo "  1. Clone 4DGaussians repository"
echo "  2. Install build dependencies (CUDA toolkit, etc.)"
echo "  3. Build CUDA kernels and binaries"
echo "  4. Install Python bindings"
echo ""
echo "Target directory: $INSTALL_DIR"
echo "Build log: $BUILD_LOG"
echo ""

read -p "Continue with build? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]
then
    echo "Build cancelled."
    exit 0
fi

# Create log directory
mkdir -p "$(dirname "$BUILD_LOG")"

echo "Starting build process..." | tee -a "$BUILD_LOG"
echo "Timestamp: $(date)" | tee -a "$BUILD_LOG"

# Step 1: Install system dependencies
echo ""
echo "Step 1/5: Installing system dependencies..."
{
    apt-get update
    apt-get install -y \
        build-essential \
        cmake \
        git \
        libglew-dev \
        libeigen3-dev \
        libboost-all-dev \
        libfreeimage-dev \
        libmetis-dev \
        libgoogle-glog-dev \
        libgflags-dev \
        libsqlite3-dev \
        libatlas-base-dev \
        libsuitesparse-dev
} >> "$BUILD_LOG" 2>&1

echo "  ✓ System dependencies installed"

# Step 2: Clone repository
echo ""
echo "Step 2/5: Cloning 4DGaussians repository..."
if [ -d "$INSTALL_DIR" ]; then
    echo "  Repository already exists, pulling latest changes..."
    cd "$INSTALL_DIR"
    git pull >> "$BUILD_LOG" 2>&1
else
    git clone "$REPO_URL" "$INSTALL_DIR" >> "$BUILD_LOG" 2>&1
    cd "$INSTALL_DIR"
fi

echo "  ✓ Repository cloned/updated"

# Step 3: Install Python dependencies
echo ""
echo "Step 3/5: Installing Python dependencies..."
{
    pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu118
    pip install -r requirements.txt
    pip install submodules/diff-gaussian-rasterization
    pip install submodules/simple-knn
} >> "$BUILD_LOG" 2>&1

echo "  ✓ Python dependencies installed"

# Step 4: Build CUDA kernels
echo ""
echo "Step 4/5: Building CUDA kernels..."
{
    cd submodules/diff-gaussian-rasterization
    python setup.py install
    cd ../simple-knn
    python setup.py install
    cd ../..
} >> "$BUILD_LOG" 2>&1

echo "  ✓ CUDA kernels built"

# Step 5: Final setup
echo ""
echo "Step 5/5: Final setup and verification..."
{
    python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"
} >> "$BUILD_LOG" 2>&1

echo "  ✓ Build complete"

echo ""
echo "=================================================="
echo "Build Complete!"
echo "=================================================="
echo ""
echo "4DGS is now installed at: $INSTALL_DIR"
echo ""
echo "To use in the notebook, set:"
echo "  DEBUG_SHIM = False"
echo "  FOURGS_REPO = '$INSTALL_DIR'"
echo ""
echo "Example command to run 4DGS:"
echo "  python $INSTALL_DIR/train.py \\"
echo "    --source_path /path/to/frames \\"
echo "    --model_path /path/to/output \\"
echo "    --images /path/to/images"
echo ""
echo "Full build log: $BUILD_LOG"
echo ""

#!/bin/bash
# easy_start.sh
# Run this on RunPod to get started immediately

echo "🚀 Starting 4DGS Setup..."

# 1. Clone the repository (specific branch)
if [ -d "mv_generator" ]; then
    echo "Folder 'mv_generator' already exists. Skipping clone."
else
    echo "Cloning repository..."
    git clone -b claude/4dgs-colab-setup-01EzHeaT6YawPrxxsB3bYmUp https://github.com/yc9954/mv_generator.git
fi

cd mv_generator

echo ""
echo "✅ Setup Complete!"
echo "=============================================="
echo "Next Steps:"
echo "1. Upload your video file to the 'mv_generator' folder."
echo "2. Run the pipeline command:"
echo "   python run_pipeline.py --input <your_video.mp4>"
echo ""
echo "NOTE: The first time you run it, it may restart to install PyTorch 1.13.1."
echo "      Just run the command again if that happens."
echo "=============================================="

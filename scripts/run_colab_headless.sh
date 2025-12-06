#!/bin/bash
# Run Colab notebook headlessly using papermill
#
# This script allows running the Colab notebook in headless mode for automation
# or CI/CD pipelines. Uses papermill to execute notebook cells programmatically.
#
# Usage:
#   bash scripts/run_colab_headless.sh [notebook_path] [output_path]
#
# Example:
#   bash scripts/run_colab_headless.sh notebooks/colab_pipeline.ipynb output_executed.ipynb

set -e  # Exit on error

NOTEBOOK_PATH="${1:-notebooks/colab_pipeline.ipynb}"
OUTPUT_PATH="${2:-notebooks/colab_pipeline_executed.ipynb}"

echo "=================================================="
echo "Running Colab Notebook Headlessly"
echo "=================================================="
echo ""
echo "Input notebook:  $NOTEBOOK_PATH"
echo "Output notebook: $OUTPUT_PATH"
echo ""

# Check if papermill is installed
if ! command -v papermill &> /dev/null; then
    echo "papermill not found. Installing..."
    pip install papermill ipykernel
fi

# Check if notebook exists
if [ ! -f "$NOTEBOOK_PATH" ]; then
    echo "Error: Notebook not found at $NOTEBOOK_PATH"
    exit 1
fi

# Set parameters for notebook execution
# These can be overridden via environment variables
DEBUG_SHIM="${DEBUG_SHIM:-True}"
ROOT_DIR="${ROOT_DIR:-/content/drive/MyDrive/mvp_4dgs_job}"

echo "Execution parameters:"
echo "  DEBUG_SHIM: $DEBUG_SHIM"
echo "  ROOT_DIR: $ROOT_DIR"
echo ""

# Run notebook with papermill
echo "Executing notebook..."
papermill \
    "$NOTEBOOK_PATH" \
    "$OUTPUT_PATH" \
    -p DEBUG_SHIM "$DEBUG_SHIM" \
    -p ROOT "$ROOT_DIR" \
    --log-output \
    --progress-bar

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "=================================================="
    echo "Notebook executed successfully!"
    echo "=================================================="
    echo ""
    echo "Output saved to: $OUTPUT_PATH"
    echo ""
    echo "To view results, check the executed notebook or logs at:"
    echo "  $ROOT_DIR/logs/"
    echo ""
else
    echo ""
    echo "=================================================="
    echo "Notebook execution failed!"
    echo "=================================================="
    echo ""
    echo "Exit code: $EXIT_CODE"
    echo "Check the output notebook for error details: $OUTPUT_PATH"
    echo ""
    exit $EXIT_CODE
fi

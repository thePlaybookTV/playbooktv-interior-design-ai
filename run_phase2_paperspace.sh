#!/bin/bash
# Run Phase 2 Training on Gradient Paperspace
# This script sets up the environment and runs Phase 2 training

set -e  # Exit on error

echo "========================================================================"
echo "🚀 PHASE 2 TRAINING - GRADIENT PAPERSPACE"
echo "========================================================================"

# Set model paths for Gradient
export YOLO_MODEL_PATH=/datasets/yolo/yolov8m.pt
export SAM2_CHECKPOINT=/datasets/sam2/sam2_hiera_large.pt
export SAM2_CONFIG=sam2_hiera_l.yaml

echo ""
echo "📁 Working directory: $(pwd)"
echo "🔧 YOLO model: $YOLO_MODEL_PATH"
echo "🔧 SAM2 checkpoint: $SAM2_CHECKPOINT"
echo ""

# Verify models exist
if [ ! -f "$YOLO_MODEL_PATH" ]; then
    echo "❌ YOLO model not found: $YOLO_MODEL_PATH"
    exit 1
fi

echo "✅ YOLO model verified"
echo ""

# Run Phase 2 training
python scripts/run_phase2_training.py \
    --db database_metadata.duckdb \
    --output ./phase2_outputs \
    --yolo-epochs 100 \
    --style-epochs 30 \
    --batch-size 50

echo ""
echo "========================================================================"
echo "✅ PHASE 2 TRAINING COMPLETE!"
echo "========================================================================"

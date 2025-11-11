#!/bin/bash
# ============================================
# Run Pristine Detection on Paperspace
# Automatically sets up models and runs detection
# ============================================

set -e  # Exit on error

echo "=============================================="
echo "🚀 PRISTINE DETECTOR - PAPERSPACE SETUP"
echo "=============================================="

# Configuration
MODELS_DIR="${MODELS_DIR:-/models/detection-models}"
DB_PATH="${DB_PATH:-interior_design_data_hybrid/processed/metadata.duckdb}"
YOLO_MODEL="${YOLO_MODEL:-yolov8m.pt}"
SAM2_CHECKPOINT="${SAM2_CHECKPOINT:-sam2_hiera_large.pt}"

echo ""
echo "📋 Configuration:"
echo "   Models directory: $MODELS_DIR"
echo "   Database: $DB_PATH"
echo "   YOLO model: $YOLO_MODEL"
echo "   SAM2 checkpoint: $SAM2_CHECKPOINT"
echo ""

# Step 1: Create checkpoints directory
echo "📁 Creating checkpoints directory..."
mkdir -p checkpoints

# Step 2: Link models from Gradient storage
echo "🔗 Linking models from Gradient storage..."

if [ -f "$MODELS_DIR/$YOLO_MODEL" ]; then
    ln -sf "$MODELS_DIR/$YOLO_MODEL" ./"$YOLO_MODEL"
    echo "   ✅ YOLO model linked"
    ls -lh "$YOLO_MODEL"
else
    echo "   ⚠️  YOLO model not found at $MODELS_DIR/$YOLO_MODEL"
    echo "   Will attempt to download..."
fi

if [ -f "$MODELS_DIR/$SAM2_CHECKPOINT" ]; then
    ln -sf "$MODELS_DIR/$SAM2_CHECKPOINT" ./checkpoints/"$SAM2_CHECKPOINT"
    echo "   ✅ SAM2 checkpoint linked"
    ls -lh checkpoints/"$SAM2_CHECKPOINT"
else
    echo "   ⚠️  SAM2 checkpoint not found at $MODELS_DIR/$SAM2_CHECKPOINT"
    echo "   Will attempt to download..."
fi

# Step 3: Check if database exists
echo ""
echo "💾 Checking database..."
if [ -f "$DB_PATH" ]; then
    echo "   ✅ Database found: $DB_PATH"
else
    echo "   ❌ Database not found: $DB_PATH"
    echo "   Please ensure your database exists before running detection"
    exit 1
fi

# Step 4: Check GPU
echo ""
echo "🎮 Checking GPU..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    echo "   ✅ GPU detected"
else
    echo "   ⚠️  No GPU detected - will use CPU (slower)"
fi

# Step 5: Install SAM2 if not already installed
echo ""
echo "📦 Checking SAM2 installation..."
if python -c "import sam2" 2>/dev/null; then
    echo "   ✅ SAM2 already installed"
else
    echo "   📥 Installing SAM2..."
    pip install git+https://github.com/facebookresearch/segment-anything-2.git
    echo "   ✅ SAM2 installed"
fi

# Step 6: Run detection
echo ""
echo "=============================================="
echo "🎬 STARTING DETECTION"
echo "=============================================="
echo ""

# Set environment variables for model paths
export YOLO_MODEL_PATH="./$YOLO_MODEL"
export SAM2_CHECKPOINT="./checkpoints/$SAM2_CHECKPOINT"
export SAM2_CONFIG="sam2_hiera_l.yaml"

# Run the detection script
python src/models/pristine_detector.py

echo ""
echo "=============================================="
echo "✅ DETECTION COMPLETE"
echo "=============================================="
echo ""
echo "📊 Next steps:"
echo "   1. Check statistics with:"
echo "      python -c 'from src.models.pristine_detector import CheckpointProcessor; p = CheckpointProcessor(\"$DB_PATH\"); p.show_stats(); p.close()'"
echo ""
echo "   2. Run Phase 2 training:"
echo "      python scripts/run_phase2_training.py --db $DB_PATH --output ./phase2_outputs"
echo ""

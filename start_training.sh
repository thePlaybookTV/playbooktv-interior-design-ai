#!/bin/bash

###############################################################################
# Phase 2 Training Launcher for Paperspace
# Automatically sets up environment and launches training
###############################################################################

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print header
echo ""
echo "=============================================================================="
echo -e "${BLUE}PlaybookTV Interior Design AI - Phase 2 Training${NC}"
echo "=============================================================================="
echo ""

# Get project root
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

echo -e "${BLUE}📁 Project root:${NC} $PROJECT_ROOT"
echo ""

# Check Python version
echo -e "${BLUE}🐍 Checking Python...${NC}"
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo -e "   ${GREEN}✓${NC} Python $PYTHON_VERSION"
echo ""

# Check dependencies
echo -e "${BLUE}📦 Checking dependencies...${NC}"

REQUIRED_PACKAGES=("torch" "ultralytics" "duckdb" "pillow" "numpy" "tqdm")
MISSING_PACKAGES=()

for package in "${REQUIRED_PACKAGES[@]}"; do
    if python3 -c "import $package" 2>/dev/null; then
        echo -e "   ${GREEN}✓${NC} $package"
    else
        echo -e "   ${RED}✗${NC} $package (missing)"
        MISSING_PACKAGES+=("$package")
    fi
done

if [ ${#MISSING_PACKAGES[@]} -gt 0 ]; then
    echo ""
    echo -e "${YELLOW}⚠️  Missing packages detected. Installing...${NC}"
    pip install -q -r requirements.txt
    echo -e "${GREEN}✓${NC} Dependencies installed"
fi

echo ""

# Check database
echo -e "${BLUE}🗄️  Checking database...${NC}"
DB_PATH="$PROJECT_ROOT/interior_design_data_hybrid/processed/metadata.duckdb"

if [ -f "$DB_PATH" ]; then
    DB_SIZE=$(du -h "$DB_PATH" | cut -f1)
    echo -e "   ${GREEN}✓${NC} Database found ($DB_SIZE)"
else
    echo -e "   ${RED}✗${NC} Database not found at:"
    echo "     $DB_PATH"
    echo ""
    echo -e "${YELLOW}Please upload your DuckDB file to:${NC}"
    echo "   $PROJECT_ROOT/interior_design_data_hybrid/processed/metadata.duckdb"
    echo ""
    exit 1
fi

echo ""

# Check GPU
echo -e "${BLUE}🎮 Checking GPU...${NC}"
if command -v nvidia-smi &> /dev/null; then
    GPU_INFO=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
    GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -n 1)
    echo -e "   ${GREEN}✓${NC} GPU: $GPU_INFO"
    echo -e "   ${GREEN}✓${NC} VRAM: $GPU_MEM"
else
    echo -e "   ${YELLOW}⚠️${NC}  No GPU detected (training will be very slow)"
    echo ""
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo ""

# Training configuration
echo -e "${BLUE}⚙️  Training Configuration${NC}"
echo ""
echo "Select training mode:"
echo "  1) Full Phase 2 (YOLO + Style Classifier) - Recommended"
echo "  2) YOLO only (8-12 hours)"
echo "  3) Style Classifier only (2-4 hours)"
echo "  4) Custom (specify parameters)"
echo ""

read -p "Enter choice [1-4]: " TRAINING_MODE

case $TRAINING_MODE in
    1)
        echo -e "${GREEN}Selected: Full Phase 2 Training${NC}"
        SKIP_YOLO=""
        SKIP_STYLE=""
        YOLO_EPOCHS=100
        STYLE_EPOCHS=30
        BATCH_SIZE=16
        ;;
    2)
        echo -e "${GREEN}Selected: YOLO Only${NC}"
        SKIP_YOLO=""
        SKIP_STYLE="--skip-style"
        YOLO_EPOCHS=100
        STYLE_EPOCHS=0
        BATCH_SIZE=16
        ;;
    3)
        echo -e "${GREEN}Selected: Style Classifier Only${NC}"
        SKIP_YOLO="--skip-yolo"
        SKIP_STYLE=""
        YOLO_EPOCHS=0
        STYLE_EPOCHS=30
        BATCH_SIZE=32
        ;;
    4)
        echo -e "${GREEN}Selected: Custom Configuration${NC}"
        echo ""
        read -p "Train YOLO? (Y/n): " TRAIN_YOLO
        read -p "Train Style Classifier? (Y/n): " TRAIN_STYLE
        read -p "YOLO epochs [100]: " YOLO_EPOCHS
        read -p "Style epochs [30]: " STYLE_EPOCHS
        read -p "Batch size [16]: " BATCH_SIZE

        YOLO_EPOCHS=${YOLO_EPOCHS:-100}
        STYLE_EPOCHS=${STYLE_EPOCHS:-30}
        BATCH_SIZE=${BATCH_SIZE:-16}

        SKIP_YOLO=""
        SKIP_STYLE=""
        [[ ! $TRAIN_YOLO =~ ^[Yy]$ ]] && SKIP_YOLO="--skip-yolo"
        [[ ! $TRAIN_STYLE =~ ^[Yy]$ ]] && SKIP_STYLE="--skip-style"
        ;;
    *)
        echo -e "${RED}Invalid choice${NC}"
        exit 1
        ;;
esac

echo ""
echo "=============================================================================="
echo -e "${BLUE}🚀 Starting Training${NC}"
echo "=============================================================================="
echo ""
echo "Parameters:"
echo "  • Database: $DB_PATH"
echo "  • Output: $PROJECT_ROOT/phase2_outputs"
echo "  • YOLO Epochs: $YOLO_EPOCHS"
echo "  • Style Epochs: $STYLE_EPOCHS"
echo "  • Batch Size: $BATCH_SIZE"
echo ""

# Estimate time
if [ -z "$SKIP_YOLO" ] && [ -z "$SKIP_STYLE" ]; then
    echo -e "${YELLOW}⏱️  Estimated time: 10-16 hours${NC}"
elif [ -z "$SKIP_YOLO" ]; then
    echo -e "${YELLOW}⏱️  Estimated time: 8-12 hours${NC}"
else
    echo -e "${YELLOW}⏱️  Estimated time: 2-4 hours${NC}"
fi

echo ""
echo "Training will start in 5 seconds... (Ctrl+C to cancel)"
sleep 5

echo ""
echo "=============================================================================="
echo ""

# Build command
CMD="python3 scripts/run_phase2_training.py"
CMD="$CMD --db $DB_PATH"
CMD="$CMD --output $PROJECT_ROOT/phase2_outputs"

if [ -n "$SKIP_YOLO" ]; then
    CMD="$CMD $SKIP_YOLO"
else
    CMD="$CMD --yolo-epochs $YOLO_EPOCHS"
fi

if [ -n "$SKIP_STYLE" ]; then
    CMD="$CMD $SKIP_STYLE"
else
    CMD="$CMD --style-epochs $STYLE_EPOCHS"
fi

CMD="$CMD --batch-size $BATCH_SIZE"

# Log start time
START_TIME=$(date +%s)
echo -e "${GREEN}Training started at $(date)${NC}" | tee training.log
echo "" | tee -a training.log

# Run training
eval $CMD 2>&1 | tee -a training.log

# Check if training succeeded
if [ ${PIPESTATUS[0]} -eq 0 ]; then
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    HOURS=$((DURATION / 3600))
    MINUTES=$(((DURATION % 3600) / 60))

    echo ""
    echo "=============================================================================="
    echo -e "${GREEN}✅ Training completed successfully!${NC}"
    echo "=============================================================================="
    echo ""
    echo -e "Duration: ${HOURS}h ${MINUTES}m"
    echo ""
    echo "Output files:"
    echo "  • YOLO model: phase2_outputs/yolo_training_runs/finetune_294_classes/weights/best.pt"
    echo "  • Style models: phase2_outputs/style_classifier_outputs/"
    echo "  • Training report: phase2_outputs/training_report_*.txt"
    echo ""
    echo -e "${BLUE}Next steps:${NC}"
    echo "  1. Review training report:"
    echo "     cat phase2_outputs/training_report_*.txt"
    echo ""
    echo "  2. Upload models to Modal Volume:"
    echo "     See PAPERSPACE_TRAINING_QUICKSTART.md section 'Upload Models to Modal'"
    echo ""
    echo "  3. Test your upgraded pipeline!"
    echo ""
    echo -e "${GREEN}Training log saved to: training.log${NC}"
    echo ""
else
    echo ""
    echo "=============================================================================="
    echo -e "${RED}❌ Training failed${NC}"
    echo "=============================================================================="
    echo ""
    echo "Check the log for errors:"
    echo "  tail -100 training.log"
    echo ""
    exit 1
fi

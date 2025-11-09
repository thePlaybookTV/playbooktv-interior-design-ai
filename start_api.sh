#!/bin/bash
# Startup script for PlaybookTV API
# Run this from anywhere - it will handle paths correctly

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "========================================="
echo "PlaybookTV Interior Design AI - API"
echo "========================================="
echo ""

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "📁 Project root: $SCRIPT_DIR"
echo ""

# Set Python path
export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"

# Check Python
if ! command -v python &> /dev/null; then
    echo -e "${RED}❌ Python not found!${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Python found:${NC} $(python --version)"

# Check required packages
echo ""
echo "📦 Checking dependencies..."

MISSING_DEPS=0

for package in fastapi uvicorn torch ultralytics pillow; do
    if python -c "import $package" 2>/dev/null; then
        echo -e "  ${GREEN}✅${NC} $package"
    else
        echo -e "  ${RED}❌${NC} $package - Missing!"
        MISSING_DEPS=1
    fi
done

if [ $MISSING_DEPS -eq 1 ]; then
    echo ""
    echo -e "${YELLOW}⚠️  Missing dependencies detected${NC}"
    echo "Installing required packages..."
    pip install -q fastapi uvicorn python-multipart pillow torch torchvision ultralytics python-dotenv
fi

# Check for .env file
echo ""
if [ -f "api/.env" ]; then
    echo -e "${GREEN}✅ .env file found${NC}"
else
    echo -e "${YELLOW}⚠️  No .env file found${NC}"
    if [ -f "api/.env.example" ]; then
        echo "Creating .env from template..."
        cp api/.env.example api/.env
        echo -e "${YELLOW}⚠️  Please edit api/.env and add your API keys!${NC}"
    fi
fi

# Check for database
echo ""
if [ -f "database_metadata.duckdb" ]; then
    DB_SIZE=$(du -h database_metadata.duckdb | cut -f1)
    echo -e "${GREEN}✅ Database found${NC} ($DB_SIZE)"
else
    echo -e "${YELLOW}⚠️  Database not found: database_metadata.duckdb${NC}"
    echo "   Upload it to: $SCRIPT_DIR/database_metadata.duckdb"
fi

# Check for models
echo ""
echo "🤖 Checking for models..."

MODEL_FOUND=0

# Check Phase 1 model
if [ -f "models_best_interior_model.pth" ]; then
    MODEL_SIZE=$(du -h models_best_interior_model.pth | cut -f1)
    echo -e "  ${GREEN}✅${NC} Phase 1 model found ($MODEL_SIZE)"
    MODEL_FOUND=1
fi

# Check Phase 2 models
if [ -f "phase2_outputs/yolo_training_runs/finetune_294_classes/weights/best.pt" ]; then
    echo -e "  ${GREEN}✅${NC} Phase 2 YOLO model found"
    MODEL_FOUND=1
fi

if [ -f "best_efficientnet_style_classifier.pth" ]; then
    echo -e "  ${GREEN}✅${NC} EfficientNet model found"
    MODEL_FOUND=1
fi

if [ $MODEL_FOUND -eq 0 ]; then
    echo -e "  ${YELLOW}⚠️  No models found${NC}"
    echo "     API will start but model loading may fail"
    echo "     Either run Phase 2 training or upload pre-trained models"
fi

# Start server
echo ""
echo "========================================="
echo "🚀 Starting API Server"
echo "========================================="
echo ""

cd "$SCRIPT_DIR"

# Get port from .env or use default
PORT=${PORT:-8000}

echo "Server will be available at:"
echo "  • Local: http://localhost:$PORT"
echo "  • Docs:  http://localhost:$PORT/docs"
echo ""
echo "Press Ctrl+C to stop"
echo ""

# Start with uvicorn
python -m uvicorn api.main:app --host 0.0.0.0 --port $PORT --reload

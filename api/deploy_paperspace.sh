#!/bin/bash
# Deploy FastAPI server to Paperspace
# Run this in your Paperspace terminal

set -e

echo "========================================="
echo "PlaybookTV API - Paperspace Deployment"
echo "========================================="

# Navigate to API directory
cd /notebooks/playbooktv-interior-design-ai/api

# Install dependencies
echo "📦 Installing dependencies..."
pip install -r requirements.txt

# Copy environment variables
if [ ! -f .env ]; then
    echo "📝 Creating .env file..."
    cp .env.example .env
    echo "⚠️  Please edit .env and update MODEL_PATHS!"
fi

# Check if models exist
echo "🔍 Checking for model files..."
MODELS_MISSING=0

if [ ! -f "../phase2_outputs/yolo_training_runs/finetune_294_classes/weights/best.pt" ]; then
    echo "❌ YOLO model not found"
    MODELS_MISSING=1
fi

if [ ! -f "../best_efficientnet_style_classifier.pth" ]; then
    echo "❌ EfficientNet model not found"
    MODELS_MISSING=1
fi

if [ $MODELS_MISSING -eq 1 ]; then
    echo ""
    echo "⚠️  Models not found. You need to either:"
    echo "   1. Run Phase 2 training first, OR"
    echo "   2. Upload pre-trained models to the correct paths"
    echo ""
    echo "Continuing anyway..."
fi

# Start server
echo ""
echo "========================================="
echo "Starting FastAPI server..."
echo "========================================="
echo ""
echo "Server will be available at:"
echo "  http://localhost:8000"
echo ""
echo "API docs at:"
echo "  http://localhost:8000/docs"
echo ""

python main.py

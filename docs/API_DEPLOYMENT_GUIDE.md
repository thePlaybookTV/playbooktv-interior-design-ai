# FastAPI Server Deployment Guide

## Overview

This guide shows you how to deploy the PlaybookTV Interior Design AI as a **FastAPI server** that your Modomo app can access.

---

## 🎯 What You're Building

A REST API server with these endpoints:

```
POST /analyze          - Complete analysis (detection + style)
POST /detect           - Object detection only
POST /classify/style   - Style classification only
GET  /health           - Health check
GET  /models/info      - Model information
GET  /                 - API info
```

---

## 📋 Prerequisites

You need **trained models** first! Two options:

### Option A: Train Models (Phase 2)
Run Phase 2 training to get the models:
```bash
python scripts/run_phase2_training.py \
    --db database_metadata.duckdb \
    --output ./phase2_outputs
```

### Option B: Use Phase 1 Model (Quick Start)
Use your existing `models_best_interior_model.pth` temporarily
(Less accurate but works immediately)

---

## 🚀 Deployment Options

### Option 1: Paperspace Deployment (Recommended)

**Why Paperspace?**
- ✅ Already has GPU
- ✅ You already have it running
- ✅ Can expose public URL

#### Step 1: Prepare Models

```bash
# In Paperspace terminal
cd /notebooks/playbooktv-interior-design-ai

# If Phase 2 is complete, copy models to api folder
mkdir -p api/models
cp phase2_outputs/yolo_training_runs/finetune_294_classes/weights/best.pt api/models/yolo_best.pt
cp best_efficientnet_style_classifier.pth api/models/
cp best_resnet_style_classifier.pth api/models/
cp best_vit_style_classifier.pth api/models/
```

#### Step 2: Configure Environment

```bash
cd api
cp .env.example .env

# Edit .env
nano .env

# Update these paths:
YOLO_MODEL_PATH=./models/yolo_best.pt
EFFICIENTNET_PATH=./models/best_efficientnet_style_classifier.pth
RESNET_PATH=./models/best_resnet_style_classifier.pth
VIT_PATH=./models/best_vit_style_classifier.pth
```

#### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

#### Step 4: Start Server

```bash
# Test locally first
python main.py

# Or use the deployment script
chmod +x deploy_paperspace.sh
./deploy_paperspace.sh
```

#### Step 5: Expose to Internet

**In Paperspace**, you can use:

```bash
# Option A: Use ngrok (easiest)
pip install pyngrok
python << EOF
from pyngrok import ngrok
import os

# Start ngrok tunnel
public_url = ngrok.connect(8000)
print(f"🌐 Public URL: {public_url}")
print("Share this with your Modomo app!")

# Keep running
input("Press Enter to stop...")
EOF
```

**Option B: Use Paperspace's built-in port forwarding**
- Paperspace automatically exposes ports
- Your URL will be something like:
  `https://n8mwcw7u7e.clg07azjl.paperspacegradient.com:8000`

---

### Option 2: Local Development

For testing on your Mac (without GPU):

```bash
# Navigate to API folder
cd /Users/leslieisah/MDMv3/playbooktv-interior-design-ai/api

# Install dependencies
pip install -r requirements.txt

# Start server
python main.py
```

Server runs at: `http://localhost:8000`

---

### Option 3: Cloud Deployment (Production)

For production deployment:

**Render.com** (Easiest):
1. Create account at render.com
2. Connect GitHub repo
3. Choose "Web Service"
4. Set build command: `pip install -r api/requirements.txt`
5. Set start command: `cd api && python main.py`
6. Add environment variables
7. Deploy!

**AWS/GCP** (More control):
- Use EC2/Compute Engine with GPU
- Set up Docker container
- Configure load balancer
- See production deployment section below

---

## 📡 Using the API

### From Modomo App

```javascript
// Send image for analysis
const formData = new FormData();
formData.append('file', imageFile);

const response = await fetch('https://your-api-url.com/analyze', {
  method: 'POST',
  body: formData
});

const result = await response.json();

console.log('Detections:', result.detections);
console.log('Style:', result.style.style);
console.log('Confidence:', result.style.confidence);
```

### cURL Examples

```bash
# Complete analysis
curl -X POST "http://localhost:8000/analyze" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/path/to/image.jpg"

# Object detection only
curl -X POST "http://localhost:8000/detect" \
  -F "file=@/path/to/image.jpg"

# Style classification
curl -X POST "http://localhost:8000/classify/style" \
  -F "file=@/path/to/image.jpg"

# Health check
curl "http://localhost:8000/health"
```

### Response Format

**`/analyze` response**:
```json
{
  "detections": [
    {
      "item_type": "sectional_sofa",
      "confidence": 0.89,
      "bbox": [120.5, 200.3, 450.2, 380.7],
      "area_percentage": 25.3
    },
    {
      "item_type": "coffee_table",
      "confidence": 0.85,
      "bbox": [300.1, 350.2, 500.5, 450.8],
      "area_percentage": 12.5
    }
  ],
  "detection_count": 2,
  "style": {
    "style": "modern",
    "confidence": 0.78,
    "all_probabilities": {
      "modern": 0.78,
      "contemporary": 0.12,
      "minimalist": 0.05,
      "scandinavian": 0.03,
      "traditional": 0.02
    }
  },
  "processing_time_ms": 234.5
}
```

---

## 🔧 Configuration

### Environment Variables

Create `api/.env`:

```bash
# Server
PORT=8000
HOST=0.0.0.0

# Model Paths (IMPORTANT - Update these!)
YOLO_MODEL_PATH=./models/yolo_best.pt
EFFICIENTNET_PATH=./models/best_efficientnet_style_classifier.pth
RESNET_PATH=./models/best_resnet_style_classifier.pth
VIT_PATH=./models/best_vit_style_classifier.pth

# CORS (Update for production)
ALLOWED_ORIGINS=https://modomo.app,http://localhost:3000

# Your API Keys (for future data collection)
ROBOFLOW_API_KEY=qgdh7zxmWdGi2N8xHsVF
KAGGLE_USERNAME=pearlisa
KAGGLE_KEY=ec87fb61ac251c5076abb1418da07c5b
HUGGINGFACE_TOKEN=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
UNSPLASH_ACCESS_KEY=_emTInR1snz-qBJYltXwot6fBsfZkJfEjLsEGn-s7U0
PEXELS_API_KEY=Ncl1eB8fS4A8qPj5yViznt7FlJ36k672yoOihJvXg9ZwqbZMHCChSBhj
```

---

## 🧪 Testing the API

### 1. Start the Server

```bash
cd api
python main.py
```

### 2. Open API Docs

Go to: `http://localhost:8000/docs`

You'll see **Swagger UI** with interactive API documentation!

### 3. Test with Sample Image

```python
import requests

# Test analysis endpoint
with open('test_image.jpg', 'rb') as f:
    files = {'file': f}
    response = requests.post('http://localhost:8000/analyze', files=files)
    print(response.json())
```

---

## 🐳 Docker Deployment (Optional)

Create `api/Dockerfile`:

```dockerfile
FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy API code
COPY . .

# Expose port
EXPOSE 8000

# Start server
CMD ["python", "main.py"]
```

Build and run:
```bash
docker build -t playbooktv-api .
docker run -p 8000:8000 -v $(pwd)/models:/app/models playbooktv-api
```

---

## 📊 Performance Optimization

### 1. Batch Processing

For multiple images, modify API to accept batch uploads.

### 2. Caching

Add Redis caching for frequently analyzed images.

### 3. GPU Optimization

```python
# In main.py, add batch inference
with torch.cuda.amp.autocast():
    predictions = model(images)
```

### 4. Load Balancing

Use Gunicorn with multiple workers:
```bash
gunicorn -w 4 -k uvicorn.workers.UvicornWorker api.main:app
```

---

## 🔒 Security Considerations

### 1. API Key Authentication

Add to `main.py`:
```python
from fastapi import Security, HTTPException
from fastapi.security.api_key import APIKeyHeader

API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME)

async def get_api_key(api_key: str = Security(api_key_header)):
    if api_key != os.getenv("API_KEY"):
        raise HTTPException(status_code=403, detail="Invalid API Key")
    return api_key

# Add to endpoints
@app.post("/analyze")
async def analyze_image(
    file: UploadFile = File(...),
    api_key: str = Depends(get_api_key)
):
    ...
```

### 2. Rate Limiting

```bash
pip install slowapi

# Add to main.py
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

@app.post("/analyze")
@limiter.limit("10/minute")
async def analyze_image(...):
    ...
```

### 3. HTTPS

Use reverse proxy (nginx) with SSL certificate.

---

## 🚨 Troubleshooting

### Models Not Loading

```python
# Check model paths
import os
print("YOLO exists:", os.path.exists(config.YOLO_MODEL_PATH))
print("EfficientNet exists:", os.path.exists(config.EFFICIENTNET_PATH))
```

### GPU Not Detected

```python
import torch
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")
```

### CORS Errors

Update `main.py`:
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://your-modomo-app.com"],  # Specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

---

## 📱 Modomo App Integration Example

```javascript
// modomo-app/src/services/api.js
const API_BASE_URL = 'https://your-paperspace-url.com';

export async function analyzeImage(imageFile) {
  const formData = new FormData();
  formData.append('file', imageFile);

  const response = await fetch(`${API_BASE_URL}/analyze`, {
    method: 'POST',
    body: formData,
    headers: {
      'X-API-Key': process.env.REACT_APP_API_KEY // If using auth
    }
  });

  if (!response.ok) {
    throw new Error(`API error: ${response.statusText}`);
  }

  return await response.json();
}

// Usage in component
import { analyzeImage } from './services/api';

async function handleImageUpload(file) {
  try {
    setLoading(true);
    const result = await analyzeImage(file);

    console.log('Detected items:', result.detections);
    console.log('Style:', result.style.style);
    console.log('Confidence:', result.style.confidence);

    // Update UI with results
    setDetections(result.detections);
    setStyle(result.style);
  } catch (error) {
    console.error('Analysis failed:', error);
  } finally {
    setLoading(false);
  }
}
```

---

## 📈 Monitoring

### Health Checks

```bash
# Continuous health monitoring
while true; do
  curl -s http://localhost:8000/health | jq
  sleep 60
done
```

### Logging

Logs are automatically generated. View with:
```bash
tail -f api.log
```

---

## ✅ Quick Deployment Checklist

- [ ] Train Phase 2 models (or use Phase 1)
- [ ] Copy models to `api/models/` folder
- [ ] Create `.env` file with correct paths
- [ ] Install dependencies: `pip install -r requirements.txt`
- [ ] Test locally: `python main.py`
- [ ] Verify at `http://localhost:8000/docs`
- [ ] Deploy to Paperspace
- [ ] Expose with ngrok or Paperspace URL
- [ ] Update Modomo app with API URL
- [ ] Test from Modomo app
- [ ] Set up monitoring and alerts

---

## 🎉 You're Done!

Your API is now ready for the Modomo app to use!

**API Endpoints for Modomo**:
- `POST /analyze` - Main endpoint (detection + style)
- `POST /detect` - Detection only (faster)
- `POST /classify/style` - Style only

**Next Steps**:
1. Deploy to Paperspace
2. Get public URL
3. Update Modomo app config
4. Start sending images!

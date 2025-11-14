# Paperspace API Deployment Guide

Deploy your Modomo Interior Design AI API on Paperspace with GPU support.

## Overview

**Architecture:**
- **Paperspace Gradient Deployment** - Runs your FastAPI server with GPU access
- **Models on Persistent Storage** - YOLO, SAM2, and SD models on volume
- **Redis** - External (Railway, Redis Cloud, or Upstash)
- **R2 Storage** - Cloudflare R2 for images

## Prerequisites

1. Paperspace account with GPU access
2. Cloudflare R2 bucket (for image storage)
3. Redis instance (Railway, Redis Cloud, or Upstash)
4. Model files downloaded locally

## Option 1: Paperspace Deployment (Production)

### Step 1: Prepare Your Repository

```bash
# Make sure all changes are committed
git add .
git commit -m "Prepare for Paperspace deployment"
git push origin main
```

### Step 2: Create Paperspace Deployment

1. Go to https://console.paperspace.com
2. Click **Deployments** → **Create Deployment**
3. Choose **Custom Container**
4. Configure:
   - **Name**: `modomo-api`
   - **Machine Type**: Choose GPU (e.g., `P4000`, `RTX4000`, or `A4000`)
   - **Container**: Build from Dockerfile
   - **Dockerfile Path**: `Dockerfile.paperspace`
   - **Port**: `8000`

### Step 3: Add Environment Variables

In the Paperspace deployment settings, add:

```bash
# Redis Configuration
REDIS_URL=redis://default:PASSWORD@HOST:PORT

# Cloudflare R2 Storage
R2_BUCKET_NAME=your-bucket-name
R2_ENDPOINT_URL=https://ACCOUNT_ID.r2.cloudflarestorage.com
R2_ACCESS_KEY_ID=your-access-key
R2_SECRET_ACCESS_KEY=your-secret-key

# Model Paths (on persistent volume)
YOLO_MODEL_PATH=/storage/models/yolov8m.pt
SAM2_CHECKPOINT=/storage/models/sam2_hiera_large.pt
PHASE1_MODEL_PATH=/storage/models/best_interior_model.pth

# API Configuration
ENVIRONMENT=production
API_SECRET_KEY=your-secret-key-generate-with-openssl
```

### Step 4: Attach Persistent Storage

1. In deployment settings, click **Add Volume**
2. Create or attach existing volume:
   - **Mount Path**: `/storage`
   - **Size**: 50GB (for models)

### Step 5: Upload Models to Volume

Via Paperspace console or CLI:

```bash
# SSH into your deployment
paperspace deployments ssh DEPLOYMENT_ID

# Create models directory
mkdir -p /storage/models

# Upload models (use scp, wget, or download from R2)
cd /storage/models

# Download YOLO model
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m.pt

# Download SAM2 checkpoint
wget https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt

# Copy your Phase1 model if you have it
# scp best_interior_model.pth user@host:/storage/models/
```

### Step 6: Deploy

1. Click **Deploy**
2. Wait for build to complete (~10-15 minutes first time)
3. Once deployed, you'll get a public URL like: `https://your-deployment.gradient.run`

### Step 7: Test Your Deployment

```bash
# Health check
curl https://your-deployment.gradient.run/health

# Test API docs
open https://your-deployment.gradient.run/docs
```

---

## Option 2: Paperspace Notebook (Development/Testing)

### Quick Start

1. Create a new Gradient Notebook
2. Choose GPU machine (P4000 or better)
3. Clone your repo:

```bash
cd /notebooks
git clone https://github.com/yourusername/playbooktv-interior-design-ai app
cd app
```

4. Run the startup script:

```bash
chmod +x start_api.sh
./start_api.sh
```

5. Get public URL with ngrok:

```bash
# In a new terminal
pip install pyngrok

python << 'EOF'
from pyngrok import ngrok
import time

public_url = ngrok.connect(8000)
print(f"\n🌐 Public URL: {public_url}\n")

try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("\nStopped")
EOF
```

---

## Cost Comparison

| Option | Cost (approx) | Pros | Cons |
|--------|---------------|------|------|
| **Paperspace P4000** | ~$0.51/hr | Good GPU, persistent | Hourly billing |
| **Paperspace RTX4000** | ~$0.56/hr | Better GPU | Hourly billing |
| **Paperspace A4000** | ~$0.76/hr | Fast GPU | Higher cost |
| **Paperspace Notebook** | ~$0.47/hr | Easy setup | Manual, ngrok needed |

**Monthly costs** (24/7):
- P4000: ~$367/month
- RTX4000: ~$403/month
- A4000: ~$547/month

**Alternative**: Use Paperspace's scale-to-zero feature - only pay when requests come in.

---

## Redis Options

### Option 1: Railway Redis (Recommended)
- $5/month for 1GB
- Easy integration
- Low latency

### Option 2: Redis Cloud
- Free tier: 30MB
- Paid: $7/month for 500MB

### Option 3: Upstash
- Serverless, pay-per-request
- Good for low traffic

---

## Architecture Diagram

```
┌─────────────┐
│   Client    │
│   (Mobile)  │
└──────┬──────┘
       │
       │ HTTPS
       ▼
┌─────────────────────────────────┐
│  Paperspace Deployment (GPU)    │
│  ┌───────────────────────────┐  │
│  │  FastAPI Server           │  │
│  │  - YOLO Detection         │  │
│  │  - SAM2 Segmentation      │  │
│  │  - Stable Diffusion 1.5   │  │
│  └───────────────────────────┘  │
│                                  │
│  /storage (Persistent Volume)   │
│  └─ models/                     │
│     ├─ yolov8m.pt              │
│     ├─ sam2_hiera_large.pt     │
│     └─ best_interior_model.pth │
└─────────────────────────────────┘
       │              │
       │              │
       ▼              ▼
  ┌─────────┐   ┌──────────┐
  │  Redis  │   │  R2      │
  │ (Railway)│   │(Cloudflare)│
  └─────────┘   └──────────┘
```

---

## Monitoring & Logs

### View Logs
```bash
paperspace deployments logs DEPLOYMENT_ID --tail 100
```

### Monitor GPU Usage
```bash
# SSH into deployment
paperspace deployments ssh DEPLOYMENT_ID

# Check GPU
nvidia-smi

# Monitor continuously
watch -n 1 nvidia-smi
```

---

## Troubleshooting

### Build Fails
- Check Dockerfile.paperspace syntax
- Verify requirements.txt is complete
- Check build logs for specific errors

### Out of Memory
- Reduce batch sizes
- Use smaller models (yolov8n instead of yolov8m)
- Increase GPU machine size

### Slow Response Times
- Check GPU is being used: `nvidia-smi`
- Monitor logs for bottlenecks
- Consider model optimization (quantization, pruning)

### Models Not Loading
- Verify paths in environment variables
- Check /storage mount is attached
- Verify models are uploaded: `ls -lh /storage/models/`

---

## Next Steps

1. **Deploy to Paperspace** using steps above
2. **Upload models** to persistent storage
3. **Configure environment variables** with Redis and R2
4. **Test the API** with health check
5. **Update mobile app** with new API URL
6. **Monitor performance** and optimize as needed

---

## Support

- Paperspace Docs: https://docs.paperspace.com
- Gradient CLI: https://docs.paperspace.com/gradient/cli/
- Community: https://community.paperspace.com

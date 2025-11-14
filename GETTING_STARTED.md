# Getting Started with Modomo Interior Design AI

**Welcome!** This guide will help you understand what you have and how to use it.

---

## 📋 What You Have Right Now

Based on your current setup:

1. ✅ **Production API** - Deployed on Railway + Modal
2. ✅ **DuckDB Database** - `database_metadata.duckdb` (18 MB, 74,872 images)
3. ✅ **Phase 1 Model** - `models_best_interior_model.pth` (130 MB)
4. ✅ **Modal GPU Processing** - Stable Diffusion + ControlNet deployed
5. ✅ **R2 Storage** - Cloudflare R2 for image storage

**Status:** You have a fully working production system! 🎉

---

## 🎯 What This System Does

### Current Capabilities

**1. Image Transformation (Production API)**
- Upload room photo → Get AI-transformed design in 15 seconds
- 5 design styles: modern, scandinavian, boho, industrial, minimalist
- Cost: ~£0.02 per transformation
- Quality: 92% quality score

**2. Object Detection (Phase 1)**
- Detects 14 COCO furniture classes
- Uses YOLO + SAM2 for precise segmentation
- 25,497 detections in database

**3. Room & Style Classification (Phase 1)**
- Room type: 68.7% accuracy (6 classes)
- Style classification: 53.8% accuracy (9 styles)

---

## 🚀 Quick Start Options

### Option 1: Use the Production API (Recommended)

Your API is already deployed and working!

```bash
# Test the health endpoint
curl https://playbooktv-interior-design-ai-production.up.railway.app/health

# Transform an image
curl -X POST https://playbooktv-interior-design-ai-production.up.railway.app/transform/submit \
  -F "file=@your-room-image.jpg" \
  -F "style=modern"
```

**See:** [docs/deployment/README.md](docs/deployment/README.md) for full API documentation

---

### Option 2: Train Phase 2 Models (Optional)

Want better accuracy? Train upgraded models:

**Improvements:**
- Object detection: 14 classes → 294 specific categories
- Style classification: 53.8% → 70%+ accuracy

**Requirements:**
- NVIDIA GPU (8GB+ VRAM)
- 50GB storage
- 10-16 hours training time

**Quick command:**
```bash
python scripts/run_phase2_training.py \
    --db database_metadata.duckdb \
    --output ./phase2_outputs \
    --yolo-epochs 100 \
    --style-epochs 30
```

**See:** [docs/training/QUICKSTART.md](docs/training/QUICKSTART.md) for Phase 2 training

---

### Option 3: Develop Mobile App

Build a mobile app that connects to your API:

**Architecture:**
```
React Native App
    ↓ Upload image
Railway API
    ↓ Submit to GPU
Modal GPU Processing
    ↓ Return result
Display in App
```

**API Endpoints:**
- `POST /transform/submit` - Upload image
- `GET /transform/status/{job_id}` - Check status
- `WS /ws/transform/{job_id}` - Real-time updates

**See:** [docs/deployment/README.md#api-endpoints](docs/deployment/README.md#api-endpoints)

---

## 📊 System Architecture

```
┌─────────────────┐
│   Mobile App    │
└────────┬────────┘
         │
         ↓
┌─────────────────────────────────────┐
│  Railway API (Minimal ~500MB)       │
│  • Job queue (Redis)                │
│  • Storage (Cloudflare R2)          │
│  • WebSocket updates                │
└─────────┬───────────────────────────┘
          │
          ↓
┌─────────────────────────────────────┐
│  Modal GPU (T4)                     │
│  • YOLO + SAM2 detection            │
│  • Depth map generation             │
│  • Stable Diffusion + ControlNet    │
│  • ~15 seconds processing           │
└─────────┬───────────────────────────┘
          │
          ↓
┌─────────────────────────────────────┐
│  Cloudflare R2 Storage              │
│  • Original images                  │
│  • Transformed results              │
│  • Thumbnails                       │
└─────────────────────────────────────┘
```

---

## 📁 Project Structure

```
playbooktv-interior-design-ai/
│
├── api/
│   ├── main.py                      # Full API (with Phase 1 models)
│   └── main_minimal.py              # ✅ DEPLOYED (minimal, production)
│
├── modal_functions/
│   └── sd_inference_complete.py    # ✅ DEPLOYED (GPU processing)
│
├── src/
│   ├── models/                     # ML model definitions
│   ├── processing/                 # Image processing
│   ├── services/                   # API services (job queue, storage)
│   └── utils/                      # Utilities
│
├── scripts/
│   └── run_phase2_training.py      # Phase 2 training script
│
├── docs/
│   ├── deployment/                 # Deployment guides
│   ├── training/                   # Training guides
│   ├── api/                        # API documentation
│   └── status/                     # Current status reports
│
├── database_metadata.duckdb        # Image database (gitignored)
├── models_best_interior_model.pth  # Phase 1 model (gitignored)
└── README.md                       # Project overview
```

---

## 🎓 Learning Path

### If You Want To...

**...Use the API:**
1. Read [docs/deployment/README.md](docs/deployment/README.md)
2. Test endpoints with curl
3. Build a client application

**...Train better models:**
1. Read [docs/training/QUICKSTART.md](docs/training/QUICKSTART.md)
2. Set up GPU environment
3. Run Phase 2 training

**...Understand the system:**
1. Read [docs/deployment/ARCHITECTURE.md](docs/deployment/ARCHITECTURE.md)
2. Read [README.md](README.md)
3. Explore the codebase

**...Deploy your own:**
1. Read [docs/deployment/QUICKSTART.md](docs/deployment/QUICKSTART.md)
2. Set up Railway account
3. Deploy in 3 steps

---

## 🔍 Verifying Your Setup

### Check Database

```bash
python check_database.py
```

**Expected output:**
```
✅ Images: 74,872
✅ Detections: 25,497
✅ Room classifications: 5,262
✅ Style classifications: 5,262
🎉 DATABASE IS READY!
```

---

### Check Phase 1 Model

```python
import torch

# Load model
checkpoint = torch.load('models_best_interior_model.pth')

# See metrics
print(f"Room accuracy: {checkpoint['val_room_acc']:.1%}")
print(f"Style accuracy: {checkpoint['val_style_acc']:.1%}")
```

**Expected output:**
```
Room accuracy: 68.7%
Style accuracy: 53.8%
```

---

### Check Production API

```bash
curl https://playbooktv-interior-design-ai-production.up.railway.app/health
```

**Expected output:**
```json
{
  "status": "healthy",
  "services": {
    "redis": "connected",
    "modal": "connected",
    "storage": "connected",
    "websocket": "ready"
  }
}
```

---

## 💰 Cost Breakdown

### Current Costs (Production)

| Service | Cost | Usage |
|---------|------|-------|
| Railway API | $5/month | Fixed (Hobby plan) |
| Modal GPU | $0.30/hour | Pay-per-use (~£0.02/image) |
| Cloudflare R2 | ~$5/month | Storage + operations |
| **Total Fixed** | **~$10/month** | Base cost |
| **Per Image** | **~£0.02** | Variable cost |

**Example:**
- 1,000 transformations/month = $10 + (1000 × £0.02) = ~$35/month
- With 8% conversion @ £15 commission = £1,200 revenue
- **Profit margin: 97%** 🎉

---

## ❓ FAQ

### Q: Do I need to train Phase 2 models?
**A:** No! Your production API already works. Phase 2 just makes it more accurate.

### Q: Can I use this in production?
**A:** Yes! It's already deployed and working. 85% production-ready.

### Q: What's missing for full production?
**A:** Rate limiting, user authentication, monitoring. See [docs/status/PRODUCTION_STATUS.md](docs/status/PRODUCTION_STATUS.md)

### Q: How do I add more design styles?
**A:** Modify the style prompts in `modal_functions/sd_inference_complete.py` and redeploy.

### Q: Can I run this locally?
**A:** The API yes (see `api/main_minimal.py`). GPU processing requires Modal or local GPU setup.

### Q: What if I don't have a GPU for training?
**A:** Use Paperspace, RunPod, or Google Colab. See archived docs in `docs/archive/paperspace/`

---

## 🎯 Next Steps

### Immediate (Today)
1. ✅ Test the production API
2. ✅ Review [docs/status/PRODUCTION_STATUS.md](docs/status/PRODUCTION_STATUS.md)
3. ✅ Decide: build app or train Phase 2?

### Short-term (This Week)
1. Choose your path (see "Learning Path" above)
2. Set up development environment
3. Start building/training

### Long-term (This Month)
1. Build mobile application
2. Add authentication & rate limiting
3. Set up monitoring
4. Launch to users!

---

## 📚 Documentation Index

### Core Documentation
- [README.md](README.md) - Project overview
- **[GETTING_STARTED.md](GETTING_STARTED.md)** ← You are here
- [docs/deployment/README.md](docs/deployment/README.md) - Complete deployment guide
- [docs/training/QUICKSTART.md](docs/training/QUICKSTART.md) - Phase 2 training guide

### Deployment
- [docs/deployment/QUICKSTART.md](docs/deployment/QUICKSTART.md) - 3-step deployment
- [docs/deployment/ARCHITECTURE.md](docs/deployment/ARCHITECTURE.md) - System architecture
- [docs/deployment/ENVIRONMENT_SETUP.md](docs/deployment/ENVIRONMENT_SETUP.md) - Environment variables

### Training
- [docs/training/QUICKSTART.md](docs/training/QUICKSTART.md) - Quick start
- [docs/training/PHASE2_TRAINING_GUIDE.md](docs/training/PHASE2_TRAINING_GUIDE.md) - Comprehensive guide

### Status
- [docs/status/PRODUCTION_STATUS.md](docs/status/PRODUCTION_STATUS.md) - Current production state
- [docs/status/DEPLOYMENT_STATUS.md](docs/status/DEPLOYMENT_STATUS.md) - Deployment status

---

## 🆘 Getting Help

**Issues?**
1. Check [docs/deployment/README.md#troubleshooting](docs/deployment/README.md#troubleshooting)
2. Review error logs on Railway dashboard
3. Check Modal app status: `modal app list`

**Questions about the code?**
- Read the documentation in `docs/`
- Check inline code comments
- Review [README.md](README.md)

---

**Status:** ✅ Ready to use
**Last Updated:** November 14, 2025
**Production URL:** https://playbooktv-interior-design-ai-production.up.railway.app

**Happy building! 🚀**

# 🚀 DEPLOY NOW - Quick Start Guide

**Status**: ✅ All code complete - Ready to deploy!

---

## Quick Deploy (5 minutes)

### Step 1: Deploy to Modal (2 minutes)

```bash
# Deploy the GPU processing function
modal deploy modal_functions/sd_inference_complete.py
```

✅ **Expected output**:
```
✓ Created deployment modomo-sd-inference
✓ App is now deployed!
```

---

### Step 2: Commit and Push to Railway (2 minutes)

```bash
# Commit all changes
git add .
git commit -m "Complete backend: Modal App API migration + minimal Railway deployment"
git push origin main
```

Railway will automatically:
- Build Docker image (~5 min)
- Deploy FastAPI server
- Run health checks

---

### Step 3: Verify Deployment (1 minute)

```bash
# Check Railway health
curl https://playbooktv-interior-design-ai-production.up.railway.app/health

# Check Modal apps
modal app list
```

---

## Environment Variables (Already Set)

Railway already has these configured:

✅ Redis: `redis://default:CiFsKXyXMUqdtPVvAiuwiFtYWAZtRchY@metro.proxy.rlwy.net:25118`
✅ Modal Token: `ak-mdhwVsEGW46OtIdFT7j0FH`
✅ Modal App: `modomo-sd-inference`
✅ R2 Credentials: Configured

**Just need to add** (if not already set):
- `MODAL_APP_NAME=modomo-sd-inference`
- `API_SECRET_KEY=<generate with: openssl rand -hex 32>`

---

## Test Transformation

After deployment, test with:

```bash
# Submit test transformation
curl -X POST "https://playbooktv-interior-design-ai-production.up.railway.app/transform/submit" \
  -F "file=@test_image.jpg" \
  -F "style=modern"
```

Expected response:
```json
{
  "success": true,
  "job_id": "uuid-here",
  "estimated_time": 15,
  "websocket_url": "ws://...",
  "status_url": "https://..."
}
```

---

## What Changed (Modal API Update)

✅ Migrated from `modal.Stub` → `modal.App`
✅ Updated all decorators: `@stub.cls()` → `@app.cls()`
✅ Updated service lookup methods
✅ Backward compatible with existing config

See [docs/MODAL_API_MIGRATION.md](docs/MODAL_API_MIGRATION.md) for details.

---

## Architecture

```
Mobile App (React Native)
    ↓
Railway API (~500MB, $5/month)
  - FastAPI endpoints
  - Redis job queue
  - WebSocket updates
  - R2 storage
    ↓
Modal GPU (T4, £0.03/image)
  - YOLO + SAM2 detection
  - Depth map (DPT-Large)
  - Edge detection (Canny)
  - SD 1.5 + ControlNet
  - Result upload
    ↓
Results (Cloudflare R2)
```

---

## Costs

| Service | Cost |
|---------|------|
| Railway (Base) | $5/month |
| Modal (per image) | £0.03/image |
| R2 Storage | ~$0-5/month |
| **Total (1000 images)** | **~$35-40/month** |

**Profit margin**: 97.5% at 8% conversion with £15 commission

---

## Next Steps After Deployment

1. **Monitor logs** (first 24 hours)
   - Railway: Check for API errors
   - Modal: Check GPU usage
   - Redis: Check job queue

2. **Test thoroughly**
   - Submit 10-20 test transformations
   - Measure processing time (<15s target)
   - Verify costs match estimates

3. **Set up monitoring**
   - Sentry for error tracking
   - UptimeRobot for uptime
   - Cost alerts in Railway + Modal

4. **Build mobile app** (Week 3-4)
   - React Native interface
   - Camera integration
   - WebSocket real-time updates
   - Style selection UI

---

## Troubleshooting

### Modal deployment fails
```bash
# Check Modal auth
modal token list

# Force redeploy
modal deploy --force modal_functions/sd_inference_complete.py
```

### Railway build fails
```bash
# Check Railway logs
railway logs

# Force rebuild
railway up --force
```

### Health check fails
1. Check environment variables in Railway
2. Verify Modal app is deployed: `modal app list`
3. Check Redis connection: `redis-cli -u $REDIS_URL ping`

---

## Documentation

- **Complete Guide**: [READY_TO_DEPLOY.md](READY_TO_DEPLOY.md)
- **Architecture**: [docs/MINIMAL_ARCHITECTURE.md](docs/MINIMAL_ARCHITECTURE.md)
- **API Migration**: [docs/MODAL_API_MIGRATION.md](docs/MODAL_API_MIGRATION.md)
- **Railway Setup**: [docs/RAILWAY_DEPLOYMENT.md](docs/RAILWAY_DEPLOYMENT.md)

---

## ✨ You're Ready!

All backend code is complete. Just run these 3 commands:

```bash
# 1. Deploy Modal
modal deploy modal_functions/sd_inference_complete.py

# 2. Push to Railway
git add . && git commit -m "Ready to deploy!" && git push origin main

# 3. Test
curl https://playbooktv-interior-design-ai-production.up.railway.app/health
```

**Let's ship it!** 🚀

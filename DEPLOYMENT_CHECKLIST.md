# 🚀 Deployment Checklist

Quick checklist for deploying Modomo backend to Railway.

---

## ✅ Pre-Deployment (Complete)

- [x] Railway project created: `playbooktv-interior-design-ai-production`
- [x] Railway Redis provisioned: `redis://metro.proxy.rlwy.net:25118`
- [x] Cloudflare R2 configured and working
- [x] Dockerfile created
- [x] Procfile created
- [x] railway.json created
- [x] .dockerignore created
- [x] .env updated with Railway Redis
- [x] Backend services created (control generators, job queue, storage)

---

## 📋 Deployment Steps

### 1. Set Up Modal Account (5 minutes)

```bash
# Install Modal
pip install modal

# Get token
modal token new
```

**Action**: Copy `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET` for Railway

---

### 2. Configure Railway Environment Variables (10 minutes)

Go to Railway → Your Service → Variables, add:

```bash
# Generate API secret key first
openssl rand -hex 32

# Then add to Railway:
API_HOST=0.0.0.0
API_SECRET_KEY=<generated-key>
ENVIRONMENT=production

# Redis (use Railway reference)
REDIS_URL=${{Redis.REDIS_URL}}
REDIS_PASSWORD=${{Redis.REDIS_PASSWORD}}
REDIS_DB=0

# R2 (already configured)
CLOUDFLARE_ACCOUNT_ID=9bbdb3861142e65685d23f4955b88ebe
R2_ACCESS_KEY_ID=6c8abdff2cdad89323e36b258b1d0f4b
R2_SECRET_ACCESS_KEY=2a2bb806281b1b321803f91cbe8fbc4180536cd87cf745ad4fef368011c3a1d1
R2_BUCKET_NAME=reroom
R2_ENDPOINT_URL=https://9bbdb3861142e65685d23f4955b88ebe.r2.cloudflarestorage.com
CDN_DOMAIN=9bbdb3861142e65685d23f4955b88ebe.r2.cloudflarestorage.com

# Modal
MODAL_TOKEN_ID=<from-modal-token-new>
MODAL_TOKEN_SECRET=<from-modal-token-new>
MODAL_STUB_NAME=modomo-sd-inference

# Models
YOLO_MODEL_PATH=yolov8n.pt
SAM2_CHECKPOINT=checkpoints/sam2_hiera_large.pt
PHASE1_MODEL_PATH=models/best_interior_model.pth
```

---

### 3. Connect GitHub to Railway (2 minutes)

1. Railway Dashboard → Your Service → Settings → Source
2. Connect GitHub repository
3. Select branch: `main`
4. Enable auto-deploy

---

### 4. Push Code to GitHub (1 minute)

```bash
git add .
git commit -m "Add Railway deployment configuration"
git push origin main
```

Railway will auto-deploy! ⚡

---

### 5. Verify Deployment (5 minutes)

**Health Check**:
```bash
curl https://playbooktv-interior-design-ai-production.up.railway.app/health
```

**Expected Response**:
```json
{
  "status": "healthy",
  "redis_connected": true
}
```

**Check Logs**:
1. Railway Dashboard → Deployments
2. Click latest deployment
3. View logs for errors

---

### 6. Deploy Modal Function (10 minutes)

```bash
# Deploy SD inference function
modal deploy modal_functions/sd_inference.py
```

**Verify Modal**:
```bash
modal app list
# Should see: modomo-sd-inference
```

---

## 🧪 Testing

### Test API Endpoints

```bash
# Base endpoint
curl https://playbooktv-interior-design-ai-production.up.railway.app/

# Health check
curl https://playbooktv-interior-design-ai-production.up.railway.app/health

# Models info (once API is complete)
curl https://playbooktv-interior-design-ai-production.up.railway.app/models/info
```

---

## 📊 Post-Deployment

### Monitor

- [ ] Check Railway metrics (CPU, Memory, Network)
- [ ] Monitor Railway logs for errors
- [ ] Test Redis connection from logs
- [ ] Verify R2 storage uploads work

### Performance

- [ ] Test image upload speed
- [ ] Test transformation end-to-end (once complete)
- [ ] Check Modal GPU cold start time
- [ ] Measure API response times

### Security

- [ ] Verify HTTPS is enabled (Railway provides)
- [ ] Check API_SECRET_KEY is set
- [ ] Confirm .env is not in git (check .gitignore)
- [ ] Test CORS for mobile app (once configured)

---

## ⚡ Quick Commands

```bash
# View Railway logs
railway logs

# SSH into Railway container
railway shell

# Restart deployment
railway up --force

# Check Railway status
railway status

# Deploy Modal function
modal deploy modal_functions/sd_inference.py

# View Modal logs
modal app logs modomo-sd-inference
```

---

## 🆘 Troubleshooting

| Issue | Solution |
|-------|----------|
| Build fails | Check Dockerfile, verify requirements.txt |
| Redis connection error | Verify `REDIS_URL` variable, check Redis logs |
| Out of memory | Upgrade Railway plan or optimize models |
| Timeout on requests | Move to async processing with job queue |
| Modal not working | Check `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET` |

---

## 📞 Support

- **Railway Docs**: https://docs.railway.app
- **Modal Docs**: https://modal.com/docs
- **Detailed Guide**: See `docs/RAILWAY_DEPLOYMENT.md`

---

## ✨ Success Criteria

Deployment is successful when:

- ✅ `/health` endpoint returns 200 OK
- ✅ Redis connection confirmed in logs
- ✅ No errors in Railway deployment logs
- ✅ Modal function deployed and listed
- ✅ Test image upload works
- ✅ API responds within 2 seconds

---

**Production URL**: https://playbooktv-interior-design-ai-production.up.railway.app

**Status**: Ready to deploy once remaining backend services are complete

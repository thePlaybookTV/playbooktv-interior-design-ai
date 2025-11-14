# Railway Deployment Guide

Complete guide to deploying Modomo backend to Railway.

**Production URL**: `playbooktv-interior-design-ai-production.up.railway.app`
**Redis URL**: `redis://default:CiFsKXyXMUqdtPVvAiuwiFtYWAZtRchY@metro.proxy.rlwy.net:25118`

---

## Prerequisites

- [x] Railway account (already set up)
- [x] Railway Redis instance (already provisioned)
- [x] Cloudflare R2 bucket (already configured)
- [ ] Modal account and token
- [x] GitHub repository (for auto-deployment)

---

## Step 1: Configure Environment Variables in Railway

Go to your Railway project settings and add these environment variables:

### Required Variables

```bash
# API Configuration
API_HOST=0.0.0.0
API_SECRET_KEY=<generate-with-openssl-rand-hex-32>
ENVIRONMENT=production

# Redis (Already connected via Railway)
REDIS_URL=${{Redis.REDIS_URL}}
REDIS_PASSWORD=${{Redis.REDIS_PASSWORD}}
REDIS_DB=0

# Cloudflare R2
CLOUDFLARE_ACCOUNT_ID=9bbdb3861142e65685d23f4955b88ebe
R2_ACCESS_KEY_ID=6c8abdff2cdad89323e36b258b1d0f4b
R2_SECRET_ACCESS_KEY=2a2bb806281b1b321803f91cbe8fbc4180536cd87cf745ad4fef368011c3a1d1
R2_BUCKET_NAME=reroom
R2_ENDPOINT_URL=https://9bbdb3861142e65685d23f4955b88ebe.r2.cloudflarestorage.com
CDN_DOMAIN=9bbdb3861142e65685d23f4955b88ebe.r2.cloudflarestorage.com

# Modal Configuration
MODAL_TOKEN_ID=<your-modal-token-id>
MODAL_TOKEN_SECRET=<your-modal-token-secret>
MODAL_STUB_NAME=modomo-sd-inference

# Model Configuration
YOLO_MODEL_PATH=yolov8n.pt
SAM2_CHECKPOINT=checkpoints/sam2_hiera_large.pt
PHASE1_MODEL_PATH=models/best_interior_model.pth

# Training Configuration
DEFAULT_ROOM_TYPE=living_room
DEFAULT_STYLE=modern
```

### How to Set Environment Variables

1. Go to Railway dashboard: https://railway.app/project
2. Select your project: `playbooktv-interior-design-ai-production`
3. Click on your service (backend)
4. Go to "Variables" tab
5. Add each variable listed above
6. Click "Add" or "Deploy" to apply changes

### Generate Secure API Key

Run this command locally to generate a secure API key:

```bash
openssl rand -hex 32
```

Copy the output and use it for `API_SECRET_KEY`.

---

## Step 2: Connect Redis to Backend Service

Railway should auto-connect Redis, but verify:

1. In Railway dashboard, go to your Redis service
2. Click "Connect" tab
3. Copy the connection string (already done above)
4. Add to backend service variables as `REDIS_URL`

**Reference Variables**: Railway allows you to reference other services:
- Use `${{Redis.REDIS_URL}}` to automatically pull Redis connection
- This updates automatically if Redis restarts

---

## Step 3: Set Up Modal Account

Modal is required for GPU processing (Stable Diffusion inference).

### Create Modal Account

1. Go to https://modal.com
2. Sign up with GitHub or email
3. Verify your email

### Get Modal Token

Run locally:

```bash
# Install Modal
pip install modal

# Authenticate
modal token new

# This will open browser and create token
# Copy the token ID and secret shown in terminal
```

### Add Modal Token to Railway

Add these to Railway environment variables:
- `MODAL_TOKEN_ID`: Your Modal token ID
- `MODAL_TOKEN_SECRET`: Your Modal token secret

---

## Step 4: Deploy to Railway

Railway auto-deploys on git push. Here's how:

### Option A: Auto-Deploy (Recommended)

1. **Connect GitHub Repository**
   - In Railway dashboard, go to your service
   - Click "Settings" → "Source"
   - Connect to your GitHub repo
   - Select branch: `main` or `master`

2. **Push Code to GitHub**
   ```bash
   git add .
   git commit -m "Add Railway deployment configuration"
   git push origin main
   ```

3. **Railway Auto-Deploys**
   - Railway detects changes
   - Builds Docker image using Dockerfile
   - Deploys automatically
   - Check "Deployments" tab for progress

### Option B: Railway CLI Deploy

```bash
# Install Railway CLI
npm install -g @railway/cli

# Login
railway login

# Link to project
railway link

# Deploy
railway up
```

---

## Step 5: Verify Deployment

Once deployed, test your endpoints:

### Health Check

```bash
curl https://playbooktv-interior-design-ai-production.up.railway.app/health
```

Expected response:
```json
{
  "status": "healthy",
  "redis_connected": true,
  "models_loaded": true
}
```

### Test API

```bash
curl https://playbooktv-interior-design-ai-production.up.railway.app/
```

Expected response:
```json
{
  "message": "Modomo Interior Design AI API",
  "version": "1.0.0",
  "status": "running"
}
```

---

## Step 6: Monitor Deployment

### View Logs

1. Go to Railway dashboard
2. Select your service
3. Click "Deployments" tab
4. Click on latest deployment
5. View real-time logs

### Check Metrics

Railway provides:
- CPU usage
- Memory usage
- Network usage
- Request count

Access via "Metrics" tab in Railway dashboard.

---

## Step 7: Set Up Custom Domain (Optional)

### Add Custom Domain

1. Go to "Settings" in Railway
2. Click "Domains"
3. Click "Generate Domain" or "Custom Domain"
4. Add your domain (e.g., `api.modomo.com`)
5. Update DNS with provided CNAME record

### DNS Configuration

In your DNS provider (Cloudflare, etc.):

```
Type: CNAME
Name: api
Value: playbooktv-interior-design-ai-production.up.railway.app
TTL: Auto
```

---

## Troubleshooting

### Build Fails

**Issue**: Docker build fails with dependency errors

**Solution**:
1. Check Dockerfile syntax
2. Verify requirements.txt is correct
3. Check Railway logs for specific error
4. Rebuild: `railway up --force`

### Redis Connection Fails

**Issue**: Backend can't connect to Redis

**Solution**:
1. Verify `REDIS_URL` is set correctly
2. Use Railway reference: `${{Redis.REDIS_URL}}`
3. Check Redis service is running
4. Check Redis logs in Railway

### Models Not Loading

**Issue**: YOLO/SAM2 models fail to load

**Solution**:
1. Models download on first request (takes time)
2. Check Railway memory limit (increase if needed)
3. Consider pre-downloading models in Dockerfile
4. Check model paths in environment variables

### Out of Memory

**Issue**: Railway container runs out of memory

**Solution**:
1. Upgrade Railway plan (more RAM)
2. Optimize model loading (lazy load)
3. Use Modal for heavy processing (SD inference)
4. Reduce concurrent requests

### Timeout Issues

**Issue**: Requests timeout after 30 seconds

**Solution**:
1. Move slow operations to background (async with Redis)
2. Use WebSocket for long-running tasks
3. Increase Railway timeout (in Pro plan)
4. Optimize processing pipeline

---

## Cost Optimization

### Railway Costs

**Hobby Plan**: $5/month
- 500 hours
- 8GB RAM
- 8GB disk
- Perfect for MVP

**Pro Plan**: $20/month (if needed)
- More resources
- Custom domains
- Better support

### Estimated Monthly Costs

```
Railway Hobby Plan:     $5
Cloudflare R2:          $0-5 (minimal storage)
Modal GPU:              £20-50 (£0.03/image × 1000 images)
Total:                  ~$30-60/month
```

### Cost Reduction Tips

1. **Use Modal efficiently**
   - Scale to zero when idle
   - Use T4 GPUs (not A100)
   - Cache models in Modal

2. **Optimize Railway**
   - Use sleep mode for dev environments
   - Monitor resource usage
   - Upgrade only when needed

3. **Cloudflare R2**
   - Set lifecycle rules (delete after 30 days)
   - Use CDN for caching
   - Compress images

---

## Scaling Strategy

### Current Setup (MVP)
- Railway Hobby: 1 instance
- Modal: 0-10 containers (auto-scale)
- Redis: Single instance

### Growth to 100 Users/Day
- Railway Pro: 2-3 instances
- Modal: 10-20 containers
- Redis: Upgrade to larger plan

### Growth to 1000 Users/Day
- Consider AWS ECS or Kubernetes
- Modal: 50+ containers
- Redis Cluster
- Load balancer

---

## CI/CD Pipeline

Railway automatically deploys on git push, but you can add GitHub Actions for testing:

### .github/workflows/test.yml

```yaml
name: Test

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v3

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'

    - name: Install dependencies
      run: |
        pip install -r requirements.txt

    - name: Run tests
      run: |
        pytest tests/ -v

    - name: Run linting
      run: |
        black --check .
        flake8 .
```

---

## Security Checklist

- [x] API_SECRET_KEY is randomly generated
- [x] Redis password is set
- [x] R2 credentials are not in git
- [x] .env is in .gitignore
- [ ] HTTPS enabled (Railway provides automatically)
- [ ] Rate limiting configured (TODO)
- [ ] Input validation on all endpoints (TODO)
- [ ] CORS configured for mobile app (TODO)

---

## Next Steps

1. **Deploy Modal SD Function**
   - See `modal_functions/sd_inference.py`
   - Deploy: `modal deploy modal_functions/sd_inference.py`

2. **Test End-to-End Flow**
   - Upload image
   - Transform with style
   - Check result URL

3. **Set Up Monitoring**
   - Add Sentry for error tracking
   - Set up uptime monitoring (UptimeRobot)
   - Configure alerts

4. **Mobile App Integration**
   - Update mobile app with production URL
   - Test from React Native app
   - Deploy to TestFlight/Play Store beta

---

## Support

- **Railway Docs**: https://docs.railway.app
- **Railway Discord**: https://discord.gg/railway
- **Modal Docs**: https://modal.com/docs
- **Modomo Team**: [Your contact info]

---

**Last Updated**: November 2025
**Deployment Status**: Ready for deployment
**Production URL**: https://playbooktv-interior-design-ai-production.up.railway.app

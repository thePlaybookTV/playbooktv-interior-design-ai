# 🚀 Deployment Quickstart

**3-step deployment to Railway + Modal**

---

## Step 1: Deploy to Railway (5 minutes)

```bash
# 1. Push to GitHub
git push origin main

# 2. Go to Railway
https://railway.app/new

# 3. Select "Deploy from GitHub repo"
# 4. Choose your repository
# 5. Railway auto-deploys!
```

**Add environment variables in Railway dashboard:**
- Go to Variables tab
- Add all variables from `.env` file
- Click "Redeploy"

See [ENVIRONMENT_SETUP.md](ENVIRONMENT_SETUP.md) for complete list.

---

## Step 2: Deploy to Modal (2 minutes)

```bash
# 1. Install Modal CLI
pip install modal

# 2. Authenticate
modal token new

# 3. Create R2 secret
modal secret create modomo-r2-credentials \
  R2_ENDPOINT_URL=https://your-account-id.r2.cloudflarestorage.com \
  R2_ACCESS_KEY_ID=your_key \
  R2_SECRET_ACCESS_KEY=your_secret \
  R2_BUCKET_NAME=your-bucket \
  CDN_DOMAIN=your-cdn-domain

# 4. Deploy
modal deploy modal_functions/sd_inference_complete.py
```

---

## Step 3: Verify (1 minute)

```bash
# Test health endpoint
curl https://your-app.up.railway.app/health

# Expected:
{
  "status": "healthy",
  "services": {
    "redis": "connected",
    "modal": "connected",
    "storage": "connected"
  }
}
```

---

## ✅ Done!

Your API is live at:
```
https://your-app-name.up.railway.app
```

**Next steps:**
- Test transformation: `POST /transform/submit`
- Monitor costs on Modal dashboard
- Set up uptime monitoring

**Troubleshooting:** See [README.md](README.md#troubleshooting)

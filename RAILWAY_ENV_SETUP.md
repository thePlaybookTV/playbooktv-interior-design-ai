# Railway Environment Variables Setup

## 🚨 **IMPORTANT: Configure These in Railway Dashboard**

Railway deployment is failing because environment variables are not set. Follow these steps:

---

## 📋 **Step 1: Access Railway Dashboard**

1. Go to https://railway.app/dashboard
2. Select your `playbooktv-interior-design-ai` project
3. Click on your service
4. Go to the **Variables** tab

---

## 🔑 **Step 2: Add Environment Variables**

Copy the values from your `.env` or `solar.env` file and add them to Railway:

### **Redis Configuration**
```bash
REDIS_URL=redis://default:CiFsKXyXMUqdtPVvAiuwiFtYWAZtRchY@metro.proxy.rlwy.net:25118
```

### **Cloudflare R2 Storage**
```bash
R2_BUCKET_NAME=reroom
R2_ENDPOINT_URL=https://9bbdb3861142e65685d23f4955b88ebe.r2.cloudflarestorage.com
R2_ACCESS_KEY_ID=6c8abdff2cdad89323e36b258b1d0f4b
R2_SECRET_ACCESS_KEY=2a2bb806281b1b321803f91cbe8fbc4180536cd87cf745ad4fef368011c3a1d1
CDN_DOMAIN=9bbdb3861142e65685d23f4955b88ebe.r2.cloudflarestorage.com
CLOUDFLARE_ACCOUNT_ID=9bbdb3861142e65685d23f4955b88ebe
```

### **Modal GPU Configuration**
```bash
MODAL_TOKEN_ID=ak-mdhwVsEGW46OtIdFT7j0FH
MODAL_TOKEN_SECRET=as-lf9wqUIlMhi65hrgiWsd8q
MODAL_APP_NAME=modomo-sd-inference
MODAL_STUB_NAME=modomo-sd-inference
```

### **API Configuration**
```bash
API_HOST=0.0.0.0
API_PORT=8000
API_SECRET_KEY=change-this-to-a-secure-random-key-in-production
API_BASE_URL=https://playbooktv-interior-design-ai-production.up.railway.app
ENVIRONMENT=production
```

---

## 📝 **Step 3: Click "Add" for Each Variable**

Railway UI:
1. Enter **Variable Name** (e.g., `REDIS_URL`)
2. Enter **Value** (paste the value from above)
3. Click **Add**
4. Repeat for all variables

---

## 🔄 **Step 4: Redeploy**

After adding all variables:
1. Railway will automatically trigger a new deployment
2. Or manually click **Deploy** → **Redeploy**

---

## ✅ **Step 5: Verify Deployment**

Wait 2-3 minutes, then test:

```bash
# Test health endpoint
curl https://playbooktv-interior-design-ai-production.up.railway.app/health

# Expected response:
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

## 🔧 **Quick Copy-Paste Format for Railway**

If Railway has a bulk import feature, use this format:

```
REDIS_URL=redis://default:CiFsKXyXMUqdtPVvAiuwiFtYWAZtRchY@metro.proxy.rlwy.net:25118
R2_BUCKET_NAME=reroom
R2_ENDPOINT_URL=https://9bbdb3861142e65685d23f4955b88ebe.r2.cloudflarestorage.com
R2_ACCESS_KEY_ID=6c8abdff2cdad89323e36b258b1d0f4b
R2_SECRET_ACCESS_KEY=2a2bb806281b1b321803f91cbe8fbc4180536cd87cf745ad4fef368011c3a1d1
CDN_DOMAIN=9bbdb3861142e65685d23f4955b88ebe.r2.cloudflarestorage.com
CLOUDFLARE_ACCOUNT_ID=9bbdb3861142e65685d23f4955b88ebe
MODAL_TOKEN_ID=ak-mdhwVsEGW46OtIdFT7j0FH
MODAL_TOKEN_SECRET=as-lf9wqUIlMhi65hrgiWsd8q
MODAL_APP_NAME=modomo-sd-inference
MODAL_STUB_NAME=modomo-sd-inference
API_HOST=0.0.0.0
API_PORT=8000
API_SECRET_KEY=change-this-to-a-secure-random-key-in-production
API_BASE_URL=https://playbooktv-interior-design-ai-production.up.railway.app
ENVIRONMENT=production
```

---

## ⚠️ **Current Error**

Railway logs show:
```
SSL validation failed for https://your-account-id.r2.cloudflarestorage.com/your-bucket-name
```

This means Railway is using placeholder values. Once you set the real environment variables above, this error will disappear.

---

## 🎯 **After Configuration**

Once env vars are set, Railway will:
1. ✅ Connect to Redis for job queue
2. ✅ Connect to Cloudflare R2 for storage
3. ✅ Connect to Modal GPU for processing
4. ✅ Start accepting transformation requests

Your API will be fully operational!

# Paperspace Quick Start - Copy & Paste

**Problem**: `ModuleNotFoundError: No module named 'src'`
**Solution**: Updated code + easy startup script

---

## 🚀 Run This in Paperspace Terminal

Copy and paste this **entire block**:

```bash
# Navigate to project
cd /notebooks/app

# Pull latest code (with the fix)
git pull origin main

# Or if that fails, force update:
# git fetch origin
# git reset --hard origin/main

# Make startup script executable
chmod +x start_api.sh

# Start the API
./start_api.sh
```

That's it! The script will:
- ✅ Check all dependencies
- ✅ Install missing packages
- ✅ Create .env file if needed
- ✅ Check for database and models
- ✅ Start the server with correct paths

---

## 📝 Expected Output

You should see:

```
=========================================
PlaybookTV Interior Design AI - API
=========================================

📁 Project root: /notebooks/app

✅ Python found: Python 3.10.12

📦 Checking dependencies...
  ✅ fastapi
  ✅ uvicorn
  ✅ torch
  ✅ ultralytics
  ✅ pillow

✅ .env file found
⚠️  Database not found: database_metadata.duckdb
   Upload it to: /notebooks/app/database_metadata.duckdb

🤖 Checking for models...
  ⚠️  No models found
     API will start but model loading may fail

=========================================
🚀 Starting API Server
=========================================

Server will be available at:
  • Local: http://localhost:8000
  • Docs:  http://localhost:8000/docs

INFO:     Uvicorn running on http://0.0.0.0:8000
INFO:     Started server process
INFO:     Waiting for application startup.
INFO:     Application startup complete.
```

---

## 📤 Upload Missing Files

If you see warnings, upload these via Paperspace file manager:

**1. Database** (Required):
- File: `database_metadata.duckdb` (18 MB)
- Upload to: `/notebooks/app/`

**2. Phase 1 Model** (Optional - for quick start):
- File: `models_best_interior_model.pth` (130 MB)
- Upload to: `/notebooks/app/`

---

## 🌐 Get Public URL

In a **new Paperspace terminal** (while server is running):

```bash
# Install ngrok
pip install pyngrok

# Start tunnel
python << 'EOF'
from pyngrok import ngrok
import time

public_url = ngrok.connect(8000)
print("\n" + "="*50)
print("🌐 YOUR PUBLIC API URL:")
print(f"   {public_url}")
print("="*50)
print("\nUse this URL in your Modomo app!")
print("Example: https://abc123.ngrok.io/analyze")
print("\nPress Ctrl+C to stop tunnel")
print("="*50 + "\n")

try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("\nTunnel stopped")
EOF
```

---

## 🧪 Test the API

```bash
# Health check
curl http://localhost:8000/health

# Or from your browser:
# http://localhost:8000/docs
```

---

## ❓ Troubleshooting

### Still getting ModuleNotFoundError?

```bash
cd /notebooks/app

# Check Python path
python -c "import sys; print('\n'.join(sys.path))"

# Manually add to path and run
export PYTHONPATH=/notebooks/app:$PYTHONPATH
python api/main.py
```

### Missing dependencies?

```bash
pip install fastapi uvicorn python-multipart pillow torch torchvision ultralytics python-dotenv
```

### Database not found?

```bash
# Check if database is uploaded
ls -lh /notebooks/app/*.duckdb

# If not, upload via Paperspace file browser
```

### Can't access from outside?

Make sure ngrok tunnel is running in a separate terminal.

---

## ✅ Quick Checklist

- [ ] Code updated: `git pull`
- [ ] Database uploaded to `/notebooks/app/`
- [ ] API started: `./start_api.sh`
- [ ] API running at http://localhost:8000
- [ ] Ngrok tunnel created
- [ ] Got public URL
- [ ] Tested with curl or browser
- [ ] Updated Modomo app with URL

---

## 🎯 Next Steps

1. **Test locally**: Visit http://localhost:8000/docs
2. **Get public URL**: Run ngrok script above
3. **Connect Modomo**: Update API_URL in your app
4. **Test upload**: Send an image from Modomo
5. **See results**: Get detections and style!

---

**Need help?** The server logs will show any errors. Check them for debugging.

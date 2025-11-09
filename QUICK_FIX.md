# Quick Fix for Paperspace Errors

## Error: `ModuleNotFoundError: No module named 'duckdb'`

**Solution**: Install missing dependencies

---

## 🚀 Run This Now (Copy-Paste into Paperspace)

```bash
# 1. Stop the server if it's running (Ctrl+C)

# 2. Go to project directory
cd /notebooks/app

# 3. Pull latest fixes
git pull origin main

# 4. Install missing dependencies
pip install duckdb pandas tqdm

# OR install everything from requirements:
pip install -r api/requirements.txt

# 5. Start the server
./start_api.sh
```

---

## ✅ What Was Fixed

1. **Logger order** - Now defined before use
2. **Missing dependencies** - Added duckdb, pandas, tqdm
3. **Better error handling** - Shows helpful messages

---

## 📦 All Required Dependencies

```bash
pip install fastapi uvicorn python-multipart python-dotenv pillow \
    torch torchvision ultralytics numpy pydantic duckdb pandas tqdm
```

---

## 🧪 Test It Works

After installation:

```bash
# Quick test
python -c "import duckdb; import pandas; print('✅ All imports work!')"

# Start API
cd /notebooks/app
./start_api.sh
```

**You should see**:
```
✅ Model imports successful
INFO:     Uvicorn running on http://0.0.0.0:8000
INFO:     Application startup complete.
```

---

## ❓ Still Having Issues?

### Import Error on Different Module

```bash
# Install whatever is missing
pip install [missing-module-name]

# Or reinstall everything
pip install -r api/requirements.txt --force-reinstall
```

### Path Issues

```bash
# Make sure you're in the right directory
cd /notebooks/app
pwd  # Should show /notebooks/app

# Check Python can find src
python -c "import sys; sys.path.insert(0, '/notebooks/app'); import src.models.improved_style_classifier"
```

### Still can't start?

Run with debug mode:
```bash
cd /notebooks/app
python api/main.py
```

This will show you the exact error.

---

## 🎯 Complete Clean Install

If nothing works, do a complete clean install:

```bash
cd /notebooks/app

# Update code
git pull origin main

# Install Python packages
pip install --upgrade pip
pip install -r api/requirements.txt

# Verify installations
python << 'EOF'
import sys
required = ['fastapi', 'uvicorn', 'torch', 'ultralytics', 'duckdb', 'pandas', 'pillow']
missing = []

for pkg in required:
    try:
        __import__(pkg)
        print(f"✅ {pkg}")
    except ImportError:
        print(f"❌ {pkg}")
        missing.append(pkg)

if missing:
    print(f"\n❌ Missing: {', '.join(missing)}")
    print(f"Install with: pip install {' '.join(missing)}")
else:
    print("\n✅ All dependencies installed!")
EOF

# Start server
./start_api.sh
```

---

## 🆘 Last Resort

If you still get errors, try this minimal version:

```bash
cd /notebooks/app

# Start with minimal dependencies
pip install fastapi uvicorn duckdb

# Run directly (bypassing imports)
python << 'EOF'
from fastapi import FastAPI
import uvicorn

app = FastAPI()

@app.get("/")
def root():
    return {"status": "API is running!"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
EOF
```

If this works, then slowly add back the full API.

---

**Your error is now fixed!** Just pull the latest code and install duckdb. 🚀

# Paperspace Quick Setup Guide

## The CUDA Version Issue

Paperspace Gradient base images use **CUDA 12.1/12.2** runtime, but the default `requirements.txt` installs PyTorch 2.5.1 with **CUDA 12.4** libraries. This causes the import error:

```
ImportError: undefined symbol: __nvJitLinkComplete_12_4
```

## Solution: Use Paperspace-Specific Requirements

### Option 1: Automated (Recommended)

The `start_training.sh` script now auto-detects Paperspace and uses the correct packages:

```bash
cd /notebooks/app
git pull origin Update-ML  # Get latest changes
./start_training.sh
```

### Option 2: Manual Install

If you need to set up manually:

```bash
# 1. Remove incompatible PyTorch
pip uninstall -y torch torchvision torchaudio nvidia-cusparse-cu12 nvidia-cuda-runtime-cu12 nvidia-cudnn-cu12

# 2. Install CUDA 12.1 compatible versions
pip install --index-url https://download.pytorch.org/whl/cu121 -r requirements.paperspace.txt

# 3. Verify
python3 -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.version.cuda); print('Available:', torch.cuda.is_available())"
```

Expected output:
```
PyTorch: 2.4.1+cu121
CUDA: 12.1
Available: True
```

## What's Different in requirements.paperspace.txt?

| Package | requirements.txt | requirements.paperspace.txt | Why |
|---------|-----------------|----------------------------|-----|
| torch | 2.5.1 (CUDA 12.4) | 2.4.1 (CUDA 12.1) | Matches Paperspace runtime |
| torchvision | 0.20.1 | 0.19.1 | Compatible with torch 2.4.1 |
| numpy | 1.24.3 | <2 (flexible) | Avoids NumPy 2.x conflicts |

## Complete Setup Flow

### 1. Clone/Update Repository
```bash
cd /notebooks
git clone https://github.com/YOUR_REPO/playbooktv-interior-design-ai.git app
# OR if already cloned:
cd /notebooks/app
git checkout Update-ML
git pull origin Update-ML
```

### 2. Install Dependencies
```bash
# Automatic (detects Paperspace)
./start_training.sh

# OR Manual
pip install --index-url https://download.pytorch.org/whl/cu121 -r requirements.paperspace.txt
```

### 3. Upload Database
Upload `metadata.duckdb` to:
```
/notebooks/app/interior_design_data_hybrid/processed/metadata.duckdb
```

### 4. Start Training
```bash
./start_training.sh
```

## Troubleshooting

### Issue: Still Getting CUDA Import Error

**Cause**: Old torch installation not fully removed

**Fix**:
```bash
# Nuclear option - remove all torch packages
pip freeze | grep -i torch | xargs pip uninstall -y
pip freeze | grep -i nvidia | xargs pip uninstall -y
pip cache purge

# Fresh install
pip install --index-url https://download.pytorch.org/whl/cu121 torch==2.4.1 torchvision==0.19.1

# Verify
python3 -c "import torch; print(torch.__version__, torch.version.cuda)"
```

### Issue: Machine Runs Out of Memory During Install

**Cause**: Installing too many packages at once

**Fix**: Install in smaller batches
```bash
pip install --index-url https://download.pytorch.org/whl/cu121 torch==2.4.1 torchvision==0.19.1
sleep 5
pip install "numpy<2" ultralytics==8.0.0
sleep 5
pip install pillow duckdb pandas tqdm python-dotenv
```

### Issue: GPU Not Detected (`nvidia-smi` fails)

**Cause**: GPU driver not loaded

**Fix**: Restart Paperspace machine
- Stop the machine
- Start it again
- Re-run setup

## Why Not Just Update requirements.txt?

We keep two separate files because:
1. **Modal deployment** uses the base `requirements.txt` with latest PyTorch
2. **Paperspace training** uses `requirements.paperspace.txt` with CUDA 12.1 compatible versions
3. **Local development** may need different versions

The training script auto-detects the environment and uses the right one.

## Verifying Everything Works

After setup, verify all components:

```bash
# Check PyTorch
python3 << 'EOF'
import torch
print(f"✅ PyTorch {torch.__version__}")
print(f"✅ CUDA {torch.version.cuda}")
print(f"✅ GPU available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
EOF

# Check other packages
python3 -c "from ultralytics import YOLO; import duckdb; print('✅ All packages working')"

# Check database
ls -lh interior_design_data_hybrid/processed/metadata.duckdb
```

Expected output:
```
✅ PyTorch 2.4.1+cu121
✅ CUDA 12.1
✅ GPU available: True
✅ GPU: NVIDIA A4000
✅ All packages working
-rw-r--r-- 1 root root 1.8M metadata.duckdb
```

## Ready to Train!

Once everything checks out:
```bash
./start_training.sh
```

Choose your training mode and let it run for 10-16 hours. The script handles everything automatically!

#!/usr/bin/env python3
"""
Quick GPU and memory check
"""

import torch
import subprocess
import sys

print("="*60)
print("🔍 SYSTEM CHECK")
print("="*60)

# Check PyTorch
print(f"\n1️⃣ PyTorch Version: {torch.__version__}")

# Check CUDA
print(f"\n2️⃣ CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   CUDA Version: {torch.version.cuda}")
    print(f"   Device Count: {torch.cuda.device_count()}")
    print(f"   Current Device: {torch.cuda.current_device()}")
    print(f"   Device Name: {torch.cuda.get_device_name(0)}")

    # Check memory
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    reserved = torch.cuda.memory_reserved(0) / 1024**3
    allocated = torch.cuda.memory_allocated(0) / 1024**3
    free = total - reserved

    print(f"\n3️⃣ GPU Memory:")
    print(f"   Total: {total:.2f} GB")
    print(f"   Reserved: {reserved:.2f} GB")
    print(f"   Allocated: {allocated:.2f} GB")
    print(f"   Free: {free:.2f} GB")
else:
    print("   ❌ No CUDA available - will use CPU (slow!)")

# Check nvidia-smi
print(f"\n4️⃣ nvidia-smi Output:")
try:
    result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
    print(result.stdout)
except FileNotFoundError:
    print("   ❌ nvidia-smi not found")

# Check RAM
try:
    import psutil
    ram = psutil.virtual_memory()
    print(f"\n5️⃣ System RAM:")
    print(f"   Total: {ram.total / 1024**3:.2f} GB")
    print(f"   Available: {ram.available / 1024**3:.2f} GB")
    print(f"   Used: {ram.used / 1024**3:.2f} GB")
    print(f"   Percent: {ram.percent}%")
except ImportError:
    print("\n5️⃣ System RAM: (install psutil to see)")

print("\n" + "="*60)
print("✅ CHECK COMPLETE")
print("="*60)

# Recommendations
if not torch.cuda.is_available():
    print("\n⚠️  WARNING: No GPU detected!")
    print("   - Check if you're on a GPU instance")
    print("   - Try: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")
elif torch.cuda.is_available():
    print("\n✅ GPU detected - should be using GPU for training")

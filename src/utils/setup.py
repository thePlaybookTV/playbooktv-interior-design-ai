"""Setup and dependency installation utilities"""

import os
import sys


def install_dependencies():
    """Install YOLO and SAM2 dependencies"""
    print("📦 Installing dependencies...\n")
    
    # Install ultralytics (YOLO)
    try:
        import ultralytics
        print("✅ Ultralytics already installed")
    except ImportError:
        print("Installing ultralytics (YOLO)...")
        os.system(f"{sys.executable} -m pip install ultralytics --quiet")
        print("✅ Ultralytics installed")
    
    # Install SAM2
    try:
        import sam2
        print("✅ SAM2 already installed")
    except ImportError:
        print("Installing SAM2...")
        os.system(f"{sys.executable} -m pip install git+https://github.com/facebookresearch/segment-anything-2.git --quiet")
        print("✅ SAM2 installed")
    
    print("\n✅ All dependencies installed!")

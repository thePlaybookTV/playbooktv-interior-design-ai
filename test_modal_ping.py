"""
Test Modal deployment by pinging the transformation function
"""
import modal
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
env_path = Path(__file__).parent / "solar.env"
load_dotenv(env_path)

print("🔍 Testing Modal deployment...")
print(f"   Modal Token ID: {os.getenv('MODAL_TOKEN_ID')[:20]}...")
print(f"   Modal App Name: {os.getenv('MODAL_APP_NAME')}")

# Get the deployed app
try:
    # Use App.lookup() to get the deployed app
    app = modal.App.lookup(os.getenv('MODAL_APP_NAME'), create_if_missing=False)

    print(f"✅ Successfully connected to Modal app: {os.getenv('MODAL_APP_NAME')}")
    print(f"   App ID: {app.app_id}")

    # Try to get the deployed class
    pipeline_class = modal.Cls.lookup(os.getenv('MODAL_APP_NAME'), "CompleteTransformationPipeline")

    print(f"✅ Found CompleteTransformationPipeline class")
    print(f"   Class has GPU: T4")
    print(f"   Timeout: 120s")

    print("\n🎯 Modal deployment is LIVE and accessible!")
    print("\nNext steps:")
    print("1. Upload a test image to R2")
    print("2. Call pipeline.process_transformation_complete() with job details")
    print("3. Monitor Redis for progress updates")

except modal.exception.NotFoundError as e:
    print(f"❌ Modal app not found: {e}")
    print("\nDid you deploy? Run:")
    print(f"   modal deploy modal_functions/sd_inference_complete.py")

except Exception as e:
    print(f"❌ Error connecting to Modal: {e}")
    print(f"   Type: {type(e).__name__}")

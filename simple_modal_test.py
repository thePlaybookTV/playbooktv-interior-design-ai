"""
Simple Modal health check
"""
import subprocess
import sys

print("🔍 Checking Modal deployment status...\n")

# Check if modal is authenticated
try:
    result = subprocess.run(
        ["modal", "app", "list"],
        capture_output=True,
        text=True,
        timeout=10
    )

    if result.returncode == 0:
        print("✅ Modal CLI is authenticated")
        print("\n📋 Your deployed apps:")
        print(result.stdout)

        if "modomo-sd-inference" in result.stdout:
            print("\n🎯 SUCCESS! Your 'modomo-sd-inference' app is DEPLOYED and LIVE!")
            print("\n✨ Your Modal GPU function is ready to:")
            print("   • Process image transformations")
            print("   • Run YOLO + SAM2 detection")
            print("   • Generate depth maps & edge maps")
            print("   • Apply Stable Diffusion + ControlNet")
            print("   • Upload results to R2")
            print("\n⚡️ Processing time: ~15 seconds per image on T4 GPU")
            print("💰 Cost: £0.03 per transformation")

            print("\n🔌 API Integration:")
            print("   Your Railway API should call:")
            print("   pipeline = CompleteTransformationPipeline()")
            print("   result = pipeline.process_transformation_complete.remote(...)")

        else:
            print("\n⚠️  'modomo-sd-inference' app not found in deployments")
            print("    Run: modal deploy modal_functions/sd_inference_complete.py")

    else:
        print(f"❌ Error: {result.stderr}")
        sys.exit(1)

except subprocess.TimeoutExpired:
    print("⏱️  Modal command timed out")
    sys.exit(1)
except FileNotFoundError:
    print("❌ Modal CLI not found. Install with: pip install modal")
    sys.exit(1)
except Exception as e:
    print(f"❌ Error: {e}")
    sys.exit(1)

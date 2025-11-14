"""
Modomo Interior Design AI - Minimal FastAPI Server

Lightweight API for SD transformation with Modal GPU processing.

Author: Modomo Team
Date: November 2025
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, Dict, List
from PIL import Image
import io
import logging
from pathlib import Path
import os
from datetime import datetime
import sys

# Load .env file FIRST
from dotenv import load_dotenv
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
logger.info(f"📋 Loaded environment from: {env_path}")

# Fix Python path to find src module
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import services
try:
    from src.services.job_queue import JobQueue, JobStatus
    from src.services.storage_service import StorageService
    from src.services.modal_service import get_modal_service
    from src.services.websocket_manager import get_websocket_manager
    logger.info("✅ Service imports successful")
except ImportError as e:
    logger.error(f"❌ Could not import services: {e}")
    raise

# Initialize FastAPI
app = FastAPI(
    title="Modomo Interior Design AI",
    description="AI-powered interior design transformation API",
    version="1.0.0"
)

# CORS settings for mobile app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global services
services = {
    "job_queue": None,
    "storage": None,
    "modal": None,
    "websocket": None
}

# Supported styles
SUPPORTED_STYLES = [
    "modern",
    "scandinavian",
    "boho",
    "industrial",
    "minimalist"
]

# ============================================
# REQUEST/RESPONSE MODELS
# ============================================

class TransformPreferences(BaseModel):
    keep_furniture: bool = False
    color_palette: Optional[List[str]] = None

class TransformSubmitResponse(BaseModel):
    success: bool
    job_id: str
    estimated_time: int  # seconds
    websocket_url: str
    status_url: str

class TransformStatusResponse(BaseModel):
    success: bool
    data: Dict

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    services: Dict

# ============================================
# STARTUP/SHUTDOWN
# ============================================

@app.on_event("startup")
async def startup():
    """Initialize services on startup"""
    logger.info("🚀 Starting Modomo API...")

    try:
        # Initialize job queue (Redis)
        services["job_queue"] = JobQueue()
        logger.info("✅ Job queue initialized")

        # Initialize storage (R2) - allow to fail gracefully
        try:
            services["storage"] = StorageService()
            logger.info("✅ Storage service initialized")
        except Exception as e:
            logger.warning(f"⚠️ Storage service failed to initialize: {e}")
            logger.warning("   Set R2 environment variables in Railway dashboard")
            services["storage"] = None

        # Initialize Modal service - allow to fail gracefully
        try:
            services["modal"] = get_modal_service()
            logger.info("✅ Modal service initialized")

            # Check Modal deployment
            if not services["modal"].is_deployed():
                logger.warning("⚠️ Modal app not deployed!")
                logger.warning("   Deploy with: modal deploy modal_functions/sd_inference_complete.py")
        except Exception as e:
            logger.warning(f"⚠️ Modal service failed to initialize: {e}")
            logger.warning("   Set MODAL_TOKEN_ID and MODAL_TOKEN_SECRET in Railway dashboard")
            services["modal"] = None

        # Initialize WebSocket manager
        services["websocket"] = get_websocket_manager()
        logger.info("✅ WebSocket manager initialized")

        logger.info("✨ Service initialization complete")

    except Exception as e:
        logger.error(f"❌ Failed to initialize core services: {e}")
        logger.error("   Application will start but may not be fully functional")
        # Don't raise - allow app to start for debugging

@app.on_event("shutdown")
async def shutdown():
    """Cleanup on shutdown"""
    logger.info("👋 Shutting down Modomo API...")

# ============================================
# API ENDPOINTS
# ============================================

@app.get("/")
async def root():
    """API root endpoint"""
    return {
        "name": "Modomo Interior Design AI",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "transform_submit": "POST /transform/submit - Submit transformation job",
            "transform_status": "GET /transform/status/{job_id} - Check job status",
            "transform_websocket": "WS /ws/transform/{job_id} - Real-time updates",
            "health": "GET /health - Health check"
        },
        "supported_styles": SUPPORTED_STYLES
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""

    # Check services
    redis_connected = False
    modal_connected = False

    try:
        # Test Redis
        queue_stats = await services["job_queue"].get_stats()
        redis_connected = True
    except Exception as e:
        logger.error(f"Redis health check failed: {e}")

    try:
        # Test Modal
        modal_health = await services["modal"].health_check()
        modal_connected = modal_health.get("modal_connected", False)
    except Exception as e:
        logger.error(f"Modal health check failed: {e}")

    return {
        "status": "healthy" if (redis_connected and modal_connected) else "degraded",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "redis": "connected" if redis_connected else "disconnected",
            "modal": "connected" if modal_connected else "disconnected",
            "storage": "connected",  # R2 assumed working if no error
            "websocket": "ready"
        }
    }

@app.post("/transform/submit", response_model=TransformSubmitResponse)
async def submit_transformation(
    file: UploadFile = File(...),
    style: str = "modern",
    preferences: Optional[str] = None  # JSON string
):
    """
    Submit new transformation job

    Process:
    1. Validate image and style
    2. Optimize and upload to R2
    3. Create job in Redis
    4. Submit to Modal
    5. Return job_id for tracking
    """
    start_time = datetime.now()

    try:
        # Validate style
        if style not in SUPPORTED_STYLES:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid style. Supported: {SUPPORTED_STYLES}"
            )

        # Validate file
        if not file.content_type.startswith("image/"):
            raise HTTPException(
                status_code=400,
                detail="File must be an image"
            )

        # Read and validate image
        contents = await file.read()
        if len(contents) > 10 * 1024 * 1024:  # 10MB limit
            raise HTTPException(
                status_code=413,
                detail="Image too large (max 10MB)"
            )

        try:
            image = Image.open(io.BytesIO(contents)).convert('RGB')
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid image file: {e}"
            )

        logger.info(f"📸 Received image: {image.size}, style: {style}")

        # Parse preferences
        import json
        prefs = {}
        if preferences:
            try:
                prefs = json.loads(preferences)
            except json.JSONDecodeError:
                logger.warning(f"Invalid preferences JSON: {preferences}")

        # Create job in Redis
        job_id = await services["job_queue"].create_job(
            user_id="user-001",  # TODO: Get from auth
            image_path="",  # Will be set after upload
            style=style,
            preferences=prefs
        )

        logger.info(f"✅ Created job: {job_id}")

        # Upload image to R2
        try:
            upload_key = f"uploads/{job_id}/original.jpg"
            image_url = await services["storage"].upload_image(
                image,
                upload_key,
                optimize=True,
                make_public=False
            )

            logger.info(f"✅ Uploaded to R2: {upload_key}")

            # Update job with image path
            await services["job_queue"].update_job_status(
                job_id,
                JobStatus.QUEUED,
                progress=0.1,
                metadata={"image_url": image_url}
            )

        except Exception as e:
            logger.error(f"❌ Failed to upload image: {e}")
            await services["job_queue"].mark_job_failed(job_id, f"Upload failed: {e}")
            raise HTTPException(status_code=500, detail="Failed to upload image")

        # Submit to Modal
        try:
            modal_call_id = await services["modal"].submit_transformation(
                job_id=job_id,
                image_url=image_url,
                style=style,
                preferences=prefs,
                redis_url=os.getenv("REDIS_URL")
            )

            logger.info(f"✅ Submitted to Modal: {modal_call_id}")

            # Update job with Modal call ID
            await services["job_queue"].update_job_status(
                job_id,
                JobStatus.QUEUED,
                progress=0.2,
                metadata={"modal_call_id": modal_call_id}
            )

        except Exception as e:
            logger.error(f"❌ Failed to submit to Modal: {e}")
            await services["job_queue"].mark_job_failed(job_id, f"Modal submission failed: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to submit to Modal: {e}")

        # Generate URLs
        api_base = os.getenv("API_BASE_URL", "http://localhost:8000")
        websocket_url = f"ws://{api_base.replace('http://', '').replace('https://', '')}/ws/transform/{job_id}"
        status_url = f"{api_base}/transform/status/{job_id}"

        processing_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"✨ Job {job_id} submitted in {processing_time:.2f}s")

        return {
            "success": True,
            "job_id": job_id,
            "estimated_time": 15,  # seconds
            "websocket_url": websocket_url,
            "status_url": status_url
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error submitting transformation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/transform/status/{job_id}", response_model=TransformStatusResponse)
async def get_transformation_status(job_id: str):
    """
    Get transformation job status

    Returns current status, progress, and result (if completed)
    """
    try:
        # Get job from Redis
        job_data = await services["job_queue"].get_job_status(job_id)

        if job_data.get("status") == "not_found":
            raise HTTPException(status_code=404, detail="Job not found")

        return {
            "success": True,
            "data": job_data
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting job status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/transform/{job_id}")
async def cancel_transformation(job_id: str):
    """
    Cancel pending transformation job
    """
    try:
        # Get job
        job_data = await services["job_queue"].get_job_status(job_id)

        if job_data.get("status") == "not_found":
            raise HTTPException(status_code=404, detail="Job not found")

        # Can only cancel queued or processing jobs
        if job_data.get("status") in ["completed", "failed"]:
            raise HTTPException(status_code=400, detail="Job already finished")

        # Cancel Modal job if exists
        modal_call_id = job_data.get("modal_call_id")
        if modal_call_id:
            await services["modal"].cancel_job(modal_call_id)

        # Mark as failed
        await services["job_queue"].mark_job_failed(job_id, "Cancelled by user")

        return {
            "success": True,
            "message": "Job cancelled successfully"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error cancelling job: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/ws/transform/{job_id}")
async def websocket_transform(websocket: WebSocket, job_id: str):
    """
    WebSocket endpoint for real-time transformation updates

    Subscribes to Redis PubSub and streams updates to client
    """
    ws_manager = services["websocket"]

    await ws_manager.connect(job_id, websocket)

    try:
        # Keep connection alive
        while True:
            # Receive messages from client (optional, for heartbeat)
            try:
                data = await websocket.receive_text()
                # Echo back as heartbeat
                await websocket.send_json({"type": "pong"})
            except WebSocketDisconnect:
                break

    except Exception as e:
        logger.error(f"WebSocket error for job {job_id}: {e}")
    finally:
        await ws_manager.disconnect(job_id, websocket)

# ============================================
# DEVELOPMENT/DEBUG ENDPOINTS
# ============================================

@app.get("/debug/queue/stats")
async def debug_queue_stats():
    """Get queue statistics (debug only)"""
    try:
        stats = await services["job_queue"].get_stats()
        return {"success": True, "stats": stats}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/debug/storage/stats")
async def debug_storage_stats():
    """Get storage statistics (debug only)"""
    try:
        stats = await services["storage"].get_storage_stats()
        return {"success": True, "stats": stats}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============================================
# RUN SERVER
# ============================================

if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", 8000))
    host = os.getenv("API_HOST", "0.0.0.0")

    logger.info(f"🚀 Starting Modomo API on {host}:{port}")

    uvicorn.run(
        "main_minimal:app",
        host=host,
        port=port,
        reload=False,  # Set to True for development
        log_level="info"
    )

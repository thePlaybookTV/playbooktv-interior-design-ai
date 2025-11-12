# Modomo: SD 1.5 + ControlNet Integration Architecture

**Version:** 1.0
**Date:** November 2025
**Status:** Implementation Blueprint
**Author:** Technical Architecture Team

---

## 📋 Executive Summary

This document outlines the complete architecture for integrating Stable Diffusion 1.5 with ControlNet into the Modomo interior design AI platform. The system enables users to transform room photos into styled designs using advanced AI, with real-time progress updates delivered to a React Native mobile app.

**Key Design Decisions:**
- **AI Model**: SD 1.5 + ControlNet (cost-optimized at £0.03-0.05/image)
- **Infrastructure**: Modal for serverless GPU compute
- **Processing**: Async job queue with Redis
- **Mobile**: React Native with WebSocket progress updates
- **Storage**: S3/Cloudflare R2 for image CDN

**Target Performance:**
- Processing Time: <15 seconds average
- Cost: <£0.05 per transformation
- Concurrent Users: 100+ simultaneous transformations
- Quality: >85% user satisfaction

---

## 🏗️ System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         MOBILE APP (React Native)                    │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │
│  │   Camera     │→ │    Style     │→ │  Processing  │→ Results     │
│  │   Capture    │  │  Selection   │  │   Progress   │              │
│  └──────────────┘  └──────────────┘  └──────────────┘              │
│         ↓                                     ↑                       │
│    HTTPS Upload                        WebSocket Updates            │
└─────────────────────────────────────────────────────────────────────┘
                              ↓                  ↑
┌─────────────────────────────────────────────────────────────────────┐
│                      FASTAPI BACKEND (api/main.py)                   │
│                                                                       │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │  POST /transform/submit                                       │  │
│  │    • Receive image + style                                   │  │
│  │    • Run existing detection (YOLO + SAM2)                    │  │
│  │    • Generate control images (depth, edges, segmentation)    │  │
│  │    • Create job in Redis queue                               │  │
│  │    • Return job_id                                           │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                                                       │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │  GET /transform/status/{job_id}                              │  │
│  │    • Check Redis for job status                              │  │
│  │    • Return progress, result URL, or error                   │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                                                       │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │  WebSocket /ws/transform/{job_id}                            │  │
│  │    • Real-time progress streaming                            │  │
│  │    • Status: analyzing → generating → complete               │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                              ↓                                        │
│                    ┌──────────────────┐                              │
│                    │  Redis Job Queue │                              │
│                    │  • Job tracking  │                              │
│                    │  • Status updates│                              │
│                    │  • Result cache  │                              │
│                    └──────────────────┘                              │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│                    MODAL (Serverless GPU Compute)                    │
│                                                                       │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │  modal_functions/sd_inference.py                             │  │
│  │                                                               │  │
│  │  @stub.function(gpu="T4", timeout=60)                        │  │
│  │  def transform_room(control_images, style, room_type):       │  │
│  │      • Load SD 1.5 pipeline                                  │  │
│  │      • Load ControlNet models (depth, canny, segmentation)   │  │
│  │      • Apply style-specific prompts                          │  │
│  │      • Generate transformation (20 steps)                    │  │
│  │      • Quality validation (>70% confidence)                  │  │
│  │      • Return image + metadata                               │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                                                       │
│  Auto-scaling: 0 → 10 containers based on queue length              │
│  GPU: NVIDIA T4 (16GB VRAM, £0.30/hour)                             │
│  Cold start: ~10s, Warm inference: ~12s                             │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│                   S3 / CLOUDFLARE R2 STORAGE                         │
│                                                                       │
│  /uploads/{job_id}/original.jpg       - User uploaded image         │
│  /uploads/{job_id}/depth.jpg          - Depth control map           │
│  /uploads/{job_id}/edges.jpg          - Canny edge map              │
│  /uploads/{job_id}/segmentation.jpg   - SAM2 segmentation           │
│  /results/{job_id}/transformed.jpg    - SD generated result         │
│  /results/{job_id}/thumbnail.jpg      - Mobile-optimized preview    │
│                                                                       │
│  CDN: CloudFront for fast global delivery                           │
│  Expiration: 30 days for uploads, permanent for results             │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 🔧 Backend Infrastructure Design

### 1. Modal SD Service

**File**: `src/services/modal_sd_service.py`

**Purpose**: Interface between FastAPI backend and Modal GPU compute

```python
class ModalSDService:
    """
    Manages communication with Modal serverless SD inference
    """
    def __init__(self):
        self.modal_client = modal.Client()
        self.stub = modal.Stub.lookup("modomo-sd-inference")

    async def submit_transformation(
        self,
        job_id: str,
        control_images: dict,  # depth, edges, segmentation
        style: str,
        room_type: str,
        room_analysis: dict
    ) -> str:
        """
        Submit transformation job to Modal

        Returns:
            modal_call_id: Unique ID for tracking Modal execution
        """
        pass

    async def check_modal_status(self, modal_call_id: str) -> dict:
        """
        Check status of Modal function execution

        Returns:
            {
                'status': 'running' | 'completed' | 'failed',
                'progress': 0.0-1.0,
                'result': <image_data> if completed
            }
        """
        pass
```

**Key Features:**
- Async communication with Modal
- Automatic retry on transient failures
- Progress estimation based on SD steps
- Cost tracking per transformation

**Cost Optimization:**
- Use T4 GPUs (£0.30/hour) instead of A100s (£1.50/hour)
- 20 inference steps (good quality, fast)
- Warm container reuse (avoid cold starts)
- Batch processing when possible

---

### 2. Control Image Generation

**File**: `src/services/control_generators.py`

**Purpose**: Generate control images for ControlNet conditioning

```python
class ControlImageGenerator:
    """
    Generates depth, edge, and segmentation control images
    """
    def __init__(self):
        # Depth estimation model
        self.depth_model = pipeline(
            "depth-estimation",
            model="Intel/dpt-large"
        )
        # Reuse existing SAM2 segmentation from detection

    async def generate_all_controls(
        self,
        image_path: str,
        segmentation_masks: List[np.ndarray]  # From existing SAM2
    ) -> dict:
        """
        Generate all control images needed for SD

        Returns:
            {
                'depth': PIL.Image,      # Depth map
                'edges': PIL.Image,      # Canny edges
                'segmentation': PIL.Image # Processed SAM2 masks
            }
        """
        pass

    async def generate_depth_map(self, image: Image) -> Image:
        """
        Generate depth map using DPT-Large

        Processing:
            1. Run depth estimation
            2. Normalize to 0-255
            3. Apply colormap for visualization
            4. Resize to target resolution (512x512)
        """
        pass

    async def generate_canny_edges(
        self,
        image: Image,
        low_threshold: int = 100,
        high_threshold: int = 200
    ) -> Image:
        """
        Generate Canny edge map

        Optimized for interior scenes:
            - Higher thresholds to avoid noise
            - Focus on architectural lines
            - Preserve furniture boundaries
        """
        pass

    async def process_segmentation_map(
        self,
        masks: List[np.ndarray],
        image_size: tuple
    ) -> Image:
        """
        Convert SAM2 masks to ControlNet-compatible format

        Processing:
            1. Combine all furniture masks
            2. Create semantic segmentation map
            3. Color-code by furniture category
            4. Resize and format for ControlNet
        """
        pass
```

**Control Image Specifications:**
- Resolution: 512x512 (SD 1.5 native)
- Format: RGB PNG
- Depth: Normalized grayscale with colormap
- Edges: Binary (0/255) black and white
- Segmentation: Multi-class color-coded

**Performance:**
- Depth generation: ~2 seconds (DPT-Large on CPU)
- Edge detection: <0.5 seconds (OpenCV)
- Segmentation processing: <0.5 seconds
- Total control generation: ~3 seconds

---

### 3. Redis Job Queue System

**File**: `src/services/job_queue.py`

**Purpose**: Track transformation jobs and enable async processing

```python
class JobQueue:
    """
    Redis-based job queue for async transformation processing
    """
    def __init__(self, redis_url: str):
        self.redis = redis.from_url(redis_url)

    async def create_job(
        self,
        user_id: str,
        image_path: str,
        style: str,
        preferences: dict
    ) -> str:
        """
        Create new transformation job

        Returns:
            job_id: UUID for tracking
        """
        job_id = str(uuid.uuid4())
        job_data = {
            'job_id': job_id,
            'user_id': user_id,
            'status': 'queued',
            'created_at': datetime.utcnow().isoformat(),
            'image_path': image_path,
            'style': style,
            'preferences': json.dumps(preferences),
            'progress': 0.0,
            'estimated_time': 15.0  # seconds
        }

        # Store in Redis with 1-hour TTL
        self.redis.setex(
            f"job:{job_id}",
            3600,
            json.dumps(job_data)
        )

        # Add to processing queue
        self.redis.lpush('job_queue', job_id)

        return job_id

    async def update_job_status(
        self,
        job_id: str,
        status: str,
        progress: float = None,
        result_url: str = None,
        error: str = None
    ):
        """
        Update job status and notify WebSocket listeners
        """
        job_data = json.loads(self.redis.get(f"job:{job_id}"))
        job_data['status'] = status
        if progress is not None:
            job_data['progress'] = progress
        if result_url:
            job_data['result_url'] = result_url
        if error:
            job_data['error'] = error

        self.redis.setex(f"job:{job_id}", 3600, json.dumps(job_data))

        # Publish update for WebSocket subscribers
        self.redis.publish(
            f"job_updates:{job_id}",
            json.dumps({
                'status': status,
                'progress': progress,
                'result_url': result_url
            })
        )

    async def get_job_status(self, job_id: str) -> dict:
        """
        Retrieve current job status
        """
        job_data = self.redis.get(f"job:{job_id}")
        if not job_data:
            return {'status': 'not_found'}
        return json.loads(job_data)
```

**Redis Schema:**

```
Keys:
  job:{job_id}           - Job metadata (JSON, 1-hour TTL)
  job_queue              - List of pending job IDs
  job_updates:{job_id}   - PubSub channel for real-time updates

Job Statuses:
  - queued       : Waiting in queue
  - analyzing    : Running room detection
  - generating   : Modal SD processing
  - completed    : Transformation complete
  - failed       : Error occurred
```

**Scaling Considerations:**
- Redis Cluster for >10K concurrent jobs
- Separate queues for free/premium users (priority)
- Job TTL to prevent memory bloat
- Automatic cleanup of expired jobs

---

### 4. Storage Service

**File**: `src/services/storage_service.py`

**Purpose**: Manage image uploads and CDN delivery

```python
class StorageService:
    """
    S3/Cloudflare R2 storage with CDN integration
    """
    def __init__(self, bucket_name: str, cdn_domain: str):
        self.s3_client = boto3.client('s3')
        self.bucket = bucket_name
        self.cdn_domain = cdn_domain

    async def upload_image(
        self,
        image: Image,
        key: str,
        optimize: bool = True
    ) -> str:
        """
        Upload image to S3/R2 with optional optimization

        Returns:
            cdn_url: Public CDN URL
        """
        if optimize:
            image = self.optimize_for_mobile(image)

        buffer = io.BytesIO()
        image.save(buffer, format='JPEG', quality=85, optimize=True)
        buffer.seek(0)

        self.s3_client.upload_fileobj(
            buffer,
            self.bucket,
            key,
            ExtraArgs={
                'ContentType': 'image/jpeg',
                'CacheControl': 'public, max-age=31536000'
            }
        )

        return f"https://{self.cdn_domain}/{key}"

    def optimize_for_mobile(self, image: Image) -> Image:
        """
        Optimize image for mobile delivery

        - Resize to max 1080p
        - Compress to 85% quality
        - Strip metadata
        """
        max_size = (1920, 1080)
        image.thumbnail(max_size, Image.LANCZOS)
        return image

    async def generate_thumbnail(self, image: Image) -> Image:
        """
        Generate thumbnail for quick preview

        - 480x360 resolution
        - 70% quality
        """
        thumbnail = image.copy()
        thumbnail.thumbnail((480, 360), Image.LANCZOS)
        return thumbnail

    async def get_signed_url(self, key: str, expires_in: int = 3600) -> str:
        """
        Generate signed URL for private access

        Used for:
            - Original uploads (private)
            - In-progress transformations
        """
        return self.s3_client.generate_presigned_url(
            'get_object',
            Params={'Bucket': self.bucket, 'Key': key},
            ExpiresIn=expires_in
        )
```

**Storage Structure:**

```
s3://modomo-images/
├── uploads/
│   └── {job_id}/
│       ├── original.jpg          (user upload)
│       ├── depth.jpg              (control image)
│       ├── edges.jpg              (control image)
│       └── segmentation.jpg       (control image)
├── results/
│   └── {job_id}/
│       ├── transformed.jpg        (SD output, 1920x1080)
│       └── thumbnail.jpg          (preview, 480x360)
└── temp/
    └── {job_id}/                  (cleanup after 24h)
```

**CDN Configuration:**
- Cloudflare R2 with CDN (zero egress fees)
- Cache-Control: 1 year for results, 1 hour for uploads
- WebP format for browsers that support it
- Lazy loading with progressive JPEGs

---

## 🌐 API Specification

### Endpoint 1: Submit Transformation

**POST** `/transform/submit`

**Purpose**: Upload image and request style transformation

**Request:**
```typescript
Content-Type: multipart/form-data

{
  file: File,           // JPEG/PNG, max 10MB
  style: string,        // "modern" | "scandinavian" | "boho" | "industrial" | "minimalist"
  preferences?: {       // Optional
    color_palette?: string[],
    budget_range?: { min: number, max: number },
    keep_existing_furniture?: boolean
  }
}
```

**Response:**
```typescript
{
  success: true,
  data: {
    job_id: string,              // UUID for tracking
    estimated_time: number,      // seconds
    websocket_url: string,       // For real-time updates
    status_url: string           // Polling endpoint
  }
}
```

**Processing Flow:**
1. Validate image (format, size, quality)
2. Optimize and upload to S3
3. Run existing detection pipeline (YOLO + SAM2)
4. Generate control images (depth, edges, segmentation)
5. Create job in Redis queue
6. Return job_id immediately
7. Process in background

**Error Responses:**
- 400: Invalid image format or size
- 413: File too large (>10MB)
- 429: Rate limit exceeded
- 500: Internal server error

---

### Endpoint 2: Check Job Status

**GET** `/transform/status/{job_id}`

**Purpose**: Poll for job status and results

**Response:**
```typescript
{
  success: true,
  data: {
    job_id: string,
    status: "queued" | "analyzing" | "generating" | "completed" | "failed",
    progress: number,          // 0.0 - 1.0
    estimated_time_remaining: number,  // seconds
    result?: {
      transformed_url: string,
      thumbnail_url: string,
      original_url: string,
      metadata: {
        style_applied: string,
        processing_time: number,
        quality_score: number
      }
    },
    error?: string
  }
}
```

**Status Transitions:**
```
queued (0%)
   ↓
analyzing (20%) - Room detection, 2-3 seconds
   ↓
generating (50%) - Control image generation, 2-3 seconds
   ↓
transforming (80%) - Modal SD inference, 10-12 seconds
   ↓
completed (100%) - Upload to S3, return URLs
```

---

### Endpoint 3: WebSocket Progress

**WebSocket** `/ws/transform/{job_id}`

**Purpose**: Real-time progress updates

**Connection Flow:**
```typescript
// Client connects
const ws = new WebSocket(`wss://api.modomo.com/ws/transform/${job_id}`)

// Server sends updates
ws.onmessage = (event) => {
  const update = JSON.parse(event.data)
  // {
  //   status: "analyzing",
  //   progress: 0.2,
  //   message: "Detecting furniture in your room...",
  //   timestamp: "2025-11-12T10:30:45Z"
  // }
}

// Progress messages
1. "Analyzing your room..." (0.1)
2. "Detected 5 furniture items" (0.2)
3. "Generating depth map..." (0.3)
4. "Creating edge map..." (0.4)
5. "Preparing style transformation..." (0.5)
6. "Applying modern style..." (0.6-0.9, live updates from Modal)
7. "Finalizing your design..." (0.95)
8. "Complete! Your new room is ready." (1.0)
```

**WebSocket Implementation:**
```python
# api/websocket_manager.py

class WebSocketManager:
    def __init__(self):
        self.active_connections: Dict[str, List[WebSocket]] = {}
        self.redis_subscriber = redis.Redis().pubsub()

    async def connect(self, job_id: str, websocket: WebSocket):
        await websocket.accept()
        if job_id not in self.active_connections:
            self.active_connections[job_id] = []
        self.active_connections[job_id].append(websocket)

        # Subscribe to Redis pub/sub for this job
        await self.subscribe_to_job_updates(job_id, websocket)

    async def broadcast_update(self, job_id: str, update: dict):
        """
        Broadcast update to all connected clients for this job
        """
        if job_id in self.active_connections:
            for connection in self.active_connections[job_id]:
                await connection.send_json(update)

    async def subscribe_to_job_updates(self, job_id: str, websocket: WebSocket):
        """
        Listen to Redis pub/sub and forward to WebSocket
        """
        channel = self.redis_subscriber.subscribe(f"job_updates:{job_id}")
        async for message in channel.listen():
            if message['type'] == 'message':
                await websocket.send_json(json.loads(message['data']))
```

---

## 🎨 Modal SD Service Detailed Design

### Modal Function Architecture

**File**: `modal_functions/sd_inference.py`

```python
import modal
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from diffusers import DPMSolverMultistepScheduler
import torch
from PIL import Image
import io

# Create Modal stub
stub = modal.Stub("modomo-sd-inference")

# Define container image with all dependencies
image = (
    modal.Image.debian_slim()
    .pip_install(
        "diffusers==0.27.0",
        "transformers==4.38.0",
        "accelerate==0.27.0",
        "torch==2.2.0",
        "torchvision==0.17.0",
        "safetensors==0.4.0",
        "Pillow==10.2.0"
    )
    .run_commands(
        # Pre-download models during image build (faster cold starts)
        "python -c 'from diffusers import StableDiffusionControlNetPipeline, ControlNetModel; "
        "ControlNetModel.from_pretrained(\"lllyasviel/control_v11p_sd15_canny\"); "
        "ControlNetModel.from_pretrained(\"lllyasviel/control_v11f1p_sd15_depth\"); "
        "StableDiffusionControlNetPipeline.from_pretrained(\"runwayml/stable-diffusion-v1-5\")'"
    )
)

# Global model cache (persists across warm invocations)
@stub.cls(
    gpu="T4",           # NVIDIA T4 (16GB VRAM, £0.30/hour)
    image=image,
    container_idle_timeout=300,  # Keep warm for 5 minutes
    timeout=60,         # Max 60 seconds per invocation
    retries=2           # Retry on failure
)
class SDTransformer:
    """
    Stable Diffusion 1.5 + ControlNet transformer

    Uses multi-ControlNet for better room structure preservation
    """

    def __enter__(self):
        """
        Load models when container starts (runs once)
        """
        print("Loading SD 1.5 + ControlNet models...")

        # Load ControlNet models
        self.controlnet_depth = ControlNetModel.from_pretrained(
            "lllyasviel/control_v11f1p_sd15_depth",
            torch_dtype=torch.float16
        )
        self.controlnet_canny = ControlNetModel.from_pretrained(
            "lllyasviel/control_v11p_sd15_canny",
            torch_dtype=torch.float16
        )

        # Load SD 1.5 pipeline with multiple ControlNets
        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            controlnet=[self.controlnet_depth, self.controlnet_canny],
            torch_dtype=torch.float16,
            safety_checker=None,  # Disable for interior design
            requires_safety_checker=False
        )

        # Optimize scheduler for speed
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            self.pipe.scheduler.config
        )

        # Move to GPU
        self.pipe = self.pipe.to("cuda")

        # Enable memory optimizations
        self.pipe.enable_model_cpu_offload()
        self.pipe.enable_attention_slicing()

        print("Models loaded successfully!")

    @modal.method()
    def transform(
        self,
        control_images_bytes: dict,  # {depth: bytes, canny: bytes}
        style: str,
        room_type: str,
        seed: int = 42
    ) -> dict:
        """
        Transform room with specified style

        Args:
            control_images_bytes: Dict of control images as bytes
            style: Style name (modern, scandinavian, etc.)
            room_type: Type of room (living_room, bedroom, etc.)
            seed: Random seed for reproducibility

        Returns:
            {
                'image_bytes': bytes,
                'metadata': {...}
            }
        """
        # Load control images
        depth_image = Image.open(io.BytesIO(control_images_bytes['depth']))
        canny_image = Image.open(io.BytesIO(control_images_bytes['canny']))

        # Generate style-specific prompt
        prompt = self.generate_prompt(style, room_type)
        negative_prompt = self.get_negative_prompt()

        # Run SD inference
        generator = torch.Generator(device="cuda").manual_seed(seed)

        output = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=[depth_image, canny_image],
            num_inference_steps=20,  # Good quality, fast (12s)
            guidance_scale=7.5,       # Balanced creativity/structure
            controlnet_conditioning_scale=[0.8, 0.6],  # Depth stronger than edges
            generator=generator
        )

        result_image = output.images[0]

        # Convert to bytes
        img_buffer = io.BytesIO()
        result_image.save(img_buffer, format='JPEG', quality=95)
        img_bytes = img_buffer.getvalue()

        return {
            'image_bytes': img_bytes,
            'metadata': {
                'style': style,
                'room_type': room_type,
                'inference_steps': 20,
                'guidance_scale': 7.5,
                'seed': seed,
                'model': 'sd15_multi_controlnet'
            }
        }

    def generate_prompt(self, style: str, room_type: str) -> str:
        """
        Generate style-specific prompt
        """
        style_templates = {
            'modern': (
                f"modern minimalist {room_type}, clean lines, neutral colors, "
                "contemporary furniture, uncluttered space, sleek design, "
                "professional interior photography, high quality, 8k"
            ),
            'scandinavian': (
                f"Scandinavian {room_type}, light wood furniture, white walls, "
                "hygge atmosphere, cozy minimalist, natural textiles, "
                "soft lighting, Nordic design, professional photography"
            ),
            'boho': (
                f"bohemian {room_type}, eclectic mix, warm textures, "
                "plants, vintage furniture, colorful textiles, "
                "relaxed atmosphere, artistic, professional photography"
            ),
            'industrial': (
                f"industrial {room_type}, exposed brick, metal fixtures, "
                "raw materials, urban loft style, concrete, steel beams, "
                "modern rustic, professional interior photography"
            ),
            'minimalist': (
                f"minimalist {room_type}, extremely clean, simple furniture, "
                "white and black, uncluttered, zen-like, modern, "
                "professional photography, high quality"
            )
        }

        return style_templates.get(style, style_templates['modern'])

    def get_negative_prompt(self) -> str:
        """
        Negative prompt to avoid unwanted elements
        """
        return (
            "blurry, low quality, distorted, deformed, ugly, bad proportions, "
            "cluttered, messy, unrealistic, cartoon, anime, "
            "oversaturated, noise, artifacts"
        )

# Deployment function
@stub.local_entrypoint()
def main():
    """
    Test the SD transformer locally
    """
    transformer = SDTransformer()

    # Test with dummy control images
    # (In production, these come from backend)
    print("Testing SD transformer...")
```

### Modal Integration in Backend

**File**: `src/services/modal_sd_service.py`

```python
import modal
import io
from PIL import Image

class ModalSDService:
    """
    Interface to Modal SD service
    """

    def __init__(self):
        # Connect to deployed Modal stub
        self.stub = modal.Stub.lookup("modomo-sd-inference")
        self.transformer = self.stub.cls.lookup("SDTransformer")

    async def transform_room(
        self,
        depth_image: Image,
        canny_image: Image,
        style: str,
        room_type: str
    ) -> Image:
        """
        Submit transformation to Modal and get result
        """
        # Convert images to bytes
        depth_bytes = self.image_to_bytes(depth_image)
        canny_bytes = self.image_to_bytes(canny_image)

        # Call Modal function (auto-scales containers)
        result = await self.transformer.transform.remote.aio(
            control_images_bytes={
                'depth': depth_bytes,
                'canny': canny_bytes
            },
            style=style,
            room_type=room_type
        )

        # Convert bytes back to image
        result_image = Image.open(io.BytesIO(result['image_bytes']))

        return result_image

    def image_to_bytes(self, image: Image) -> bytes:
        """
        Convert PIL Image to bytes
        """
        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        return buffer.getvalue()
```

### Modal Cost Analysis

**GPU Options:**

| GPU    | VRAM | Performance  | Cost/Hour | Cost/Image (12s) | Recommendation |
|--------|------|--------------|-----------|------------------|----------------|
| T4     | 16GB | ~12s/image   | £0.30     | £0.03            | ✅ Best choice |
| A10G   | 24GB | ~8s/image    | £0.70     | £0.045           | Premium tier   |
| A100   | 40GB | ~5s/image    | £1.50     | £0.06            | Overkill       |

**Scaling Economics:**

```python
# Cost calculations for SD 1.5 on Modal T4

INFERENCE_TIME = 12  # seconds per image
GPU_COST_PER_HOUR = 0.30  # £0.30 for T4
IMAGES_PER_HOUR = 3600 / INFERENCE_TIME  # 300 images/hour

COST_PER_IMAGE = GPU_COST_PER_HOUR / IMAGES_PER_HOUR  # £0.03

# Monthly budget scenarios
MONTHLY_BUDGET = 500  # £500/month

MAX_IMAGES_PER_MONTH = MONTHLY_BUDGET / COST_PER_IMAGE  # 16,666 images

# With 8% conversion rate at £15 commission
REVENUE_PER_IMAGE = 15 * 0.08  # £1.20
PROFIT_PER_IMAGE = REVENUE_PER_IMAGE - COST_PER_IMAGE  # £1.17 (97.5% margin!)

# Break-even point
IMAGES_TO_BREAK_EVEN = MONTHLY_BUDGET / PROFIT_PER_IMAGE  # 427 converting users
```

**Auto-Scaling Strategy:**

```python
# Modal auto-scaling configuration

@stub.cls(
    gpu="T4",
    min_containers=0,    # Scale to zero when idle (save costs)
    max_containers=10,   # Max 10 concurrent (£3/hour max)
    container_idle_timeout=300,  # Keep warm for 5 min
    concurrency=1        # 1 job per container (avoid VRAM issues)
)

# Scaling behavior:
# - 0 jobs: 0 containers (£0/hour)
# - 1-10 jobs: 1-10 containers (£0.30-£3/hour)
# - 10+ jobs: Queue at 10 containers (manage with Redis)
```

---

## 📱 Mobile App Architecture (React Native)

### Project Structure

```
mobile/
├── package.json
├── tsconfig.json
├── babel.config.js
├── metro.config.js
├── ios/                          # iOS native code
├── android/                      # Android native code
└── src/
    ├── App.tsx                   # Root component
    ├── navigation/
    │   └── AppNavigator.tsx      # React Navigation setup
    ├── screens/
    │   ├── CameraScreen.tsx      # Photo capture
    │   ├── StyleSelectionScreen.tsx  # Style picker
    │   ├── ProcessingScreen.tsx  # Real-time progress
    │   └── ResultsScreen.tsx     # Final result viewer
    ├── services/
    │   ├── api.ts                # API client
    │   ├── websocket.ts          # WebSocket manager
    │   ├── camera.ts             # Camera utilities
    │   └── storage.ts            # Local storage
    ├── stores/
    │   └── appStore.ts           # Zustand state management
    ├── components/
    │   ├── StyleCard.tsx         # Style selection card
    │   ├── ProgressBar.tsx       # Progress indicator
    │   └── QualityGuide.tsx      # Camera quality hints
    └── types/
        └── index.ts              # TypeScript definitions
```

### Core Dependencies

**package.json:**
```json
{
  "name": "modomo-mobile",
  "version": "1.0.0",
  "dependencies": {
    "react": "18.2.0",
    "react-native": "0.73.0",
    "react-native-vision-camera": "^3.8.0",
    "react-native-image-resizer": "^3.0.5",
    "@react-navigation/native": "^6.1.9",
    "@react-navigation/native-stack": "^6.9.17",
    "zustand": "^4.4.7",
    "axios": "^1.6.0",
    "react-native-fs": "^2.20.0",
    "react-native-reanimated": "^3.6.0",
    "react-native-gesture-handler": "^2.14.0"
  }
}
```

### Screen 1: Camera Screen

**File**: `mobile/src/screens/CameraScreen.tsx`

```typescript
import React, { useRef, useState } from 'react'
import { Camera, useCameraDevice } from 'react-native-vision-camera'
import { View, TouchableOpacity, Text } from 'react-native'
import { useNavigation } from '@react-navigation/native'

export function CameraScreen() {
  const device = useCameraDevice('back')
  const camera = useRef<Camera>(null)
  const navigation = useNavigation()

  const [qualityScore, setQualityScore] = useState(0)

  const capturePhoto = async () => {
    if (!camera.current) return

    const photo = await camera.current.takePhoto({
      qualityPrioritization: 'quality',
      enableShutterSound: true
    })

    // Navigate to style selection with photo URI
    navigation.navigate('StyleSelection', { photoUri: photo.path })
  }

  // Real-time quality assessment (simplified)
  const onFrameProcessed = (frame) => {
    // Check lighting, focus, composition
    // Update quality score (0-100)
    // Show hints: "Move closer", "Need more light", etc.
  }

  return (
    <View style={{ flex: 1 }}>
      <Camera
        ref={camera}
        style={{ flex: 1 }}
        device={device}
        isActive={true}
        photo={true}
        frameProcessor={onFrameProcessed}
      />

      {/* Quality indicator overlay */}
      <View style={styles.qualityIndicator}>
        <Text>Quality: {qualityScore}%</Text>
        {qualityScore < 70 && (
          <Text style={styles.hint}>💡 Need better lighting</Text>
        )}
      </View>

      {/* Capture button */}
      <TouchableOpacity
        style={styles.captureButton}
        onPress={capturePhoto}
        disabled={qualityScore < 70}
      >
        <View style={styles.captureButtonInner} />
      </TouchableOpacity>
    </View>
  )
}
```

### Screen 2: Style Selection

**File**: `mobile/src/screens/StyleSelectionScreen.tsx`

```typescript
import React, { useState } from 'react'
import { View, ScrollView, TouchableOpacity, Image } from 'react-native'
import { useNavigation, useRoute } from '@react-navigation/native'
import { apiService } from '../services/api'
import { useAppStore } from '../stores/appStore'

const STYLES = [
  {
    id: 'modern',
    name: 'Modern',
    description: 'Clean lines and minimalist design',
    preview: require('../assets/styles/modern.jpg')
  },
  {
    id: 'scandinavian',
    name: 'Scandinavian',
    description: 'Light wood and cozy atmosphere',
    preview: require('../assets/styles/scandinavian.jpg')
  },
  {
    id: 'boho',
    name: 'Bohemian',
    description: 'Eclectic mix with warm textures',
    preview: require('../assets/styles/boho.jpg')
  },
  {
    id: 'industrial',
    name: 'Industrial',
    description: 'Raw materials and urban style',
    preview: require('../assets/styles/industrial.jpg')
  },
  {
    id: 'minimalist',
    name: 'Minimalist',
    description: 'Simple and uncluttered',
    preview: require('../assets/styles/minimalist.jpg')
  }
]

export function StyleSelectionScreen() {
  const route = useRoute()
  const navigation = useNavigation()
  const { photoUri } = route.params

  const [selectedStyle, setSelectedStyle] = useState(null)
  const [loading, setLoading] = useState(false)

  const submitTransformation = async () => {
    if (!selectedStyle) return

    setLoading(true)

    try {
      // Submit to backend API
      const result = await apiService.submitTransformation(
        photoUri,
        selectedStyle.id
      )

      // Navigate to processing screen with job ID
      navigation.navigate('Processing', {
        jobId: result.job_id,
        websocketUrl: result.websocket_url
      })
    } catch (error) {
      console.error('Failed to submit:', error)
      // Show error to user
    } finally {
      setLoading(false)
    }
  }

  return (
    <View style={styles.container}>
      <ScrollView horizontal showsHorizontalScrollIndicator={false}>
        {STYLES.map(style => (
          <TouchableOpacity
            key={style.id}
            style={[
              styles.styleCard,
              selectedStyle?.id === style.id && styles.styleCardSelected
            ]}
            onPress={() => setSelectedStyle(style)}
          >
            <Image source={style.preview} style={styles.stylePreview} />
            <Text style={styles.styleName}>{style.name}</Text>
            <Text style={styles.styleDescription}>{style.description}</Text>
          </TouchableOpacity>
        ))}
      </ScrollView>

      <TouchableOpacity
        style={[styles.submitButton, !selectedStyle && styles.submitButtonDisabled]}
        onPress={submitTransformation}
        disabled={!selectedStyle || loading}
      >
        <Text style={styles.submitButtonText}>
          {loading ? 'Submitting...' : 'Transform My Room'}
        </Text>
      </TouchableOpacity>
    </View>
  )
}
```

### Screen 3: Processing Screen

**File**: `mobile/src/screens/ProcessingScreen.tsx`

```typescript
import React, { useEffect, useState } from 'react'
import { View, Text, ActivityIndicator } from 'react-native'
import { useRoute, useNavigation } from '@react-navigation/native'
import { websocketService } from '../services/websocket'

export function ProcessingScreen() {
  const route = useRoute()
  const navigation = useNavigation()
  const { jobId, websocketUrl } = route.params

  const [status, setStatus] = useState('queued')
  const [progress, setProgress] = useState(0)
  const [message, setMessage] = useState('Starting transformation...')

  useEffect(() => {
    // Connect to WebSocket for real-time updates
    const ws = websocketService.connect(websocketUrl)

    ws.onMessage((update) => {
      setStatus(update.status)
      setProgress(update.progress)
      setMessage(update.message)

      // Navigate to results when complete
      if (update.status === 'completed') {
        navigation.replace('Results', {
          jobId,
          resultUrl: update.result_url
        })
      }

      // Show error if failed
      if (update.status === 'failed') {
        // Show error modal
        console.error('Transformation failed:', update.error)
      }
    })

    return () => {
      ws.disconnect()
    }
  }, [jobId])

  return (
    <View style={styles.container}>
      <ActivityIndicator size="large" color="#0066FF" />

      <Text style={styles.message}>{message}</Text>

      {/* Progress bar */}
      <View style={styles.progressBarContainer}>
        <View style={[styles.progressBar, { width: `${progress * 100}%` }]} />
      </View>

      <Text style={styles.percentage}>{Math.round(progress * 100)}%</Text>

      {/* Status messages */}
      <View style={styles.statusList}>
        <StatusItem
          done={progress > 0.1}
          text="Analyzing your room"
        />
        <StatusItem
          done={progress > 0.3}
          text="Detecting furniture"
        />
        <StatusItem
          done={progress > 0.5}
          text="Preparing style transformation"
        />
        <StatusItem
          done={progress > 0.8}
          text="Applying your selected style"
        />
        <StatusItem
          done={progress >= 1.0}
          text="Finalizing your new design"
        />
      </View>
    </View>
  )
}

function StatusItem({ done, text }) {
  return (
    <View style={styles.statusItem}>
      <View style={[styles.statusIcon, done && styles.statusIconDone]}>
        {done && <Text>✓</Text>}
      </View>
      <Text style={[styles.statusText, done && styles.statusTextDone]}>
        {text}
      </Text>
    </View>
  )
}
```

### Screen 4: Results Screen

**File**: `mobile/src/screens/ResultsScreen.tsx`

```typescript
import React, { useState } from 'react'
import { View, Image, TouchableOpacity, Text, Share } from 'react-native'
import { useRoute, useNavigation } from '@react-navigation/native'
import { ReactNativeZoomableView } from '@openspacelabs/react-native-zoomable-view'

export function ResultsScreen() {
  const route = useRoute()
  const navigation = useNavigation()
  const { jobId, resultUrl } = route.params

  const [showOriginal, setShowOriginal] = useState(false)

  const shareResult = async () => {
    try {
      await Share.share({
        message: 'Check out my room makeover with Modomo!',
        url: resultUrl
      })
    } catch (error) {
      console.error('Failed to share:', error)
    }
  }

  const saveToGallery = async () => {
    // Download and save to device gallery
    // Implementation with react-native-fs
  }

  const transformAgain = () => {
    navigation.navigate('Camera')
  }

  return (
    <View style={styles.container}>
      {/* Zoomable result image */}
      <ReactNativeZoomableView
        maxZoom={3}
        minZoom={1}
        zoomStep={0.5}
        initialZoom={1}
        bindToBorders={true}
        style={styles.imageContainer}
      >
        <Image
          source={{ uri: resultUrl }}
          style={styles.resultImage}
          resizeMode="contain"
        />
      </ReactNativeZoomableView>

      {/* Action buttons */}
      <View style={styles.actionBar}>
        <TouchableOpacity
          style={styles.actionButton}
          onPress={() => setShowOriginal(!showOriginal)}
        >
          <Text>👁️ Compare</Text>
        </TouchableOpacity>

        <TouchableOpacity
          style={styles.actionButton}
          onPress={shareResult}
        >
          <Text>📤 Share</Text>
        </TouchableOpacity>

        <TouchableOpacity
          style={styles.actionButton}
          onPress={saveToGallery}
        >
          <Text>💾 Save</Text>
        </TouchableOpacity>
      </View>

      {/* Transform again button */}
      <TouchableOpacity
        style={styles.primaryButton}
        onPress={transformAgain}
      >
        <Text style={styles.primaryButtonText}>
          Transform Another Room
        </Text>
      </TouchableOpacity>
    </View>
  )
}
```

### API Service

**File**: `mobile/src/services/api.ts`

```typescript
import axios from 'axios'
import ImageResizer from 'react-native-image-resizer'
import RNFS from 'react-native-fs'

const API_BASE_URL = 'https://api.modomo.com/v1'

class APIService {
  private client = axios.create({
    baseURL: API_BASE_URL,
    timeout: 30000,
    headers: {
      'Content-Type': 'multipart/form-data'
    }
  })

  async submitTransformation(
    photoUri: string,
    style: string
  ): Promise<{ job_id: string; websocket_url: string }> {
    // Optimize image before upload
    const optimizedImage = await this.optimizeImage(photoUri)

    // Create form data
    const formData = new FormData()
    formData.append('file', {
      uri: optimizedImage.uri,
      type: 'image/jpeg',
      name: 'room.jpg'
    })
    formData.append('style', style)

    // Submit to backend
    const response = await this.client.post('/transform/submit', formData)

    return response.data.data
  }

  async checkJobStatus(jobId: string): Promise<any> {
    const response = await this.client.get(`/transform/status/${jobId}`)
    return response.data.data
  }

  private async optimizeImage(uri: string) {
    // Resize to max 1920x1080
    // Compress to 85% quality
    return ImageResizer.createResizedImage(
      uri,
      1920,
      1080,
      'JPEG',
      85,
      0
    )
  }
}

export const apiService = new APIService()
```

### WebSocket Service

**File**: `mobile/src/services/websocket.ts`

```typescript
class WebSocketService {
  private ws: WebSocket | null = null
  private reconnectAttempts = 0
  private maxReconnectAttempts = 5

  connect(url: string): WebSocketConnection {
    this.ws = new WebSocket(url)

    this.ws.onopen = () => {
      console.log('WebSocket connected')
      this.reconnectAttempts = 0
    }

    this.ws.onerror = (error) => {
      console.error('WebSocket error:', error)
    }

    this.ws.onclose = () => {
      console.log('WebSocket closed')
      this.attemptReconnect(url)
    }

    return {
      onMessage: (callback) => {
        if (this.ws) {
          this.ws.onmessage = (event) => {
            const data = JSON.parse(event.data)
            callback(data)
          }
        }
      },
      disconnect: () => {
        if (this.ws) {
          this.ws.close()
          this.ws = null
        }
      }
    }
  }

  private attemptReconnect(url: string) {
    if (this.reconnectAttempts < this.maxReconnectAttempts) {
      this.reconnectAttempts++
      setTimeout(() => {
        console.log(`Reconnecting... Attempt ${this.reconnectAttempts}`)
        this.connect(url)
      }, 2000 * this.reconnectAttempts)
    }
  }
}

export const websocketService = new WebSocketService()
```

---

## 📊 Data Flow Diagrams

### Complete Transformation Flow

```
┌──────────────────────────────────────────────────────────────────────┐
│ PHASE 1: PHOTO CAPTURE & UPLOAD (3-5 seconds)                        │
└──────────────────────────────────────────────────────────────────────┘
                              ↓
Mobile App (CameraScreen)
  → Capture photo with quality checks
  → Optimize image (resize to 1920x1080, compress to 85%)
  → Navigate to StyleSelectionScreen
                              ↓
Mobile App (StyleSelectionScreen)
  → User selects style (modern, scandinavian, etc.)
  → Submit to API: POST /transform/submit
                              ↓
┌──────────────────────────────────────────────────────────────────────┐
│ PHASE 2: BACKEND PROCESSING (5-8 seconds)                            │
└──────────────────────────────────────────────────────────────────────┘
                              ↓
FastAPI Backend (api/main.py)
  → Receive upload, validate image
  → Upload original to S3: /uploads/{job_id}/original.jpg
  → Create job in Redis: job:{job_id} = {status: 'queued', ...}
  → Return job_id to mobile app
  → Process in background (async)
                              ↓
Background Processing (BackgroundTasks)
  1. Run existing detection pipeline
     → YOLO + SAM2 detection (2-3s)
     → Room classification
     → Furniture segmentation

  2. Generate control images (src/services/control_generators.py)
     → Depth map with DPT-Large (2s)
     → Canny edge detection (0.5s)
     → Process SAM2 masks for segmentation control (0.5s)
     → Upload controls to S3

  3. Update Redis: status = 'generating', progress = 0.5

  4. Submit to Modal (src/services/modal_sd_service.py)
     → Call Modal function with control images + style
                              ↓
┌──────────────────────────────────────────────────────────────────────┐
│ PHASE 3: MODAL SD INFERENCE (10-12 seconds)                          │
└──────────────────────────────────────────────────────────────────────┘
                              ↓
Modal Container (modal_functions/sd_inference.py)
  → Load control images
  → Generate style-specific prompt
  → Run SD 1.5 + Multi-ControlNet
     • 20 inference steps
     • Guidance scale: 7.5
     • ControlNet scales: [0.8 depth, 0.6 canny]
  → Quality validation (>70% confidence)
  → Return image bytes
                              ↓
┌──────────────────────────────────────────────────────────────────────┐
│ PHASE 4: RESULT PROCESSING & STORAGE (2-3 seconds)                   │
└──────────────────────────────────────────────────────────────────────┘
                              ↓
FastAPI Backend
  → Receive result from Modal
  → Upload to S3: /results/{job_id}/transformed.jpg
  → Generate thumbnail (480x360)
  → Upload thumbnail to S3
  → Update Redis: status = 'completed', result_url = <CDN_URL>
  → Publish to Redis PubSub: job_updates:{job_id}
                              ↓
┌──────────────────────────────────────────────────────────────────────┐
│ PHASE 5: MOBILE APP DISPLAY (instant)                                │
└──────────────────────────────────────────────────────────────────────┘
                              ↓
Mobile App (ProcessingScreen)
  → WebSocket receives update: status = 'completed'
  → Navigate to ResultsScreen
  → Display transformed image
  → Enable share, save, compare actions

TOTAL TIME: 20-28 seconds (target: <15s in production)
```

### WebSocket Progress Updates Flow

```
┌─────────────────────┐
│ Mobile App          │
│ (ProcessingScreen)  │
└─────────────────────┘
         ↓ WebSocket Connect: /ws/transform/{job_id}
┌─────────────────────┐
│ FastAPI             │
│ WebSocketManager    │
└─────────────────────┘
         ↓ Subscribe to Redis PubSub: job_updates:{job_id}
┌─────────────────────┐
│ Redis PubSub        │
└─────────────────────┘
         ↑ Publish updates
┌─────────────────────┐
│ Background Worker   │
└─────────────────────┘

Update Timeline:
  0s   → "queued" (progress: 0.0)
  2s   → "analyzing" (progress: 0.2) - "Detecting furniture..."
  5s   → "generating" (progress: 0.4) - "Creating depth map..."
  8s   → "transforming" (progress: 0.6) - "Applying modern style..."
  15s  → "transforming" (progress: 0.85) - "Almost done..."
  20s  → "completed" (progress: 1.0) - "Your new room is ready!"
```

### Error Handling Flow

```
Error Scenarios:

1. Upload Failure (Network Issues)
   Mobile → API (FAILED)
   ↓
   Show error: "Upload failed. Please check connection."
   ↓
   Retry with exponential backoff (3 attempts)
   ↓
   If still fails: Save to local queue, retry when online

2. Detection Failure (Poor Image Quality)
   API → Detection Pipeline (FAILED)
   ↓
   Update Redis: status = 'failed', error = 'Low quality image'
   ↓
   WebSocket → Mobile: Show error
   ↓
   Prompt: "Please retake photo with better lighting"

3. Modal Timeout (Infrastructure Issues)
   API → Modal (TIMEOUT after 60s)
   ↓
   Automatic retry (2 attempts)
   ↓
   If still fails: Update Redis: status = 'failed'
   ↓
   WebSocket → Mobile: "Transformation failed. Please try again."
   ↓
   Offer: "Retry" or "Contact Support"

4. Storage Failure (S3 Down)
   Modal → API → S3 (FAILED)
   ↓
   Retry with exponential backoff
   ↓
   If fails: Store result temporarily in Redis (max 1 hour)
   ↓
   Background job retries S3 upload
   ↓
   Update Redis with final URL when successful
```

---

## 🚀 Implementation Roadmap

### Week 1: Backend Infrastructure Setup

**Days 1-2: Environment & Dependencies**
- [ ] Set up Modal account, get API key
- [ ] Set up Redis (local or Redis Cloud free tier)
- [ ] Set up S3/Cloudflare R2 bucket
- [ ] Update requirements.txt with new dependencies
- [ ] Create .env.example with all required variables

**Days 3-4: Control Image Generation**
- [ ] Create `src/services/control_generators.py`
- [ ] Implement depth map generation (DPT-Large)
- [ ] Implement Canny edge detection
- [ ] Implement segmentation map processing
- [ ] Unit tests for control generation

**Days 5-7: Job Queue & Storage**
- [ ] Create `src/services/job_queue.py` (Redis integration)
- [ ] Create `src/services/storage_service.py` (S3/R2 integration)
- [ ] Implement job status tracking
- [ ] Implement image upload/download
- [ ] Integration tests

**Success Criteria:**
- ✅ All control images generate correctly
- ✅ Jobs tracked in Redis
- ✅ Images upload to S3/R2
- ✅ Unit test coverage >80%

---

### Week 2: Modal SD Service & API Endpoints

**Days 1-3: Modal SD Function**
- [ ] Create `modal_functions/sd_inference.py`
- [ ] Implement SD 1.5 + ControlNet pipeline
- [ ] Add style-specific prompt templates
- [ ] Test with sample images locally
- [ ] Deploy to Modal, test cold start and warm inference

**Days 4-5: Backend Integration**
- [ ] Create `src/services/modal_sd_service.py`
- [ ] Implement Modal API communication
- [ ] Add error handling and retries
- [ ] Test end-to-end: control images → Modal → result

**Days 6-7: API Endpoints**
- [ ] Update `api/main.py` with new endpoints
  - POST /transform/submit
  - GET /transform/status/{job_id}
- [ ] Create `api/websocket_manager.py`
- [ ] Add WebSocket endpoint: /ws/transform/{job_id}
- [ ] Integration tests for all endpoints

**Success Criteria:**
- ✅ Modal function deploys successfully
- ✅ Processing time <15 seconds
- ✅ Cost <£0.05 per image
- ✅ WebSocket updates work in real-time

---

### Week 3: React Native Mobile App

**Days 1-2: Project Setup**
- [ ] Initialize React Native project
- [ ] Install dependencies (camera, navigation, etc.)
- [ ] Set up TypeScript configuration
- [ ] Configure iOS and Android builds

**Days 3-4: Core Screens**
- [ ] CameraScreen with quality guidance
- [ ] StyleSelectionScreen with 5 styles
- [ ] ProcessingScreen with progress bar
- [ ] ResultsScreen with zoom/share

**Days 5-7: API & WebSocket Integration**
- [ ] Create `mobile/src/services/api.ts`
- [ ] Create `mobile/src/services/websocket.ts`
- [ ] Implement image optimization
- [ ] Test end-to-end flow on device
- [ ] Add error handling and retry logic

**Success Criteria:**
- ✅ App builds on iOS and Android
- ✅ Photo capture works smoothly
- ✅ Real-time progress updates display
- ✅ Results load and display correctly

---

### Week 4: Testing, Optimization & Polish

**Days 1-2: Backend Load Testing**
- [ ] Test with 10, 50, 100 concurrent requests
- [ ] Validate Modal auto-scaling
- [ ] Test Redis queue under load
- [ ] Optimize bottlenecks

**Days 3-4: Mobile Testing**
- [ ] Test on multiple iOS devices (iPhone 12, 13, 14, 15)
- [ ] Test on Android devices (Samsung, Pixel)
- [ ] Test different network conditions (3G, 4G, WiFi)
- [ ] Fix UI/UX issues

**Days 5-7: Production Preparation**
- [ ] Set up monitoring (Sentry, Datadog)
- [ ] Add analytics events (transformation start, completion, errors)
- [ ] Security audit (API keys, image validation)
- [ ] Documentation (API docs, deployment guide)
- [ ] Beta launch with 10 users

**Success Criteria:**
- ✅ Handle 100 concurrent users
- ✅ Average processing time <15s
- ✅ <2% error rate
- ✅ App works on all target devices
- ✅ Beta users satisfied (4.5+ rating)

---

## 📈 Success Metrics & Monitoring

### Performance Metrics

**Backend:**
```python
# Key metrics to track

METRICS = {
    'api_request_duration': Histogram(
        'api_request_duration_seconds',
        'API request duration',
        ['endpoint', 'status']
    ),
    'modal_inference_duration': Histogram(
        'modal_inference_duration_seconds',
        'Modal SD inference time'
    ),
    'job_queue_length': Gauge(
        'job_queue_length',
        'Number of jobs in queue'
    ),
    'transformation_success_rate': Counter(
        'transformation_success_total',
        'Number of successful transformations'
    ),
    'transformation_failure_rate': Counter(
        'transformation_failure_total',
        'Number of failed transformations',
        ['error_type']
    ),
    'cost_per_transformation': Histogram(
        'cost_per_transformation_gbp',
        'Cost per transformation in GBP'
    )
}

# Target SLAs
TARGETS = {
    'api_response_time_p95': 200,  # ms
    'transformation_time_avg': 15,  # seconds
    'transformation_time_p95': 25,  # seconds
    'success_rate': 0.92,           # 92%
    'cost_per_image': 0.05,         # £0.05
    'concurrent_capacity': 100      # users
}
```

**Mobile:**
```typescript
// Analytics events to track

enum AnalyticsEvent {
  APP_OPENED = 'app_opened',
  PHOTO_CAPTURED = 'photo_captured',
  STYLE_SELECTED = 'style_selected',
  TRANSFORMATION_STARTED = 'transformation_started',
  TRANSFORMATION_COMPLETED = 'transformation_completed',
  TRANSFORMATION_FAILED = 'transformation_failed',
  RESULT_SHARED = 'result_shared',
  RESULT_SAVED = 'result_saved'
}

// Track performance
interface PerformanceMetrics {
  photo_capture_time: number      // ms
  upload_time: number              // ms
  transformation_time: number      // seconds
  websocket_latency: number        // ms
  app_crash_rate: number           // %
}
```

### Cost Monitoring

**Daily Dashboard:**
```python
# Track costs in real-time

def calculate_daily_costs():
    return {
        'modal_gpu_hours': get_modal_usage(),          # Hours
        'modal_cost': get_modal_cost(),                # £
        'redis_cost': 0,                               # Free tier
        's3_storage': get_s3_storage_cost(),           # £
        's3_bandwidth': get_s3_bandwidth_cost(),       # £
        'total_images_processed': get_image_count(),
        'cost_per_image': get_cost_per_image(),
        'revenue_per_image': calculate_revenue(),
        'profit_margin': calculate_margin()
    }

# Alert thresholds
COST_ALERTS = {
    'daily_budget': 20,              # £20/day max
    'cost_per_image_max': 0.08,      # £0.08/image max
    'profit_margin_min': 0.85        # 85% min margin
}
```

### Quality Monitoring

**Image Quality Dashboard:**
```python
def track_quality_metrics():
    return {
        'upload_quality_score': get_avg_quality(),     # 0-1
        'sd_confidence_score': get_sd_confidence(),    # 0-1
        'user_satisfaction': get_user_ratings(),       # 1-5 stars
        'retry_rate': calculate_retry_rate(),          # %
        'error_types': get_error_distribution()        # {type: count}
    }

# Quality gates
QUALITY_GATES = {
    'min_upload_quality': 0.70,       # Reject below 70%
    'min_sd_confidence': 0.70,        # Regenerate below 70%
    'target_satisfaction': 4.5,       # 4.5+ star rating
    'max_retry_rate': 0.10            # <10% retries
}
```

---

## 🔒 Security & Privacy Considerations

### Data Privacy

**Image Handling:**
- Images stored with UUID-based paths (no user info in path)
- Automatic deletion after 30 days (GDPR compliance)
- No facial recognition or personal data extraction
- User consent for image processing (terms of service)

**API Security:**
- Rate limiting: 10 requests/minute per user (free tier)
- JWT authentication (future: premium tier)
- Image validation: file type, size, malware scanning
- HTTPS-only communication

**Storage Security:**
- S3 bucket: Private by default
- Signed URLs with expiration (1 hour)
- Encryption at rest (AES-256)
- Encryption in transit (TLS 1.3)

### Infrastructure Security

**Modal:**
- Container isolation (each job separate)
- No persistent storage (images deleted after processing)
- Environment variables for secrets (never in code)

**Redis:**
- Password-protected
- SSL/TLS enabled
- Job TTL (1 hour max)
- No sensitive data in Redis

---

## 📚 Configuration Files

### Backend Environment Variables

**.env.example:**
```bash
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_SECRET_KEY=your-secret-key-here

# Modal Configuration
MODAL_TOKEN_ID=your-modal-token-id
MODAL_TOKEN_SECRET=your-modal-token-secret
MODAL_STUB_NAME=modomo-sd-inference

# Redis Configuration
REDIS_URL=redis://localhost:6379
REDIS_PASSWORD=your-redis-password
REDIS_DB=0

# Storage Configuration (S3/R2)
S3_BUCKET_NAME=modomo-images
S3_ACCESS_KEY_ID=your-access-key
S3_SECRET_ACCESS_KEY=your-secret-key
S3_REGION=auto  # For Cloudflare R2
S3_ENDPOINT_URL=https://your-account.r2.cloudflarestorage.com

# CDN Configuration
CDN_DOMAIN=cdn.modomo.com

# Monitoring
SENTRY_DSN=your-sentry-dsn
ENVIRONMENT=development  # development | staging | production
```

### Mobile App Configuration

**mobile/.env:**
```bash
API_BASE_URL=https://api.modomo.com/v1
WEBSOCKET_BASE_URL=wss://api.modomo.com/v1
SENTRY_DSN=your-mobile-sentry-dsn
ANALYTICS_KEY=your-analytics-key
```

---

## 🎓 Training & Documentation

### Developer Onboarding

**Quick Start Guide:**
1. Clone repository
2. Install dependencies: `pip install -r requirements.txt`
3. Set up .env file with credentials
4. Start Redis: `redis-server`
5. Run API: `uvicorn api.main:app --reload`
6. Deploy Modal: `modal deploy modal_functions/sd_inference.py`
7. Test: `pytest tests/`

**Architecture Documentation:**
- This document (SD_INTEGRATION_ARCHITECTURE.md)
- API documentation (auto-generated with FastAPI)
- Modal function documentation
- Mobile app README

### Troubleshooting Guide

**Common Issues:**

1. **Modal cold starts taking >10s**
   - Solution: Increase `container_idle_timeout` to keep warm
   - Pre-download models during container build

2. **Redis connection timeout**
   - Check REDIS_URL is correct
   - Verify Redis is running: `redis-cli ping`
   - Check firewall rules

3. **S3 upload fails**
   - Verify credentials in .env
   - Check bucket permissions
   - Test with AWS CLI: `aws s3 ls s3://modomo-images`

4. **WebSocket disconnects frequently**
   - Check network stability
   - Increase reconnect attempts
   - Add heartbeat/ping messages

5. **High Modal costs**
   - Check `container_idle_timeout` (not too high)
   - Verify auto-scaling max limit
   - Monitor concurrent container count

---

## ✅ Checklist: Pre-Launch

### Backend
- [ ] All API endpoints tested and documented
- [ ] Modal function deployed and tested
- [ ] Redis queue working correctly
- [ ] S3/R2 storage configured with CDN
- [ ] WebSocket real-time updates working
- [ ] Error handling covers all scenarios
- [ ] Monitoring and logging set up
- [ ] Cost tracking implemented
- [ ] Security audit completed
- [ ] Load testing passed (100+ concurrent users)

### Mobile
- [ ] App builds on iOS and Android
- [ ] Camera works on all test devices
- [ ] Style selection UI polished
- [ ] Progress updates smooth and responsive
- [ ] Results display with zoom/share
- [ ] Offline queue works
- [ ] Error messages user-friendly
- [ ] Analytics events tracked
- [ ] Crash reporting set up
- [ ] Beta testing completed with 10+ users

### Infrastructure
- [ ] Modal account set up and funded
- [ ] Redis instance running (local or cloud)
- [ ] S3/R2 bucket created and configured
- [ ] CDN domain configured
- [ ] Monitoring dashboards created
- [ ] Alert rules configured
- [ ] Backup strategy documented
- [ ] Disaster recovery plan in place

---

## 🎯 Next Steps

1. **Review this document** with the team
2. **Set up accounts**: Modal, Redis Cloud, Cloudflare R2
3. **Create project structure** as outlined
4. **Start Week 1 implementation** (Backend Infrastructure)
5. **Daily standups** to track progress
6. **Weekly demos** to stakeholders

---

**Document Maintained By:** Technical Team
**Last Updated:** November 2025
**Next Review:** After Week 2 Implementation

**Questions or feedback?** Open an issue in the GitHub repository or contact the technical lead.

---

**This document serves as the complete blueprint for implementing SD 1.5 + ControlNet integration with Modal and React Native mobile app for Modomo's interior design AI platform.**

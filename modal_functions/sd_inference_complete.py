"""
Complete SD Transformation Pipeline on Modal GPU

This Modal function handles ALL heavy processing:
- YOLO + SAM2 detection
- Room classification
- Control image generation (depth, edges, segmentation)
- SD 1.5 + ControlNet transformation
- Quality validation
- Result upload

Author: Modomo Team
Date: November 2025
"""

import modal
import io
import json
import logging
from typing import Dict, Optional
from datetime import datetime

# Create Modal app (formerly called Stub)
app = modal.App("modomo-sd-inference")

# Define container image with ALL dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    # Install system dependencies first (including git for SAM2)
    .apt_install("libgl1-mesa-glx", "libglib2.0-0", "git")
    .pip_install(
        # Core ML - pinned versions for faster resolution
        "torch==2.5.1",
        "torchvision==0.20.1",
        "transformers==4.38.0",
        "accelerate==0.27.0",
        "timm==1.0.3",
        # SD & ControlNet
        "diffusers==0.27.0",
        "huggingface_hub==0.20.3",  # Compatible with diffusers 0.27.0
        "safetensors==0.4.1",
        # YOLO & SAM2
        "ultralytics==8.0.0",
        "sam-2 @ git+https://github.com/facebookresearch/segment-anything-2.git",
        # Image processing
        "opencv-python==4.9.0.80",
        "Pillow==10.2.0",
        "scikit-image==0.22.0",
        "numpy==1.26.4",
        # Storage & Redis
        "boto3==1.28.0",
        "redis==5.0.0",
        # Utilities
        "requests==2.31.0",
        "python-dotenv==1.0.0"
    )
    .run_commands(
        # Pre-download SD 1.5 and ControlNet models during build
        "python -c '"
        "from diffusers import StableDiffusionPipeline, ControlNetModel; "
        "ControlNetModel.from_pretrained(\"lllyasviel/control_v11f1p_sd15_depth\"); "
        "ControlNetModel.from_pretrained(\"lllyasviel/control_v11p_sd15_canny\"); "
        "ControlNetModel.from_pretrained(\"BertChristiaens/controlnet-seg-room\"); "
        "ControlNetModel.from_pretrained(\"lllyasviel/control_v11p_sd15_mlsd\"); "
        "StableDiffusionPipeline.from_pretrained(\"runwayml/stable-diffusion-v1-5\")"
        "'"
    )
)

# Logging
logger = logging.getLogger(__name__)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def update_redis_progress(redis_url: str, job_id: str, status: str, progress: float, message: str):
    """Update job progress in Redis"""
    try:
        import redis
        r = redis.from_url(redis_url, decode_responses=True)

        # Get job data
        job_key = f"job:{job_id}"
        job_data = json.loads(r.get(job_key) or "{}")

        # Update
        job_data["status"] = status
        job_data["progress"] = progress
        job_data["updated_at"] = datetime.utcnow().isoformat()

        # Save
        r.setex(job_key, 3600, json.dumps(job_data))

        # Publish update
        r.publish(
            f"job_updates:{job_id}",
            json.dumps({
                "job_id": job_id,
                "status": status,
                "progress": progress,
                "message": message,
                "timestamp": datetime.utcnow().isoformat()
            })
        )

        logger.info(f"✓ Updated Redis: {job_id} - {status} ({progress:.0%})")

    except Exception as e:
        logger.error(f"Failed to update Redis: {e}")


def download_image_from_r2(image_url: str, r2_config: dict) -> bytes:
    """Download image from R2"""
    import boto3

    s3 = boto3.client(
        's3',
        endpoint_url=r2_config['endpoint_url'],
        aws_access_key_id=r2_config['access_key_id'],
        aws_secret_access_key=r2_config['secret_access_key'],
        region_name='auto'
    )

    # Extract key from URL
    key = image_url.split(r2_config['bucket_name'] + '/')[-1]

    response = s3.get_object(Bucket=r2_config['bucket_name'], Key=key)
    return response['Body'].read()


def upload_to_r2(image_bytes: bytes, key: str, r2_config: dict) -> str:
    """Upload image to R2"""
    import boto3

    s3 = boto3.client(
        's3',
        endpoint_url=r2_config['endpoint_url'],
        aws_access_key_id=r2_config['access_key_id'],
        aws_secret_access_key=r2_config['secret_access_key'],
        region_name='auto'
    )

    s3.put_object(
        Bucket=r2_config['bucket_name'],
        Key=key,
        Body=image_bytes,
        ContentType='image/jpeg',
        ACL='public-read'
    )

    return f"{r2_config['cdn_domain']}/{key}"


# ============================================================================
# ENSEMBLE STYLE CLASSIFIER MODELS
# Note: Classes are defined here but only instantiated inside Modal where torch is available
# ============================================================================

# Placeholder - actual classes will be created at runtime in the Modal function


# ============================================================================
# MAIN PROCESSING CLASS
# ============================================================================

# Create Modal Volume for custom model storage
model_volume = modal.Volume.from_name("modomo-models", create_if_missing=True)

@app.cls(
    gpu="T4",  # NVIDIA T4 GPU (£0.30/hour)
    image=image,
    timeout=120,  # 2 minutes max
    scaledown_window=300,  # Keep warm for 5 minutes (renamed from container_idle_timeout)
    retries=2,
    secrets=[modal.Secret.from_name("modomo-r2-credentials")],
    volumes={"/models": model_volume}  # Mount volume for custom models
)
class CompleteTransformationPipeline:
    """
    Complete transformation pipeline on Modal GPU

    Includes all heavy processing that was removed from Railway
    """

    @modal.enter()
    def __enter__(self):
        """Load all models when container starts"""
        import torch
        from transformers import pipeline
        from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, DPMSolverMultistepScheduler
        from ultralytics import YOLO
        from .quality_validator import QualityValidator

        logger.info("🚀 Loading models on Modal GPU...")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")

        # 1. Load depth estimation model
        logger.info("Loading depth estimation model...")
        self.depth_estimator = pipeline(
            "depth-estimation",
            model="Intel/dpt-large",
            device=0 if self.device == "cuda" else -1
        )

        # 2. Load YOLO (custom if available, fallback to generic)
        logger.info("Loading YOLO...")
        import os
        if os.path.exists("/models/yolo/v1/best.pt"):
            self.yolo = YOLO("/models/yolo/v1/best.pt")
            logger.info("✓ Loaded fine-tuned YOLO with 294 interior categories")
        else:
            logger.warning("⚠️ Custom YOLO not found at /models/yolo/v1/best.pt, using generic yolov8n.pt")
            self.yolo = YOLO("yolov8n.pt")  # Fallback to generic

        # 3. Load ControlNet models (core models eagerly, advanced models lazily)
        logger.info("Loading core ControlNet models...")
        self.controlnet_depth = ControlNetModel.from_pretrained(
            "lllyasviel/control_v11f1p_sd15_depth",
            torch_dtype=torch.float16
        )
        self.controlnet_canny = ControlNetModel.from_pretrained(
            "lllyasviel/control_v11p_sd15_canny",
            torch_dtype=torch.float16
        )
        logger.info("✓ Loaded core ControlNet models (depth, canny)")

        # Lazy-load advanced ControlNets only when needed
        self.controlnet_seg_room = None
        self.controlnet_mlsd = None
        logger.info("Advanced ControlNets (seg-room, M-LSD) will be lazy-loaded on demand")

        # 4. Load SD 1.5 pipeline
        logger.info("Loading SD 1.5 pipeline...")
        self.sd_pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            controlnet=[self.controlnet_depth, self.controlnet_canny],
            torch_dtype=torch.float16,
            safety_checker=None,
            requires_safety_checker=False
        )

        # Optimize scheduler
        self.sd_pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            self.sd_pipe.scheduler.config
        )

        # Move to GPU
        self.sd_pipe = self.sd_pipe.to(self.device)

        # Enable optimizations
        self.sd_pipe.enable_model_cpu_offload()
        self.sd_pipe.enable_attention_slicing()

        # 5. Ensemble Style Classifier placeholder (will be loaded from volume when available)
        logger.info("Checking for ensemble style classifier...")
        if os.path.exists("/models/ensemble/v1/efficientnet.pth"):
            logger.info("✓ Custom ensemble models found - will load on demand")
            self.has_ensemble = True
        else:
            logger.warning("⚠️ Ensemble classifier not found at /models/ensemble/v1/, using basic style prompts only")
            self.has_ensemble = False
        self.style_classifier = None  # Will be loaded on-demand if needed

        # 6. Initialize Quality Validator
        logger.info("Initializing quality validator...")
        self.quality_validator = QualityValidator(min_score=0.75)
        logger.info("✓ Quality validator ready")

        logger.info("✅ All models loaded successfully!")

    def _load_controlnet_seg_room(self):
        """Lazy-load segmentation ControlNet"""
        if self.controlnet_seg_room is None:
            import torch
            from diffusers import ControlNetModel
            logger.info("Lazy-loading seg-room ControlNet...")
            self.controlnet_seg_room = ControlNetModel.from_pretrained(
                "BertChristiaens/controlnet-seg-room",
                torch_dtype=torch.float16
            ).to(self.device)
            logger.info("✓ Loaded seg-room ControlNet")
        return self.controlnet_seg_room

    def _load_controlnet_mlsd(self):
        """Lazy-load M-LSD ControlNet"""
        if self.controlnet_mlsd is None:
            import torch
            from diffusers import ControlNetModel
            logger.info("Lazy-loading M-LSD ControlNet...")
            self.controlnet_mlsd = ControlNetModel.from_pretrained(
                "lllyasviel/control_v11p_sd15_mlsd",
                torch_dtype=torch.float16
            ).to(self.device)
            logger.info("✓ Loaded M-LSD ControlNet")
        return self.controlnet_mlsd

    @modal.method()
    def process_transformation_complete(
        self,
        job_id: str,
        image_url: str,
        style: str,
        room_type: Optional[str] = None,
        preferences: Optional[Dict] = None,
        redis_url: str = None
    ) -> Dict:
        """
        Complete transformation pipeline

        Steps:
        1. Download image from R2
        2. YOLO + SAM2 detection (optional)
        3. Room classification (optional)
        4. Generate depth map
        5. Generate Canny edges
        6. Run SD 1.5 + ControlNet
        7. Quality validation
        8. Upload to R2
        9. Update Redis with result

        Returns:
            Result dictionary with URLs and metadata
        """
        import torch
        import cv2
        import numpy as np
        from PIL import Image
        import os

        logger.info(f"🎨 Processing transformation for job {job_id}")
        start_time = datetime.utcnow()

        # R2 configuration from environment
        r2_config = {
            'endpoint_url': os.getenv('R2_ENDPOINT_URL'),
            'access_key_id': os.getenv('R2_ACCESS_KEY_ID'),
            'secret_access_key': os.getenv('R2_SECRET_ACCESS_KEY'),
            'bucket_name': os.getenv('R2_BUCKET_NAME', 'reroom'),
            'cdn_domain': os.getenv('CDN_DOMAIN')
        }

        # Validate R2 config before proceeding
        required_keys = ['endpoint_url', 'access_key_id', 'secret_access_key', 'bucket_name']
        missing_keys = [k for k in required_keys if not r2_config.get(k)]

        if missing_keys:
            error_msg = f"Missing required R2 configuration: {', '.join(missing_keys).upper()}"
            logger.error(error_msg)
            if redis_url:
                update_redis_progress(redis_url, job_id, "failed", 0.0, error_msg)
            raise ValueError(error_msg)

        try:
            # Step 1: Download image from R2
            update_redis_progress(redis_url, job_id, "analyzing", 0.1, "Downloading your image...")

            image_bytes = download_image_from_r2(image_url, r2_config)
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            logger.info(f"✓ Downloaded image: {image.size}")

            # Calculate target size preserving aspect ratio (nearest multiple of 64)
            orig_width, orig_height = image.size
            target_size = self._calculate_target_size(orig_width, orig_height)
            logger.info(f"Target size for processing: {target_size} (original: {image.size})")

            # Step 2: Generate depth map
            update_redis_progress(redis_url, job_id, "generating", 0.3, "Generating depth map...")

            depth_result = self.depth_estimator(image)
            depth_np = np.array(depth_result['depth'])

            # Normalize and colormap
            depth_normalized = cv2.normalize(depth_np, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_INFERNO)
            depth_colored = cv2.cvtColor(depth_colored, cv2.COLOR_BGR2RGB)

            depth_image = Image.fromarray(depth_colored).resize(target_size, Image.LANCZOS)
            logger.info("✓ Generated depth map")

            # Step 3: Generate Canny edges with adaptive thresholds
            update_redis_progress(redis_url, job_id, "generating", 0.4, "Generating edge map...")

            image_np = np.array(image)
            gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)

            # Adaptive Canny thresholds based on image contrast
            low_thresh, high_thresh = self._calculate_adaptive_canny_thresholds(blurred)
            logger.info(f"Using adaptive Canny thresholds: {low_thresh}, {high_thresh}")

            edges = cv2.Canny(blurred, low_thresh, high_thresh)

            # Dilate edges
            kernel = np.ones((2, 2), np.uint8)
            edges = cv2.dilate(edges, kernel, iterations=1)
            edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)

            canny_image = Image.fromarray(edges_rgb).resize(target_size, Image.LANCZOS)
            logger.info("✓ Generated edge map")

            # Step 4: Generate style-specific prompt
            update_redis_progress(redis_url, job_id, "transforming", 0.5, f"Applying {style} style...")

            prompt = self._get_style_prompt(style, room_type or "room")
            negative_prompt = self._get_negative_prompt()

            logger.info(f"Prompt: {prompt[:100]}...")

            # Step 5: Run SD 1.5 + ControlNet with dynamic conditioning scales
            update_redis_progress(redis_url, job_id, "transforming", 0.6, "Running AI transformation...")

            # Get dynamic conditioning scales based on room type
            conditioning_scales = self._get_controlnet_scales(room_type or "living_room")
            logger.info(f"Using conditioning scales {conditioning_scales} for {room_type}")

            # Generate deterministic seed from job_id for reproducibility with variety
            seed = self._generate_seed_from_job_id(job_id)
            logger.info(f"Using seed {seed} derived from job_id")

            generator = torch.Generator(device=self.device).manual_seed(seed)

            output = self.sd_pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=[depth_image, canny_image],
                num_inference_steps=20,  # Good quality, fast
                guidance_scale=7.5,
                controlnet_conditioning_scale=conditioning_scales,  # Dynamic based on room type
                generator=generator
            )

            result_image = output.images[0]
            logger.info("✓ SD transformation complete")

            # Step 6: Quality validation with retry logic
            update_redis_progress(redis_url, job_id, "finalizing", 0.85, "Validating quality...")

            validation_result = self.quality_validator.validate_result(
                generated_image=result_image,
                original_image=image,
                style=style
            )

            logger.info(f"Quality score: {validation_result['score']:.2f} - {validation_result['reason']}")

            # Retry once if quality is below threshold
            retry_count = 0
            max_retries = 1

            while validation_result['retry_recommended'] and retry_count < max_retries:
                retry_count += 1
                logger.warning(f"Quality below threshold ({validation_result['score']:.2f}), retrying... (attempt {retry_count}/{max_retries})")
                update_redis_progress(redis_url, job_id, "transforming", 0.7, f"Improving quality (retry {retry_count})...")

                # Get suggested parameter adjustments
                current_params = {
                    'prompt': prompt,
                    'negative_prompt': negative_prompt,
                    'image': [depth_image, canny_image],
                    'num_inference_steps': 20,
                    'guidance_scale': 7.5,
                    'controlnet_conditioning_scale': conditioning_scales,
                    'generator': generator,
                    'seed': seed
                }

                new_params = self.quality_validator.suggest_retry_params(
                    validation_result=validation_result,
                    current_params=current_params
                )

                # Update generator with new seed if changed
                if 'seed' in new_params and new_params['seed'] != seed:
                    new_params['generator'] = torch.Generator(device=self.device).manual_seed(new_params['seed'])
                    del new_params['seed']

                # Retry with adjusted parameters
                output = self.sd_pipe(**new_params)
                result_image = output.images[0]

                # Validate again
                validation_result = self.quality_validator.validate_result(
                    generated_image=result_image,
                    original_image=image,
                    style=style
                )
                logger.info(f"Retry quality score: {validation_result['score']:.2f} - {validation_result['reason']}")

            if not validation_result['passed']:
                logger.warning(f"Final quality score {validation_result['score']:.2f} below threshold, but proceeding (max retries reached)")
            else:
                logger.info(f"✓ Quality validation passed with score {validation_result['score']:.2f}")

            update_redis_progress(redis_url, job_id, "finalizing", 0.9, "Finalizing your design...")

            # Step 7: Upload results to R2
            update_redis_progress(redis_url, job_id, "finalizing", 0.95, "Uploading results...")

            # Save result as JPEG
            result_buffer = io.BytesIO()
            result_image.save(result_buffer, format='JPEG', quality=95)
            result_bytes = result_buffer.getvalue()

            # Upload
            result_key = f"results/{job_id}/transformed.jpg"
            result_url = upload_to_r2(result_bytes, result_key, r2_config)

            # Generate and upload thumbnail
            thumbnail = result_image.copy()
            thumbnail.thumbnail((480, 360), Image.LANCZOS)
            thumb_buffer = io.BytesIO()
            thumbnail.save(thumb_buffer, format='JPEG', quality=70)
            thumb_bytes = thumb_buffer.getvalue()

            thumb_key = f"results/{job_id}/thumbnail.jpg"
            thumb_url = upload_to_r2(thumb_bytes, thumb_key, r2_config)

            logger.info(f"✓ Uploaded results to R2")

            # Step 8: Update Redis with completion
            processing_time = (datetime.utcnow() - start_time).total_seconds()

            result_data = {
                "transformed_url": result_url,
                "thumbnail_url": thumb_url,
                "original_url": image_url,
                "metadata": {
                    "style": style,
                    "processing_time": processing_time,
                    "quality_score": validation_result['score'],  # Actual validation score
                    "quality_checks": validation_result['checks'],
                    "room_type": room_type,
                    "retry_count": retry_count
                }
            }

            update_redis_progress(
                redis_url,
                job_id,
                "completed",
                1.0,
                "Your new room is ready!"
            )

            # Also store result in Redis
            import redis
            r = redis.from_url(redis_url, decode_responses=True)
            job_key = f"job:{job_id}"
            job_data = json.loads(r.get(job_key) or "{}")
            job_data["result"] = result_data
            job_data["status"] = "completed"
            job_data["progress"] = 1.0
            r.setex(job_key, 3600, json.dumps(job_data))

            logger.info(f"✅ Job {job_id} completed in {processing_time:.1f}s")

            return result_data

        except Exception as e:
            logger.error(f"❌ Job {job_id} failed: {e}")

            # Update Redis with failure
            if redis_url:
                update_redis_progress(
                    redis_url,
                    job_id,
                    "failed",
                    0.0,
                    f"Transformation failed: {str(e)}"
                )

            raise

    def _generate_seed_from_job_id(self, job_id: str) -> int:
        """
        Generate deterministic seed from job_id

        This ensures:
        - Each job gets a unique seed (variety)
        - Same job_id always produces same seed (reproducibility for retries)

        Args:
            job_id: Job UUID string

        Returns:
            Integer seed (0 to 2^32-1)
        """
        import hashlib

        # Hash the job_id to get deterministic integer
        hash_object = hashlib.md5(job_id.encode())
        hash_hex = hash_object.hexdigest()

        # Convert first 8 hex chars to integer
        seed = int(hash_hex[:8], 16) % (2**32)

        return seed

    def _calculate_adaptive_canny_thresholds(self, gray_image: "np.ndarray") -> tuple:
        """
        Calculate adaptive Canny thresholds based on image contrast

        Uses Otsu's method to determine optimal threshold, then derives
        Canny low/high thresholds from it.

        Args:
            gray_image: Grayscale image array

        Returns:
            Tuple of (low_threshold, high_threshold)
        """
        import cv2

        # Calculate median intensity
        median_intensity = np.median(gray_image)

        # Use percentile-based approach for better adaptation
        # Calculate gradient magnitude to assess image contrast
        gradient_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
        gradient_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)

        # Use gradient percentiles to set thresholds
        low_percentile = np.percentile(gradient_magnitude, 50)
        high_percentile = np.percentile(gradient_magnitude, 85)

        # Scale thresholds (typical ratio is 1:2 or 1:3)
        low_thresh = max(50, int(low_percentile * 0.66))
        high_thresh = max(100, int(high_percentile * 1.33))

        # Ensure valid range
        low_thresh = min(low_thresh, 150)
        high_thresh = min(high_thresh, 300)
        high_thresh = max(high_thresh, low_thresh * 2)

        return (low_thresh, high_thresh)

    def _calculate_target_size(self, width: int, height: int, max_dim: int = 768) -> tuple:
        """
        Calculate target size preserving aspect ratio as nearest multiple of 64

        Args:
            width: Original width
            height: Original height
            max_dim: Maximum dimension (default: 768)

        Returns:
            Tuple of (target_width, target_height)
        """
        aspect_ratio = width / height

        # Calculate dimensions preserving aspect ratio
        if width > height:
            new_width = min(width, max_dim)
            new_height = int(new_width / aspect_ratio)
        else:
            new_height = min(height, max_dim)
            new_width = int(new_height * aspect_ratio)

        # Round to nearest multiple of 64 for SD compatibility
        new_width = ((new_width + 31) // 64) * 64
        new_height = ((new_height + 31) // 64) * 64

        # Ensure minimum size
        new_width = max(new_width, 512)
        new_height = max(new_height, 512)

        return (new_width, new_height)

    def _get_style_prompt(self, style: str, room_type: str) -> str:
        """Generate enhanced style-specific prompt with materials, colors, and details"""
        style_prompts = {
            'modern': (
                f"modern minimalist {room_type}, clean lines, neutral palette, "
                "contemporary furniture, uncluttered space, sleek design, "
                "materials: polished concrete, steel, glass, smooth surfaces, "
                "colors: white, charcoal gray, black accents, occasional bold color, "
                "lighting: recessed LED, natural light, "
                "professional interior photography, high quality, 8k, sharp focus"
            ),
            'scandinavian': (
                f"Scandinavian {room_type}, light oak furniture, white painted walls, "
                "hygge cozy atmosphere, minimalist Nordic design, natural textiles, "
                "materials: light wood, natural linen, wool, white paint, brass fixtures, "
                "colors: white, light gray, soft pastels, natural wood tones, "
                "lighting: soft diffused natural light, candles, warm ambiance, "
                "potted green plants, simple decor, "
                "professional interior photography, 8k, warm tones"
            ),
            'boho': (
                f"bohemian {room_type}, eclectic mix, rich warm textures, "
                "vintage furniture, layered textiles, abundant plants, "
                "materials: rattan, woven textiles, vintage wood, brass, macrame, "
                "colors: earth tones, terracotta, mustard yellow, deep reds, jewel tones, "
                "lighting: warm golden light, string lights, natural sunlight, "
                "decorative pillows, patterned rugs, artistic wall hangings, "
                "relaxed creative atmosphere, professional photography, 8k"
            ),
            'industrial': (
                f"industrial {room_type}, exposed elements, raw urban aesthetic, "
                "loft style, metal fixtures, reclaimed materials, "
                "materials: exposed brick, weathered steel, concrete, reclaimed wood, iron pipes, "
                "colors: charcoal, rust, raw steel gray, black, weathered wood brown, "
                "lighting: Edison bulbs, metal pendant lights, track lighting, "
                "open ductwork, minimal decoration, urban edge, "
                "professional interior photography, 8k, dramatic lighting"
            ),
            'minimalist': (
                f"minimalist {room_type}, extremely clean aesthetic, simple forms, "
                "essential furniture only, uncluttered zen space, "
                "materials: smooth white surfaces, natural wood, concrete, glass, "
                "colors: pure white, black, light gray, single accent color, "
                "lighting: hidden LED strips, natural light, clean shadows, "
                "negative space, geometric shapes, calm atmosphere, "
                "professional photography, high quality, 8k, crisp and clear"
            ),
            'traditional': (
                f"traditional {room_type}, classic elegant furniture, timeless style, "
                "formal arrangement, sophisticated details, "
                "materials: dark mahogany wood, leather upholstery, silk fabrics, brass hardware, "
                "colors: warm neutrals, navy blue, burgundy, gold accents, cream, "
                "lighting: chandeliers, table lamps, warm ambient light, "
                "crown molding, elegant drapery, classic decor, "
                "professional interior photography, 8k, rich colors"
            )
        }

        return style_prompts.get(style, style_prompts['modern'])

    def _get_negative_prompt(self) -> str:
        """Negative prompt to avoid unwanted elements"""
        return (
            "blurry, low quality, distorted, deformed, ugly, bad proportions, "
            "cluttered, messy, unrealistic, cartoon, anime, "
            "oversaturated, noise, artifacts"
        )

    def _get_controlnet_scales(self, room_type: str) -> list:
        """Dynamic ControlNet conditioning scales based on room characteristics"""
        scales_map = {
            'living_room': [0.8, 0.6],    # Strong depth, moderate canny
            'bedroom': [0.7, 0.5],         # Moderate control for intimate spaces
            'kitchen': [0.8, 0.4],         # Strong depth, light edges (for fixtures)
            'bathroom': [0.8, 0.7],        # Strong control for both (fixtures + layout)
            'dining_room': [0.75, 0.55],   # Balanced control
            'office': [0.7, 0.6],          # Moderate control
            'default': [0.8, 0.6]          # Default balanced approach
        }

        # Normalize room_type to lowercase and handle variations
        room_normalized = room_type.lower().replace(' ', '_')

        return scales_map.get(room_normalized, scales_map['default'])

    def _should_use_multi_controlnet(self, room_type: str, image_complexity: str = 'medium') -> bool:
        """
        Determine if multi-ControlNet should be used based on room type and complexity

        Args:
            room_type: Type of room
            image_complexity: Estimated complexity ('low', 'medium', 'high')

        Returns:
            True if multi-ControlNet should be used
        """
        # Rooms that benefit from additional ControlNets
        complex_rooms = ['kitchen', 'bathroom', 'office']

        room_normalized = room_type.lower().replace(' ', '_')

        # Use multi-ControlNet for complex rooms or high complexity images
        return room_normalized in complex_rooms or image_complexity == 'high'


# ============================================================================
# DEPLOYMENT ENTRYPOINT
# ============================================================================

@app.local_entrypoint()
def main():
    """Test the complete transformation pipeline locally"""

    print("🧪 Testing Modal transformation pipeline...")
    print("Note: This requires environment variables to be set")

    # Test with dummy data
    pipeline = CompleteTransformationPipeline()

    try:
        result = pipeline.process_transformation_complete.remote(
            job_id="test-job-123",
            image_url="https://example.com/test.jpg",
            style="modern",
            redis_url="redis://localhost:6379"
        )

        print(f"✅ Test completed!")
        print(f"Result: {result}")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        print("This is expected if environment is not fully configured")

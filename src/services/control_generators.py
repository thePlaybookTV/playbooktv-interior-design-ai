"""
Control Image Generation Service

Generates control images for ControlNet:
- Depth maps using DPT-Large
- Canny edge detection
- Segmentation maps from SAM2 masks

Author: Modomo Team
Date: November 2025
"""

import cv2
import numpy as np
from PIL import Image
from typing import List, Dict, Tuple
import torch
from transformers import pipeline
import logging

logger = logging.getLogger(__name__)


class ControlImageGenerator:
    """
    Generates control images for Stable Diffusion ControlNet

    Supports three types of control:
    1. Depth maps - for spatial structure preservation
    2. Canny edges - for architectural lines
    3. Segmentation maps - for furniture layout
    """

    def __init__(self, device: str = "auto"):
        """
        Initialize control image generators

        Args:
            device: Device to run models on ("cpu", "cuda", or "auto")
        """
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        logger.info(f"Initializing ControlImageGenerator on {self.device}")

        # Initialize depth estimation model
        self._init_depth_model()

    def _init_depth_model(self):
        """Initialize DPT-Large depth estimation model"""
        try:
            logger.info("Loading depth estimation model (Intel/dpt-large)...")
            self.depth_estimator = pipeline(
                "depth-estimation",
                model="Intel/dpt-large",
                device=0 if self.device == "cuda" else -1
            )
            logger.info("Depth model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load depth model: {e}")
            # Fallback to a smaller model
            logger.info("Falling back to DPT-hybrid model...")
            self.depth_estimator = pipeline(
                "depth-estimation",
                model="Intel/dpt-hybrid-midas",
                device=0 if self.device == "cuda" else -1
            )

    async def generate_all_controls(
        self,
        image: Image.Image,
        segmentation_masks: List[np.ndarray] = None,
        target_size: Tuple[int, int] = (512, 512)
    ) -> Dict[str, Image.Image]:
        """
        Generate all control images needed for SD ControlNet

        Args:
            image: Input PIL Image
            segmentation_masks: List of SAM2 segmentation masks (optional)
            target_size: Target resolution for control images

        Returns:
            Dictionary with keys: 'depth', 'canny', 'segmentation'
        """
        logger.info("Generating all control images...")

        controls = {}

        # Generate depth map
        try:
            controls['depth'] = await self.generate_depth_map(image, target_size)
            logger.info("✓ Depth map generated")
        except Exception as e:
            logger.error(f"Failed to generate depth map: {e}")
            controls['depth'] = None

        # Generate canny edges
        try:
            controls['canny'] = await self.generate_canny_edges(image, target_size)
            logger.info("✓ Canny edges generated")
        except Exception as e:
            logger.error(f"Failed to generate canny edges: {e}")
            controls['canny'] = None

        # Generate segmentation map (if masks provided)
        if segmentation_masks is not None and len(segmentation_masks) > 0:
            try:
                controls['segmentation'] = await self.process_segmentation_map(
                    segmentation_masks,
                    image.size,
                    target_size
                )
                logger.info("✓ Segmentation map generated")
            except Exception as e:
                logger.error(f"Failed to generate segmentation map: {e}")
                controls['segmentation'] = None
        else:
            controls['segmentation'] = None

        return controls

    async def generate_depth_map(
        self,
        image: Image.Image,
        target_size: Tuple[int, int] = (512, 512)
    ) -> Image.Image:
        """
        Generate depth map using DPT-Large

        Args:
            image: Input PIL Image
            target_size: Target resolution

        Returns:
            Depth map as PIL Image (RGB with colormap applied)
        """
        logger.info("Generating depth map...")

        # Resize image for faster inference
        original_size = image.size
        inference_image = image.resize((384, 384), Image.LANCZOS)

        # Run depth estimation
        depth_result = self.depth_estimator(inference_image)

        # Get depth as numpy array
        depth_np = np.array(depth_result['depth'])

        # Normalize to 0-255
        depth_normalized = cv2.normalize(
            depth_np,
            None,
            0, 255,
            cv2.NORM_MINMAX,
            dtype=cv2.CV_8U
        )

        # Apply colormap for better visualization
        depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_INFERNO)
        depth_colored = cv2.cvtColor(depth_colored, cv2.COLOR_BGR2RGB)

        # Convert to PIL and resize to target
        depth_image = Image.fromarray(depth_colored)
        depth_image = depth_image.resize(target_size, Image.LANCZOS)

        return depth_image

    async def generate_canny_edges(
        self,
        image: Image.Image,
        target_size: Tuple[int, int] = (512, 512),
        low_threshold: int = 100,
        high_threshold: int = 200
    ) -> Image.Image:
        """
        Generate Canny edge map optimized for interior scenes

        Args:
            image: Input PIL Image
            target_size: Target resolution
            low_threshold: Lower threshold for hysteresis (default: 100)
            high_threshold: Upper threshold for hysteresis (default: 200)

        Returns:
            Canny edge map as PIL Image (binary: 0 or 255)
        """
        logger.info("Generating Canny edge map...")

        # Convert to numpy array
        image_np = np.array(image)

        # Convert to grayscale
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Apply Canny edge detection
        edges = cv2.Canny(blurred, low_threshold, high_threshold)

        # Dilate edges slightly for better visibility
        kernel = np.ones((2, 2), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)

        # Convert to RGB (ControlNet expects RGB)
        edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)

        # Convert to PIL and resize
        edges_image = Image.fromarray(edges_rgb)
        edges_image = edges_image.resize(target_size, Image.LANCZOS)

        return edges_image

    async def process_segmentation_map(
        self,
        masks: List[np.ndarray],
        original_size: Tuple[int, int],
        target_size: Tuple[int, int] = (512, 512)
    ) -> Image.Image:
        """
        Convert SAM2 masks to ControlNet-compatible segmentation map

        Args:
            masks: List of binary masks from SAM2
            original_size: Original image size (width, height)
            target_size: Target resolution

        Returns:
            Color-coded segmentation map as PIL Image
        """
        logger.info(f"Processing {len(masks)} segmentation masks...")

        # Create empty segmentation map
        width, height = original_size
        seg_map = np.zeros((height, width, 3), dtype=np.uint8)

        # Define colors for different furniture categories
        # Use distinct colors for better differentiation
        colors = [
            (255, 0, 0),      # Red - Sofas/Chairs
            (0, 255, 0),      # Green - Tables
            (0, 0, 255),      # Blue - Cabinets/Storage
            (255, 255, 0),    # Yellow - Beds
            (255, 0, 255),    # Magenta - Lighting
            (0, 255, 255),    # Cyan - Decorative items
            (128, 0, 0),      # Dark Red - Curtains
            (0, 128, 0),      # Dark Green - Plants
            (0, 0, 128),      # Dark Blue - Electronics
            (128, 128, 0),    # Olive - Rugs
        ]

        # Apply each mask with a different color
        for idx, mask in enumerate(masks):
            if mask.shape[:2] != (height, width):
                # Resize mask to match original image size
                mask = cv2.resize(
                    mask.astype(np.uint8),
                    (width, height),
                    interpolation=cv2.INTER_NEAREST
                )

            # Get color for this mask (cycle through colors)
            color = colors[idx % len(colors)]

            # Apply color to mask region
            for c in range(3):
                seg_map[:, :, c][mask > 0] = color[c]

        # Convert to PIL Image
        seg_image = Image.fromarray(seg_map)

        # Resize to target size
        seg_image = seg_image.resize(target_size, Image.NEAREST)

        return seg_image

    def validate_control_images(
        self,
        controls: Dict[str, Image.Image]
    ) -> Tuple[bool, List[str]]:
        """
        Validate generated control images

        Args:
            controls: Dictionary of control images

        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []

        # Check if at least one control image is present
        valid_controls = [k for k, v in controls.items() if v is not None]
        if len(valid_controls) == 0:
            errors.append("No valid control images generated")
            return False, errors

        # Check image sizes
        target_size = (512, 512)
        for name, img in controls.items():
            if img is not None and img.size != target_size:
                errors.append(f"{name} has incorrect size: {img.size} != {target_size}")

        # Check if depth map has sufficient detail
        if controls.get('depth') is not None:
            depth_np = np.array(controls['depth'].convert('L'))
            if depth_np.std() < 10:
                errors.append("Depth map has insufficient detail (std < 10)")

        # Check if canny edges are not empty
        if controls.get('canny') is not None:
            edges_np = np.array(controls['canny'].convert('L'))
            if edges_np.sum() < 1000:
                errors.append("Canny edge map is too sparse")

        is_valid = len(errors) == 0
        return is_valid, errors


def create_control_image_preview(
    controls: Dict[str, Image.Image],
    save_path: str = None
) -> Image.Image:
    """
    Create a preview image showing all control images side by side

    Args:
        controls: Dictionary of control images
        save_path: Optional path to save the preview

    Returns:
        Combined preview image
    """
    valid_images = [(k, v) for k, v in controls.items() if v is not None]

    if not valid_images:
        raise ValueError("No valid control images to preview")

    # Get dimensions
    width = valid_images[0][1].width
    height = valid_images[0][1].height

    # Create combined image
    combined_width = width * len(valid_images)
    combined = Image.new('RGB', (combined_width, height))

    # Paste each control image
    for idx, (name, img) in enumerate(valid_images):
        combined.paste(img, (idx * width, 0))

    # Save if path provided
    if save_path:
        combined.save(save_path)
        logger.info(f"Control image preview saved to {save_path}")

    return combined


# Example usage
if __name__ == "__main__":
    import asyncio

    async def test_control_generation():
        """Test control image generation"""

        # Load test image
        test_image = Image.open("test_room.jpg")

        # Initialize generator
        generator = ControlImageGenerator(device="auto")

        # Generate controls
        controls = await generator.generate_all_controls(
            test_image,
            segmentation_masks=None,
            target_size=(512, 512)
        )

        # Validate
        is_valid, errors = generator.validate_control_images(controls)

        if is_valid:
            print("✓ All control images generated successfully!")

            # Create preview
            preview = create_control_image_preview(
                controls,
                save_path="control_preview.jpg"
            )
        else:
            print("✗ Control image validation failed:")
            for error in errors:
                print(f"  - {error}")

    # Run test
    asyncio.run(test_control_generation())

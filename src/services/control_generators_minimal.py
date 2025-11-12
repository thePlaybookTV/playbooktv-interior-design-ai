"""
Minimal Control Image Generation Service (for Railway)

This is a lightweight version that only handles basic image operations.
Heavy processing (depth estimation) is done on Modal.

Author: Modomo Team
Date: November 2025
"""

import cv2
import numpy as np
from PIL import Image
from typing import Dict, Tuple
import logging

logger = logging.getLogger(__name__)


class MinimalControlImageGenerator:
    """
    Lightweight control image generator for Railway API

    Only generates:
    - Canny edges (OpenCV, fast)

    Depth maps are generated on Modal (requires transformers + torch)
    """

    def __init__(self):
        """Initialize minimal generator"""
        logger.info("Initializing MinimalControlImageGenerator")

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
        masks: list,
        original_size: Tuple[int, int],
        target_size: Tuple[int, int] = (512, 512)
    ) -> Image.Image:
        """
        Convert SAM2 masks to ControlNet-compatible segmentation map

        Note: Masks come from Modal, this just processes them for display

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


def optimize_image_for_upload(
    image: Image.Image,
    max_size: Tuple[int, int] = (1920, 1080),
    quality: int = 85
) -> Image.Image:
    """
    Optimize image for uploading to Modal

    Args:
        image: Input PIL Image
        max_size: Maximum dimensions
        quality: JPEG quality (1-100)

    Returns:
        Optimized image
    """
    # Convert RGBA to RGB if needed
    if image.mode == 'RGBA':
        rgb_image = Image.new('RGB', image.size, (255, 255, 255))
        rgb_image.paste(image, mask=image.split()[3])
        image = rgb_image

    # Resize if too large
    if image.width > max_size[0] or image.height > max_size[1]:
        image.thumbnail(max_size, Image.LANCZOS)

    return image


# Example usage
if __name__ == "__main__":
    import asyncio

    async def test_minimal_generator():
        """Test minimal control generation"""

        # Load test image
        test_image = Image.open("test_room.jpg")

        # Initialize generator
        generator = MinimalControlImageGenerator()

        # Generate edges only (depth done on Modal)
        edges = await generator.generate_canny_edges(
            test_image,
            target_size=(512, 512)
        )

        # Save result
        edges.save("test_edges.jpg")
        print("✓ Edge map generated successfully!")

    # Run test
    asyncio.run(test_minimal_generator())

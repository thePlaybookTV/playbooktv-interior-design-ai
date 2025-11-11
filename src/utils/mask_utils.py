"""
Mask utility functions for SAM2 segmentation masks
Helps process, filter, and enhance masks for training
"""

import numpy as np
import cv2
from typing import Tuple, Optional, List, Dict
from pathlib import Path
import duckdb


def get_tight_bbox_from_mask(mask: np.ndarray) -> Optional[Tuple[float, float, float, float]]:
    """
    Calculate tight bounding box from mask contour

    Args:
        mask: Binary mask array (H, W) where 1 = foreground

    Returns:
        Tuple of (x1, y1, x2, y2) in pixel coordinates or None if invalid mask
    """
    if mask is None or mask.sum() == 0:
        return None

    # Find contours
    contours, _ = cv2.findContours(
        mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) == 0:
        return None

    # Get bounding box from largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)

    return (float(x), float(y), float(x + w), float(y + h))


def calculate_mask_quality_score(mask: np.ndarray, bbox: Tuple[float, float, float, float]) -> float:
    """
    Calculate quality score for a mask based on coverage of bounding box

    Args:
        mask: Binary mask array
        bbox: Bounding box (x1, y1, x2, y2)

    Returns:
        Quality score 0-1, where 1 = mask perfectly fills bbox
    """
    if mask is None or mask.sum() == 0:
        return 0.0

    x1, y1, x2, y2 = [int(v) for v in bbox]

    # Get mask region within bbox
    bbox_mask = mask[y1:y2, x1:x2]

    # Calculate fill ratio
    mask_area = bbox_mask.sum()
    bbox_area = (x2 - x1) * (y2 - y1)

    if bbox_area == 0:
        return 0.0

    fill_ratio = mask_area / bbox_area

    return min(fill_ratio, 1.0)


def filter_detections_by_mask_quality(
    db_path: str,
    min_mask_score: float = 0.7,
    min_area: int = 100
) -> List[Dict]:
    """
    Get high-quality detections from database

    Args:
        db_path: Path to DuckDB database
        min_mask_score: Minimum SAM2 mask score (0-1)
        min_area: Minimum mask area in pixels

    Returns:
        List of detection dictionaries with all fields
    """
    conn = duckdb.connect(str(db_path))

    results = conn.execute("""
        SELECT
            detection_id,
            image_id,
            item_type,
            confidence,
            bbox_x1, bbox_y1, bbox_x2, bbox_y2,
            area_percentage,
            mask_area,
            mask_score
        FROM furniture_detections
        WHERE has_mask = true
        AND mask_score >= ?
        AND mask_area >= ?
        ORDER BY mask_score DESC
    """, [min_mask_score, min_area]).fetchall()

    conn.close()

    # Convert to list of dicts
    detections = []
    for row in results:
        detections.append({
            'detection_id': row[0],
            'image_id': row[1],
            'item_type': row[2],
            'confidence': row[3],
            'bbox': (row[4], row[5], row[6], row[7]),
            'area_percentage': row[8],
            'mask_area': row[9],
            'mask_score': row[10]
        })

    return detections


def refine_bbox_with_mask(
    original_bbox: Tuple[float, float, float, float],
    mask: np.ndarray,
    padding: float = 0.05
) -> Tuple[float, float, float, float]:
    """
    Refine YOLO bounding box using SAM2 mask for tighter fit

    Args:
        original_bbox: Original YOLO bbox (x1, y1, x2, y2)
        mask: SAM2 binary mask
        padding: Fractional padding to add (0.05 = 5% on each side)

    Returns:
        Refined bounding box (x1, y1, x2, y2)
    """
    tight_bbox = get_tight_bbox_from_mask(mask)

    if tight_bbox is None:
        return original_bbox

    x1, y1, x2, y2 = tight_bbox

    # Add padding
    width = x2 - x1
    height = y2 - y1

    x1 = max(0, x1 - width * padding)
    y1 = max(0, y1 - height * padding)
    x2 = x2 + width * padding
    y2 = y2 + height * padding

    # Ensure within image bounds (if we know them)
    img_h, img_w = mask.shape
    x2 = min(img_w, x2)
    y2 = min(img_h, y2)

    return (x1, y1, x2, y2)


def get_mask_center_of_mass(mask: np.ndarray) -> Optional[Tuple[float, float]]:
    """
    Calculate center of mass of mask

    Args:
        mask: Binary mask array

    Returns:
        (x, y) coordinates of center of mass or None
    """
    if mask is None or mask.sum() == 0:
        return None

    moments = cv2.moments(mask.astype(np.uint8))

    if moments['m00'] == 0:
        return None

    cx = moments['m10'] / moments['m00']
    cy = moments['m01'] / moments['m00']

    return (cx, cy)


def compute_mask_compactness(mask: np.ndarray) -> float:
    """
    Compute compactness (circularity) of mask

    Compactness = (4 * π * area) / (perimeter^2)
    Circle = 1.0, more irregular shapes < 1.0

    Args:
        mask: Binary mask array

    Returns:
        Compactness score 0-1
    """
    if mask is None or mask.sum() == 0:
        return 0.0

    contours, _ = cv2.findContours(
        mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) == 0:
        return 0.0

    contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)

    if perimeter == 0:
        return 0.0

    compactness = (4 * np.pi * area) / (perimeter ** 2)

    return min(compactness, 1.0)


def get_detection_statistics(db_path: str) -> Dict:
    """
    Get overall statistics about detections in database

    Args:
        db_path: Path to DuckDB database

    Returns:
        Dictionary with detection statistics
    """
    conn = duckdb.connect(str(db_path))

    # Total detections
    total = conn.execute("SELECT COUNT(*) FROM furniture_detections").fetchone()[0]

    # Detections with masks
    with_masks = conn.execute(
        "SELECT COUNT(*) FROM furniture_detections WHERE has_mask = true"
    ).fetchone()[0]

    # Average mask score
    avg_score = conn.execute("""
        SELECT AVG(mask_score)
        FROM furniture_detections
        WHERE has_mask = true
    """).fetchone()[0]

    # Average mask area
    avg_area = conn.execute("""
        SELECT AVG(mask_area)
        FROM furniture_detections
        WHERE has_mask = true
    """).fetchone()[0]

    # Detection counts by type
    type_counts = conn.execute("""
        SELECT item_type, COUNT(*) as count
        FROM furniture_detections
        WHERE has_mask = true
        GROUP BY item_type
        ORDER BY count DESC
        LIMIT 20
    """).fetchall()

    conn.close()

    return {
        'total_detections': total,
        'detections_with_masks': with_masks,
        'mask_percentage': (with_masks / total * 100) if total > 0 else 0,
        'avg_mask_score': avg_score or 0.0,
        'avg_mask_area': int(avg_area) if avg_area else 0,
        'top_detected_types': [(t[0], t[1]) for t in type_counts]
    }


def visualize_mask_overlay(
    image: np.ndarray,
    mask: np.ndarray,
    bbox: Optional[Tuple[float, float, float, float]] = None,
    color: Tuple[int, int, int] = (0, 255, 0),
    alpha: float = 0.4
) -> np.ndarray:
    """
    Overlay mask and bounding box on image for visualization

    Args:
        image: RGB image array (H, W, 3)
        mask: Binary mask array (H, W)
        bbox: Optional bounding box (x1, y1, x2, y2)
        color: RGB color for overlay
        alpha: Transparency of mask overlay

    Returns:
        Image with mask overlay
    """
    result = image.copy()

    # Create colored mask overlay
    if mask is not None and mask.sum() > 0:
        mask_colored = np.zeros_like(image)
        mask_colored[mask > 0] = color

        # Blend with original image
        result = cv2.addWeighted(result, 1 - alpha, mask_colored, alpha, 0)

    # Draw bounding box
    if bbox is not None:
        x1, y1, x2, y2 = [int(v) for v in bbox]
        cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)

    return result


def export_masks_for_yolo(
    db_path: str,
    output_dir: Path,
    min_mask_score: float = 0.7,
    format: str = 'yolo_seg'
) -> int:
    """
    Export mask detections in YOLO segmentation format

    YOLO segmentation format:
    <class_id> <x1> <y1> <x2> <y2> ... <xn> <yn>

    Where coordinates are normalized polygon points from mask contour

    Args:
        db_path: Path to DuckDB database
        output_dir: Directory to save label files
        min_mask_score: Minimum mask quality
        format: Export format ('yolo_seg' or 'coco')

    Returns:
        Number of label files created
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get high-quality detections
    detections = filter_detections_by_mask_quality(db_path, min_mask_score)

    # Group by image_id
    images_dict = {}
    for det in detections:
        img_id = det['image_id']
        if img_id not in images_dict:
            images_dict[img_id] = []
        images_dict[img_id].append(det)

    # Export each image's labels
    # Note: This is a placeholder - actual implementation would need:
    # 1. Access to actual mask polygon data
    # 2. Image dimensions for normalization
    # 3. Class ID mapping

    count = 0
    for img_id, img_detections in images_dict.items():
        label_path = output_dir / f"{img_id}.txt"

        # Would write YOLO format labels here
        # Format: <class_id> <x1> <y1> <x2> <y2> ... (normalized polygon points)

        count += 1

    return count

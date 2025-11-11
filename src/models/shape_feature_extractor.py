"""
Shape and Geometry Feature Extractor using SAM2 Masks
Extracts shape-based features from furniture masks for style classification
"""

import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import duckdb
from PIL import Image


class ShapeFeatureExtractor:
    """Extract shape and geometry features from SAM2 masks"""

    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize shape feature extractor

        Args:
            db_path: Optional path to DuckDB for batch processing
        """
        self.db_path = db_path
        if db_path:
            self.conn = duckdb.connect(str(db_path))

    def extract_shape_features(self, mask: np.ndarray) -> Dict[str, float]:
        """
        Extract comprehensive shape features from a single mask

        Args:
            mask: Binary mask array (H, W)

        Returns:
            Dictionary of shape features
        """
        if mask is None or mask.sum() == 0:
            return self._empty_features()

        features = {}

        # Find contours
        contours, _ = cv2.findContours(
            mask.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        if len(contours) == 0:
            return self._empty_features()

        # Use largest contour
        contour = max(contours, key=cv2.contourArea)

        # Basic geometric properties
        features.update(self._compute_basic_geometry(contour, mask))

        # Shape descriptors
        features.update(self._compute_shape_descriptors(contour))

        # Hu moments (rotation invariant)
        features.update(self._compute_hu_moments(mask))

        # Spatial features
        features.update(self._compute_spatial_features(mask))

        return features

    def _compute_basic_geometry(self, contour, mask: np.ndarray) -> Dict[str, float]:
        """Compute basic geometric properties"""

        features = {}

        # Area and perimeter
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)

        features['area'] = area
        features['perimeter'] = perimeter

        # Bounding box
        x, y, w, h = cv2.boundingRect(contour)
        features['bbox_width'] = w
        features['bbox_height'] = h
        features['bbox_area'] = w * h
        features['aspect_ratio'] = w / h if h > 0 else 0

        # Fill ratio (how much of bbox is filled)
        features['fill_ratio'] = area / (w * h) if (w * h) > 0 else 0

        # Compactness (circularity)
        # Circle = 1.0, more irregular shapes < 1.0
        if perimeter > 0:
            features['compactness'] = (4 * np.pi * area) / (perimeter ** 2)
        else:
            features['compactness'] = 0

        # Solidity (area / convex hull area)
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        features['solidity'] = area / hull_area if hull_area > 0 else 0

        # Extent (area / bounding box area)
        features['extent'] = features['fill_ratio']

        return features

    def _compute_shape_descriptors(self, contour) -> Dict[str, float]:
        """Compute advanced shape descriptors"""

        features = {}

        # Fit ellipse (if contour has enough points)
        if len(contour) >= 5:
            try:
                ellipse = cv2.fitEllipse(contour)
                (center_x, center_y), (major_axis, minor_axis), angle = ellipse

                features['ellipse_major_axis'] = max(major_axis, minor_axis)
                features['ellipse_minor_axis'] = min(major_axis, minor_axis)
                features['ellipse_ratio'] = features['ellipse_minor_axis'] / features['ellipse_major_axis'] if features['ellipse_major_axis'] > 0 else 0
                features['ellipse_angle'] = angle

                # Eccentricity
                if major_axis > 0:
                    eccentricity = np.sqrt(1 - (minor_axis / major_axis) ** 2)
                    features['eccentricity'] = eccentricity
                else:
                    features['eccentricity'] = 0

            except:
                features['ellipse_major_axis'] = 0
                features['ellipse_minor_axis'] = 0
                features['ellipse_ratio'] = 0
                features['ellipse_angle'] = 0
                features['eccentricity'] = 0
        else:
            features['ellipse_major_axis'] = 0
            features['ellipse_minor_axis'] = 0
            features['ellipse_ratio'] = 0
            features['ellipse_angle'] = 0
            features['eccentricity'] = 0

        # Convexity defects (complexity of shape)
        hull = cv2.convexHull(contour, returnPoints=False)
        try:
            defects = cv2.convexityDefects(contour, hull)
            if defects is not None:
                # Average depth of defects (how "jagged" the shape is)
                depths = defects[:, 0, 3] / 256.0  # Normalize
                features['avg_convexity_defect'] = np.mean(depths)
                features['max_convexity_defect'] = np.max(depths)
                features['num_convexity_defects'] = len(defects)
            else:
                features['avg_convexity_defect'] = 0
                features['max_convexity_defect'] = 0
                features['num_convexity_defects'] = 0
        except:
            features['avg_convexity_defect'] = 0
            features['max_convexity_defect'] = 0
            features['num_convexity_defects'] = 0

        return features

    def _compute_hu_moments(self, mask: np.ndarray) -> Dict[str, float]:
        """
        Compute Hu moments (rotation, scale, translation invariant)

        These are useful for matching similar shapes regardless of orientation
        """

        moments = cv2.moments(mask.astype(np.uint8))
        hu_moments = cv2.HuMoments(moments)

        # Log transform for better scale
        hu_moments = -np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-10)

        features = {}
        for i, hu in enumerate(hu_moments.flatten()):
            features[f'hu_moment_{i+1}'] = float(hu)

        return features

    def _compute_spatial_features(self, mask: np.ndarray) -> Dict[str, float]:
        """Compute spatial distribution features"""

        features = {}
        h, w = mask.shape

        # Center of mass
        moments = cv2.moments(mask.astype(np.uint8))
        if moments['m00'] > 0:
            cx = moments['m10'] / moments['m00']
            cy = moments['m01'] / moments['m00']

            # Normalized position (0-1)
            features['center_x'] = cx / w
            features['center_y'] = cy / h

            # Distance from image center
            img_center_x = w / 2
            img_center_y = h / 2
            dist = np.sqrt((cx - img_center_x)**2 + (cy - img_center_y)**2)
            max_dist = np.sqrt((w/2)**2 + (h/2)**2)
            features['distance_from_center'] = dist / max_dist

        else:
            features['center_x'] = 0.5
            features['center_y'] = 0.5
            features['distance_from_center'] = 0

        # Coverage (what % of image is occupied)
        features['coverage'] = mask.sum() / (h * w)

        return features

    def _empty_features(self) -> Dict[str, float]:
        """Return empty feature dict with all keys set to 0"""

        return {
            'area': 0, 'perimeter': 0, 'bbox_width': 0, 'bbox_height': 0,
            'bbox_area': 0, 'aspect_ratio': 0, 'fill_ratio': 0,
            'compactness': 0, 'solidity': 0, 'extent': 0,
            'ellipse_major_axis': 0, 'ellipse_minor_axis': 0,
            'ellipse_ratio': 0, 'ellipse_angle': 0, 'eccentricity': 0,
            'avg_convexity_defect': 0, 'max_convexity_defect': 0,
            'num_convexity_defects': 0,
            'hu_moment_1': 0, 'hu_moment_2': 0, 'hu_moment_3': 0,
            'hu_moment_4': 0, 'hu_moment_5': 0, 'hu_moment_6': 0,
            'hu_moment_7': 0,
            'center_x': 0.5, 'center_y': 0.5, 'distance_from_center': 0,
            'coverage': 0
        }

    def extract_room_shape_features(
        self,
        image_id: str,
        min_mask_score: float = 0.7
    ) -> Dict[str, any]:
        """
        Extract aggregated shape features for entire room (all furniture)

        Args:
            image_id: Image ID from database
            min_mask_score: Minimum mask quality

        Returns:
            Dictionary with room-level shape statistics
        """
        if not self.db_path:
            raise ValueError("Database path required for room feature extraction")

        # Get all high-quality detections for this image
        detections = self.conn.execute("""
            SELECT
                detection_id,
                item_type,
                bbox_x1, bbox_y1, bbox_x2, bbox_y2,
                mask_score,
                mask_area
            FROM furniture_detections
            WHERE image_id = ?
            AND has_mask = TRUE
            AND mask_score >= ?
        """, [image_id, min_mask_score]).fetchall()

        if len(detections) == 0:
            return {
                'num_items': 0,
                'avg_compactness': 0,
                'avg_aspect_ratio': 0,
                'avg_solidity': 0,
                'total_coverage': 0,
                'furniture_density': 0
            }

        # Aggregate features
        compactness_values = []
        aspect_ratios = []
        solidity_values = []
        total_area = 0

        for det in detections:
            # For now, compute basic stats from bbox
            # In full implementation, would load actual masks
            x1, y1, x2, y2 = det[2:6]
            width = x2 - x1
            height = y2 - y1
            area = det[7]  # mask_area

            total_area += area

            if height > 0:
                aspect_ratios.append(width / height)

            # Placeholder values - would compute from actual mask
            # compactness_values.append(...)
            # solidity_values.append(...)

        return {
            'num_items': len(detections),
            'avg_aspect_ratio': np.mean(aspect_ratios) if aspect_ratios else 0,
            'total_coverage': total_area,
            'furniture_density': len(detections)  # items per image
        }

    def extract_spatial_arrangement_features(
        self,
        image_id: str,
        min_mask_score: float = 0.7
    ) -> Dict[str, float]:
        """
        Extract features describing spatial arrangement of furniture

        Useful for style classification (e.g., minimalist = sparse, maximalist = dense)

        Args:
            image_id: Image ID from database
            min_mask_score: Minimum mask quality

        Returns:
            Dictionary with spatial arrangement features
        """
        if not self.db_path:
            raise ValueError("Database path required")

        # Get detections with positions
        detections = self.conn.execute("""
            SELECT
                bbox_x1, bbox_y1, bbox_x2, bbox_y2,
                mask_area,
                area_percentage
            FROM furniture_detections
            WHERE image_id = ?
            AND has_mask = TRUE
            AND mask_score >= ?
        """, [image_id, min_mask_score]).fetchall()

        if len(detections) == 0:
            return self._empty_arrangement_features()

        # Calculate centers
        centers = []
        areas = []

        for det in detections:
            x1, y1, x2, y2 = det[0:4]
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            centers.append((cx, cy))
            areas.append(det[4])  # mask_area

        centers = np.array(centers)
        areas = np.array(areas)

        features = {}

        # Density: number of items
        features['item_count'] = len(detections)

        # Average distance between items
        if len(centers) > 1:
            from scipy.spatial.distance import pdist
            distances = pdist(centers)
            features['avg_distance_between_items'] = np.mean(distances)
            features['min_distance_between_items'] = np.min(distances)
            features['std_distance_between_items'] = np.std(distances)
        else:
            features['avg_distance_between_items'] = 0
            features['min_distance_between_items'] = 0
            features['std_distance_between_items'] = 0

        # Coverage variance (how evenly distributed are items)
        features['area_variance'] = np.var(areas)
        features['area_std'] = np.std(areas)

        # Spatial distribution (spread in x and y)
        features['x_spread'] = np.std(centers[:, 0])
        features['y_spread'] = np.std(centers[:, 1])

        # Total coverage
        features['total_coverage_pct'] = sum(det[5] for det in detections)

        return features

    def _empty_arrangement_features(self) -> Dict[str, float]:
        """Return empty arrangement features"""

        return {
            'item_count': 0,
            'avg_distance_between_items': 0,
            'min_distance_between_items': 0,
            'std_distance_between_items': 0,
            'area_variance': 0,
            'area_std': 0,
            'x_spread': 0,
            'y_spread': 0,
            'total_coverage_pct': 0
        }

    def create_shape_vocabulary(
        self,
        n_clusters: int = 50,
        min_mask_score: float = 0.7
    ) -> Dict:
        """
        Create a "visual vocabulary" of common furniture shapes using clustering

        This can be used as a Bag-of-Shapes feature for style classification

        Args:
            n_clusters: Number of shape clusters
            min_mask_score: Minimum mask quality

        Returns:
            Dictionary with cluster centers and statistics
        """
        if not self.db_path:
            raise ValueError("Database path required")

        from sklearn.cluster import KMeans

        print(f"🔍 Creating shape vocabulary with {n_clusters} clusters...")

        # Get all high-quality detections
        detections = self.conn.execute("""
            SELECT detection_id, image_id, item_type
            FROM furniture_detections
            WHERE has_mask = TRUE
            AND mask_score >= ?
        """, [min_mask_score]).fetchall()

        print(f"   Found {len(detections)} detections with high-quality masks")

        # Extract features for each detection
        # NOTE: This is a placeholder - would need actual mask data
        # For now, return structure

        return {
            'n_clusters': n_clusters,
            'num_samples': len(detections),
            'message': 'Shape vocabulary creation requires access to mask pixel data'
        }

    def close(self):
        """Close database connection"""
        if self.db_path and self.conn:
            self.conn.close()


# ============================================
# USAGE EXAMPLES
# ============================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Extract shape features from SAM2 masks')
    parser.add_argument('--db', type=str, required=True,
                        help='Path to DuckDB database')
    parser.add_argument('--image-id', type=str,
                        help='Extract features for specific image')
    parser.add_argument('--min-mask-score', type=float, default=0.7,
                        help='Minimum mask score')

    args = parser.parse_args()

    print("=" * 70)
    print("🔷 SHAPE FEATURE EXTRACTOR")
    print("=" * 70)

    extractor = ShapeFeatureExtractor(args.db)

    if args.image_id:
        # Extract room features
        print(f"\n📊 Extracting features for image: {args.image_id}")

        room_features = extractor.extract_room_shape_features(
            args.image_id,
            args.min_mask_score
        )

        print("\n🏠 Room Shape Features:")
        for key, value in room_features.items():
            print(f"   {key}: {value:.3f}" if isinstance(value, float) else f"   {key}: {value}")

        arrangement_features = extractor.extract_spatial_arrangement_features(
            args.image_id,
            args.min_mask_score
        )

        print("\n📐 Spatial Arrangement Features:")
        for key, value in arrangement_features.items():
            print(f"   {key}: {value:.3f}" if isinstance(value, float) else f"   {key}: {value}")

    else:
        print("\nℹ️  Provide --image-id to extract features for a specific image")

    extractor.close()

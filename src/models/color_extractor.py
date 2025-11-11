"""
Mask-Based Color Extraction for Furniture
Uses SAM2 masks to extract per-furniture color palettes
"""

import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import duckdb
from PIL import Image
from sklearn.cluster import KMeans
import colorsys


class MaskBasedColorExtractor:
    """Extract color features from furniture using SAM2 masks"""

    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize color extractor

        Args:
            db_path: Optional path to DuckDB for batch processing
        """
        self.db_path = db_path
        if db_path:
            self.conn = duckdb.connect(str(db_path))

    def extract_colors_from_mask(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        n_colors: int = 3
    ) -> Dict[str, any]:
        """
        Extract dominant colors from masked region of image

        Args:
            image: RGB image array (H, W, 3)
            mask: Binary mask array (H, W)
            n_colors: Number of dominant colors to extract

        Returns:
            Dictionary with color palette and statistics
        """
        if mask is None or mask.sum() == 0:
            return self._empty_color_features()

        # Extract pixels within mask
        masked_pixels = image[mask > 0]

        if len(masked_pixels) == 0:
            return self._empty_color_features()

        # Downsample if too many pixels (for performance)
        if len(masked_pixels) > 10000:
            indices = np.random.choice(len(masked_pixels), 10000, replace=False)
            masked_pixels = masked_pixels[indices]

        # Run KMeans clustering
        kmeans = KMeans(n_clusters=min(n_colors, len(masked_pixels)), random_state=42, n_init=10)
        kmeans.fit(masked_pixels)

        # Get cluster centers (colors) and their frequencies
        colors = kmeans.cluster_centers_.astype(int)
        labels = kmeans.labels_
        unique, counts = np.unique(labels, return_counts=True)

        # Sort by frequency
        sorted_indices = np.argsort(-counts)
        colors = colors[sorted_indices]
        frequencies = counts[sorted_indices] / counts.sum()

        # Convert to hex
        color_hex = [self._rgb_to_hex(color) for color in colors]

        # Compute color statistics
        color_stats = self._compute_color_statistics(colors, frequencies)

        return {
            'palette': color_hex,
            'palette_rgb': colors.tolist(),
            'frequencies': frequencies.tolist(),
            'dominant_color': color_hex[0],
            'dominant_color_rgb': colors[0].tolist(),
            **color_stats
        }

    def _compute_color_statistics(
        self,
        colors: np.ndarray,
        frequencies: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute statistical color features

        Args:
            colors: RGB colors array (N, 3)
            frequencies: Frequency of each color

        Returns:
            Dictionary with color statistics
        """
        stats = {}

        # Average color (weighted by frequency)
        avg_color = np.average(colors, axis=0, weights=frequencies)
        stats['avg_r'] = int(avg_color[0])
        stats['avg_g'] = int(avg_color[1])
        stats['avg_b'] = int(avg_color[2])

        # Brightness (luminance)
        luminance = 0.299 * avg_color[0] + 0.587 * avg_color[1] + 0.114 * avg_color[2]
        stats['brightness'] = luminance / 255.0  # Normalize to 0-1

        # Convert to HSV for more features
        hsv_colors = []
        for color in colors:
            r, g, b = color / 255.0
            h, s, v = colorsys.rgb_to_hsv(r, g, b)
            hsv_colors.append([h, s, v])

        hsv_colors = np.array(hsv_colors)

        # Average HSV (weighted)
        avg_hsv = np.average(hsv_colors, axis=0, weights=frequencies)
        stats['avg_hue'] = avg_hsv[0]
        stats['avg_saturation'] = avg_hsv[1]
        stats['avg_value'] = avg_hsv[2]

        # Color diversity (std of hue)
        stats['hue_std'] = np.std(hsv_colors[:, 0])
        stats['saturation_std'] = np.std(hsv_colors[:, 1])

        # Color temperature (warm vs cool)
        # Red/Orange/Yellow = warm (hue 0-60), Blue/Green = cool (hue 180-300)
        warm_weight = sum(
            freq for (h, s, v), freq in zip(hsv_colors, frequencies)
            if h < 0.17 or h > 0.83  # Red to yellow range
        )
        stats['warm_cool_ratio'] = warm_weight  # >0.5 = warm, <0.5 = cool

        # Colorfulness (saturation * value)
        colorfulness = np.average(hsv_colors[:, 1] * hsv_colors[:, 2], weights=frequencies)
        stats['colorfulness'] = colorfulness

        # Dominant color category
        dominant_hue = hsv_colors[0, 0]
        stats['color_category'] = self._hue_to_category(dominant_hue)

        return stats

    def _hue_to_category(self, hue: float) -> int:
        """
        Convert hue to color category

        Categories:
        0: Red, 1: Orange, 2: Yellow, 3: Green, 4: Cyan, 5: Blue, 6: Purple, 7: Pink, 8: Neutral

        Args:
            hue: Hue value 0-1

        Returns:
            Category index
        """
        # Hue is in range [0, 1] where 0 and 1 are red
        if hue < 0.05 or hue > 0.95:
            return 0  # Red
        elif hue < 0.12:
            return 1  # Orange
        elif hue < 0.18:
            return 2  # Yellow
        elif hue < 0.45:
            return 3  # Green
        elif hue < 0.55:
            return 4  # Cyan
        elif hue < 0.75:
            return 5  # Blue
        elif hue < 0.85:
            return 6  # Purple
        else:
            return 7  # Pink

    def _rgb_to_hex(self, rgb: np.ndarray) -> str:
        """Convert RGB to hex color string"""
        return '#{:02x}{:02x}{:02x}'.format(int(rgb[0]), int(rgb[1]), int(rgb[2]))

    def _empty_color_features(self) -> Dict:
        """Return empty color features"""
        return {
            'palette': [],
            'palette_rgb': [],
            'frequencies': [],
            'dominant_color': '#000000',
            'dominant_color_rgb': [0, 0, 0],
            'avg_r': 0, 'avg_g': 0, 'avg_b': 0,
            'brightness': 0,
            'avg_hue': 0, 'avg_saturation': 0, 'avg_value': 0,
            'hue_std': 0, 'saturation_std': 0,
            'warm_cool_ratio': 0.5,
            'colorfulness': 0,
            'color_category': 8  # Neutral
        }

    def extract_room_color_features(
        self,
        image_path: str,
        image_id: str,
        min_mask_score: float = 0.7
    ) -> Dict[str, any]:
        """
        Extract color features for entire room (furniture + background)

        Args:
            image_path: Path to image file
            image_id: Image ID from database
            min_mask_score: Minimum mask quality

        Returns:
            Dictionary with room color features and per-furniture colors
        """
        if not self.db_path:
            raise ValueError("Database path required")

        # Load image
        image = np.array(Image.open(image_path).convert('RGB'))

        # Get all detections with masks
        detections = self.conn.execute("""
            SELECT
                detection_id,
                item_type,
                bbox_x1, bbox_y1, bbox_x2, bbox_y2,
                mask_score
            FROM furniture_detections
            WHERE image_id = ?
            AND has_mask = TRUE
            AND mask_score >= ?
        """, [image_id, min_mask_score]).fetchall()

        if len(detections) == 0:
            # No furniture detected - extract whole room colors
            whole_mask = np.ones((image.shape[0], image.shape[1]), dtype=bool)
            room_colors = self.extract_colors_from_mask(image, whole_mask, n_colors=5)
            return {
                'room_palette': room_colors['palette'],
                'room_brightness': room_colors['brightness'],
                'room_colorfulness': room_colors['colorfulness'],
                'furniture_colors': []
            }

        # Create combined furniture mask
        combined_mask = np.zeros((image.shape[0], image.shape[1]), dtype=bool)

        # Extract colors for each furniture item
        furniture_colors = []

        for det in detections:
            detection_id = det[0]
            item_type = det[1]
            x1, y1, x2, y2 = [int(v) for v in det[2:6]]

            # Create bbox mask (placeholder - would use actual SAM2 mask)
            mask = np.zeros_like(combined_mask)
            mask[y1:y2, x1:x2] = True

            # Extract colors for this furniture item
            colors = self.extract_colors_from_mask(image, mask, n_colors=3)

            furniture_colors.append({
                'detection_id': detection_id,
                'item_type': item_type,
                'colors': colors
            })

            # Add to combined mask
            combined_mask |= mask

        # Extract background colors (inverse of furniture mask)
        background_mask = ~combined_mask
        background_colors = self.extract_colors_from_mask(image, background_mask, n_colors=5)

        # Extract overall room colors
        whole_mask = np.ones_like(combined_mask)
        room_colors = self.extract_colors_from_mask(image, whole_mask, n_colors=5)

        return {
            'room_palette': room_colors['palette'],
            'room_brightness': room_colors['brightness'],
            'room_colorfulness': room_colors['colorfulness'],
            'room_warm_cool': room_colors['warm_cool_ratio'],
            'background_palette': background_colors['palette'],
            'background_brightness': background_colors['brightness'],
            'furniture_colors': furniture_colors,
            'num_furniture_items': len(furniture_colors)
        }

    def compute_color_harmony_score(self, colors_rgb: List[Tuple[int, int, int]]) -> float:
        """
        Compute color harmony score based on color theory

        Checks for complementary, analogous, or triadic color schemes

        Args:
            colors_rgb: List of RGB tuples

        Returns:
            Harmony score 0-1 (1 = highly harmonious)
        """
        if len(colors_rgb) < 2:
            return 1.0

        # Convert to HSV
        hsv_colors = []
        for r, g, b in colors_rgb:
            h, s, v = colorsys.rgb_to_hsv(r/255, g/255, b/255)
            hsv_colors.append((h, s, v))

        # Check for complementary colors (opposite on color wheel)
        complementary_score = 0
        for i in range(len(hsv_colors)):
            for j in range(i+1, len(hsv_colors)):
                hue_diff = abs(hsv_colors[i][0] - hsv_colors[j][0])
                # Complementary if ~180 degrees apart (0.5 in 0-1 range)
                if 0.4 < hue_diff < 0.6:
                    complementary_score += 1

        # Check for analogous colors (adjacent on color wheel)
        analogous_score = 0
        for i in range(len(hsv_colors)):
            for j in range(i+1, len(hsv_colors)):
                hue_diff = abs(hsv_colors[i][0] - hsv_colors[j][0])
                # Analogous if close (< 60 degrees = 0.17)
                if hue_diff < 0.17:
                    analogous_score += 1

        # Normalize scores
        max_pairs = len(hsv_colors) * (len(hsv_colors) - 1) / 2
        complementary_norm = complementary_score / max_pairs if max_pairs > 0 else 0
        analogous_norm = analogous_score / max_pairs if max_pairs > 0 else 0

        # Harmony is presence of either scheme
        harmony = max(complementary_norm, analogous_norm)

        return min(harmony, 1.0)

    def create_color_histogram_features(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        bins: int = 8
    ) -> np.ndarray:
        """
        Create color histogram features for the masked region

        Useful for ML models

        Args:
            image: RGB image array
            mask: Binary mask
            bins: Number of bins per channel

        Returns:
            Flattened histogram array (bins^3 dimensions)
        """
        if mask is None or mask.sum() == 0:
            return np.zeros(bins ** 3)

        # Extract masked pixels
        masked_pixels = image[mask > 0]

        if len(masked_pixels) == 0:
            return np.zeros(bins ** 3)

        # Compute 3D histogram (R, G, B)
        hist, _ = np.histogramdd(
            masked_pixels,
            bins=(bins, bins, bins),
            range=((0, 256), (0, 256), (0, 256))
        )

        # Normalize
        hist = hist / hist.sum()

        return hist.flatten()

    def batch_extract_furniture_colors(
        self,
        output_path: Optional[str] = None,
        min_mask_score: float = 0.7,
        limit: Optional[int] = None
    ) -> List[Dict]:
        """
        Extract colors for all furniture items in database

        Args:
            output_path: Optional path to save results as JSON
            min_mask_score: Minimum mask quality
            limit: Optional limit on number of items to process

        Returns:
            List of color feature dictionaries
        """
        if not self.db_path:
            raise ValueError("Database path required")

        print("🎨 Batch extracting furniture colors...")

        # Get all detections
        query = """
            SELECT
                fd.detection_id,
                fd.image_id,
                fd.item_type,
                i.original_path,
                fd.bbox_x1, fd.bbox_y1, fd.bbox_x2, fd.bbox_y2
            FROM furniture_detections fd
            INNER JOIN images i ON fd.image_id = i.image_id
            WHERE fd.has_mask = TRUE
            AND fd.mask_score >= ?
            AND i.original_path IS NOT NULL
        """

        if limit:
            query += f" LIMIT {limit}"

        detections = self.conn.execute(query, [min_mask_score]).fetchall()

        print(f"   Found {len(detections)} furniture items to process")

        results = []

        for det in detections:
            detection_id, image_id, item_type, image_path = det[0:4]
            x1, y1, x2, y2 = [int(v) for v in det[4:8]]

            try:
                # Load image
                image = np.array(Image.open(image_path).convert('RGB'))

                # Create bbox mask (placeholder - would use actual SAM2 mask)
                mask = np.zeros((image.shape[0], image.shape[1]), dtype=bool)
                mask[y1:y2, x1:x2] = True

                # Extract colors
                colors = self.extract_colors_from_mask(image, mask, n_colors=3)

                result = {
                    'detection_id': detection_id,
                    'image_id': image_id,
                    'item_type': item_type,
                    **colors
                }

                results.append(result)

            except Exception as e:
                print(f"   ⚠️  Error processing {detection_id}: {e}")
                continue

        print(f"✅ Extracted colors for {len(results)} items")

        # Save if requested
        if output_path:
            import json
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"💾 Saved to {output_path}")

        return results

    def close(self):
        """Close database connection"""
        if self.db_path and self.conn:
            self.conn.close()


# ============================================
# USAGE EXAMPLES
# ============================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Extract mask-based color features')
    parser.add_argument('--db', type=str, required=True,
                        help='Path to DuckDB database')
    parser.add_argument('--image-id', type=str,
                        help='Extract colors for specific image')
    parser.add_argument('--image-path', type=str,
                        help='Path to image file')
    parser.add_argument('--batch', action='store_true',
                        help='Batch process all furniture items')
    parser.add_argument('--output', type=str,
                        help='Output JSON file for batch processing')
    parser.add_argument('--limit', type=int,
                        help='Limit number of items to process')
    parser.add_argument('--min-mask-score', type=float, default=0.7,
                        help='Minimum mask score')

    args = parser.parse_args()

    print("=" * 70)
    print("🎨 MASK-BASED COLOR EXTRACTOR")
    print("=" * 70)

    extractor = MaskBasedColorExtractor(args.db)

    if args.batch:
        # Batch process
        results = extractor.batch_extract_furniture_colors(
            output_path=args.output,
            min_mask_score=args.min_mask_score,
            limit=args.limit
        )

        print(f"\n✅ Processed {len(results)} furniture items")

    elif args.image_id and args.image_path:
        # Single image
        print(f"\n🖼️  Extracting colors for: {args.image_id}")

        room_colors = extractor.extract_room_color_features(
            args.image_path,
            args.image_id,
            args.min_mask_score
        )

        print("\n🏠 Room Colors:")
        print(f"   Palette: {room_colors['room_palette']}")
        print(f"   Brightness: {room_colors['room_brightness']:.3f}")
        print(f"   Colorfulness: {room_colors['room_colorfulness']:.3f}")
        print(f"   Warm/Cool: {room_colors['room_warm_cool']:.3f}")

        print(f"\n🪑 Furniture Colors ({len(room_colors['furniture_colors'])} items):")
        for item in room_colors['furniture_colors'][:5]:  # Show first 5
            print(f"   {item['item_type']}: {item['colors']['dominant_color']}")

    else:
        print("\nℹ️  Provide --batch for batch processing or --image-id + --image-path for single image")

    extractor.close()

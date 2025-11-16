"""
Quality Validator for SD-Generated Images
Validates output quality and provides retry recommendations
"""

import numpy as np
from PIL import Image
from typing import Dict, Tuple
import logging

logger = logging.getLogger(__name__)


class QualityValidator:
    """Validates quality of generated interior design images"""

    def __init__(self, min_score: float = 0.75, use_clip_scorer: bool = False):
        """
        Initialize validator

        Args:
            min_score: Minimum acceptable quality score (0-1)
            use_clip_scorer: Whether to use CLIP-based aesthetic scoring (requires model loading)
        """
        self.min_score = min_score
        self.use_clip_scorer = use_clip_scorer
        self.clip_model = None
        self.clip_processor = None

        if use_clip_scorer:
            self._load_clip_scorer()

    def _load_clip_scorer(self):
        """Load CLIP model for aesthetic scoring"""
        try:
            from transformers import CLIPProcessor, CLIPModel
            import torch

            logger.info("Loading CLIP model for aesthetic scoring...")
            self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

            # Move to GPU if available
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.clip_model = self.clip_model.to(device)
            self.clip_model.eval()

            logger.info(f"✓ CLIP model loaded on {device}")

        except Exception as e:
            logger.warning(f"Failed to load CLIP model: {e}. Falling back to heuristics.")
            self.use_clip_scorer = False

    def _calculate_clip_aesthetic_score(self, image: Image.Image, style: str) -> float:
        """
        Calculate aesthetic score using CLIP

        Compares the image against positive/negative aesthetic prompts
        specific to interior design.

        Args:
            image: Generated image
            style: Style name

        Returns:
            Aesthetic score (0-1)
        """
        if not self.use_clip_scorer or self.clip_model is None:
            return 0.5  # Neutral score if CLIP not available

        try:
            import torch

            # Aesthetic quality prompts for interior design
            positive_prompts = [
                f"professional high-quality {style} interior design photograph",
                "beautiful, well-lit, harmonious interior space",
                "aesthetically pleasing room with good composition",
                "realistic interior design with excellent details"
            ]

            negative_prompts = [
                "blurry, low quality interior photograph",
                "distorted, unrealistic room",
                "artificial, fake-looking interior",
                "poorly composed cluttered space"
            ]

            # Process image and text
            device = self.clip_model.device

            inputs = self.clip_processor(
                text=positive_prompts + negative_prompts,
                images=image,
                return_tensors="pt",
                padding=True
            ).to(device)

            # Get CLIP scores
            with torch.no_grad():
                outputs = self.clip_model(**inputs)
                logits_per_image = outputs.logits_per_image
                probs = logits_per_image.softmax(dim=1).cpu().numpy()[0]

            # Calculate score as ratio of positive to total
            num_positive = len(positive_prompts)
            positive_score = probs[:num_positive].sum()
            negative_score = probs[num_positive:].sum()

            # Normalize to 0-1 range
            aesthetic_score = positive_score / (positive_score + negative_score)

            logger.info(f"CLIP aesthetic score: {aesthetic_score:.3f}")
            return float(aesthetic_score)

        except Exception as e:
            logger.warning(f"CLIP scoring failed: {e}")
            return 0.5  # Neutral score on error

    def validate_result(
        self,
        generated_image: Image.Image,
        original_image: Image.Image,
        style: str
    ) -> Dict:
        """
        Comprehensive quality validation

        Args:
            generated_image: The SD-generated result
            original_image: The original input image
            style: The requested style

        Returns:
            Dictionary with validation results:
            {
                'passed': bool,
                'score': float,
                'reason': str,
                'checks': dict,
                'retry_recommended': bool
            }
        """
        checks = {}

        # Check 1: Image not blank/uniform
        checks['not_blank'] = self._check_not_blank(generated_image)

        # Check 2: Reasonable color distribution
        checks['color_variance'] = self._check_color_variance(generated_image)

        # Check 3: No major artifacts
        checks['no_artifacts'] = self._detect_artifacts(generated_image)

        # Check 4: Structural similarity to original (room layout preserved)
        checks['structural_similarity'] = self._check_structural_similarity(
            generated_image, original_image
        )

        # Check 5: Image sharpness/detail
        checks['sharpness'] = self._check_sharpness(generated_image)

        # Check 6: CLIP aesthetic score (if enabled)
        if self.use_clip_scorer:
            checks['aesthetic'] = self._calculate_clip_aesthetic_score(generated_image, style)
            weights = {
                'not_blank': 0.20,
                'color_variance': 0.10,
                'no_artifacts': 0.20,
                'structural_similarity': 0.20,
                'sharpness': 0.10,
                'aesthetic': 0.20  # CLIP aesthetic score
            }
        else:
            # Calculate overall score (weighted average) - without CLIP
            weights = {
                'not_blank': 0.25,
                'color_variance': 0.15,
                'no_artifacts': 0.25,
                'structural_similarity': 0.25,
                'sharpness': 0.10
            }

        overall_score = sum(
            checks[key] * weights[key]
            for key in weights.keys()
        )

        # Determine pass/fail
        passed = overall_score >= self.min_score

        # Determine reason if failed
        if not passed:
            reason = self._get_failure_reason(checks, weights)
        else:
            reason = "Quality check passed"

        # Recommend retry if quality is poor but not terrible
        retry_recommended = 0.5 < overall_score < self.min_score

        return {
            'passed': passed,
            'score': overall_score,
            'reason': reason,
            'checks': checks,
            'retry_recommended': retry_recommended
        }

    def _check_not_blank(self, image: Image.Image) -> float:
        """
        Check if image has sufficient detail (not blank/uniform)

        Returns:
            Score 0-1 (1 = good detail)
        """
        image_array = np.array(image)

        # Calculate standard deviation across all channels
        std_dev = np.std(image_array)

        # If std dev is very low, image is too uniform
        if std_dev < 5:
            return 0.0
        elif std_dev < 15:
            return 0.3
        elif std_dev < 25:
            return 0.6
        else:
            return 1.0

    def _check_color_variance(self, image: Image.Image) -> float:
        """
        Check for reasonable color distribution

        Returns:
            Score 0-1 (1 = good color variance)
        """
        image_array = np.array(image)

        # Calculate variance per channel
        variances = []
        for channel in range(3):  # RGB
            var = np.var(image_array[:, :, channel])
            variances.append(var)

        avg_variance = np.mean(variances)

        # Normalize variance to 0-1 scale
        # Good interior images typically have variance in range 500-3000
        if avg_variance < 100:
            return 0.0
        elif avg_variance < 500:
            return 0.4
        elif avg_variance < 3000:
            return 1.0
        else:
            # Too much variance might indicate noise
            return 0.8

    def _detect_artifacts(self, image: Image.Image) -> float:
        """
        Detect common AI artifacts

        Returns:
            Score 0-1 (1 = no artifacts detected)
        """
        image_array = np.array(image)

        # Check for unusual patterns
        # 1. Check for extreme values (pure black/white patches)
        extreme_pixels = np.sum((image_array < 10) | (image_array > 245))
        total_pixels = image_array.size
        extreme_ratio = extreme_pixels / total_pixels

        if extreme_ratio > 0.3:
            return 0.2  # Too many extreme values
        elif extreme_ratio > 0.15:
            return 0.6
        else:
            return 1.0

    def _check_structural_similarity(
        self,
        generated: Image.Image,
        original: Image.Image
    ) -> float:
        """
        Check if room structure is preserved
        Uses simple pixel-wise comparison as proxy

        Returns:
            Score 0-1 (1 = structure well preserved)
        """
        try:
            # Resize both to same size for comparison
            size = (256, 256)
            gen_resized = generated.resize(size, Image.LANCZOS)
            orig_resized = original.resize(size, Image.LANCZOS)

            # Convert to grayscale for structural comparison
            gen_gray = np.array(gen_resized.convert('L'))
            orig_gray = np.array(orig_resized.convert('L'))

            # Calculate edge maps (approximate structural features)
            gen_edges = self._simple_edge_detect(gen_gray)
            orig_edges = self._simple_edge_detect(orig_gray)

            # Compare edge maps
            edge_similarity = 1 - (np.abs(gen_edges - orig_edges).mean() / 255.0)

            return max(0.0, edge_similarity)

        except Exception as e:
            logger.warning(f"Structural similarity check failed: {e}")
            return 0.7  # Default to acceptable

    def _simple_edge_detect(self, gray_image: np.ndarray) -> np.ndarray:
        """
        Simple edge detection using gradients

        Args:
            gray_image: Grayscale image array

        Returns:
            Edge map
        """
        # Compute gradients
        grad_x = np.abs(np.diff(gray_image, axis=1, prepend=0))
        grad_y = np.abs(np.diff(gray_image, axis=0, prepend=0))

        # Combine gradients
        edges = np.sqrt(grad_x**2 + grad_y**2)

        return edges

    def _check_sharpness(self, image: Image.Image) -> float:
        """
        Check image sharpness using Laplacian variance

        Returns:
            Score 0-1 (1 = sharp)
        """
        # Convert to grayscale
        gray = np.array(image.convert('L'))

        # Simple Laplacian kernel
        laplacian_kernel = np.array([[0, 1, 0],
                                     [1, -4, 1],
                                     [0, 1, 0]])

        # Convolve (simple implementation)
        h, w = gray.shape
        laplacian = np.zeros_like(gray, dtype=float)

        for i in range(1, h - 1):
            for j in range(1, w - 1):
                region = gray[i-1:i+2, j-1:j+2]
                laplacian[i, j] = np.sum(region * laplacian_kernel)

        # Variance of Laplacian (measure of sharpness)
        variance = np.var(laplacian)

        # Normalize to 0-1
        # Good sharpness typically has variance > 100
        if variance < 10:
            return 0.0
        elif variance < 50:
            return 0.5
        elif variance < 100:
            return 0.8
        else:
            return 1.0

    def _get_failure_reason(self, checks: Dict, weights: Dict) -> str:
        """
        Determine the main reason for quality failure

        Args:
            checks: Dictionary of check results
            weights: Dictionary of check weights

        Returns:
            Human-readable failure reason
        """
        # Find the check with lowest score (weighted)
        worst_check = min(
            checks.items(),
            key=lambda x: x[1] * weights[x[0]]
        )

        reasons = {
            'not_blank': "Image appears too uniform or blank",
            'color_variance': "Insufficient color variation",
            'no_artifacts': "AI artifacts detected in output",
            'structural_similarity': "Room structure not preserved from original",
            'sharpness': "Image lacks detail/sharpness"
        }

        return reasons.get(worst_check[0], "Quality below threshold")

    def suggest_retry_params(
        self,
        validation_result: Dict,
        current_params: Dict
    ) -> Dict:
        """
        Suggest parameter adjustments for retry

        Args:
            validation_result: Result from validate_result()
            current_params: Current generation parameters

        Returns:
            Suggested parameters for retry
        """
        new_params = current_params.copy()
        checks = validation_result['checks']

        # If structural similarity is low, increase ControlNet strength
        if checks['structural_similarity'] < 0.6:
            if 'controlnet_conditioning_scale' in new_params:
                scales = new_params['controlnet_conditioning_scale']
                new_params['controlnet_conditioning_scale'] = [
                    min(s + 0.1, 1.0) for s in scales
                ]

        # If sharpness is low, adjust guidance scale
        if checks['sharpness'] < 0.6:
            if 'guidance_scale' in new_params:
                new_params['guidance_scale'] = min(
                    new_params['guidance_scale'] + 0.5,
                    10.0
                )

        # If artifacts detected, try different seed
        if checks['no_artifacts'] < 0.7:
            import random
            new_params['seed'] = random.randint(0, 999999)

        return new_params

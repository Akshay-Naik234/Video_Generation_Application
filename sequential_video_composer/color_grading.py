"""Professional color grading effects for video clips."""

import numpy as np


class ColorGrading:
    """Professional color grading effects.
    
    Supports both global color grading (one grade for all images) and
    section-aware grading where each image gets a color treatment
    matched to its narrative section and emotional tone.
    """

    # Mapping from narrative sections to recommended color grades
    SECTION_GRADE_MAP = {
        'COLD_OPEN': 'cool',
        'EARLY_LIFE': 'warm',
        'THE_SPARK': 'documentary',
        'THE_RISE': 'cinematic',
        'THE_CONFLICT': 'high_contrast',
        'THE_CLIMAX': 'dramatic',
        'THE_FALL': 'soft',
        'LEGACY': 'warm',
        'CTA': 'natural',
    }

    # Mapping from emotional tones to color grade overrides
    EMOTION_GRADE_MAP = {
        'tension': 'high_contrast',
        'nostalgia': 'vintage',
        'hope': 'warm',
        'darkness': 'dramatic',
        'devastation': 'cool',
        'triumph': 'cinematic',
        'bittersweet': 'soft',
    }

    GRADE_TYPES = [
        'cinematic',
        'documentary',
        'vintage',
        'modern',
        'warm',
        'cool',
        'high_contrast',
        'soft',
        'dramatic',
        'natural'
    ]

    def apply_grade(self, image: np.ndarray, grade_type: str) -> np.ndarray:
        """Apply color grading to an image."""
        if grade_type == 'cinematic':
            return self._cinematic_grade(image)
        elif grade_type == 'documentary':
            return self._documentary_grade(image)
        elif grade_type == 'vintage':
            return self._vintage_grade(image)
        elif grade_type == 'modern':
            return self._modern_grade(image)
        elif grade_type == 'warm':
            return self._warm_grade(image)
        elif grade_type == 'cool':
            return self._cool_grade(image)
        elif grade_type == 'high_contrast':
            return self._high_contrast_grade(image)
        elif grade_type == 'soft':
            return self._soft_grade(image)
        elif grade_type == 'dramatic':
            return self._dramatic_grade(image)
        elif grade_type == 'natural':
            return self._natural_grade(image)
        else:
            return image

    def _cinematic_grade(self, image: np.ndarray) -> np.ndarray:
        """Cinematic color grading with enhanced shadows and highlights."""
        img = image.astype(np.float64)
        shadows = np.where(img < 128, img + 10, img)
        highlights = np.where(shadows > 180, shadows * 0.95, shadows)
        result = (highlights - 128) * 1.15 + 128
        return np.clip(result, 0, 255).astype(np.uint8)

    def _documentary_grade(self, image: np.ndarray) -> np.ndarray:
        """Documentary style with natural contrast."""
        img = image.astype(np.float64)
        img = (img - 128) * 1.08 + 128
        img[:, :, 0] *= 1.02
        img[:, :, 2] *= 0.98
        return np.clip(img, 0, 255).astype(np.uint8)

    def _vintage_grade(self, image: np.ndarray) -> np.ndarray:
        """Vintage/retro color grading."""
        img = image.astype(np.float64)
        img[:, :, 0] *= 1.1
        img[:, :, 1] *= 1.05
        img[:, :, 2] *= 0.9
        img = (img - 128) * 0.9 + 128
        return np.clip(img, 0, 255).astype(np.uint8)

    def _modern_grade(self, image: np.ndarray) -> np.ndarray:
        """Modern high-contrast look."""
        img = image.astype(np.float64)
        img = (img - 128) * 1.25 + 128
        gray = np.dot(img, [0.299, 0.587, 0.114])
        img = img * 0.85 + gray[..., np.newaxis] * 0.15
        return np.clip(img, 0, 255).astype(np.uint8)

    def _warm_grade(self, image: np.ndarray) -> np.ndarray:
        """Warm, golden tones."""
        img = image.astype(np.float64)
        img[:, :, 0] *= 1.08
        img[:, :, 1] *= 1.03
        img[:, :, 2] *= 0.92
        return np.clip(img, 0, 255).astype(np.uint8)

    def _cool_grade(self, image: np.ndarray) -> np.ndarray:
        """Cool, blue tones."""
        img = image.astype(np.float64)
        img[:, :, 0] *= 0.95
        img[:, :, 1] *= 1.0
        img[:, :, 2] *= 1.1
        return np.clip(img, 0, 255).astype(np.uint8)

    def _high_contrast_grade(self, image: np.ndarray) -> np.ndarray:
        """High contrast dramatic look."""
        img = image.astype(np.float64)
        img = (img - 128) * 1.4 + 128
        img = np.where(img < 40, img * 0.5, img)
        img = np.where(img > 215, 215 + (img - 215) * 0.3, img)
        return np.clip(img, 0, 255).astype(np.uint8)

    def _soft_grade(self, image: np.ndarray) -> np.ndarray:
        """Soft, dreamy look."""
        img = image.astype(np.float64)
        img = (img - 128) * 0.85 + 128 + 15
        return np.clip(img, 0, 255).astype(np.uint8)

    def _dramatic_grade(self, image: np.ndarray) -> np.ndarray:
        """Dramatic, intense look."""
        img = image.astype(np.float64)
        img = (img - 128) * 1.3 + 128
        img[:, :, 0] *= 1.05
        return np.clip(img, 0, 255).astype(np.uint8)

    def _natural_grade(self, image: np.ndarray) -> np.ndarray:
        """Natural, balanced look."""
        img = image.astype(np.float64)
        img = (img - 128) * 1.05 + 128
        return np.clip(img, 0, 255).astype(np.uint8)

    def normalize_brightness(self, image: np.ndarray, target_mean: float = 110.0, strength: float = 0.5) -> np.ndarray:
        """Normalize image brightness toward a target mean to prevent jarring jumps.

        AI-generated images can vary wildly in brightness. This gently nudges
        each image toward a consistent baseline so adjacent clips don't create
        a distracting bright→dark→bright flicker pattern.

        Args:
            image: Input image as numpy array (H, W, 3)
            target_mean: Target mean brightness (0-255). 110 is neutral cinematic.
            strength: How aggressively to normalize (0.0=none, 1.0=full). 0.5 keeps
                      the original character while reducing extreme outliers.
        """
        img = image.astype(np.float64)
        current_mean = img.mean()
        if current_mean < 1.0:
            return image  # Nearly black image, don't adjust

        # Calculate adjustment factor, clamped to avoid extreme shifts
        ratio = target_mean / current_mean
        ratio = max(0.7, min(1.4, ratio))  # Never shift more than 40%

        # Blend between original and adjusted based on strength
        adjusted = img * (1.0 - strength + strength * ratio)
        return np.clip(adjusted, 0, 255).astype(np.uint8)

    def smooth_color_transition(self, prev_image: np.ndarray, curr_image: np.ndarray, blend_strength: float = 0.15) -> np.ndarray:
        """Smooth color continuity between adjacent images.

        Prevents jarring color temperature jumps between consecutive clips by
        blending a small portion of the previous image's color characteristics
        into the current image. This mimics how professional editors use
        color continuity to maintain visual flow.

        Args:
            prev_image: Previous image as numpy array (H, W, 3)
            curr_image: Current image as numpy array (H, W, 3)
            blend_strength: How much of prev color to blend (0.0-0.3). Higher
                           values create smoother transitions but may wash out
                           intentional color changes between sections.
        """
        prev_mean = prev_image.astype(np.float64).mean(axis=(0, 1))  # Per-channel mean
        curr_mean = curr_image.astype(np.float64).mean(axis=(0, 1))

        # Only smooth if the difference is noticeable but not intentional
        diff = np.abs(prev_mean - curr_mean).mean()
        if diff < 5.0 or diff > 60.0:
            # Too similar (no need) or too different (probably intentional section change)
            return curr_image

        # Blend channel means gently
        target_mean = prev_mean * blend_strength + curr_mean * (1.0 - blend_strength)
        img = curr_image.astype(np.float64)

        for c in range(3):
            if curr_mean[c] > 1.0:
                ratio = target_mean[c] / curr_mean[c]
                ratio = max(0.85, min(1.15, ratio))
                img[:, :, c] *= ratio

        return np.clip(img, 0, 255).astype(np.uint8)

    def auto_grade_for_section(self, image: np.ndarray, section: str, emotional_tone: str = '') -> np.ndarray:
        """Automatically select and apply color grading based on section and emotional tone.
        
        Priority: emotional_tone override > section mapping > no grading.
        This enables each image to have its own color treatment that matches
        the narrative arc without manual configuration.
        
        Args:
            image: Input image as numpy array
            section: Narrative section name (e.g., 'COLD_OPEN', 'THE_RISE')
            emotional_tone: Optional emotional tone override (e.g., 'tension', 'hope')
        
        Returns:
            Color-graded image as numpy array
        """
        # Emotional tone takes priority
        if emotional_tone and emotional_tone in self.EMOTION_GRADE_MAP:
            grade = self.EMOTION_GRADE_MAP[emotional_tone]
            return self.apply_grade(image, grade)
        
        # Section-based grading
        if section and section in self.SECTION_GRADE_MAP:
            grade = self.SECTION_GRADE_MAP[section]
            return self.apply_grade(image, grade)
        
        # No metadata available, return ungraded
        return image

"""Professional color grading effects for video clips.

Includes section-aware grading that automatically applies the right colour
tone for each documentary section — warm golden for nostalgia, cool
desaturated for conflict, rich dramatic for climax, etc.
"""

import numpy as np


class ColorGrading:
    """Professional color grading effects."""

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
        'natural',
        'teal_orange',
        'noir',
        'golden_hour',
    ]

    # Maps each documentary section to its ideal colour grade.
    SECTION_GRADES = {
        'COLD_OPEN': 'dramatic',
        'EARLY_LIFE': 'warm',
        'THE_SPARK': 'golden_hour',
        'THE_RISE': 'cinematic',
        'THE_CONFLICT': 'teal_orange',
        'THE_CLIMAX': 'high_contrast',
        'THE_FALL': 'cool',
        'LEGACY': 'warm',
        'CTA': 'modern',
    }

    def grade_for_section(self, section: str) -> str:
        """Return the recommended grade name for a documentary section."""
        return self.SECTION_GRADES.get(section, 'cinematic')

    def _enforce_min_brightness(self, image: np.ndarray) -> np.ndarray:
        """Ensure no frame is excessively dark by lifting shadows.

        Frames with a large amount of content below luminance 60 are treated
        as underexposed.  Shadows are lifted before a final floor clamp so
        dark scenes stay readable without washing out highlights.
        """
        img = image.astype(np.float64)
        luminance = np.dot(img[..., :3], [0.299, 0.587, 0.114])
        dark_ratio = np.mean(luminance < 60)
        if dark_ratio > 0.3:
            boost = 40 + 30 * dark_ratio
            img = img + boost
            img = self._apply_clahe(np.clip(img, 0, 255).astype(np.uint8)).astype(np.float64)
        img = np.clip(img, 30, 255)
        return img.astype(np.uint8)

    @staticmethod
    def measure_luminance(image: np.ndarray) -> float:
        """Return average perceived luminance for an RGB image."""
        img = image.astype(np.float64)
        return float(np.mean(np.dot(img[..., :3], [0.299, 0.587, 0.114])))

    def normalize_luminance(self, image: np.ndarray, target: float = 120.0) -> np.ndarray:
        """Move an image toward a target luminance while preserving contrast."""
        img = image.astype(np.float64)
        current = self.measure_luminance(image)
        if current <= 1:
            return self._enforce_min_brightness(image)

        delta = np.clip(target - current, -35, 55)
        # Lift dark frames more than bright frames are reduced.
        if delta > 0:
            img += delta
            img = np.where(img < 90, img + delta * 0.35, img)
        else:
            img += delta * 0.45

        return self._enforce_min_brightness(np.clip(img, 0, 255).astype(np.uint8))

    @staticmethod
    def measure_channel_means(image: np.ndarray) -> np.ndarray:
        """Return RGB channel means for sequence color matching."""
        return image.astype(np.float64).reshape(-1, 3).mean(axis=0)

    def match_color_to_reference(
        self,
        image: np.ndarray,
        reference_means: np.ndarray,
        strength: float = 0.25,
    ) -> np.ndarray:
        """Gently blend image color balance toward a sequence reference."""
        if reference_means is None:
            return image
        img = image.astype(np.float64)
        current = self.measure_channel_means(image)
        delta = (reference_means - current) * np.clip(strength, 0.0, 1.0)
        img += delta.reshape(1, 1, 3)
        return self._enforce_min_brightness(np.clip(img, 0, 255).astype(np.uint8))

    def _apply_clahe(self, image: np.ndarray) -> np.ndarray:
        """Apply adaptive histogram equalization when OpenCV is available."""
        try:
            import cv2
        except ImportError:
            return image

        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_channel = clahe.apply(l_channel)
        merged = cv2.merge((l_channel, a_channel, b_channel))
        return cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)

    def apply_grade(self, image: np.ndarray, grade_type: str) -> np.ndarray:
        """Apply color grading to an image then enforce minimum brightness."""
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
        elif grade_type == 'teal_orange':
            return self._teal_orange_grade(image)
        elif grade_type == 'noir':
            return self._noir_grade(image)
        elif grade_type == 'golden_hour':
            return self._golden_hour_grade(image)
        else:
            return self._enforce_min_brightness(image)

    def _cinematic_grade(self, image: np.ndarray) -> np.ndarray:
        """Cinematic color grading with lifted shadows and gentle contrast."""
        img = image.astype(np.float64)
        img = np.where(img < 40, img * 0.7 + 15, img)
        highlights = np.where(img > 180, img * 0.96, img)
        result = (highlights - 128) * 1.10 + 128
        return self._enforce_min_brightness(np.clip(result, 0, 255).astype(np.uint8))

    def _documentary_grade(self, image: np.ndarray) -> np.ndarray:
        """Documentary style with natural contrast."""
        img = image.astype(np.float64)
        img = (img - 128) * 1.08 + 128
        img[:, :, 0] *= 1.02
        img[:, :, 2] *= 0.98
        return self._enforce_min_brightness(np.clip(img, 0, 255).astype(np.uint8))

    def _vintage_grade(self, image: np.ndarray) -> np.ndarray:
        """Vintage/retro color grading."""
        img = image.astype(np.float64)
        img[:, :, 0] *= 1.1
        img[:, :, 1] *= 1.05
        img[:, :, 2] *= 0.9
        img = (img - 128) * 0.9 + 128
        return self._enforce_min_brightness(np.clip(img, 0, 255).astype(np.uint8))

    def _modern_grade(self, image: np.ndarray) -> np.ndarray:
        """Modern look with gentler contrast."""
        img = image.astype(np.float64)
        img = (img - 128) * 1.15 + 128
        gray = np.dot(img, [0.299, 0.587, 0.114])
        img = img * 0.85 + gray[..., np.newaxis] * 0.15
        return self._enforce_min_brightness(np.clip(img, 0, 255).astype(np.uint8))

    def _warm_grade(self, image: np.ndarray) -> np.ndarray:
        """Warm, golden tones."""
        img = image.astype(np.float64)
        img[:, :, 0] *= 1.08
        img[:, :, 1] *= 1.03
        img[:, :, 2] *= 0.92
        return self._enforce_min_brightness(np.clip(img, 0, 255).astype(np.uint8))

    def _cool_grade(self, image: np.ndarray) -> np.ndarray:
        """Cool, blue tones."""
        img = image.astype(np.float64)
        img[:, :, 0] *= 0.95
        img[:, :, 1] *= 1.0
        img[:, :, 2] *= 1.1
        return self._enforce_min_brightness(np.clip(img, 0, 255).astype(np.uint8))

    def _high_contrast_grade(self, image: np.ndarray) -> np.ndarray:
        """High contrast dramatic look — shadows lifted, not crushed."""
        img = image.astype(np.float64)
        img = (img - 128) * 1.25 + 128
        img = np.where(img < 40, img * 0.7 + 15, img)
        img = np.where(img > 215, 215 + (img - 215) * 0.3, img)
        return self._enforce_min_brightness(np.clip(img, 0, 255).astype(np.uint8))

    def _soft_grade(self, image: np.ndarray) -> np.ndarray:
        """Soft, dreamy look."""
        img = image.astype(np.float64)
        img = (img - 128) * 0.85 + 128 + 15
        return self._enforce_min_brightness(np.clip(img, 0, 255).astype(np.uint8))

    def _dramatic_grade(self, image: np.ndarray) -> np.ndarray:
        """Dramatic, intense look — contrast reduced to prevent black crush."""
        img = image.astype(np.float64)
        img = (img - 128) * 1.15 + 128
        img[:, :, 0] *= 1.05
        return self._enforce_min_brightness(np.clip(img, 0, 255).astype(np.uint8))

    def _natural_grade(self, image: np.ndarray) -> np.ndarray:
        """Natural, balanced look."""
        img = image.astype(np.float64)
        img = (img - 128) * 1.05 + 128
        return self._enforce_min_brightness(np.clip(img, 0, 255).astype(np.uint8))

    def _teal_orange_grade(self, image: np.ndarray) -> np.ndarray:
        """Hollywood teal-and-orange split-tone — gentler to preserve shadows."""
        img = image.astype(np.float64)
        luminance = np.dot(img[..., :3], [0.299, 0.587, 0.114])
        shadow_mask = np.clip(1.0 - luminance / 128.0, 0, 1)[..., np.newaxis]
        highlight_mask = np.clip((luminance - 128.0) / 128.0, 0, 1)[..., np.newaxis]
        img[:, :, 0] -= shadow_mask[:, :, 0] * 8
        img[:, :, 1] += shadow_mask[:, :, 0] * 6
        img[:, :, 2] += shadow_mask[:, :, 0] * 12
        img[:, :, 0] += highlight_mask[:, :, 0] * 12
        img[:, :, 1] += highlight_mask[:, :, 0] * 4
        img[:, :, 2] -= highlight_mask[:, :, 0] * 8
        img = (img - 128) * 1.08 + 128
        return self._enforce_min_brightness(np.clip(img, 0, 255).astype(np.uint8))

    def _noir_grade(self, image: np.ndarray) -> np.ndarray:
        """Near-monochrome with subtle color hint — shadows preserved."""
        img = image.astype(np.float64)
        gray = np.dot(img[..., :3], [0.299, 0.587, 0.114])[..., np.newaxis]
        img = gray * 0.85 + img * 0.15
        img = (img - 128) * 1.20 + 128
        return self._enforce_min_brightness(np.clip(img, 0, 255).astype(np.uint8))

    def _golden_hour_grade(self, image: np.ndarray) -> np.ndarray:
        """Warm golden-hour glow for nostalgic/inspirational moments."""
        img = image.astype(np.float64)
        img[:, :, 0] *= 1.12
        img[:, :, 1] *= 1.06
        img[:, :, 2] *= 0.88
        img = (img - 128) * 1.08 + 128
        img += 8
        return self._enforce_min_brightness(np.clip(img, 0, 255).astype(np.uint8))

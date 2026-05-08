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

        If more than 50% of pixels are below brightness 40, the frame is
        considered underexposed and gets an automatic brightness boost.
        Minimum pixel value is clamped to 15 so nothing is pure black.
        """
        img = image.astype(np.float64)
        luminance = np.dot(img[..., :3], [0.299, 0.587, 0.114])
        dark_ratio = np.mean(luminance < 40)
        if dark_ratio > 0.5:
            boost = 25 + 20 * (dark_ratio - 0.5)
            img = img + boost
        img = np.clip(img, 15, 255)
        return img.astype(np.uint8)

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

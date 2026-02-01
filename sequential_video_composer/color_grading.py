"""Professional color grading effects for video clips."""

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

"""Professional color grading effects for video clips.

Includes section-aware grading that automatically applies the right colour
tone for each documentary section — warm golden for nostalgia, cool
desaturated for conflict, rich dramatic for climax, etc.

Features (v2.0):
- 13 color grades with section-aware mapping
- Per-shot adaptive grading (histogram-aware intensity scaling)
- Smooth cross-grade transitions between sections
- Emotional tone → color science mapping
- Split-toning (independent shadow/highlight color control)
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
        """Lift deep shadows to preserve detail without altering overall brightness.

        Only pixels below value 20 are gently lifted so pure-black areas
        retain some visible detail. This is NOT brightness enforcement —
        the overall image brightness is untouched.
        """
        img = image.astype(np.float64)
        img = np.where(img < 20, img * 1.3 + 10, img)
        return np.clip(img, 0, 255).astype(np.uint8)

    def apply_grade(self, image: np.ndarray, grade_type: str) -> np.ndarray:
        """Apply color grading to an image."""
        dispatch = {
            'cinematic': self._cinematic_grade,
            'documentary': self._documentary_grade,
            'vintage': self._vintage_grade,
            'modern': self._modern_grade,
            'warm': self._warm_grade,
            'cool': self._cool_grade,
            'high_contrast': self._high_contrast_grade,
            'soft': self._soft_grade,
            'dramatic': self._dramatic_grade,
            'natural': self._natural_grade,
            'teal_orange': self._teal_orange_grade,
            'noir': self._noir_grade,
            'golden_hour': self._golden_hour_grade,
        }
        grader = dispatch.get(grade_type, self._enforce_min_brightness)
        return grader(image)

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

    # ---- Per-shot adaptive color science (v2.0) ----

    EMOTION_GRADE_MAP = {
        'tension': 'teal_orange',
        'nostalgia': 'warm',
        'hope': 'golden_hour',
        'darkness': 'noir',
        'devastation': 'cool',
        'triumph': 'high_contrast',
        'bittersweet': 'vintage',
    }

    def grade_for_emotion(self, emotional_tone: str) -> str:
        """Return the recommended grade for an emotional tone."""
        return self.EMOTION_GRADE_MAP.get(emotional_tone, 'cinematic')

    def apply_adaptive_grade(
        self, image: np.ndarray, grade_type: str, emotional_tone: str = ''
    ) -> np.ndarray:
        """Apply histogram-adaptive color grading.

        Analyzes the source image histogram before applying the grade and
        scales intensity to prevent double-warming (warm grade on warm image)
        or over-cooling (cool grade on cool image). This preserves the
        natural feel of well-lit AI-generated images.
        """
        img_f = image.astype(np.float64)
        mean_r = np.mean(img_f[:, :, 0])
        mean_g = np.mean(img_f[:, :, 1])
        mean_b = np.mean(img_f[:, :, 2])
        mean_lum = 0.299 * mean_r + 0.587 * mean_g + 0.114 * mean_b

        warmth = (mean_r - mean_b) / max(mean_lum, 1.0)

        # Reduce warm grade intensity if image is already warm
        warm_grades = {'warm', 'golden_hour', 'vintage'}
        cool_grades = {'cool', 'teal_orange', 'noir'}

        graded = self.apply_grade(image, grade_type)

        if grade_type in warm_grades and warmth > 0.15:
            blend = max(0.5, 1.0 - warmth)
            graded = self._blend_images(image, graded, blend)
        elif grade_type in cool_grades and warmth < -0.15:
            blend = max(0.5, 1.0 - abs(warmth))
            graded = self._blend_images(image, graded, blend)

        return graded

    def cross_grade(
        self,
        image: np.ndarray,
        grade_from: str,
        grade_to: str,
        blend: float,
    ) -> np.ndarray:
        """Smoothly transition between two color grades.

        Args:
            image: Source image array.
            grade_from: Starting grade name.
            grade_to: Target grade name.
            blend: 0.0 = fully grade_from, 1.0 = fully grade_to.

        Used at section boundaries to create smooth 3-5 second color
        transitions instead of abrupt grade switches.
        """
        blend = max(0.0, min(1.0, blend))
        if blend <= 0.0:
            return self.apply_grade(image, grade_from)
        if blend >= 1.0:
            return self.apply_grade(image, grade_to)

        graded_from = self.apply_grade(image, grade_from)
        graded_to = self.apply_grade(image, grade_to)
        return self._blend_images(graded_from, graded_to, blend)

    def apply_split_tone(
        self,
        image: np.ndarray,
        shadow_color: tuple = (0, 10, 20),
        highlight_color: tuple = (15, 5, -5),
        strength: float = 0.5,
    ) -> np.ndarray:
        """Apply split-toning: independent shadow and highlight color shifts.

        Args:
            image: Source image.
            shadow_color: (R, G, B) offset for shadow regions.
            highlight_color: (R, G, B) offset for highlight regions.
            strength: Overall intensity 0-1.
        """
        img = image.astype(np.float64)
        luminance = np.dot(img[..., :3], [0.299, 0.587, 0.114])
        shadow_mask = np.clip(1.0 - luminance / 128.0, 0, 1)[..., np.newaxis]
        highlight_mask = np.clip((luminance - 128.0) / 128.0, 0, 1)[..., np.newaxis]

        for ch in range(3):
            img[:, :, ch] += shadow_mask[:, :, 0] * shadow_color[ch] * strength
            img[:, :, ch] += highlight_mask[:, :, 0] * highlight_color[ch] * strength

        return np.clip(img, 0, 255).astype(np.uint8)

    @staticmethod
    def _blend_images(
        img_a: np.ndarray, img_b: np.ndarray, blend: float
    ) -> np.ndarray:
        """Blend two images with a scalar factor."""
        a = img_a.astype(np.float64)
        b = img_b.astype(np.float64)
        result = a * (1.0 - blend) + b * blend
        return np.clip(result, 0, 255).astype(np.uint8)

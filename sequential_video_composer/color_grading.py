"""Professional color grading effects for video clips.

Includes section-aware grading that automatically applies the right colour
tone for each documentary section — warm golden for nostalgia, cool
desaturated for conflict, rich dramatic for climax, etc.

Features (v2.1):
- 13 color grades with section-aware mapping
- Per-shot adaptive grading (histogram-aware intensity scaling)
- Smooth cross-grade transitions between sections
- Emotional tone → color science mapping
- Split-toning (independent shadow/highlight color control)
- Smooth luminance-based operations (no per-channel thresholds)
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
        'COLD_OPEN': 'cinematic',
        'EARLY_LIFE': 'warm',
        'THE_SPARK': 'golden_hour',
        'THE_RISE': 'cinematic',
        'THE_CONFLICT': 'cool',
        'THE_CLIMAX': 'cinematic',
        'THE_FALL': 'cool',
        'LEGACY': 'warm',
        'CTA': 'natural',
    }

    def grade_for_section(self, section: str) -> str:
        """Return the recommended grade name for a documentary section."""
        return self.SECTION_GRADES.get(section, 'cinematic')

    @staticmethod
    def _smooth_shadow_lift(img: np.ndarray) -> np.ndarray:
        """Gently lift very dark pixels using a smooth curve.

        Uses a smooth blending function based on luminance so all three
        channels are lifted proportionally — preventing color fringing.
        """
        luminance = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
        # Smooth blend factor: 1.0 for pure black, 0.0 for luminance >= 50
        blend = np.clip(1.0 - luminance / 50.0, 0, 1)
        blend = blend[:, :, np.newaxis]
        # Lift proportionally: add a small amount to all channels equally
        lift = blend * 8.0
        return img + lift

    @staticmethod
    def _auto_brightness_correction(img: np.ndarray) -> np.ndarray:
        """Per-scene adaptive brightness that normalizes overly dark or bright images.

        Targets a mean luminance around 120–160 so no scene looks blown-out
        or lost in shadow.
        """
        luminance = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
        mean_lum = np.mean(luminance)

        if mean_lum < 80:
            boost = (120 - mean_lum) / max(mean_lum, 1)
            img = img * (1 + boost * 0.4)
        elif mean_lum > 180:
            reduction = (mean_lum - 160) / max(mean_lum, 1)
            img = img * (1 - reduction * 0.3)

        return img

    @staticmethod
    def _reduce_overexposure(img: np.ndarray) -> np.ndarray:
        """Recover detail from overexposed (washed-out) highlights.

        Applies inverse shadow-lift on the bright end and increases
        contrast for images whose mean luminance exceeds 180.
        """
        luminance = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
        mean_lum = np.mean(luminance)

        if mean_lum > 180:
            # Pull highlights back
            highlight_mask = np.clip((luminance - 160) / 95.0, 0, 1)[:, :, np.newaxis]
            img = img - highlight_mask * 20
            # Boost contrast slightly
            img = (img - 128) * 1.08 + 128

        return img

    def _finalize(self, img: np.ndarray) -> np.ndarray:
        """Final pass: brightness correction, shadow lift, highlight recovery, clip."""
        img = self._auto_brightness_correction(img)
        img = self._reduce_overexposure(img)
        img = self._smooth_shadow_lift(img)
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
        grader = dispatch.get(grade_type, self._finalize)
        return grader(image)

    def _cinematic_grade(self, image: np.ndarray) -> np.ndarray:
        """Cinematic color grading with gentle contrast."""
        img = image.astype(np.float64)
        img = (img - 128) * 1.04 + 128
        return self._finalize(img)

    def _documentary_grade(self, image: np.ndarray) -> np.ndarray:
        """Documentary style with natural contrast."""
        img = image.astype(np.float64)
        img = (img - 128) * 1.05 + 128
        img[:, :, 0] *= 1.01
        img[:, :, 2] *= 0.99
        return self._finalize(img)

    def _vintage_grade(self, image: np.ndarray) -> np.ndarray:
        """Vintage/retro color grading."""
        img = image.astype(np.float64)
        img[:, :, 0] *= 1.06
        img[:, :, 1] *= 1.03
        img[:, :, 2] *= 0.94
        img = (img - 128) * 0.92 + 128
        return self._finalize(img)

    def _modern_grade(self, image: np.ndarray) -> np.ndarray:
        """Modern look with gentle desaturation."""
        img = image.astype(np.float64)
        img = (img - 128) * 1.06 + 128
        gray = np.dot(img[..., :3], [0.299, 0.587, 0.114])
        img = img * 0.9 + gray[..., np.newaxis] * 0.1
        return self._finalize(img)

    def _warm_grade(self, image: np.ndarray) -> np.ndarray:
        """Warm, golden tones."""
        img = image.astype(np.float64)
        img[:, :, 0] *= 1.05
        img[:, :, 1] *= 1.02
        img[:, :, 2] *= 0.95
        return self._finalize(img)

    def _cool_grade(self, image: np.ndarray) -> np.ndarray:
        """Cool, blue tones."""
        img = image.astype(np.float64)
        img[:, :, 0] *= 0.97
        img[:, :, 2] *= 1.05
        return self._finalize(img)

    def _high_contrast_grade(self, image: np.ndarray) -> np.ndarray:
        """High contrast look with smooth curve."""
        img = image.astype(np.float64)
        img = (img - 128) * 1.10 + 128
        return self._finalize(img)

    def _soft_grade(self, image: np.ndarray) -> np.ndarray:
        """Soft, dreamy look."""
        img = image.astype(np.float64)
        img = (img - 128) * 0.88 + 128 + 8
        return self._finalize(img)

    def _dramatic_grade(self, image: np.ndarray) -> np.ndarray:
        """Dramatic look with mild contrast boost."""
        img = image.astype(np.float64)
        img = (img - 128) * 1.06 + 128
        return self._finalize(img)

    def _natural_grade(self, image: np.ndarray) -> np.ndarray:
        """Natural, balanced look — minimal processing."""
        img = image.astype(np.float64)
        img = (img - 128) * 1.02 + 128
        return self._finalize(img)

    def _teal_orange_grade(self, image: np.ndarray) -> np.ndarray:
        """Hollywood teal-and-orange split-tone using smooth luminance masks."""
        img = image.astype(np.float64)
        luminance = np.dot(img[..., :3], [0.299, 0.587, 0.114])
        shadow_mask = np.clip(1.0 - luminance / 128.0, 0, 1)[..., np.newaxis]
        highlight_mask = np.clip((luminance - 128.0) / 128.0, 0, 1)[..., np.newaxis]
        # Subtle color shifts
        img[:, :, 0] -= shadow_mask[:, :, 0] * 4
        img[:, :, 1] += shadow_mask[:, :, 0] * 3
        img[:, :, 2] += shadow_mask[:, :, 0] * 6
        img[:, :, 0] += highlight_mask[:, :, 0] * 6
        img[:, :, 1] += highlight_mask[:, :, 0] * 2
        img[:, :, 2] -= highlight_mask[:, :, 0] * 4
        img = (img - 128) * 1.04 + 128
        return self._finalize(img)

    def _noir_grade(self, image: np.ndarray) -> np.ndarray:
        """Near-monochrome with subtle color retained."""
        img = image.astype(np.float64)
        gray = np.dot(img[..., :3], [0.299, 0.587, 0.114])[..., np.newaxis]
        img = gray * 0.7 + img * 0.3
        img = (img - 128) * 1.06 + 128
        return self._finalize(img)

    def _golden_hour_grade(self, image: np.ndarray) -> np.ndarray:
        """Warm golden-hour glow."""
        img = image.astype(np.float64)
        img[:, :, 0] *= 1.08
        img[:, :, 1] *= 1.04
        img[:, :, 2] *= 0.92
        img += 5
        return self._finalize(img)

    # ---- Per-shot adaptive color science (v2.0) ----

    EMOTION_GRADE_MAP = {
        'tension': 'cinematic',
        'nostalgia': 'warm',
        'hope': 'golden_hour',
        'darkness': 'cool',
        'devastation': 'cool',
        'triumph': 'cinematic',
        'bittersweet': 'warm',
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
        or over-cooling (cool grade on cool image).
        """
        img_f = image.astype(np.float64)
        mean_r = np.mean(img_f[:, :, 0])
        mean_g = np.mean(img_f[:, :, 1])
        mean_b = np.mean(img_f[:, :, 2])
        mean_lum = 0.299 * mean_r + 0.587 * mean_g + 0.114 * mean_b

        warmth = (mean_r - mean_b) / max(mean_lum, 1.0)

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
        """Smoothly transition between two color grades."""
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
        highlight_color: tuple = (20, 5, 0),
        strength: float = 0.15,
    ) -> np.ndarray:
        """Apply split-toning using smooth luminance-based masks."""
        img = image.astype(np.float64)
        luminance = np.dot(img[..., :3], [0.299, 0.587, 0.114])
        shadow_mask = np.clip(1.0 - luminance / 128.0, 0, 1)
        highlight_mask = np.clip((luminance - 128.0) / 128.0, 0, 1)

        for c in range(3):
            img[:, :, c] += shadow_mask * shadow_color[c] * strength
            img[:, :, c] += highlight_mask * highlight_color[c] * strength

        return np.clip(img, 0, 255).astype(np.uint8)

    @staticmethod
    def _blend_images(
        img_a: np.ndarray, img_b: np.ndarray, alpha: float
    ) -> np.ndarray:
        a = img_a.astype(np.float64)
        b = img_b.astype(np.float64)
        result = a * (1.0 - alpha) + b * alpha
        return np.clip(result, 0, 255).astype(np.uint8)

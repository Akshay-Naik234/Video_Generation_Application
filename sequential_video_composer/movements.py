"""Documentary-style movement styles for dynamic image animations.

Includes classic Ken Burns effects plus advanced documentary techniques:
- 3D parallax / 2.5D depth simulation
- Dolly zoom (Vertigo effect)
- Orbit / arc movements
- Whip pan with motion blur
- Crane up/down
- Spiral zoom
- Handheld drift / camera shake
- Dutch tilt
- Push in / push out with rack focus
- Bounce zoom for emphasis
- Tilt shift (miniature effect)
"""

from pathlib import Path
from typing import Tuple, TYPE_CHECKING

import numpy as np
from PIL import Image as PILImage, ImageFilter

if TYPE_CHECKING:
    from .color_grading import ColorGrading


class MovementStyles:
    """Documentary-style movement styles for dynamic image animations."""

    MOVEMENT_TYPES = [
        # Classic Ken Burns
        'zoom_in',
        'zoom_out',
        'pan_left',
        'pan_right',
        'pan_up',
        'pan_down',
        'diagonal_tl_br',
        'diagonal_tr_bl',
        'breathing',
        'dramatic_zoom',
        'gentle_drift',
        'focus_center',
        'minimal',
        'static',
        # Documentary-style advanced movements
        'parallax_depth',
        'push_in',
        'push_out',
        'orbit',
        'whip_pan',
        'dolly_zoom',
        'handheld_drift',
        'crane_up',
        'crane_down',
        'spiral_zoom',
        'tilt_shift',
        'dutch_tilt',
        'rack_focus',
        'bounce_zoom',
        'float_up',
        'reveal_left',
        'reveal_right',
        # Map & geography movements
        'map_zoom',
        'map_pan',
        # Timeline / chronology
        'timeline_reveal',
    ]

    # Movements grouped by documentary section / emotional tone
    SECTION_MOVEMENTS = {
        'COLD_OPEN': ['dramatic_zoom', 'push_in', 'handheld_drift', 'rack_focus', 'dutch_tilt'],
        'EARLY_LIFE': ['gentle_drift', 'zoom_in', 'float_up', 'breathing', 'parallax_depth'],
        'THE_SPARK': ['push_in', 'orbit', 'spiral_zoom', 'zoom_in', 'reveal_left'],
        'THE_RISE': ['crane_up', 'push_in', 'orbit', 'dolly_zoom', 'parallax_depth'],
        'THE_CONFLICT': ['handheld_drift', 'dutch_tilt', 'whip_pan', 'rack_focus', 'push_in'],
        'THE_CLIMAX': ['dramatic_zoom', 'dolly_zoom', 'crane_up', 'spiral_zoom', 'parallax_depth'],
        'THE_FALL': ['crane_down', 'zoom_out', 'float_up', 'gentle_drift', 'breathing'],
        'LEGACY': ['parallax_depth', 'gentle_drift', 'zoom_out', 'float_up', 'orbit'],
        'CTA': ['zoom_out', 'gentle_drift', 'breathing', 'minimal', 'parallax_depth'],
    }

    TONE_MOVEMENTS = {
        'tension': ['handheld_drift', 'dutch_tilt', 'push_in', 'rack_focus', 'whip_pan'],
        'nostalgia': ['gentle_drift', 'parallax_depth', 'float_up', 'breathing', 'zoom_in'],
        'hope': ['crane_up', 'float_up', 'reveal_left', 'parallax_depth', 'zoom_in'],
        'darkness': ['dutch_tilt', 'handheld_drift', 'push_in', 'crane_down', 'rack_focus'],
        'devastation': ['handheld_drift', 'zoom_out', 'crane_down', 'dutch_tilt', 'breathing'],
        'triumph': ['crane_up', 'dolly_zoom', 'orbit', 'spiral_zoom', 'dramatic_zoom'],
        'bittersweet': ['parallax_depth', 'gentle_drift', 'float_up', 'zoom_out', 'breathing'],
    }

    # Section-aware zoom intensity multipliers: kept conservative to prevent
    # content from being cropped out of frame.
    SECTION_ZOOM_MULTIPLIERS = {
        'COLD_OPEN': 1.1,
        'EARLY_LIFE': 0.85,
        'THE_SPARK': 1.0,
        'THE_RISE': 1.05,
        'THE_CONFLICT': 1.1,
        'THE_CLIMAX': 1.15,
        'THE_FALL': 0.9,
        'LEGACY': 0.8,
        'CTA': 0.75,
    }

    def __init__(self, resolution: Tuple[int, int]):
        self.width, self.height = resolution
        self.resolution = resolution

    def create_animated_clip(
        self,
        image_path: Path,
        duration: float,
        movement_type: str,
        zoom_intensity: float = 1.12,
        color_grader: 'ColorGrading' = None,
        color_grade: str = None,
        enable_vignette: bool = False,
        section: str = '',
    ):
        """Create an animated clip with the specified movement style and effects.

        Uses a memory-efficient approach with make_frame instead of creating
        many sub-clips. Pre-scales the image larger to allow for smooth
        zooming without per-frame resizing.

        Accepts an optional *section* name so section-aware zoom multipliers
        can be applied automatically.
        """
        # Apply section-aware zoom boost
        section_mult = self.SECTION_ZOOM_MULTIPLIERS.get(section, 1.0)
        zoom_intensity = 1.0 + (zoom_intensity - 1.0) * section_mult
        base_img = PILImage.open(image_path)
        if base_img.mode != 'RGB':
            base_img = base_img.convert('RGB')

        max_zoom = max(zoom_intensity, 1.20)
        scaled_width = int(self.width * max_zoom * 1.4)
        scaled_height = int(self.height * max_zoom * 1.4)

        orig_width, orig_height = base_img.size
        orig_aspect = orig_width / orig_height
        target_aspect = scaled_width / scaled_height

        # Fill-crop: scale the image so it completely covers the canvas
        # (no black bars, no blur borders). Minor edges may be cropped but
        # a 10% safe margin is maintained during movements.
        if orig_aspect > target_aspect:
            fill_height = scaled_height
            fill_width = int(scaled_height * orig_aspect)
        else:
            fill_width = scaled_width
            fill_height = int(scaled_width / orig_aspect)

        resized_img = base_img.resize((fill_width, fill_height), PILImage.LANCZOS)
        crop_x = (fill_width - scaled_width) // 2
        crop_y = (fill_height - scaled_height) // 2
        canvas = resized_img.crop((crop_x, crop_y, crop_x + scaled_width, crop_y + scaled_height))

        base_img = canvas
        base_array = np.array(base_img)

        if color_grader and color_grade:
            base_array = color_grader.apply_grade(base_array, color_grade)

        if enable_vignette:
            base_array = self._apply_vignette(base_array)

        scaled_img = PILImage.fromarray(base_array)

        # For tilt_shift, pre-compute blurred version
        blurred_img = None
        if movement_type == 'tilt_shift':
            blurred_img = scaled_img.filter(
                ImageFilter.GaussianBlur(radius=int(8 * (self.height / 1080)))
            )

        center_x = scaled_width / 2.0
        center_y = scaled_height / 2.0

        out_w, out_h = self.resolution

        # Pre-convert to numpy for fast per-frame crop+resize
        base_np = np.array(scaled_img)
        blurred_np = np.array(blurred_img) if blurred_img is not None else None

        try:
            import cv2
            _has_cv2 = True
        except ImportError:
            _has_cv2 = False

        # Use smooth sine easing for all movements — gives the most fluid,
        # professional feel with no sudden acceleration or deceleration.
        _easing_map = {
            'dramatic_zoom': self._ease_in_out_sine,
            'push_in': self._ease_in_out_sine,
            'push_out': self._ease_in_out_sine,
            'whip_pan': self._ease_in_out_sine,
            'crane_up': self._ease_in_out_sine,
            'crane_down': self._ease_in_out_sine,
            'rack_focus': self._ease_in_out_sine,
            'bounce_zoom': self._ease_in_out_sine,
            'dolly_zoom': self._ease_in_out_sine,
            'spiral_zoom': self._ease_in_out_sine,
            'orbit': self._ease_in_out_sine,
            'float_up': self._ease_in_out_sine,
            'reveal_left': self._ease_in_out_sine,
            'reveal_right': self._ease_in_out_sine,
            'map_zoom': self._ease_in_out_sine,
            'map_pan': self._ease_in_out_sine,
            'timeline_reveal': self._ease_in_out_sine,
        }
        _ease_fn = _easing_map.get(movement_type, self._ease_in_out_sine)

        def make_frame(t):
            progress = t / duration if duration > 0 else 0
            progress = max(0.0, min(1.0, progress))
            eased = _ease_fn(progress)

            zoom, pan_x, pan_y = self._calculate_movement(
                movement_type, eased, zoom_intensity, progress
            )

            # Safety cap: max 15% zoom to prevent content cropping
            zoom = min(zoom, 1.15)

            crop_w = self.width / zoom
            crop_h = self.height / zoom

            offset_x = pan_x * crop_w * 0.5
            offset_y = pan_y * crop_h * 0.5

            left = center_x - crop_w / 2.0 + offset_x
            top = center_y - crop_h / 2.0 + offset_y

            left = max(0.0, min(left, scaled_width - crop_w))
            top = max(0.0, min(top, scaled_height - crop_h))

            l_i, t_i = int(left), int(top)
            r_i = min(l_i + int(crop_w), scaled_width)
            b_i = min(t_i + int(crop_h), scaled_height)

            src = base_np
            if movement_type == 'tilt_shift' and blurred_np is not None:
                src = self._apply_tilt_shift_blend_np(
                    base_np, blurred_np, progress
                )

            cropped = src[t_i:b_i, l_i:r_i]

            if _has_cv2:
                frame = cv2.resize(cropped, (out_w, out_h), interpolation=cv2.INTER_LANCZOS4)
            else:
                pil_crop = PILImage.fromarray(cropped)
                frame = np.array(pil_crop.resize((out_w, out_h), PILImage.LANCZOS))

            if movement_type in ('whip_pan', 'push_in', 'dramatic_zoom', 'dolly_zoom'):
                frame = self._apply_motion_blur(frame, progress)

            return frame

        from moviepy.video.VideoClip import VideoClip
        clip = VideoClip(make_frame, duration=duration)
        clip = clip.set_fps(30)
        return clip

    # ---- Visual effect helpers ----

    def _apply_vignette(self, image: np.ndarray) -> np.ndarray:
        """Apply an adaptive vignette that weakens on dark images."""
        avg_brightness = np.mean(image)
        if avg_brightness < 80:
            strength = 0.12
        elif avg_brightness < 120:
            strength = 0.20
        else:
            strength = 0.30
        rows, cols = image.shape[:2]
        X = np.arange(0, cols)
        Y = np.arange(0, rows)
        X, Y = np.meshgrid(X, Y)
        center_x, center_y = cols / 2, rows / 2
        distance = np.sqrt((X - center_x) ** 2 + (Y - center_y) ** 2)
        max_distance = np.sqrt(center_x ** 2 + center_y ** 2)
        vignette = 1 - (distance / max_distance) * strength
        vignette = np.clip(vignette, 1.0 - strength, 1.0)
        vignette = np.dstack([vignette] * 3)
        return (image * vignette).astype(np.uint8)

    def _apply_tilt_shift_blend(
        self, sharp: PILImage.Image, blurred: PILImage.Image, progress: float
    ) -> PILImage.Image:
        """Blend sharp center band with blurred edges for miniature effect."""
        sharp_arr = np.array(sharp).astype(np.float32)
        blur_arr = np.array(blurred).astype(np.float32)
        result = self._tilt_shift_blend_arrays(sharp_arr, blur_arr, sharp.size[1], progress)
        return PILImage.fromarray(result.astype(np.uint8))

    def _apply_tilt_shift_blend_np(
        self, sharp_np: np.ndarray, blurred_np: np.ndarray, progress: float
    ) -> np.ndarray:
        """Blend sharp center band with blurred edges (numpy arrays)."""
        h = sharp_np.shape[0]
        return self._tilt_shift_blend_arrays(
            sharp_np.astype(np.float32), blurred_np.astype(np.float32), h, progress
        ).astype(np.uint8)

    @staticmethod
    def _tilt_shift_blend_arrays(
        sharp_arr: np.ndarray, blur_arr: np.ndarray, h: int, progress: float
    ) -> np.ndarray:
        focus_center = 0.45 + 0.1 * progress
        band_width = 0.25
        y_coords = np.linspace(0, 1, h)[:, np.newaxis]
        dist = np.abs(y_coords - focus_center)
        mask = np.clip((dist - band_width * 0.5) / (band_width * 0.3 + 1e-6), 0, 1)
        mask = mask[:, :, np.newaxis]
        return sharp_arr * (1 - mask) + blur_arr * mask

    def _apply_motion_blur(self, frame: np.ndarray, progress: float) -> np.ndarray:
        """Apply directional motion blur during fast animation phases.

        Blur strength peaks in the middle 40% of the animation and scales
        with resolution so the effect looks consistent at any output size.
        """
        blur_strength = max(0, 1.0 - abs(progress - 0.5) * 4.0)
        if blur_strength < 0.05:
            return frame

        kernel_size = int(blur_strength * 3 * (self.width / 1920))
        if kernel_size < 2:
            return frame

        try:
            import cv2
            kernel = np.zeros((kernel_size, kernel_size))
            kernel[kernel_size // 2, :] = 1.0 / kernel_size
            return cv2.filter2D(frame, -1, kernel)
        except ImportError:
            img = PILImage.fromarray(frame)
            img = img.filter(ImageFilter.BoxBlur(kernel_size))
            return np.array(img)

    def _apply_handheld_shake(self, frame: np.ndarray, t: float) -> np.ndarray:
        """Apply handheld camera micro-shake by shifting pixels.

        Uses np.pad with 'edge' mode to replicate border pixels instead of
        filling with black, which prevents flickering black edges.
        """
        amplitude = 2.0 * (self.width / 1920)
        freq1, freq2 = 3.7, 5.3
        dx = int(amplitude * np.sin(t * freq1 * 2 * np.pi))
        dy = int(amplitude * np.cos(t * freq2 * 2 * np.pi))

        if dx == 0 and dy == 0:
            return frame

        h, w = frame.shape[:2]
        abs_dx, abs_dy = abs(dx), abs(dy)

        padded = np.pad(
            frame,
            ((abs_dy, abs_dy), (abs_dx, abs_dx), (0, 0)),
            mode='edge'
        )
        crop_y = abs_dy - dy
        crop_x = abs_dx - dx
        return padded[crop_y:crop_y + h, crop_x:crop_x + w]

    # ---- Movement calculations ----

    def _calculate_movement(
        self,
        movement_type: str,
        eased: float,
        zoom_intensity: float,
        raw_progress: float = 0.0,
    ) -> Tuple[float, float, float]:
        """Calculate zoom and pan values for the given movement type.

        Returns (zoom, pan_x, pan_y) where pan values are normalized offsets.
        """
        # Classic movements
        if movement_type == 'zoom_in':
            zoom = 1.0 + (zoom_intensity - 1.0) * eased * 1.4
            return zoom, 0, 0

        elif movement_type == 'zoom_out':
            zoom = zoom_intensity - (zoom_intensity - 1.0) * eased
            return zoom, 0, 0

        elif movement_type == 'pan_left':
            zoom = 1.0 + (zoom_intensity - 1.0) * 0.6
            pan_x = -0.20 * eased
            return zoom, pan_x, 0

        elif movement_type == 'pan_right':
            zoom = 1.0 + (zoom_intensity - 1.0) * 0.6
            pan_x = 0.20 * eased
            return zoom, pan_x, 0

        elif movement_type == 'pan_up':
            zoom = 1.0 + (zoom_intensity - 1.0) * 0.6
            pan_y = -0.20 * eased
            return zoom, 0, pan_y

        elif movement_type == 'pan_down':
            zoom = 1.0 + (zoom_intensity - 1.0) * 0.6
            pan_y = 0.20 * eased
            return zoom, 0, pan_y

        elif movement_type == 'diagonal_tl_br':
            zoom = 1.0 + (zoom_intensity - 1.0) * eased * 1.0
            pan_x = 0.14 * eased
            pan_y = 0.14 * eased
            return zoom, pan_x, pan_y

        elif movement_type == 'diagonal_tr_bl':
            zoom = 1.0 + (zoom_intensity - 1.0) * eased * 1.0
            pan_x = -0.14 * eased
            pan_y = 0.14 * eased
            return zoom, pan_x, pan_y

        elif movement_type == 'breathing':
            zoom = 1.0 + 0.07 * np.sin(raw_progress * np.pi)
            pan_y = 0.015 * np.sin(raw_progress * np.pi * 0.5)
            return zoom, 0, pan_y

        elif movement_type == 'dramatic_zoom':
            zoom = 1.0 + (zoom_intensity - 1.0) * 1.3 * eased
            return zoom, 0, 0

        elif movement_type == 'gentle_drift':
            zoom = 1.0 + (zoom_intensity - 1.0) * 0.6
            pan_x = 0.10 * np.sin(raw_progress * np.pi)
            pan_y = 0.06 * np.cos(raw_progress * np.pi)
            return zoom, pan_x, pan_y

        elif movement_type == 'focus_center':
            zoom = 1.0 + (zoom_intensity - 1.0) * eased * 1.0
            return zoom, 0, 0

        elif movement_type == 'minimal':
            zoom = 1.0 + (zoom_intensity - 1.0) * eased * 0.8
            return zoom, 0, 0

        elif movement_type == 'static':
            return 1.0, 0, 0

        # ---- Documentary-style advanced movements ----

        elif movement_type == 'parallax_depth':
            zoom = 1.0 + (zoom_intensity - 1.0) * eased * 1.1
            phase = raw_progress * np.pi
            pan_x = 0.18 * np.sin(phase)
            pan_y = 0.09 * np.sin(phase * 0.7 + 0.3)
            return zoom, pan_x, pan_y

        elif movement_type == 'push_in':
            zoom = 1.0 + (zoom_intensity - 1.0) * 1.2 * eased
            pan_y = -0.08 * eased
            return zoom, 0, pan_y

        elif movement_type == 'push_out':
            zoom = zoom_intensity - (zoom_intensity - 1.0) * eased
            pan_y = 0.08 * eased
            return zoom, 0, pan_y

        elif movement_type == 'orbit':
            zoom = 1.0 + (zoom_intensity - 1.0) * 0.85
            angle = eased * np.pi * 1.4
            pan_x = 0.16 * np.sin(angle)
            pan_y = 0.08 * (1 - np.cos(angle))
            return zoom, pan_x, pan_y

        elif movement_type == 'whip_pan':
            zoom = 1.0 + (zoom_intensity - 1.0) * 0.5
            pan_x = 0.22 * eased
            return zoom, pan_x, 0

        elif movement_type == 'dolly_zoom':
            zoom = 1.0 + (zoom_intensity - 1.0) * 1.1 * eased
            pan_y = -0.06 * eased
            pan_x = 0.03 * eased
            return zoom, pan_x, pan_y

        elif movement_type == 'handheld_drift':
            zoom = 1.0 + (zoom_intensity - 1.0) * 0.6
            pan_x = 0.10 * np.sin(raw_progress * np.pi * 1.2)
            pan_y = 0.06 * np.sin(raw_progress * np.pi * 0.8 + 0.5)
            return zoom, pan_x, pan_y

        elif movement_type == 'crane_up':
            zoom = 1.0 + (zoom_intensity - 1.0) * (1.0 - eased * 0.2)
            pan_y = -0.18 * eased
            return zoom, 0, pan_y

        elif movement_type == 'crane_down':
            zoom = 1.0 + (zoom_intensity - 1.0) * eased * 0.8
            pan_y = 0.18 * eased
            return zoom, 0, pan_y

        elif movement_type == 'spiral_zoom':
            zoom = 1.0 + (zoom_intensity - 1.0) * 1.15 * eased
            angle = eased * np.pi * 2.5
            radius = 0.12 * (1.0 - eased)
            pan_x = radius * np.cos(angle)
            pan_y = radius * np.sin(angle)
            return zoom, pan_x, pan_y

        elif movement_type == 'tilt_shift':
            zoom = 1.0 + (zoom_intensity - 1.0) * 0.8
            pan_x = 0.18 * eased
            pan_y = -0.08 * eased
            return zoom, pan_x, pan_y

        elif movement_type == 'dutch_tilt':
            zoom = 1.0 + (zoom_intensity - 1.0) * eased * 0.9
            tilt = 0.10 * eased
            pan_x = tilt
            pan_y = tilt * 0.5
            return zoom, pan_x, pan_y

        elif movement_type == 'rack_focus':
            zoom = 1.0 + (zoom_intensity - 1.0) * 1.3 * eased
            return zoom, 0, 0

        elif movement_type == 'bounce_zoom':
            overshoot = 1.0 + 0.12 * np.sin(eased * np.pi * 3) * (1.0 - eased)
            zoom = 1.0 + (zoom_intensity - 1.0) * eased * 1.1 * overshoot
            return zoom, 0, 0

        elif movement_type == 'float_up':
            zoom = 1.0 + (zoom_intensity - 1.0) * eased * 0.8
            pan_y = -0.14 * eased
            pan_x = 0.04 * np.sin(raw_progress * np.pi)
            return zoom, pan_x, pan_y

        elif movement_type == 'reveal_left':
            zoom = 1.0 + (zoom_intensity - 1.0) * 0.5
            pan_x = 0.18 * (1.0 - eased)
            return zoom, pan_x, 0

        elif movement_type == 'reveal_right':
            zoom = 1.0 + (zoom_intensity - 1.0) * 0.5
            pan_x = -0.18 * (1.0 - eased)
            return zoom, pan_x, 0

        elif movement_type == 'map_zoom':
            zoom = 1.0 + (zoom_intensity - 1.0) * 1.0 * eased
            pan_y = 0.06 * eased
            return zoom, 0, pan_y

        elif movement_type == 'map_pan':
            zoom = 1.0 + (zoom_intensity - 1.0) * 0.5
            pan_x = 0.20 * eased
            pan_y = 0.05 * np.sin(eased * np.pi)
            return zoom, pan_x, pan_y

        elif movement_type == 'timeline_reveal':
            zoom = 1.0 + (zoom_intensity - 1.0) * 0.5
            pan_x = -0.22 * (1.0 - eased)
            return zoom, pan_x, 0

        else:
            zoom = 1.0 + (zoom_intensity - 1.0) * eased * 0.7
            return zoom, 0, 0

    # ---- Easing functions ----

    def _ease_in_out_cubic(self, t: float) -> float:
        """Cubic easing function for smooth animations."""
        return 3 * t * t - 2 * t * t * t

    def _dramatic_ease(self, t: float) -> float:
        """Dramatic easing for impactful moments."""
        return 0.5 * (np.sin((t - 0.5) * np.pi) + 1)

    def _ease_in_quad(self, t: float) -> float:
        return t * t

    def _ease_out_quad(self, t: float) -> float:
        return 1 - (1 - t) * (1 - t)

    def _ease_in_out_quart(self, t: float) -> float:
        if t < 0.5:
            return 8 * t * t * t * t
        return 1 - (-2 * t + 2) ** 4 / 2

    def _ease_in_out_quint(self, t: float) -> float:
        if t < 0.5:
            return 16 * t ** 5
        return 1 - (-2 * t + 2) ** 5 / 2

    def _ease_out_expo(self, t: float) -> float:
        if t >= 1.0:
            return 1.0
        return 1.0 - 2.0 ** (-10.0 * t)

    def _ease_in_out_sine(self, t: float) -> float:
        return -(np.cos(np.pi * t) - 1) / 2

    def _ease_elastic_out(self, t: float) -> float:
        """Elastic easing for bouncy emphasis moments."""
        if t <= 0:
            return 0.0
        if t >= 1.0:
            return 1.0
        return 2.0 ** (-10.0 * t) * np.sin((t * 10.0 - 0.75) * (2 * np.pi / 3)) + 1.0

    def _ease_out_back(self, t: float, overshoot: float = 1.70158) -> float:
        """Back easing — overshoots then settles for punchy emphasis."""
        t -= 1.0
        return t * t * ((overshoot + 1) * t + overshoot) + 1.0

    def _speed_ramp(self, t: float) -> float:
        """Speed ramp: slow start, fast middle, slow end."""
        if t < 0.2:
            return self._ease_in_quad(t / 0.2) * 0.2
        elif t > 0.8:
            return 0.8 + self._ease_out_quad((t - 0.8) / 0.2) * 0.2
        else:
            return 0.2 + (t - 0.2) / 0.6 * 0.6

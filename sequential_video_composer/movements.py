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
        # Cinematic camera movements
        'truck_left',
        'truck_right',
        'static_motion',
        'shoulder_drift',
        'tracking_shot',
        'cinematic_reveal',
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
        'COLD_OPEN': 1.05,
        'EARLY_LIFE': 0.9,
        'THE_SPARK': 1.0,
        'THE_RISE': 1.0,
        'THE_CONFLICT': 1.05,
        'THE_CLIMAX': 1.1,
        'THE_FALL': 0.95,
        'LEGACY': 0.85,
        'CTA': 0.8,
    }

    def __init__(self, resolution: Tuple[int, int]):
        self.width, self.height = resolution
        self.resolution = resolution
        self.aspect_mode = 'letterbox'
        self.fast_mode = False
        self._easing_map = {
            # Energy → settle (moderate deceleration, avoids dead frames)
            'crane_up': self._ease_out_cubic,
            'crane_down': self._ease_out_cubic,
            'float_up': self._ease_out_cubic,
            'reveal_left': self._ease_out_cubic,
            'reveal_right': self._ease_out_cubic,
            'cinematic_reveal': self._ease_out_cubic,
            # Build → impact (smooth cubic for steady motion)
            'push_in': self._ease_in_out_cubic,
            'dramatic_zoom': self._ease_in_out_cubic,
            'rack_focus': self._ease_in_out_cubic,
            'dolly_zoom': self._ease_in_out_cubic,
            'spiral_zoom': self._ease_in_out_cubic,
            # Organic flow (smooth sinusoidal for natural movements)
            'gentle_drift': self._ease_in_out_sine,
            'breathing': self._ease_in_out_sine,
            'parallax_depth': self._ease_in_out_sine,
            'shoulder_drift': self._ease_in_out_sine,
            'static_motion': self._ease_in_out_sine,
            # Bouncy emphasis (gentle back overshoot for impact)
            'bounce_zoom': self._ease_out_back_gentle,
            # Mechanical consistency (linear for tracking)
            'tracking_shot': self._ease_in_out_sine,
            'truck_left': self._ease_in_out_sine,
            'truck_right': self._ease_in_out_sine,
            'map_zoom': self._ease_in_out_sine,
            'map_pan': self._ease_in_out_sine,
            'timeline_reveal': self._ease_in_out_sine,
            # Speed ramp for fast movements
            'whip_pan': self._speed_ramp,
            # Standard cinematic ease for orbits
            'orbit': self._ease_in_out_cubic,
            'push_out': self._ease_in_out_sine,
        }
        # Breathing base layer amplitude (subtle oscillation on all movements)
        self._breathing_amplitude = 0.0
        self._breathing_freq = 0.04  # Hz
        # Camera inertia overshoot disabled — it caused visible jitter at
        # movement endpoints, especially at low zoom intensities.
        self._inertia_duration_frac = 0.0
        self._inertia_overshoot = 0.0

    def _fit_image(
        self, img: PILImage.Image, target_w: int, target_h: int
    ) -> PILImage.Image:
        """Fit image to target dimensions using the configured aspect mode.

        Modes:
            fill — scale + center-crop (no black bars, slight edge crop)
            fit — scale to fit inside, preserving aspect ratio (may leave bars)
            letterbox — same as fit but with explicit black background
        """
        orig_w, orig_h = img.size
        orig_aspect = orig_w / orig_h
        target_aspect = target_w / target_h

        if self.aspect_mode == 'fill':
            if orig_aspect > target_aspect:
                fill_h = target_h
                fill_w = int(target_h * orig_aspect)
            else:
                fill_w = target_w
                fill_h = int(target_w / orig_aspect)
            resized = img.resize((fill_w, fill_h), PILImage.LANCZOS)
            cx = (fill_w - target_w) // 2
            cy = (fill_h - target_h) // 2
            return resized.crop((cx, cy, cx + target_w, cy + target_h))

        # fit / letterbox — scale to fit inside
        if orig_aspect > target_aspect:
            new_w = target_w
            new_h = int(target_w / orig_aspect)
        else:
            new_h = target_h
            new_w = int(target_h * orig_aspect)
        resized = img.resize((new_w, new_h), PILImage.LANCZOS)
        canvas = PILImage.new('RGB', (target_w, target_h), (0, 0, 0))
        paste_x = (target_w - new_w) // 2
        paste_y = (target_h - new_h) // 2
        canvas.paste(resized, (paste_x, paste_y))
        return canvas

    def create_animated_clip(
        self,
        image_path: Path,
        duration: float,
        movement_type: str,
        zoom_intensity: float = 1.08,
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
        # Apply section-aware zoom boost, capped to prevent excessive cropping
        section_mult = self.SECTION_ZOOM_MULTIPLIERS.get(section, 1.0)
        zoom_intensity = 1.0 + (zoom_intensity - 1.0) * section_mult
        zoom_intensity = min(zoom_intensity, 1.12)  # hard cap
        base_img = PILImage.open(image_path)
        base_img.load()
        if base_img.mode != 'RGB':
            base_img = base_img.convert('RGB')

        # Pan headroom: largest pan offset is ~15% of crop_w/2 = 7.5%,
        # so canvas needs ~1.15x the output resolution.  Previous 1.68x
        # buffer cropped away ~40% of every image.
        max_zoom = zoom_intensity + 0.05
        pan_buffer = 1.15
        scaled_width = int(self.width * max_zoom * pan_buffer)
        scaled_height = int(self.height * max_zoom * pan_buffer)

        base_img = self._fit_image(base_img, scaled_width, scaled_height)
        base_array = np.array(base_img)

        if color_grader and color_grade:
            base_array = color_grader.apply_grade(base_array, color_grade)

        if enable_vignette:
            base_array = self._apply_vignette(base_array)

        scaled_img = PILImage.fromarray(base_array)

        # For tilt_shift and rack_focus, pre-compute blurred version
        blurred_img = None
        if movement_type in ('tilt_shift', 'rack_focus'):
            blur_radius = int(8 * (self.height / 1080))
            blurred_img = scaled_img.filter(
                ImageFilter.GaussianBlur(radius=blur_radius)
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

        _ease_fn = self._easing_map.get(movement_type, self._ease_in_out_sine)
        _safe_duration = max(duration, 1e-6)
        _fast = self.fast_mode

        # In fast mode use INTER_AREA (3-5x faster, good for downscale)
        # In normal mode use INTER_LANCZOS4 (highest quality)
        if _has_cv2:
            import cv2 as _cv2
            _interp = _cv2.INTER_AREA if _fast else _cv2.INTER_LANCZOS4

        _breath_amp = self._breathing_amplitude
        _breath_freq = self._breathing_freq
        _inertia_frac = self._inertia_duration_frac
        _inertia_over = self._inertia_overshoot

        def make_frame(t):
            progress = t / _safe_duration
            progress = max(0.0, min(1.0, progress))
            eased = _ease_fn(progress)

            zoom, pan_x, pan_y = self._calculate_movement(
                movement_type, eased, zoom_intensity, progress
            )

            # Breathing base layer: subtle sinusoidal zoom on all movements
            if movement_type != 'static':
                breath = _breath_amp * np.sin(2 * np.pi * _breath_freq * t)
                zoom += breath

            # Camera inertia: slight overshoot at the end of movement
            if progress > (1.0 - _inertia_frac) and movement_type not in (
                'static', 'static_motion', 'breathing'
            ):
                inertia_t = (progress - (1.0 - _inertia_frac)) / _inertia_frac
                overshoot = _inertia_over * np.sin(inertia_t * np.pi)
                zoom += overshoot * (zoom_intensity - 1.0)

            zoom = min(zoom, 1.15)

            crop_w = self.width / zoom
            crop_h = self.height / zoom

            offset_x = pan_x * crop_w * 0.5
            offset_y = pan_y * crop_h * 0.5

            cx = center_x + offset_x
            cy = center_y + offset_y

            # Clamp so the crop window stays within the canvas
            half_cw = crop_w / 2.0
            half_ch = crop_h / 2.0
            cx = max(half_cw, min(cx, scaled_width - half_cw))
            cy = max(half_ch, min(cy, scaled_height - half_ch))

            src = base_np
            if movement_type == 'tilt_shift' and blurred_np is not None:
                src = self._apply_tilt_shift_blend_np(
                    base_np, blurred_np, progress
                )
            elif movement_type == 'rack_focus' and blurred_np is not None:
                src = self._apply_rack_focus_blend(
                    base_np, blurred_np, progress
                )

            # Sub-pixel crop using affine transform for smooth motion
            if _has_cv2:
                patch_w = int(round(crop_w))
                patch_h = int(round(crop_h))
                cropped = _cv2.getRectSubPix(
                    src, (patch_w, patch_h), (float(cx), float(cy))
                )
                frame = _cv2.resize(
                    cropped, (out_w, out_h), interpolation=_interp
                )
            else:
                l_i = int(cx - half_cw)
                t_i = int(cy - half_ch)
                l_i = max(0, l_i)
                t_i = max(0, t_i)
                r_i = min(l_i + int(crop_w), scaled_width)
                b_i = min(t_i + int(crop_h), scaled_height)
                cropped = src[t_i:b_i, l_i:r_i]
                pil_crop = PILImage.fromarray(cropped)
                frame = np.array(
                    pil_crop.resize((out_w, out_h), PILImage.LANCZOS)
                )

            if movement_type == 'whip_pan':
                frame = self._apply_motion_blur(frame, progress)

            return frame

        from moviepy.video.VideoClip import VideoClip
        clip = VideoClip(make_frame, duration=duration)
        clip = clip.set_fps(30)
        return clip

    # ---- Visual effect helpers ----

    def _apply_vignette(self, image: np.ndarray) -> np.ndarray:
        """Apply a cinematic vignette, attenuated on dark images."""
        avg_brightness = np.mean(image)
        if avg_brightness < 80:
            return image
        # Scale strength with brightness: full at 160+, reduced below
        base_strength = 0.04
        brightness_scale = min(1.0, max(0.3, (avg_brightness - 80) / 80.0))
        strength = base_strength * brightness_scale
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

    def _apply_rack_focus_blend(
        self, sharp_np: np.ndarray, blurred_np: np.ndarray, progress: float
    ) -> np.ndarray:
        """Simulate focus pull from background to foreground (or vice versa).

        Starts blurred (background in focus) and transitions to sharp (foreground
        in focus) using a radial center-weighted mask that shifts over time.
        Creates a cinematic depth-of-field shift effect.
        """
        h, w = sharp_np.shape[:2]
        sharp_f = sharp_np.astype(np.float32)
        blur_f = blurred_np.astype(np.float32)

        # Focus pull: blend from blurred to sharp over the animation
        # First 40%: mostly blurred (background focus)
        # Middle 20%: transition (rack focus moment)
        # Last 40%: mostly sharp (foreground focus)
        if progress < 0.4:
            blend = progress / 0.4 * 0.2  # 0 → 0.2
        elif progress < 0.6:
            blend = 0.2 + (progress - 0.4) / 0.2 * 0.6  # 0.2 → 0.8
        else:
            blend = 0.8 + (progress - 0.6) / 0.4 * 0.2  # 0.8 → 1.0

        # Radial mask: center is sharper, edges stay blurrier longer
        cy, cx = h // 2, w // 2
        Y, X = np.ogrid[:h, :w]
        dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2).astype(np.float32)
        max_dist = np.sqrt(cx ** 2 + cy ** 2)
        radial = np.clip(1.0 - dist / max_dist, 0, 1)

        # Combine linear blend with radial weighting
        mask = np.clip(blend + radial * 0.3, 0, 1)
        mask = mask[:, :, np.newaxis]

        result = blur_f * (1.0 - mask) + sharp_f * mask
        return np.clip(result, 0, 255).astype(np.uint8)

    def _apply_output_sharpen(self, frame: np.ndarray) -> np.ndarray:
        """Apply a gentle unsharp-mask to the final output frame.

        Counteracts softness introduced by the crop→resize pipeline.
        Uses a lightweight 3×3 kernel for speed.
        """
        try:
            import cv2
            blurred = cv2.GaussianBlur(frame, (0, 0), sigmaX=1.0)
            return cv2.addWeighted(frame, 1.4, blurred, -0.4, 0)
        except ImportError:
            img = PILImage.fromarray(frame)
            img = img.filter(ImageFilter.UnsharpMask(radius=1, percent=80, threshold=2))
            return np.array(img)

    def _apply_motion_blur(self, frame: np.ndarray, progress: float) -> np.ndarray:
        """Apply subtle directional motion blur during the fastest phase.

        Only activates in the narrow middle band of the animation and uses
        a very small kernel so the frame stays sharp.
        """
        blur_strength = max(0, 1.0 - abs(progress - 0.5) * 5.0)
        if blur_strength < 0.15:
            return frame

        kernel_size = max(2, int(blur_strength * 2 * (self.width / 1920)))
        if kernel_size < 2:
            return frame

        try:
            import cv2
            kernel = np.zeros((kernel_size, kernel_size))
            kernel[kernel_size // 2, :] = 1.0 / kernel_size
            blurred = cv2.filter2D(frame, -1, kernel)
            # Blend: keep 60% sharp, 40% blurred for subtlety
            return cv2.addWeighted(frame, 0.6, blurred, 0.4, 0)
        except ImportError:
            img = PILImage.fromarray(frame)
            img = img.filter(ImageFilter.BoxBlur(max(1, kernel_size // 2)))
            return np.array(img)

    def _apply_handheld_shake(self, frame: np.ndarray, t: float) -> np.ndarray:
        """Apply subtle handheld camera shake by shifting pixels.

        Uses a spectral profile with low-frequency sway (1-2 Hz body motion)
        and light mid-frequency drift (3-5 Hz hand tremor).
        np.pad with 'edge' mode replicates border pixels to prevent black edges.
        """
        scale = self.width / 1920
        # Low-frequency body sway (1.2 Hz, 1.8 Hz) — reduced
        lo_dx = 0.6 * scale * np.sin(t * 1.2 * 2 * np.pi + 0.3)
        lo_dy = 0.5 * scale * np.sin(t * 1.8 * 2 * np.pi + 1.1)
        # Mid-frequency hand tremor (3.7 Hz, 5.3 Hz) — reduced
        mid_dx = 0.3 * scale * np.sin(t * 3.7 * 2 * np.pi)
        mid_dy = 0.2 * scale * np.cos(t * 5.3 * 2 * np.pi)
        # High-frequency micro-tremor removed (too jittery)
        hi_dx = 0.0
        hi_dy = 0.0

        dx = int(lo_dx + mid_dx + hi_dx)
        dy = int(lo_dy + mid_dy + hi_dy)

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

        Args:
            movement_type: One of MOVEMENT_TYPES (e.g. 'zoom_in', 'orbit').
            eased: Progress 0→1 after easing function is applied.
            zoom_intensity: Base zoom multiplier (typically 1.05–1.10).
            raw_progress: Linear 0→1 progress before easing (used for
                some effects that need raw time, like sinusoidal drift).

        Returns:
            (zoom, pan_x, pan_y) where zoom is a scale factor (1.0 = no zoom)
            and pan_x/pan_y are fractional offsets of the frame centre.
        """
        # Classic movements
        if movement_type == 'zoom_in':
            zoom = 1.0 + (zoom_intensity - 1.0) * eased * 1.0
            return zoom, 0, 0

        elif movement_type == 'zoom_out':
            zoom = zoom_intensity - (zoom_intensity - 1.0) * eased * 0.8
            return zoom, 0, 0

        elif movement_type == 'pan_left':
            zoom = 1.0 + (zoom_intensity - 1.0) * 0.5
            pan_x = -0.10 * eased
            return zoom, pan_x, 0

        elif movement_type == 'pan_right':
            zoom = 1.0 + (zoom_intensity - 1.0) * 0.5
            pan_x = 0.10 * eased
            return zoom, pan_x, 0

        elif movement_type == 'pan_up':
            zoom = 1.0 + (zoom_intensity - 1.0) * 0.5
            pan_y = -0.10 * eased
            return zoom, 0, pan_y

        elif movement_type == 'pan_down':
            zoom = 1.0 + (zoom_intensity - 1.0) * 0.5
            pan_y = 0.10 * eased
            return zoom, 0, pan_y

        elif movement_type == 'diagonal_tl_br':
            zoom = 1.0 + (zoom_intensity - 1.0) * eased * 0.8
            pan_x = 0.10 * eased
            pan_y = 0.10 * eased
            return zoom, pan_x, pan_y

        elif movement_type == 'diagonal_tr_bl':
            zoom = 1.0 + (zoom_intensity - 1.0) * eased * 0.8
            pan_x = -0.10 * eased
            pan_y = 0.10 * eased
            return zoom, pan_x, pan_y

        elif movement_type == 'breathing':
            zoom = 1.0 + 0.04 * np.sin(raw_progress * np.pi)
            pan_y = 0.01 * np.sin(raw_progress * np.pi * 0.5)
            return zoom, 0, pan_y

        elif movement_type == 'dramatic_zoom':
            zoom = 1.0 + (zoom_intensity - 1.0) * 1.2 * eased
            return zoom, 0, 0

        elif movement_type == 'gentle_drift':
            zoom = 1.0 + (zoom_intensity - 1.0) * 0.6
            pan_x = 0.08 * np.sin(raw_progress * np.pi)
            pan_y = 0.05 * np.cos(raw_progress * np.pi)
            return zoom, pan_x, pan_y

        elif movement_type == 'focus_center':
            zoom = 1.0 + (zoom_intensity - 1.0) * eased * 1.0
            return zoom, 0, 0

        elif movement_type == 'minimal':
            zoom = 1.0 + (zoom_intensity - 1.0) * eased * 0.6
            return zoom, 0, 0

        elif movement_type == 'static':
            return 1.0, 0, 0

        # ---- Documentary-style advanced movements ----

        elif movement_type == 'parallax_depth':
            zoom = 1.0 + (zoom_intensity - 1.0) * eased * 0.8
            phase = raw_progress * np.pi
            pan_x = 0.08 * np.sin(phase)
            pan_y = 0.04 * np.sin(phase * 0.7 + 0.3)
            return zoom, pan_x, pan_y

        elif movement_type == 'push_in':
            zoom = 1.0 + (zoom_intensity - 1.0) * 1.0 * eased
            pan_y = -0.06 * eased
            return zoom, 0, pan_y

        elif movement_type == 'push_out':
            zoom = zoom_intensity - (zoom_intensity - 1.0) * eased * 0.8
            pan_y = 0.06 * eased
            return zoom, 0, pan_y

        elif movement_type == 'orbit':
            zoom = 1.0 + (zoom_intensity - 1.0) * 0.7
            angle = eased * np.pi * 1.0
            pan_x = 0.08 * np.sin(angle)
            pan_y = 0.04 * (1 - np.cos(angle))
            return zoom, pan_x, pan_y

        elif movement_type == 'whip_pan':
            zoom = 1.0 + (zoom_intensity - 1.0) * 0.5
            pan_x = 0.10 * eased
            return zoom, pan_x, 0

        elif movement_type == 'dolly_zoom':
            zoom = 1.0 + (zoom_intensity - 1.0) * 1.0 * eased
            pan_y = -0.05 * eased
            pan_x = 0.03 * eased
            return zoom, pan_x, pan_y

        elif movement_type == 'handheld_drift':
            zoom = 1.0 + (zoom_intensity - 1.0) * 0.4 * eased
            pan_x = 0.04 * np.sin(raw_progress * np.pi * 1.2)
            pan_y = 0.025 * np.sin(raw_progress * np.pi * 0.8 + 0.5)
            return zoom, pan_x, pan_y

        elif movement_type == 'crane_up':
            zoom = 1.0 + (zoom_intensity - 1.0) * (1.0 - eased * 0.1)
            pan_y = -0.10 * eased
            return zoom, 0, pan_y

        elif movement_type == 'crane_down':
            zoom = 1.0 + (zoom_intensity - 1.0) * eased * 0.9
            pan_y = 0.10 * eased
            pan_x = 0.015 * np.sin(raw_progress * np.pi)
            return zoom, pan_x, pan_y

        elif movement_type == 'spiral_zoom':
            zoom = 1.0 + (zoom_intensity - 1.0) * 1.0 * eased
            angle = eased * np.pi * 2.0
            radius = 0.04 * (1.0 - eased)
            pan_x = radius * np.cos(angle)
            pan_y = radius * np.sin(angle)
            return zoom, pan_x, pan_y

        elif movement_type == 'tilt_shift':
            zoom = 1.0 + (zoom_intensity - 1.0) * 0.7
            pan_x = 0.08 * eased
            pan_y = -0.04 * eased
            return zoom, pan_x, pan_y

        elif movement_type == 'dutch_tilt':
            zoom = 1.0 + (zoom_intensity - 1.0) * eased * 0.7
            tilt = 0.02 * eased
            pan_x = tilt
            pan_y = tilt * 0.5
            return zoom, pan_x, pan_y

        elif movement_type == 'rack_focus':
            zoom = 1.0 + (zoom_intensity - 1.0) * 1.0 * eased
            return zoom, 0, 0

        elif movement_type == 'bounce_zoom':
            overshoot = 1.0 + 0.015 * np.sin(eased * np.pi * 2) * (1.0 - eased)
            zoom = 1.0 + (zoom_intensity - 1.0) * eased * 0.8 * overshoot
            return zoom, 0, 0

        elif movement_type == 'float_up':
            zoom = 1.0 + (zoom_intensity - 1.0) * eased * 0.7
            pan_y = -0.08 * eased
            pan_x = 0.02 * np.sin(raw_progress * np.pi)
            return zoom, pan_x, pan_y

        elif movement_type == 'reveal_left':
            zoom = 1.0 + (zoom_intensity - 1.0) * 0.5
            pan_x = 0.10 * (1.0 - eased)
            return zoom, pan_x, 0

        elif movement_type == 'reveal_right':
            zoom = 1.0 + (zoom_intensity - 1.0) * 0.5
            pan_x = -0.10 * (1.0 - eased)
            return zoom, pan_x, 0

        elif movement_type == 'map_zoom':
            zoom = 1.0 + (zoom_intensity - 1.0) * 1.0 * eased
            pan_y = 0.05 * eased
            return zoom, 0, pan_y

        elif movement_type == 'map_pan':
            zoom = 1.0 + (zoom_intensity - 1.0) * 0.5
            pan_x = 0.12 * eased
            pan_y = 0.03 * np.sin(eased * np.pi)
            return zoom, pan_x, pan_y

        elif movement_type == 'timeline_reveal':
            zoom = 1.0 + (zoom_intensity - 1.0) * 0.5
            pan_x = -0.12 * (1.0 - eased)
            return zoom, pan_x, 0

        # ---- Cinematic camera movements (v5.0) ----

        elif movement_type == 'truck_left':
            zoom = 1.0 + (zoom_intensity - 1.0) * 0.5
            pan_x = -0.10 * eased
            pan_y = 0.015 * np.sin(raw_progress * np.pi)
            return zoom, pan_x, pan_y

        elif movement_type == 'truck_right':
            zoom = 1.0 + (zoom_intensity - 1.0) * 0.5
            pan_x = 0.10 * eased
            pan_y = 0.015 * np.sin(raw_progress * np.pi)
            return zoom, pan_x, pan_y

        elif movement_type == 'static_motion':
            zoom = 1.0 + 0.008 * np.sin(raw_progress * np.pi)
            pan_x = 0.004 * np.sin(raw_progress * np.pi * 0.8)
            pan_y = 0.004 * np.cos(raw_progress * np.pi * 0.6)
            return zoom, pan_x, pan_y

        elif movement_type == 'shoulder_drift':
            zoom = 1.0 + (zoom_intensity - 1.0) * 0.5
            pan_x = 0.01 * eased + 0.005
            pan_y = 0.005 * eased
            return zoom, pan_x, pan_y

        elif movement_type == 'tracking_shot':
            zoom = 1.0 + (zoom_intensity - 1.0) * eased * 0.8
            pan_x = 0.10 * eased
            pan_y = -0.03 * eased
            return zoom, pan_x, pan_y

        elif movement_type == 'cinematic_reveal':
            zoom = 1.0 + (zoom_intensity - 1.0) * 1.0 * (1.0 - eased)
            pan_y = -0.07 * (1.0 - eased)
            pan_x = 0.04 * (1.0 - eased)
            return zoom, pan_x, pan_y

        else:
            zoom = 1.0 + (zoom_intensity - 1.0) * eased * 0.8
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
        """Elastic easing for subtle bouncy emphasis.

        Clamped to [0.0, 1.05] to keep overshoot barely perceptible.
        """
        if t <= 0:
            return 0.0
        if t >= 1.0:
            return 1.0
        raw = 2.0 ** (-10.0 * t) * np.sin((t * 10.0 - 0.75) * (2 * np.pi / 3)) + 1.0
        return float(np.clip(raw, 0.0, 1.05))

    def _ease_out_cubic(self, t: float) -> float:
        """Cubic ease-out — decelerates smoothly without going dead."""
        return 1.0 - (1.0 - t) ** 3

    def _ease_out_back(self, t: float, overshoot: float = 1.70158) -> float:
        """Back easing — overshoots then settles for punchy emphasis."""
        t -= 1.0
        return t * t * ((overshoot + 1) * t + overshoot) + 1.0

    def _ease_out_back_gentle(self, t: float) -> float:
        """Gentle back easing with minimal overshoot for smooth bounce."""
        return self._ease_out_back(t, overshoot=0.5)

    def _speed_ramp(self, t: float) -> float:
        """Speed ramp: slow start, fast middle, slow end."""
        if t < 0.2:
            return self._ease_in_quad(t / 0.2) * 0.2
        elif t > 0.8:
            return 0.8 + self._ease_out_quad((t - 0.8) / 0.2) * 0.2
        else:
            return 0.2 + (t - 0.2) / 0.6 * 0.6

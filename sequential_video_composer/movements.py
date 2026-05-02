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

    def __init__(self, resolution: Tuple[int, int]):
        self.width, self.height = resolution
        self.resolution = resolution

    def create_animated_clip(
        self,
        image_path: Path,
        duration: float,
        movement_type: str,
        zoom_intensity: float = 1.15,
        color_grader: 'ColorGrading' = None,
        color_grade: str = None,
        enable_vignette: bool = False
    ):
        """Create an animated clip with the specified movement style and effects.

        Uses a memory-efficient approach with make_frame instead of creating
        many sub-clips. Pre-scales the image larger to allow for smooth
        zooming without per-frame resizing.
        """
        base_img = PILImage.open(image_path)
        if base_img.mode != 'RGB':
            base_img = base_img.convert('RGB')

        max_zoom = max(zoom_intensity, 1.35)
        scaled_width = int(self.width * max_zoom * 1.3)
        scaled_height = int(self.height * max_zoom * 1.3)

        orig_width, orig_height = base_img.size
        orig_aspect = orig_width / orig_height
        target_aspect = scaled_width / scaled_height

        if orig_aspect > target_aspect:
            new_width = scaled_width
            new_height = int(scaled_width / orig_aspect)
        else:
            new_height = scaled_height
            new_width = int(scaled_height * orig_aspect)

        resized_img = base_img.resize((new_width, new_height), PILImage.LANCZOS)

        canvas = PILImage.new('RGB', (scaled_width, scaled_height), (0, 0, 0))
        paste_x = (scaled_width - new_width) // 2
        paste_y = (scaled_height - new_height) // 2
        canvas.paste(resized_img, (paste_x, paste_y))

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

        def make_frame(t):
            progress = t / duration if duration > 0 else 0
            progress = max(0.0, min(1.0, progress))
            eased = self._ease_in_out_cubic(progress)

            zoom, pan_x, pan_y = self._calculate_movement(
                movement_type, eased, zoom_intensity, progress
            )

            crop_width = self.width / zoom
            crop_height = self.height / zoom

            offset_x = pan_x * crop_width * 0.5
            offset_y = pan_y * crop_height * 0.5

            left = center_x - crop_width / 2.0 + offset_x
            top = center_y - crop_height / 2.0 + offset_y

            left = max(0.0, min(left, scaled_width - crop_width))
            top = max(0.0, min(top, scaled_height - crop_height))

            sx = crop_width / out_w
            sy = crop_height / out_h

            src_img = scaled_img

            # Tilt-shift: blend sharp center with blurred edges
            if movement_type == 'tilt_shift' and blurred_img is not None:
                src_img = self._apply_tilt_shift_blend(
                    scaled_img, blurred_img, progress
                )

            final = src_img.transform(
                (out_w, out_h),
                PILImage.AFFINE,
                (sx, 0.0, left, 0.0, sy, top),
                resample=PILImage.BICUBIC,
            )

            frame = np.array(final)

            # Whip pan motion blur
            if movement_type == 'whip_pan':
                frame = self._apply_motion_blur(frame, progress)

            # Handheld micro-shake
            if movement_type == 'handheld_drift':
                frame = self._apply_handheld_shake(frame, t)

            return frame

        from moviepy.video.VideoClip import VideoClip
        clip = VideoClip(make_frame, duration=duration)
        clip = clip.set_fps(30)
        return clip

    # ---- Visual effect helpers ----

    def _apply_vignette(self, image: np.ndarray) -> np.ndarray:
        """Apply a subtle vignette effect to the image."""
        rows, cols = image.shape[:2]
        X = np.arange(0, cols)
        Y = np.arange(0, rows)
        X, Y = np.meshgrid(X, Y)
        center_x, center_y = cols / 2, rows / 2
        distance = np.sqrt((X - center_x) ** 2 + (Y - center_y) ** 2)
        max_distance = np.sqrt(center_x ** 2 + center_y ** 2)
        vignette = 1 - (distance / max_distance) * 0.4
        vignette = np.clip(vignette, 0.6, 1.0)
        vignette = np.dstack([vignette] * 3)
        return (image * vignette).astype(np.uint8)

    def _apply_tilt_shift_blend(
        self, sharp: PILImage.Image, blurred: PILImage.Image, progress: float
    ) -> PILImage.Image:
        """Blend sharp center band with blurred edges for miniature effect."""
        w, h = sharp.size
        sharp_arr = np.array(sharp).astype(np.float32)
        blur_arr = np.array(blurred).astype(np.float32)

        # The focus band slowly drifts with progress
        focus_center = 0.45 + 0.1 * progress
        band_width = 0.25

        y_coords = np.linspace(0, 1, h)[:, np.newaxis]
        dist = np.abs(y_coords - focus_center)
        mask = np.clip((dist - band_width * 0.5) / (band_width * 0.3 + 1e-6), 0, 1)
        mask = mask[:, :, np.newaxis]

        result = sharp_arr * (1 - mask) + blur_arr * mask
        return PILImage.fromarray(result.astype(np.uint8))

    def _apply_motion_blur(self, frame: np.ndarray, progress: float) -> np.ndarray:
        """Apply horizontal motion blur during the fast middle of whip pan."""
        # Blur is strongest in the middle 40% of the animation
        blur_strength = max(0, 1.0 - abs(progress - 0.5) * 4.0)
        if blur_strength < 0.05:
            return frame

        kernel_size = int(blur_strength * 25 * (self.width / 1920))
        if kernel_size < 2:
            return frame

        img = PILImage.fromarray(frame)
        img = img.filter(ImageFilter.BoxBlur(kernel_size))
        return np.array(img)

    def _apply_handheld_shake(self, frame: np.ndarray, t: float) -> np.ndarray:
        """Apply handheld camera micro-shake by shifting pixels."""
        amplitude = 4.0 * (self.width / 1920)
        freq1, freq2 = 3.7, 5.3
        dx = int(amplitude * np.sin(t * freq1 * 2 * np.pi))
        dy = int(amplitude * np.cos(t * freq2 * 2 * np.pi))

        if dx == 0 and dy == 0:
            return frame

        h, w = frame.shape[:2]
        result = np.zeros_like(frame)

        src_x1 = max(0, dx)
        src_y1 = max(0, dy)
        src_x2 = min(w, w + dx)
        src_y2 = min(h, h + dy)
        dst_x1 = max(0, -dx)
        dst_y1 = max(0, -dy)
        dst_x2 = dst_x1 + (src_x2 - src_x1)
        dst_y2 = dst_y1 + (src_y2 - src_y1)

        result[dst_y1:dst_y2, dst_x1:dst_x2] = frame[src_y1:src_y2, src_x1:src_x2]
        return result

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
            zoom = 1.0 + (zoom_intensity - 1.0) * eased
            return zoom, 0, 0

        elif movement_type == 'zoom_out':
            zoom = zoom_intensity - (zoom_intensity - 1.0) * eased
            return zoom, 0, 0

        elif movement_type == 'pan_left':
            zoom = 1.0 + (zoom_intensity - 1.0) * 0.5
            pan_x = -0.12 * eased
            return zoom, pan_x, 0

        elif movement_type == 'pan_right':
            zoom = 1.0 + (zoom_intensity - 1.0) * 0.5
            pan_x = 0.12 * eased
            return zoom, pan_x, 0

        elif movement_type == 'pan_up':
            zoom = 1.0 + (zoom_intensity - 1.0) * 0.5
            pan_y = -0.12 * eased
            return zoom, 0, pan_y

        elif movement_type == 'pan_down':
            zoom = 1.0 + (zoom_intensity - 1.0) * 0.5
            pan_y = 0.12 * eased
            return zoom, 0, pan_y

        elif movement_type == 'diagonal_tl_br':
            zoom = 1.0 + (zoom_intensity - 1.0) * eased * 0.7
            pan_x = 0.08 * eased
            pan_y = 0.08 * eased
            return zoom, pan_x, pan_y

        elif movement_type == 'diagonal_tr_bl':
            zoom = 1.0 + (zoom_intensity - 1.0) * eased * 0.7
            pan_x = -0.08 * eased
            pan_y = 0.08 * eased
            return zoom, pan_x, pan_y

        elif movement_type == 'breathing':
            zoom = 1.0 + 0.08 * np.sin(raw_progress * np.pi * 2)
            return zoom, 0, 0

        elif movement_type == 'dramatic_zoom':
            zoom = 1.0 + (zoom_intensity - 1.0) * 1.2 * self._dramatic_ease(eased)
            return zoom, 0, 0

        elif movement_type == 'gentle_drift':
            zoom = 1.0 + (zoom_intensity - 1.0) * 0.6
            pan_x = 0.06 * np.sin(raw_progress * np.pi)
            pan_y = 0.03 * np.cos(raw_progress * np.pi)
            return zoom, pan_x, pan_y

        elif movement_type == 'focus_center':
            zoom = 1.0 + (zoom_intensity - 1.0) * eased * 0.8
            return zoom, 0, 0

        elif movement_type == 'minimal':
            zoom = 1.0 + (zoom_intensity - 1.0) * eased * 0.5
            return zoom, 0, 0

        elif movement_type == 'static':
            return 1.0, 0, 0

        # ---- Documentary-style advanced movements ----

        elif movement_type == 'parallax_depth':
            # Simulates 2.5D depth: slow zoom with pronounced layered pan.
            # The foreground (pan) moves faster than the background (zoom)
            # creating a visible depth separation effect.
            zoom = 1.0 + (zoom_intensity - 1.0) * eased * 0.9
            phase = raw_progress * np.pi
            pan_x = 0.10 * np.sin(phase)
            pan_y = 0.05 * np.sin(phase * 0.7 + 0.3)
            return zoom, pan_x, pan_y

        elif movement_type == 'push_in':
            # Cinematic slow push into the subject with acceleration.
            zoom = 1.0 + (zoom_intensity - 1.0) * 1.5 * self._ease_in_quad(eased)
            pan_y = -0.03 * eased
            return zoom, 0, pan_y

        elif movement_type == 'push_out':
            # Reverse push — pulling away from the subject.
            zoom = zoom_intensity - (zoom_intensity - 1.0) * self._ease_in_quad(eased) * 1.2
            pan_y = 0.03 * eased
            return zoom, 0, pan_y

        elif movement_type == 'orbit':
            # Camera arcs around the subject in an elliptical path.
            zoom = 1.0 + (zoom_intensity - 1.0) * 0.7
            angle = eased * np.pi * 0.8
            pan_x = 0.12 * np.sin(angle)
            pan_y = 0.04 * (1 - np.cos(angle))
            return zoom, pan_x, pan_y

        elif movement_type == 'whip_pan':
            # Fast horizontal pan with ease-in / ease-out and motion blur.
            whip = self._ease_in_out_quint(eased)
            zoom = 1.0 + (zoom_intensity - 1.0) * 0.5
            pan_x = 0.20 * whip
            return zoom, pan_x, 0

        elif movement_type == 'dolly_zoom':
            # Vertigo effect: zoom in while pulling camera back (or vice versa).
            # Zoom increases while the visible area stays roughly the same size,
            # creating a disorienting perspective shift.
            zoom = 1.0 + (zoom_intensity - 1.0) * 1.8 * eased
            # Counter-pan to create vertigo effect
            pan_y = -0.04 * np.sin(eased * np.pi)
            pan_x = 0.02 * np.sin(eased * np.pi * 2)
            return zoom, pan_x, pan_y

        elif movement_type == 'handheld_drift':
            # Organic handheld camera feel with irregular micro-movements.
            zoom = 1.0 + (zoom_intensity - 1.0) * 0.5
            pan_x = 0.04 * np.sin(raw_progress * np.pi * 3.1)
            pan_x += 0.02 * np.sin(raw_progress * np.pi * 7.3)
            pan_y = 0.03 * np.cos(raw_progress * np.pi * 2.7)
            pan_y += 0.015 * np.cos(raw_progress * np.pi * 5.9)
            return zoom, pan_x, pan_y

        elif movement_type == 'crane_up':
            # Vertical crane shot moving upward, slight zoom out.
            zoom = 1.0 + (zoom_intensity - 1.0) * (1.0 - eased * 0.3)
            pan_y = -0.15 * self._ease_in_out_quart(eased)
            return zoom, 0, pan_y

        elif movement_type == 'crane_down':
            # Vertical crane shot moving downward, slight zoom in.
            zoom = 1.0 + (zoom_intensity - 1.0) * eased * 0.7
            pan_y = 0.15 * self._ease_in_out_quart(eased)
            return zoom, 0, pan_y

        elif movement_type == 'spiral_zoom':
            # Zoom in with a slow spiral pan creating a dramatic reveal.
            zoom = 1.0 + (zoom_intensity - 1.0) * 1.3 * eased
            angle = eased * np.pi * 2.0
            radius = 0.08 * (1.0 - eased)
            pan_x = radius * np.cos(angle)
            pan_y = radius * np.sin(angle)
            return zoom, pan_x, pan_y

        elif movement_type == 'tilt_shift':
            # Slow pan with tilt-shift (miniature) effect.
            zoom = 1.0 + (zoom_intensity - 1.0) * 0.6
            pan_x = 0.08 * eased
            pan_y = -0.03 * eased
            return zoom, pan_x, pan_y

        elif movement_type == 'dutch_tilt':
            # Slow zoom with diagonal drift suggesting camera tilt.
            zoom = 1.0 + (zoom_intensity - 1.0) * eased * 1.0
            tilt = 0.07 * np.sin(eased * np.pi * 0.5)
            pan_x = tilt
            pan_y = tilt * 0.6
            return zoom, pan_x, pan_y

        elif movement_type == 'rack_focus':
            # Quick zoom to simulate rack focus pull — fast initial movement
            # that decelerates.
            zoom = 1.0 + (zoom_intensity - 1.0) * self._ease_out_expo(eased)
            return zoom, 0, 0

        elif movement_type == 'bounce_zoom':
            # Zoom in with a slight overshoot and settle for emphasis.
            overshoot = 1.0 + 0.15 * np.sin(eased * np.pi * 3) * (1.0 - eased)
            zoom = 1.0 + (zoom_intensity - 1.0) * eased * overshoot
            return zoom, 0, 0

        elif movement_type == 'float_up':
            # Gentle upward drift with slow zoom — dreamy, reflective feel.
            zoom = 1.0 + (zoom_intensity - 1.0) * eased * 0.8
            pan_y = -0.10 * self._ease_in_out_sine(eased)
            pan_x = 0.025 * np.sin(raw_progress * np.pi)
            return zoom, pan_x, pan_y

        elif movement_type == 'reveal_left':
            # Pan from right to left to reveal the scene.
            zoom = 1.0 + (zoom_intensity - 1.0) * 0.5
            pan_x = 0.15 * (1.0 - self._ease_in_out_quart(eased))
            return zoom, pan_x, 0

        elif movement_type == 'reveal_right':
            # Pan from left to right to reveal the scene.
            zoom = 1.0 + (zoom_intensity - 1.0) * 0.5
            pan_x = -0.15 * (1.0 - self._ease_in_out_quart(eased))
            return zoom, pan_x, 0

        else:
            zoom = 1.0 + (zoom_intensity - 1.0) * eased * 0.5
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

"""Ken Burns movement styles for dynamic image animations."""

from pathlib import Path
from typing import Tuple, TYPE_CHECKING

import numpy as np
from PIL import Image as PILImage

if TYPE_CHECKING:
    from .color_grading import ColorGrading


class MovementStyles:
    """Ken Burns movement styles for dynamic image animations."""

    MOVEMENT_TYPES = [
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
        'static'
    ]

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
        
        Uses a memory-efficient approach with make_frame instead of creating many sub-clips.
        Pre-scales the image larger to allow for smooth zooming without per-frame resizing.
        """
        base_img = PILImage.open(image_path)
        if base_img.mode != 'RGB':
            base_img = base_img.convert('RGB')
        
        max_zoom = max(zoom_intensity, 1.1)
        scaled_width = int(self.width * max_zoom * 1.05)
        scaled_height = int(self.height * max_zoom * 1.05)
        
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
        
        center_x = scaled_width / 2.0
        center_y = scaled_height / 2.0
        
        def make_frame(t):
            progress = t / duration if duration > 0 else 0
            progress = max(0.0, min(1.0, progress))
            eased = self._ease_in_out_cubic(progress)
            
            zoom, pan_x, pan_y = self._calculate_movement(
                movement_type, eased, zoom_intensity
            )
            
            crop_width = self.width / zoom
            crop_height = self.height / zoom
            
            offset_x = pan_x * crop_width * 0.5
            offset_y = pan_y * crop_height * 0.5
            
            left = center_x - crop_width / 2.0 + offset_x
            top = center_y - crop_height / 2.0 + offset_y
            right = left + crop_width
            bottom = top + crop_height
            
            left = max(0, min(left, scaled_width - crop_width))
            top = max(0, min(top, scaled_height - crop_height))
            right = left + crop_width
            bottom = top + crop_height
            
            cropped = scaled_img.crop((int(left), int(top), int(right), int(bottom)))
            
            final = cropped.resize(self.resolution, PILImage.LANCZOS)
            return np.array(final)
        
        from moviepy.video.VideoClip import VideoClip
        clip = VideoClip(make_frame, duration=duration)
        clip = clip.set_fps(30)
        return clip

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

    def _calculate_movement(
        self,
        movement_type: str,
        progress: float,
        zoom_intensity: float
    ) -> Tuple[float, float, float]:
        """Calculate zoom and pan values for the given movement type."""
        if movement_type == 'zoom_in':
            zoom = 1.0 + (zoom_intensity - 1.0) * progress
            return zoom, 0, 0

        elif movement_type == 'zoom_out':
            zoom = zoom_intensity - (zoom_intensity - 1.0) * progress
            return zoom, 0, 0

        elif movement_type == 'pan_left':
            zoom = 1.0 + (zoom_intensity - 1.0) * 0.5
            pan_x = -0.04 * progress
            return zoom, pan_x, 0

        elif movement_type == 'pan_right':
            zoom = 1.0 + (zoom_intensity - 1.0) * 0.5
            pan_x = 0.04 * progress
            return zoom, pan_x, 0

        elif movement_type == 'pan_up':
            zoom = 1.0 + (zoom_intensity - 1.0) * 0.5
            pan_y = -0.04 * progress
            return zoom, 0, pan_y

        elif movement_type == 'pan_down':
            zoom = 1.0 + (zoom_intensity - 1.0) * 0.5
            pan_y = 0.04 * progress
            return zoom, 0, pan_y

        elif movement_type == 'diagonal_tl_br':
            zoom = 1.0 + (zoom_intensity - 1.0) * progress * 0.7
            pan_x = 0.02 * progress
            pan_y = 0.02 * progress
            return zoom, pan_x, pan_y

        elif movement_type == 'diagonal_tr_bl':
            zoom = 1.0 + (zoom_intensity - 1.0) * progress * 0.7
            pan_x = -0.02 * progress
            pan_y = 0.02 * progress
            return zoom, pan_x, pan_y

        elif movement_type == 'breathing':
            zoom = 1.0 + 0.04 * np.sin(progress * np.pi * 2)
            return zoom, 0, 0

        elif movement_type == 'dramatic_zoom':
            zoom = 1.0 + (zoom_intensity - 1.0) * 1.2 * self._dramatic_ease(progress)
            return zoom, 0, 0

        elif movement_type == 'gentle_drift':
            zoom = 1.0 + (zoom_intensity - 1.0) * 0.5
            pan_x = 0.02 * np.sin(progress * np.pi)
            pan_y = 0.01 * np.cos(progress * np.pi)
            return zoom, pan_x, pan_y

        elif movement_type == 'focus_center':
            zoom = 1.0 + (zoom_intensity - 1.0) * progress * 0.8
            return zoom, 0, 0

        elif movement_type == 'minimal':
            zoom = 1.0 + (zoom_intensity - 1.0) * progress * 0.5
            return zoom, 0, 0

        elif movement_type == 'static':
            return 1.0, 0, 0

        else:
            zoom = 1.0 + (zoom_intensity - 1.0) * progress * 0.5
            return zoom, 0, 0

    def _ease_in_out_cubic(self, t: float) -> float:
        """Cubic easing function for smooth animations."""
        return 3 * t * t - 2 * t * t * t

    def _dramatic_ease(self, t: float) -> float:
        """Dramatic easing for impactful moments."""
        return 0.5 * (np.sin((t - 0.5) * np.pi) + 1)

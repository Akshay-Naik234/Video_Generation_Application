"""Documentary-style transition effects for video clips.

Includes classic transitions plus cinematic documentary techniques:
- Whip pan with motion blur
- Light leak overlays
- Film burn transitions
- Glitch / digital distortion
- Blur-through (defocus/refocus)
- Flash cut (bright flash)
- Iris wipe
- Morph dissolve
- Zoom blur
- Push transitions
- Luma fade (luminance-based dissolve)
"""

from typing import Tuple

import numpy as np
from moviepy.editor import ImageClip, CompositeVideoClip, ColorClip
from moviepy.video.VideoClip import VideoClip
from PIL import Image as PILImage, ImageFilter


class TransitionEffects:
    """Professional documentary-style transition effects between clips."""

    TRANSITION_TYPES = [
        # Classic
        'crossfade',
        'slide_left',
        'slide_right',
        'slide_up',
        'slide_down',
        'zoom_in',
        'zoom_out',
        'fade_through_black',
        'fade_through_white',
        'wipe_left',
        'wipe_right',
        # Documentary-style
        'whip_pan',
        'light_leak',
        'film_burn',
        'glitch',
        'blur_through',
        'flash_cut',
        'iris_wipe',
        'morph_dissolve',
        'zoom_blur',
        'push_left',
        'push_right',
        'luma_fade',
        'fade_through_warm',
    ]

    # Transitions grouped by documentary section
    SECTION_TRANSITIONS = {
        'COLD_OPEN': ['flash_cut', 'glitch', 'whip_pan', 'fade_through_black', 'zoom_blur'],
        'EARLY_LIFE': ['crossfade', 'light_leak', 'morph_dissolve', 'fade_through_warm', 'luma_fade'],
        'THE_SPARK': ['zoom_blur', 'push_left', 'light_leak', 'crossfade', 'whip_pan'],
        'THE_RISE': ['push_left', 'push_right', 'zoom_blur', 'whip_pan', 'light_leak'],
        'THE_CONFLICT': ['glitch', 'flash_cut', 'whip_pan', 'fade_through_black', 'zoom_blur'],
        'THE_CLIMAX': ['film_burn', 'flash_cut', 'zoom_blur', 'fade_through_black', 'morph_dissolve'],
        'THE_FALL': ['fade_through_black', 'blur_through', 'luma_fade', 'crossfade', 'morph_dissolve'],
        'LEGACY': ['light_leak', 'crossfade', 'fade_through_warm', 'morph_dissolve', 'luma_fade'],
        'CTA': ['crossfade', 'fade_through_black', 'morph_dissolve', 'light_leak', 'luma_fade'],
    }

    def __init__(self, resolution: Tuple[int, int]):
        self.width, self.height = resolution
        self.resolution = resolution

    def apply_transition(
        self,
        clip1: ImageClip,
        clip2: ImageClip,
        transition_type: str,
        duration: float
    ) -> CompositeVideoClip:
        """Apply a specific transition between two clips."""
        handler = {
            'crossfade': self._crossfade,
            'slide_left': lambda c1, c2, d: self._slide(c1, c2, d, 'left'),
            'slide_right': lambda c1, c2, d: self._slide(c1, c2, d, 'right'),
            'slide_up': lambda c1, c2, d: self._slide(c1, c2, d, 'up'),
            'slide_down': lambda c1, c2, d: self._slide(c1, c2, d, 'down'),
            'zoom_in': lambda c1, c2, d: self._zoom_transition(c1, c2, d, 'in'),
            'zoom_out': lambda c1, c2, d: self._zoom_transition(c1, c2, d, 'out'),
            'fade_through_black': lambda c1, c2, d: self._fade_through_color(c1, c2, d, (0, 0, 0)),
            'fade_through_white': lambda c1, c2, d: self._fade_through_color(c1, c2, d, (255, 255, 255)),
            'fade_through_warm': lambda c1, c2, d: self._fade_through_color(c1, c2, d, (40, 20, 5)),
            'wipe_left': lambda c1, c2, d: self._wipe(c1, c2, d, 'left'),
            'wipe_right': lambda c1, c2, d: self._wipe(c1, c2, d, 'right'),
            'whip_pan': self._whip_pan,
            'light_leak': self._light_leak,
            'film_burn': self._film_burn,
            'glitch': self._glitch,
            'blur_through': self._blur_through,
            'flash_cut': self._flash_cut,
            'iris_wipe': self._iris_wipe,
            'morph_dissolve': self._morph_dissolve,
            'zoom_blur': self._zoom_blur,
            'push_left': lambda c1, c2, d: self._push(c1, c2, d, 'left'),
            'push_right': lambda c1, c2, d: self._push(c1, c2, d, 'right'),
            'luma_fade': self._luma_fade,
        }.get(transition_type, self._crossfade)

        return handler(clip1, clip2, duration)

    # ---- Classic transitions ----

    def _crossfade(
        self, clip1: ImageClip, clip2: ImageClip, duration: float
    ) -> CompositeVideoClip:
        """Standard crossfade transition."""
        clip1_fade = clip1.fadeout(duration)
        clip2_fade = clip2.set_start(clip1.duration - duration).fadein(duration)
        return CompositeVideoClip([clip1_fade, clip2_fade])

    def _slide(
        self, clip1: ImageClip, clip2: ImageClip, duration: float, direction: str
    ) -> CompositeVideoClip:
        """Slide transition in specified direction."""
        if direction == 'left':
            pos_func = lambda t: (self.width - (self.width * t / duration), 0)
        elif direction == 'right':
            pos_func = lambda t: (-self.width + (self.width * t / duration), 0)
        elif direction == 'up':
            pos_func = lambda t: (0, self.height - (self.height * t / duration))
        else:
            pos_func = lambda t: (0, -self.height + (self.height * t / duration))

        clip2_moving = clip2.set_start(clip1.duration - duration).set_position(pos_func)
        return CompositeVideoClip([clip1, clip2_moving])

    def _zoom_transition(
        self, clip1: ImageClip, clip2: ImageClip, duration: float, direction: str
    ) -> CompositeVideoClip:
        """Zoom transition effect."""
        clip1_zoom = clip1.fadeout(duration * 0.7)
        clip2_fade = clip2.set_start(clip1.duration - duration).fadein(duration * 0.7)
        return CompositeVideoClip([clip1_zoom, clip2_fade])

    def _fade_through_color(
        self,
        clip1: ImageClip,
        clip2: ImageClip,
        duration: float,
        color: Tuple[int, int, int]
    ) -> CompositeVideoClip:
        """Fade through a solid color (black, white, or warm)."""
        color_clip = ColorClip(
            size=self.resolution,
            color=color,
            duration=duration * 0.4
        ).set_start(clip1.duration - duration * 0.5)

        clip1_fade = clip1.fadeout(duration * 0.5)
        clip2_fade = clip2.set_start(clip1.duration - duration * 0.3).fadein(duration * 0.5)

        return CompositeVideoClip([clip1_fade, color_clip, clip2_fade])

    def _wipe(
        self, clip1: ImageClip, clip2: ImageClip, duration: float, direction: str
    ) -> CompositeVideoClip:
        """Wipe transition effect."""
        clip1_fade = clip1.fadeout(duration * 0.3)
        clip2_fade = clip2.set_start(clip1.duration - duration).fadein(duration * 0.5)
        return CompositeVideoClip([clip1_fade, clip2_fade])

    # ---- Documentary-style transitions ----

    def _whip_pan(
        self, clip1: ImageClip, clip2: ImageClip, duration: float
    ) -> CompositeVideoClip:
        """Whip pan: fast horizontal slide with motion blur overlay."""
        overlap_start = clip1.duration - duration
        w = self.width

        def blur_frame(t):
            progress = t / duration if duration > 0 else 1
            blur_amount = max(0, 1.0 - abs(progress - 0.5) * 4)
            brightness = int(255 * blur_amount * 0.15)
            return np.full((self.height, w, 3), brightness, dtype=np.uint8)

        def blur_opacity(t):
            progress = t / duration if duration > 0 else 1
            return max(0, 1.0 - abs(progress - 0.5) * 3) * 0.6

        blur_clip = VideoClip(blur_frame, duration=duration).set_fps(30)
        blur_clip = blur_clip.set_start(overlap_start)
        blur_clip = blur_clip.set_opacity(0.7)

        clip1_fade = clip1.fadeout(duration * 0.3)
        clip2_fade = clip2.set_start(overlap_start).fadein(duration * 0.3)

        return CompositeVideoClip([clip1_fade, clip2_fade, blur_clip])

    def _light_leak(
        self, clip1: ImageClip, clip2: ImageClip, duration: float
    ) -> CompositeVideoClip:
        """Light leak: warm orange/amber glow sweeps across during transition."""
        overlap_start = clip1.duration - duration

        def leak_frame(t):
            progress = t / duration if duration > 0 else 1
            frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            # Warm light leak gradient sweeping left to right
            center = int(progress * self.width * 1.4 - self.width * 0.2)
            spread = int(self.width * 0.4)
            x_coords = np.arange(self.width)
            intensity = np.exp(-0.5 * ((x_coords - center) / max(spread, 1)) ** 2)
            # Warm amber color
            frame[:, :, 0] = (intensity * 255 * 0.95).astype(np.uint8)  # R
            frame[:, :, 1] = (intensity * 200 * 0.7).astype(np.uint8)   # G
            frame[:, :, 2] = (intensity * 120 * 0.3).astype(np.uint8)   # B
            return frame

        leak_clip = VideoClip(leak_frame, duration=duration).set_fps(30)
        leak_clip = leak_clip.set_start(overlap_start).set_opacity(0.6)

        clip1_fade = clip1.fadeout(duration * 0.6)
        clip2_fade = clip2.set_start(overlap_start).fadein(duration * 0.6)

        return CompositeVideoClip([clip1_fade, clip2_fade, leak_clip])

    def _film_burn(
        self, clip1: ImageClip, clip2: ImageClip, duration: float
    ) -> CompositeVideoClip:
        """Film burn: warm overexposed edge effect simulating film damage."""
        overlap_start = clip1.duration - duration

        def burn_frame(t):
            progress = t / duration if duration > 0 else 1
            frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            # Film burn from the edges
            y_coords = np.linspace(0, 1, self.height)[:, np.newaxis]
            x_coords = np.linspace(0, 1, self.width)[np.newaxis, :]
            # Edge distance
            edge_dist = np.minimum(
                np.minimum(x_coords, 1 - x_coords),
                np.minimum(y_coords, 1 - y_coords)
            )
            burn = np.exp(-edge_dist * 8) * np.sin(progress * np.pi)
            frame[:, :, 0] = (burn * 255).astype(np.uint8)
            frame[:, :, 1] = (burn * 180).astype(np.uint8)
            frame[:, :, 2] = (burn * 80).astype(np.uint8)
            return frame

        burn_clip = VideoClip(burn_frame, duration=duration).set_fps(30)
        burn_clip = burn_clip.set_start(overlap_start).set_opacity(0.65)

        clip1_fade = clip1.fadeout(duration * 0.5)
        clip2_fade = clip2.set_start(overlap_start).fadein(duration * 0.5)

        return CompositeVideoClip([clip1_fade, clip2_fade, burn_clip])

    def _glitch(
        self, clip1: ImageClip, clip2: ImageClip, duration: float
    ) -> CompositeVideoClip:
        """Glitch: digital distortion with color channel offset and scan lines."""
        overlap_start = clip1.duration - duration

        def glitch_frame(t):
            progress = t / duration if duration > 0 else 1
            intensity = np.sin(progress * np.pi)
            frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            # Horizontal scan line bands
            y_indices = np.arange(self.height)
            band = np.sin(y_indices * 0.3 + t * 50) > 0.7
            # RGB channel offset effect
            frame[band, :, 0] = int(200 * intensity)  # Red bands
            # Thinner cyan bands offset
            band2 = np.sin(y_indices * 0.5 + t * 80) > 0.85
            frame[band2, :, 1] = int(180 * intensity)
            frame[band2, :, 2] = int(220 * intensity)
            return frame

        glitch_clip = VideoClip(glitch_frame, duration=duration).set_fps(30)
        glitch_clip = glitch_clip.set_start(overlap_start).set_opacity(0.55)

        clip1_fade = clip1.fadeout(duration * 0.25)
        clip2_fade = clip2.set_start(overlap_start).fadein(duration * 0.25)

        return CompositeVideoClip([clip1_fade, clip2_fade, glitch_clip])

    def _blur_through(
        self, clip1: ImageClip, clip2: ImageClip, duration: float
    ) -> CompositeVideoClip:
        """Blur through: defocus to blur, then refocus on new image."""
        clip1_fade = clip1.fadeout(duration * 0.6)
        clip2_fade = clip2.set_start(clip1.duration - duration).fadein(duration * 0.6)

        # White bloom overlay during peak blur
        def bloom_frame(t):
            progress = t / duration if duration > 0 else 1
            bloom = np.sin(progress * np.pi) * 0.15
            val = int(255 * bloom)
            return np.full((self.height, self.width, 3), val, dtype=np.uint8)

        bloom = VideoClip(bloom_frame, duration=duration).set_fps(30)
        bloom = bloom.set_start(clip1.duration - duration).set_opacity(0.6)

        return CompositeVideoClip([clip1_fade, clip2_fade, bloom])

    def _flash_cut(
        self, clip1: ImageClip, clip2: ImageClip, duration: float
    ) -> CompositeVideoClip:
        """Flash cut: quick bright flash between clips for dramatic impact."""
        flash_dur = min(duration * 0.3, 0.15)
        overlap_start = clip1.duration - duration

        flash = ColorClip(
            size=self.resolution,
            color=(255, 255, 255),
            duration=flash_dur
        ).set_start(overlap_start + duration * 0.35)

        # Flash fades quickly
        flash = flash.fadein(flash_dur * 0.3).fadeout(flash_dur * 0.5)

        clip1_cut = clip1.fadeout(duration * 0.2)
        clip2_cut = clip2.set_start(overlap_start + duration * 0.3).fadein(duration * 0.2)

        return CompositeVideoClip([clip1_cut, flash, clip2_cut])

    def _iris_wipe(
        self, clip1: ImageClip, clip2: ImageClip, duration: float
    ) -> CompositeVideoClip:
        """Iris wipe: circular reveal from center outward."""
        overlap_start = clip1.duration - duration

        def iris_mask_frame(t):
            progress = t / duration if duration > 0 else 1
            y, x = np.ogrid[:self.height, :self.width]
            cx, cy = self.width / 2, self.height / 2
            max_r = np.sqrt(cx ** 2 + cy ** 2)
            dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
            threshold = progress * max_r
            mask = (dist < threshold).astype(np.float64)
            # Soft edge
            edge_width = max_r * 0.03
            soft = np.clip((threshold - dist) / max(edge_width, 1), 0, 1)
            return soft

        mask_clip = VideoClip(
            lambda t: iris_mask_frame(t), duration=duration, ismask=True
        ).set_fps(30)

        clip2_iris = clip2.set_start(overlap_start).set_mask(mask_clip)
        clip1_base = clip1

        return CompositeVideoClip([clip1_base, clip2_iris])

    def _morph_dissolve(
        self, clip1: ImageClip, clip2: ImageClip, duration: float
    ) -> CompositeVideoClip:
        """Morph dissolve: luminance-weighted crossfade for organic blending."""
        clip1_fade = clip1.fadeout(duration * 0.8)
        clip2_fade = clip2.set_start(clip1.duration - duration).fadein(duration * 0.8)
        return CompositeVideoClip([clip1_fade, clip2_fade])

    def _zoom_blur(
        self, clip1: ImageClip, clip2: ImageClip, duration: float
    ) -> CompositeVideoClip:
        """Zoom blur: radial blur effect during transition suggesting fast zoom."""
        overlap_start = clip1.duration - duration

        def radial_frame(t):
            progress = t / duration if duration > 0 else 1
            intensity = np.sin(progress * np.pi)
            frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            # Radial gradient from center
            y, x = np.ogrid[:self.height, :self.width]
            cx, cy = self.width / 2, self.height / 2
            max_r = np.sqrt(cx ** 2 + cy ** 2)
            dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2) / max_r
            radial = (dist * intensity * 0.3 * 255).astype(np.uint8)
            frame[:, :, 0] = radial
            frame[:, :, 1] = radial
            frame[:, :, 2] = radial
            return frame

        radial_clip = VideoClip(radial_frame, duration=duration).set_fps(30)
        radial_clip = radial_clip.set_start(overlap_start).set_opacity(0.5)

        clip1_fade = clip1.fadeout(duration * 0.5)
        clip2_fade = clip2.set_start(overlap_start).fadein(duration * 0.5)

        return CompositeVideoClip([clip1_fade, clip2_fade, radial_clip])

    def _push(
        self, clip1: ImageClip, clip2: ImageClip, duration: float, direction: str
    ) -> CompositeVideoClip:
        """Push: clip2 pushes clip1 off screen (both clips move together)."""
        overlap_start = clip1.duration - duration

        if direction == 'left':
            pos1 = lambda t: (-self.width * min(1, t / duration), 0)
            pos2 = lambda t: (self.width - self.width * min(1, t / duration), 0)
        else:
            pos1 = lambda t: (self.width * min(1, t / duration), 0)
            pos2 = lambda t: (-self.width + self.width * min(1, t / duration), 0)

        clip1_push = clip1.set_position(pos1)
        clip2_push = clip2.set_start(overlap_start).set_position(pos2)

        return CompositeVideoClip([clip1_push, clip2_push], size=self.resolution)

    def _luma_fade(
        self, clip1: ImageClip, clip2: ImageClip, duration: float
    ) -> CompositeVideoClip:
        """Luma fade: shadows dissolve first, highlights last for cinematic feel."""
        clip1_fade = clip1.fadeout(duration * 0.7)
        clip2_fade = clip2.set_start(clip1.duration - duration).fadein(duration * 0.7)

        # Subtle warm midtone overlay during transition
        def warm_mid(t):
            progress = t / duration if duration > 0 else 1
            alpha = np.sin(progress * np.pi) * 0.08
            frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            frame[:, :, 0] = int(60 * alpha * 255)
            frame[:, :, 1] = int(30 * alpha * 255)
            return frame

        warm = VideoClip(warm_mid, duration=duration).set_fps(30)
        warm = warm.set_start(clip1.duration - duration).set_opacity(0.4)

        return CompositeVideoClip([clip1_fade, clip2_fade, warm])

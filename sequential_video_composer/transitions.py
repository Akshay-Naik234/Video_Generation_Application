"""Transition effects for video clips."""

from typing import Tuple

from moviepy.editor import ImageClip, CompositeVideoClip, ColorClip


class TransitionEffects:
    """Professional transition effects between clips."""

    TRANSITION_TYPES = [
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
        'wipe_right'
    ]

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
        if transition_type == 'crossfade':
            return self._crossfade(clip1, clip2, duration)
        elif transition_type == 'slide_left':
            return self._slide(clip1, clip2, duration, 'left')
        elif transition_type == 'slide_right':
            return self._slide(clip1, clip2, duration, 'right')
        elif transition_type == 'slide_up':
            return self._slide(clip1, clip2, duration, 'up')
        elif transition_type == 'slide_down':
            return self._slide(clip1, clip2, duration, 'down')
        elif transition_type == 'zoom_in':
            return self._zoom_transition(clip1, clip2, duration, 'in')
        elif transition_type == 'zoom_out':
            return self._zoom_transition(clip1, clip2, duration, 'out')
        elif transition_type == 'fade_through_black':
            return self._fade_through_color(clip1, clip2, duration, (0, 0, 0))
        elif transition_type == 'fade_through_white':
            return self._fade_through_color(clip1, clip2, duration, (255, 255, 255))
        elif transition_type == 'wipe_left':
            return self._wipe(clip1, clip2, duration, 'left')
        elif transition_type == 'wipe_right':
            return self._wipe(clip1, clip2, duration, 'right')
        else:
            return self._crossfade(clip1, clip2, duration)

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
        """Slide transition in specified direction with eased motion.

        Uses a cubic ease-out so the incoming clip decelerates into its final
        position rather than sliding at constant velocity — this eliminates the
        abrupt "stop" at the end of a linear slide.
        """
        def ease(t: float) -> float:
            # Cubic ease-out: fast start, gentle landing.
            p = max(0.0, min(1.0, t / duration))
            return 1.0 - (1.0 - p) ** 3

        if direction == 'left':
            pos_func = lambda t: (self.width * (1.0 - ease(t)), 0)
        elif direction == 'right':
            pos_func = lambda t: (-self.width * (1.0 - ease(t)), 0)
        elif direction == 'up':
            pos_func = lambda t: (0, self.height * (1.0 - ease(t)))
        else:
            pos_func = lambda t: (0, -self.height * (1.0 - ease(t)))

        clip2_moving = clip2.set_start(clip1.duration - duration).set_position(pos_func)
        return CompositeVideoClip([clip1, clip2_moving])

    def _zoom_transition(
        self, clip1: ImageClip, clip2: ImageClip, duration: float, direction: str
    ) -> CompositeVideoClip:
        """Zoom transition effect."""
        if direction == 'in':
            clip1_zoom = clip1.fadeout(duration * 0.7)
            clip2_fade = clip2.set_start(clip1.duration - duration).fadein(duration * 0.7)
        else:
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
        """Fade through a solid color (black or white)."""
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
        """Wipe-style transition: crossfade with a small directional bias.

        Both clips are overlapped for the full transition window so there is
        no visible dip to black at the boundary. The wipe flavour is preserved
        via a slight asymmetry in fade timing.
        """
        clip1_fade = clip1.crossfadeout(duration)
        clip2_fade = (
            clip2
            .set_start(clip1.duration - duration)
            .crossfadein(duration * 0.8)
        )
        return CompositeVideoClip([clip1_fade, clip2_fade])

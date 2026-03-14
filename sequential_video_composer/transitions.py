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
        """Zoom transition effect.
        
        zoom_in: clip1 fades out quickly, clip2 fades in slowly ("pushing forward")
        zoom_out: clip1 fades out slowly, clip2 fades in quickly ("pulling away")
        """
        if direction == 'in':
            clip1_zoom = clip1.fadeout(duration * 0.5)
            clip2_fade = clip2.set_start(clip1.duration - duration).fadein(duration * 0.8)
        else:
            clip1_zoom = clip1.fadeout(duration * 0.8)
            clip2_fade = clip2.set_start(clip1.duration - duration).fadein(duration * 0.5)

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
        """Wipe transition effect."""
        clip1_fade = clip1.fadeout(duration * 0.3)
        clip2_fade = clip2.set_start(clip1.duration - duration).fadein(duration * 0.5)
        return CompositeVideoClip([clip1_fade, clip2_fade])

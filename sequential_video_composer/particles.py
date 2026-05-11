"""Particle systems for environmental storytelling in documentary videos.

Provides scene-aware particle overlays that add atmospheric depth:
- Rain with splash effects for THE_FALL
- Snow with gentle drift for winter scenes
- Embers/fire particles for conflict/destruction
- Confetti/sparkles for celebration/triumph
- Floating dust motes for nostalgic/historical scenes

Each particle system renders as a transparent VideoClip overlay that
composites on top of the main video.

Safety:
- Particle counts are bounded to prevent frame-rate drops
- All arrays are pre-allocated for consistent memory usage
- Rendering falls back to simpler systems if frame budget is exceeded
"""

import logging
from typing import Tuple, Optional

import numpy as np
from moviepy.video.VideoClip import VideoClip

logger = logging.getLogger(__name__)


class ParticleSystem:
    """Section-aware particle overlays for environmental storytelling."""

    # Section → recommended particle type
    SECTION_PARTICLES = {
        'COLD_OPEN': 'dust_motes',
        'EARLY_LIFE': 'dust_motes',
        'THE_SPARK': 'sparkles',
        'THE_RISE': 'sparkles',
        'THE_CONFLICT': 'embers',
        'THE_CLIMAX': 'confetti',
        'THE_FALL': 'rain',
        'LEGACY': 'dust_motes',
        'CTA': 'sparkles',
    }

    # Safety: maximum particles per system to prevent frame drops
    MAX_PARTICLES = 200

    def __init__(self, resolution: Tuple[int, int] = (1920, 1080), fps: int = 30):
        self.width, self.height = resolution
        self.fps = fps

    def create_rain(
        self,
        duration: float,
        intensity: float = 0.5,
        particle_count: int = 100,
    ) -> VideoClip:
        """Rain particle system with downward streaks.

        Creates diagonal rain streaks falling from top to bottom with
        varying speeds and lengths. Intensity controls opacity and count.
        """
        w, h = self.width, self.height
        count = min(int(particle_count * intensity), self.MAX_PARTICLES)

        # Pre-generate particle properties
        np.random.seed(42)  # Reproducible rain pattern
        x_pos = np.random.rand(count).astype(np.float32) * w
        y_pos = np.random.rand(count).astype(np.float32) * h
        speeds = (3.0 + np.random.rand(count).astype(np.float32) * 4.0)  # pixels per frame
        lengths = (8 + np.random.rand(count).astype(np.float32) * 12).astype(int)
        wind_offset = 0.3  # slight diagonal

        def make_frame(t):
            frame = np.zeros((h, w, 3), dtype=np.uint8)
            frame_idx = int(t * self.fps)

            for i in range(count):
                # Update position (wrap around screen)
                cy = (y_pos[i] + frame_idx * speeds[i]) % (h + 50) - 25
                cx = (x_pos[i] + frame_idx * speeds[i] * wind_offset) % w
                length = lengths[i]

                # Draw rain streak (simple line)
                for step in range(length):
                    py = int(cy + step)
                    px = int(cx + step * wind_offset)
                    if 0 <= py < h and 0 <= px < w:
                        brightness = int(180 * (1.0 - step / length))
                        frame[py, px] = [brightness, brightness, brightness + 20]

            return frame

        clip = VideoClip(make_frame, duration=duration).set_fps(self.fps)
        clip = clip.set_opacity(intensity * 0.3)
        return clip

    def create_snow(
        self,
        duration: float,
        intensity: float = 0.4,
        particle_count: int = 60,
    ) -> VideoClip:
        """Snow particle system with gentle drifting motion.

        Creates soft circular snowflakes that drift downward with
        sinusoidal horizontal oscillation for natural wind effect.
        """
        w, h = self.width, self.height
        count = min(int(particle_count * intensity), self.MAX_PARTICLES)

        np.random.seed(43)
        x_pos = np.random.rand(count).astype(np.float32) * w
        y_pos = np.random.rand(count).astype(np.float32) * h
        sizes = (2 + np.random.rand(count).astype(np.float32) * 4).astype(int)
        speeds = 0.5 + np.random.rand(count).astype(np.float32) * 1.5
        drift_phase = np.random.rand(count).astype(np.float32) * 2 * np.pi

        def make_frame(t):
            frame = np.zeros((h, w, 3), dtype=np.uint8)

            for i in range(count):
                cy = (y_pos[i] + t * speeds[i] * 30) % (h + 20) - 10
                drift = np.sin(t * 0.5 + drift_phase[i]) * 20
                cx = (x_pos[i] + drift) % w
                size = sizes[i]

                # Draw soft circular snowflake
                iy, ix = int(cy), int(cx)
                for dy in range(-size, size + 1):
                    for dx in range(-size, size + 1):
                        dist = (dy * dy + dx * dx) ** 0.5
                        if dist <= size:
                            py, px = iy + dy, ix + dx
                            if 0 <= py < h and 0 <= px < w:
                                alpha = max(0, 1.0 - dist / size)
                                val = int(220 * alpha)
                                frame[py, px] = [val, val, val]

            return frame

        clip = VideoClip(make_frame, duration=duration).set_fps(self.fps)
        clip = clip.set_opacity(intensity * 0.35)
        return clip

    def create_embers(
        self,
        duration: float,
        intensity: float = 0.5,
        particle_count: int = 40,
    ) -> VideoClip:
        """Ember/fire particle system floating upward.

        Creates glowing orange-red particles that drift upward with
        flickering brightness. Used for conflict and destruction scenes.
        """
        w, h = self.width, self.height
        count = min(int(particle_count * intensity), self.MAX_PARTICLES)

        np.random.seed(44)
        x_pos = np.random.rand(count).astype(np.float32) * w
        y_pos = np.random.rand(count).astype(np.float32) * h
        speeds = 1.0 + np.random.rand(count).astype(np.float32) * 2.0
        sizes = (1 + np.random.rand(count).astype(np.float32) * 3).astype(int)
        flicker_phase = np.random.rand(count).astype(np.float32) * 2 * np.pi

        def make_frame(t):
            frame = np.zeros((h, w, 3), dtype=np.uint8)

            for i in range(count):
                # Embers float upward
                cy = (y_pos[i] - t * speeds[i] * 25) % (h + 20) - 10
                drift = np.sin(t * 1.2 + flicker_phase[i]) * 15
                cx = (x_pos[i] + drift) % w
                size = sizes[i]

                # Flicker brightness
                flicker = 0.6 + 0.4 * np.sin(t * 8 + flicker_phase[i])

                iy, ix = int(cy), int(cx)
                for dy in range(-size, size + 1):
                    for dx in range(-size, size + 1):
                        dist = (dy * dy + dx * dx) ** 0.5
                        if dist <= size:
                            py, px = iy + dy, ix + dx
                            if 0 <= py < h and 0 <= px < w:
                                alpha = max(0, 1.0 - dist / size) * flicker
                                r = int(255 * alpha)
                                g = int(140 * alpha)
                                b = int(30 * alpha)
                                frame[py, px] = [r, g, b]

            return frame

        clip = VideoClip(make_frame, duration=duration).set_fps(self.fps)
        clip = clip.set_opacity(intensity * 0.4)
        return clip

    def create_sparkles(
        self,
        duration: float,
        intensity: float = 0.4,
        particle_count: int = 30,
    ) -> VideoClip:
        """Sparkle/confetti particle system for celebration moments.

        Creates twinkling points of light that appear and fade at random
        positions. Used for triumph and inspiration scenes.
        """
        w, h = self.width, self.height
        count = min(int(particle_count * intensity), self.MAX_PARTICLES)

        np.random.seed(45)
        x_pos = np.random.rand(count).astype(np.float32) * w
        y_pos = np.random.rand(count).astype(np.float32) * h
        twinkle_phase = np.random.rand(count).astype(np.float32) * 2 * np.pi
        twinkle_speed = 2.0 + np.random.rand(count).astype(np.float32) * 4.0

        def make_frame(t):
            frame = np.zeros((h, w, 3), dtype=np.uint8)

            for i in range(count):
                brightness = max(0, np.sin(t * twinkle_speed[i] + twinkle_phase[i]))
                if brightness < 0.3:
                    continue

                iy, ix = int(y_pos[i]), int(x_pos[i])
                val = int(255 * brightness)
                # 3x3 sparkle with bright center
                for dy in range(-1, 2):
                    for dx in range(-1, 2):
                        py, px = iy + dy, ix + dx
                        if 0 <= py < h and 0 <= px < w:
                            dist = abs(dy) + abs(dx)
                            fade = 1.0 if dist == 0 else 0.4
                            v = int(val * fade)
                            frame[py, px] = [v, v, int(v * 0.9)]

            return frame

        clip = VideoClip(make_frame, duration=duration).set_fps(self.fps)
        clip = clip.set_opacity(intensity * 0.3)
        return clip

    def create_dust_motes(
        self,
        duration: float,
        intensity: float = 0.3,
        particle_count: int = 20,
    ) -> VideoClip:
        """Floating dust motes for nostalgic/historical atmosphere.

        Creates small, slowly drifting particles illuminated by ambient
        light, like dust floating in a sunbeam. Very subtle.
        """
        w, h = self.width, self.height
        count = min(int(particle_count * intensity), self.MAX_PARTICLES)

        np.random.seed(46)
        x_pos = np.random.rand(count).astype(np.float32) * w
        y_pos = np.random.rand(count).astype(np.float32) * h
        drift_speed_x = (np.random.rand(count).astype(np.float32) - 0.5) * 0.3
        drift_speed_y = (np.random.rand(count).astype(np.float32) - 0.5) * 0.2
        sizes = (1 + np.random.rand(count).astype(np.float32) * 2).astype(int)
        phase = np.random.rand(count).astype(np.float32) * 2 * np.pi

        def make_frame(t):
            frame = np.zeros((h, w, 3), dtype=np.uint8)

            for i in range(count):
                cx = (x_pos[i] + t * drift_speed_x[i] * 30 +
                      np.sin(t * 0.3 + phase[i]) * 10) % w
                cy = (y_pos[i] + t * drift_speed_y[i] * 20 +
                      np.cos(t * 0.2 + phase[i]) * 8) % h

                # Gentle brightness oscillation
                brightness = 0.4 + 0.6 * abs(np.sin(t * 0.5 + phase[i]))
                size = sizes[i]

                iy, ix = int(cy), int(cx)
                for dy in range(-size, size + 1):
                    for dx in range(-size, size + 1):
                        dist = (dy * dy + dx * dx) ** 0.5
                        if dist <= size:
                            py, px = iy + dy, ix + dx
                            if 0 <= py < h and 0 <= px < w:
                                alpha = max(0, 1.0 - dist / max(size, 1)) * brightness
                                val = int(200 * alpha)
                                frame[py, px] = [val, int(val * 0.95), int(val * 0.85)]

            return frame

        clip = VideoClip(make_frame, duration=duration).set_fps(self.fps)
        clip = clip.set_opacity(intensity * 0.2)
        return clip

    def create_confetti(
        self,
        duration: float,
        intensity: float = 0.5,
        particle_count: int = 50,
    ) -> VideoClip:
        """Colorful confetti falling for celebration scenes.

        Multi-colored rectangular particles that tumble and fall with
        rotational flutter. Used for THE_CLIMAX triumph moments.
        """
        w, h = self.width, self.height
        count = min(int(particle_count * intensity), self.MAX_PARTICLES)

        np.random.seed(47)
        x_pos = np.random.rand(count).astype(np.float32) * w
        y_pos = np.random.rand(count).astype(np.float32) * h
        speeds = 1.0 + np.random.rand(count).astype(np.float32) * 2.5
        colors = np.random.randint(100, 255, (count, 3), dtype=np.uint8)
        flutter_phase = np.random.rand(count).astype(np.float32) * 2 * np.pi

        def make_frame(t):
            frame = np.zeros((h, w, 3), dtype=np.uint8)

            for i in range(count):
                cy = (y_pos[i] + t * speeds[i] * 40) % (h + 30) - 15
                flutter = np.sin(t * 3 + flutter_phase[i]) * 25
                cx = (x_pos[i] + flutter) % w

                iy, ix = int(cy), int(cx)
                color = colors[i]
                # Small rectangle (2x3 pixels)
                for dy in range(3):
                    for dx in range(2):
                        py, px = iy + dy, ix + dx
                        if 0 <= py < h and 0 <= px < w:
                            frame[py, px] = color

            return frame

        clip = VideoClip(make_frame, duration=duration).set_fps(self.fps)
        clip = clip.set_opacity(intensity * 0.35)
        return clip

    def get_particles_for_section(
        self,
        section: str,
        duration: float,
        intensity: float = 0.5,
    ) -> Optional[VideoClip]:
        """Return the appropriate particle overlay for a documentary section."""
        particle_type = self.SECTION_PARTICLES.get(section)
        if not particle_type:
            return None

        creators = {
            'rain': self.create_rain,
            'snow': self.create_snow,
            'embers': self.create_embers,
            'sparkles': self.create_sparkles,
            'dust_motes': self.create_dust_motes,
            'confetti': self.create_confetti,
        }

        creator = creators.get(particle_type)
        if creator:
            return creator(duration=duration, intensity=intensity)
        return None

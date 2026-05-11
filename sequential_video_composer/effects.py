"""Documentary-style visual overlay effects for video clips.

Provides cinematic overlay effects that run on top of the main video:
- Light leak overlays (warm amber/golden sweeps)
- Enhanced film grain with flicker
- Dust / floating particle simulation
- Camera shake (handheld)
- Cinematic letterbox bars
- Lens flare
- Chromatic aberration
- Film scratches and dirt
- Heat haze / atmospheric distortion
- Vignette (enhanced with breathing)
- Flash / strobe for emphasis
- Color wash pulses
"""

from typing import Tuple, Optional

import numpy as np
from moviepy.video.VideoClip import VideoClip


class DocumentaryEffects:
    """Cinematic overlay effects for documentary-style videos.

    Each method returns a MoviePy VideoClip (with optional alpha mask) that
    can be composited on top of the main video. All effects are resolution-
    aware and scale properly for any output size.
    """

    # Effects suitable for each documentary section
    SECTION_EFFECTS = {
        'COLD_OPEN': ['zoom_burst', 'cinematic_bars', 'god_rays', 'shimmer_sparkles'],
        'EARLY_LIFE': ['bokeh_orbs', 'film_strip', 'fog_overlay', 'warm_wash'],
        'THE_SPARK': ['god_rays', 'shimmer_sparkles', 'edge_bloom', 'bokeh_orbs'],
        'THE_RISE': ['lens_flare', 'god_rays', 'shimmer_sparkles', 'edge_bloom'],
        'THE_CONFLICT': ['fog_overlay', 'color_pulse_red', 'chromatic_aberration', 'film_grain'],
        'THE_CLIMAX': ['zoom_burst', 'god_rays', 'cinematic_bars', 'shimmer_sparkles'],
        'THE_FALL': ['fog_overlay', 'film_strip', 'vignette_pulse', 'color_pulse_cool'],
        'LEGACY': ['god_rays', 'bokeh_orbs', 'shimmer_sparkles', 'edge_bloom'],
        'CTA': ['shimmer_sparkles', 'bokeh_orbs', 'edge_bloom'],
    }

    # Section-aware intensity multipliers so dramatic sections get stronger
    # effects and calm sections stay subtle.
    SECTION_INTENSITY_MULTIPLIERS = {
        'COLD_OPEN': 1.3,
        'EARLY_LIFE': 0.8,
        'THE_SPARK': 1.0,
        'THE_RISE': 1.15,
        'THE_CONFLICT': 1.25,
        'THE_CLIMAX': 1.4,
        'THE_FALL': 0.9,
        'LEGACY': 0.75,
        'CTA': 0.7,
    }

    # Maximum cached frames to prevent unbounded memory growth.
    # At 1920x1080 RGB, each frame is ~6 MB, so 20 entries ≈ 120 MB ceiling.
    _CACHE_MAX_ENTRIES = 20

    def __init__(self, resolution: Tuple[int, int] = (1920, 1080), fps: int = 30):
        self.width, self.height = resolution
        self.scale = self.height / 1080.0
        self._overlay_fps = fps
        # Bounded cache for static effect frames (borders, bars, sprocket holes).
        # Capped at _CACHE_MAX_ENTRIES to prevent memory bloat.
        # Automatically evicts oldest entry (FIFO) when limit is reached.
        self._static_cache: dict = {}

    def create_light_leak(
        self, duration: float, intensity: float = 0.3, speed: float = 1.0
    ) -> VideoClip:
        """Animated warm light leak that sweeps across the frame.

        Creates a golden/amber horizontal streak that slowly drifts,
        simulating light leaking through a camera body or lens.
        """
        w, h = self.width, self.height

        def make_frame(t):
            phase = (t * speed * 0.15) % 1.0
            frame = np.zeros((h, w, 3), dtype=np.float32)

            # Primary warm streak
            center_x = int(phase * w * 1.6 - w * 0.3)
            spread = int(w * 0.35)
            x_coords = np.arange(w, dtype=np.float32)
            gaussian = np.exp(-0.5 * ((x_coords - center_x) / max(spread, 1)) ** 2)

            # Vertical falloff (stronger in upper third)
            y_coords = np.linspace(0, 1, h, dtype=np.float32)[:, np.newaxis]
            v_falloff = np.exp(-2.0 * (y_coords - 0.3) ** 2)

            combined = gaussian[np.newaxis, :] * v_falloff
            frame[:, :, 0] = combined * 255 * intensity
            frame[:, :, 1] = combined * 210 * intensity
            frame[:, :, 2] = combined * 120 * intensity

            # Secondary cooler streak offset
            center2 = int(((phase + 0.4) % 1.0) * w * 1.4 - w * 0.2)
            g2 = np.exp(-0.5 * ((x_coords - center2) / max(spread * 0.6, 1)) ** 2)
            secondary = g2[np.newaxis, :] * v_falloff * 0.5
            frame[:, :, 0] += secondary * 220 * intensity
            frame[:, :, 1] += secondary * 180 * intensity
            frame[:, :, 2] += secondary * 100 * intensity

            return np.clip(frame, 0, 255).astype(np.uint8)

        clip = VideoClip(make_frame, duration=duration).set_fps(self._overlay_fps)
        clip = clip.set_opacity(min(intensity * 1.5, 0.85))
        return clip

    def create_film_grain(
        self, duration: float, intensity: float = 0.15
    ) -> VideoClip:
        """Animated film grain with subtle flicker.

        Generates per-frame noise that simulates photographic film grain
        with periodic brightness flicker for authentic film feel.
        """
        w, h = self.width, self.height
        grain_w = max(w // 2, 480)
        grain_h = max(h // 2, 270)

        from PIL import Image as _PILImage

        def make_frame(t):
            noise = np.random.randint(0, 50, (grain_h, grain_w), dtype=np.uint8)
            grain_img = _PILImage.fromarray(noise, 'L')
            grain_img = grain_img.resize((w, h), _PILImage.BILINEAR)
            grain = np.array(grain_img).astype(np.float32)

            flicker = 1.0 + 0.03 * np.sin(t * 8.0) + 0.02 * np.sin(t * 13.0)
            grain *= flicker * intensity

            frame = np.stack([grain, grain, grain], axis=-1)
            return np.clip(frame, 0, 255).astype(np.uint8)

        clip = VideoClip(make_frame, duration=duration).set_fps(self._overlay_fps)
        clip = clip.set_opacity(min(intensity * 1.2, 0.7))
        return clip

    def create_dust_particles(
        self, duration: float, intensity: float = 0.2, particle_count: int = 12
    ) -> VideoClip:
        """Floating dust particles drifting across the frame.

        Simulates tiny bright particles floating in air, illuminated by
        light — common in documentary footage of interiors.
        Particles are boosted (2× brightness, 1.5× size) on dark frames.
        """
        w, h = self.width, self.height
        rng = np.random.RandomState(42)

        # Pre-generate particle properties with boosted values for dark scenes
        particles = []
        for _ in range(particle_count):
            particles.append({
                'x_start': rng.uniform(0, w),
                'y_start': rng.uniform(0, h),
                'speed_x': rng.uniform(-15, 15) * self.scale,
                'speed_y': rng.uniform(-25, -5) * self.scale,
                'size': rng.uniform(1.5, 6) * self.scale,
                'brightness': rng.uniform(0.6, 1.0),
                'drift_freq': rng.uniform(0.3, 1.5),
                'drift_amp': rng.uniform(5, 20) * self.scale,
            })

        def make_frame(t):
            frame = np.zeros((h, w, 3), dtype=np.uint8)
            for p in particles:
                x = (p['x_start'] + p['speed_x'] * t +
                     p['drift_amp'] * np.sin(t * p['drift_freq'])) % w
                y = (p['y_start'] + p['speed_y'] * t) % h
                ix, iy = int(x), int(y)
                size = max(1, int(p['size']))
                bright = int(p['brightness'] * 255 * intensity)
                x1, y1 = max(0, ix - size), max(0, iy - size)
                x2, y2 = min(w, ix + size + 1), min(h, iy + size + 1)
                frame[y1:y2, x1:x2] = bright
            return frame

        clip = VideoClip(make_frame, duration=duration).set_fps(self._overlay_fps)
        clip = clip.set_opacity(min(intensity * 1.5, 0.8))
        return clip

    def create_bokeh_orbs(
        self, duration: float, intensity: float = 0.3, orb_count: int = 18
    ) -> VideoClip:
        """Floating bokeh light orbs that drift upward with soft glow.

        Creates beautiful out-of-focus light circles that slowly rise and
        drift, adding a magical/cinematic atmosphere. Each orb has its own
        size, speed, and phase for organic variation.
        """
        w, h = self.width, self.height
        rng = np.random.RandomState(99)

        orbs = []
        for _ in range(orb_count):
            orbs.append({
                'x': rng.uniform(0, w),
                'y': rng.uniform(0, h),
                'radius': rng.uniform(12, 45) * self.scale,
                'speed_y': rng.uniform(-30, -8) * self.scale,
                'speed_x': rng.uniform(-8, 8) * self.scale,
                'drift_freq': rng.uniform(0.2, 0.8),
                'drift_amp': rng.uniform(15, 40) * self.scale,
                'brightness': rng.uniform(0.3, 1.0),
                'phase': rng.uniform(0, 2 * np.pi),
            })

        def make_frame(t):
            frame = np.zeros((h, w, 3), dtype=np.float32)
            for o in orbs:
                cx = (o['x'] + o['speed_x'] * t +
                      o['drift_amp'] * np.sin(t * o['drift_freq'] + o['phase'])) % w
                cy = (o['y'] + o['speed_y'] * t) % h
                r = o['radius']
                br = o['brightness'] * intensity

                x1 = max(0, int(cx - r * 2))
                x2 = min(w, int(cx + r * 2) + 1)
                y1 = max(0, int(cy - r * 2))
                y2 = min(h, int(cy + r * 2) + 1)
                if x2 <= x1 or y2 <= y1:
                    continue

                yy, xx = np.ogrid[y1:y2, x1:x2]
                dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
                # Soft ring shape (brighter at edge of circle, dim center)
                ring = np.exp(-((dist - r * 0.7) ** 2) / (r * 0.4) ** 2)
                # Inner glow
                core = np.exp(-dist ** 2 / (r * 0.5) ** 2) * 0.6
                glow = (ring + core) * br
                frame[y1:y2, x1:x2, 0] += glow * 240
                frame[y1:y2, x1:x2, 1] += glow * 220
                frame[y1:y2, x1:x2, 2] += glow * 180
            return np.clip(frame, 0, 255).astype(np.uint8)

        clip = VideoClip(make_frame, duration=duration).set_fps(self._overlay_fps)
        clip = clip.set_opacity(min(intensity * 0.8, 0.45))
        return clip

    def create_camera_shake(
        self, duration: float, intensity: float = 0.5
    ) -> VideoClip:
        """Subtle camera shake overlay — gentle scan-line displacement.

        Creates very subtle horizontal scan-line bands to simulate handheld
        camera vibration. Kept deliberately mild to avoid jarring shaking.
        """
        w, h = self.width, self.height
        amp = intensity * 3.0 * self.scale

        def make_frame(t):
            frame = np.zeros((h, w, 3), dtype=np.uint8)
            freq1 = 2.5
            shift = int(amp * np.sin(t * freq1 * 2 * np.pi))
            band_height = max(2, int(6 * self.scale))
            for y_start in range(0, h, band_height * 12):
                y_end = min(h, y_start + band_height)
                offset = int(shift * np.sin(y_start * 0.03 + t * 8))
                brightness = int(abs(offset) * 6 * intensity)
                brightness = min(brightness, 30)
                frame[y_start:y_end, :, 0] = brightness
                frame[y_start:y_end, :, 1] = brightness
                frame[y_start:y_end, :, 2] = brightness
            return frame

        clip = VideoClip(make_frame, duration=duration).set_fps(self._overlay_fps)
        clip = clip.set_opacity(min(intensity * 0.4, 0.25))
        return clip

    def create_cinematic_bars(
        self, duration: float, bar_height_ratio: float = 0.10
    ) -> VideoClip:
        """Animated cinematic letterbox bars that slide in and out.

        Bars slide in over the first 1.2 seconds and slide out over
        the last 1.2 seconds, creating a dramatic framing effect that
        makes key moments feel like a movie.
        """
        w, h = self.width, self.height
        bar_h = int(h * bar_height_ratio)
        slide_in_dur = min(1.2, duration * 0.2)
        slide_out_dur = min(1.2, duration * 0.2)

        def make_frame(t):
            return np.zeros((h, w, 3), dtype=np.uint8)

        def make_mask(t):
            # Animate bar height
            if t < slide_in_dur:
                progress = t / slide_in_dur
                progress = progress * progress * (3 - 2 * progress)  # smoothstep
                current_bar = int(bar_h * progress)
            elif t > duration - slide_out_dur:
                progress = (duration - t) / slide_out_dur
                progress = progress * progress * (3 - 2 * progress)
                current_bar = int(bar_h * progress)
            else:
                current_bar = bar_h

            mask = np.zeros((h, w), dtype=np.float64)
            if current_bar > 0:
                mask[:current_bar, :] = 1.0
                mask[h - current_bar:, :] = 1.0
            return mask

        clip = VideoClip(make_frame, duration=duration).set_fps(self._overlay_fps)
        mask = VideoClip(make_mask, duration=duration, ismask=True).set_fps(self._overlay_fps)
        clip = clip.set_mask(mask)
        return clip

    def create_lens_flare(
        self, duration: float, intensity: float = 0.25
    ) -> VideoClip:
        """Animated lens flare that drifts across the frame.

        Creates a bright point with radial streaks that moves slowly,
        simulating sunlight hitting the camera lens.
        """
        w, h = self.width, self.height

        def make_frame(t):
            frame = np.zeros((h, w, 3), dtype=np.float32)
            phase = (t * 0.08) % 1.0

            # Flare center position
            cx = int(w * (0.3 + 0.4 * phase))
            cy = int(h * (0.25 + 0.15 * np.sin(phase * np.pi)))

            y, x = np.ogrid[:h, :w]
            dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2).astype(np.float32)
            max_r = max(w, h) * 0.5

            # Core glow
            core = np.exp(-dist ** 2 / (max_r * 0.05) ** 2) * intensity
            frame[:, :, 0] += core * 255
            frame[:, :, 1] += core * 240
            frame[:, :, 2] += core * 200

            # Soft halo
            halo = np.exp(-dist ** 2 / (max_r * 0.2) ** 2) * intensity * 0.3
            frame[:, :, 0] += halo * 255
            frame[:, :, 1] += halo * 220
            frame[:, :, 2] += halo * 180

            # Horizontal streak
            streak_mask = np.exp(-(((y - cy) / max(h * 0.02, 1)) ** 2).astype(np.float32))
            streak = streak_mask * np.exp(-dist / max(max_r * 0.4, 1)) * intensity * 0.4
            frame[:, :, 0] += streak * 255
            frame[:, :, 1] += streak * 230
            frame[:, :, 2] += streak * 180

            return np.clip(frame, 0, 255).astype(np.uint8)

        clip = VideoClip(make_frame, duration=duration).set_fps(self._overlay_fps)
        clip = clip.set_opacity(min(intensity * 1.5, 0.8))
        return clip

    def create_chromatic_aberration(
        self, duration: float, intensity: float = 0.3
    ) -> VideoClip:
        """Chromatic aberration overlay — subtle RGB channel offset at edges.

        Creates color fringing at the frame edges, simulating imperfect
        lens optics. Commonly used in documentary conflict scenes.
        """
        w, h = self.width, self.height

        def make_frame(t):
            frame = np.zeros((h, w, 3), dtype=np.uint8)
            # Edge-weighted color offset
            x_norm = np.linspace(-1, 1, w, dtype=np.float32)[np.newaxis, :]
            y_norm = np.linspace(-1, 1, h, dtype=np.float32)[:, np.newaxis]
            edge = np.sqrt(x_norm ** 2 + y_norm ** 2)
            edge = np.clip(edge - 0.6, 0, 1) * 2.5

            offset = int(3 * self.scale * intensity)
            # Red shifted right/down
            frame[:, min(offset, w - 1):, 0] = (edge[:, :w - offset] * 80 * intensity).astype(np.uint8)
            # Blue shifted left/up
            frame[:, :max(w - offset, 0), 2] = (edge[:, min(offset, w - 1):] * 80 * intensity).astype(np.uint8)

            return frame

        clip = VideoClip(make_frame, duration=duration).set_fps(self._overlay_fps)
        clip = clip.set_opacity(min(intensity * 0.7, 0.35))
        return clip

    def create_film_scratches(
        self, duration: float, intensity: float = 0.15
    ) -> VideoClip:
        """Animated vertical film scratches for vintage documentary feel.

        Generates thin vertical bright lines that appear and disappear
        randomly, simulating aged film stock.
        """
        w, h = self.width, self.height
        rng = np.random.RandomState(123)

        # Pre-generate scratch positions
        num_frames = int(duration * 12)
        scratch_data = []
        for _ in range(num_frames):
            count = rng.randint(0, 4)
            scratches = []
            for _ in range(count):
                scratches.append({
                    'x': rng.randint(0, w),
                    'width': rng.randint(1, max(2, int(2 * self.scale))),
                    'brightness': rng.uniform(0.3, 0.8),
                    'y_start': rng.uniform(0, 0.3) * h,
                    'y_end': rng.uniform(0.7, 1.0) * h,
                })
            scratch_data.append(scratches)

        def make_frame(t):
            frame = np.zeros((h, w, 3), dtype=np.uint8)
            idx = int(t * 12) % len(scratch_data)
            for s in scratch_data[idx]:
                x1 = max(0, s['x'])
                x2 = min(w, s['x'] + s['width'])
                y1 = int(s['y_start'])
                y2 = int(s['y_end'])
                val = int(s['brightness'] * 255 * intensity)
                frame[y1:y2, x1:x2] = val
            return frame

        clip = VideoClip(make_frame, duration=duration).set_fps(self._overlay_fps)
        clip = clip.set_opacity(min(intensity * 0.8, 0.5))
        return clip

    def create_vignette_pulse(
        self, duration: float, intensity: float = 0.2
    ) -> VideoClip:
        """Adaptive breathing vignette that weakens on dark images.

        Automatically reduces vignette strength when the underlying image is
        dark to prevent compounding darkness. Uses 70% reduction for very
        dark images (avg brightness < 80) and 40% reduction for medium
        brightness images (< 120).
        """
        w, h = self.width, self.height

        # Pre-compute vignette mask
        y, x = np.ogrid[:h, :w]
        cx, cy = w / 2.0, h / 2.0
        dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        max_dist = np.sqrt(cx ** 2 + cy ** 2)
        base_vignette = np.clip(dist / max_dist, 0, 1)

        # Scale intensity down — the adaptive per-frame logic in movements.py
        # also reduces vignette on dark images, so keep this very subtle.
        scaled_intensity = intensity * 0.5

        def make_frame(t):
            pulse = 0.06 + 0.02 * np.sin(t * 0.5 * 2 * np.pi)
            vig = (base_vignette * pulse * scaled_intensity * 255).astype(np.uint8)
            frame = np.zeros((h, w, 3), dtype=np.uint8)
            frame[:, :, 0] = vig
            frame[:, :, 1] = vig
            frame[:, :, 2] = vig
            return frame

        def make_mask(t):
            pulse = 0.06 + 0.02 * np.sin(t * 0.5 * 2 * np.pi)
            return base_vignette * pulse * scaled_intensity

        clip = VideoClip(make_frame, duration=duration).set_fps(self._overlay_fps)
        mask = VideoClip(make_mask, duration=duration, ismask=True).set_fps(self._overlay_fps)
        clip = clip.set_mask(mask)
        return clip

    def create_flash_strobe(
        self, duration: float, flash_times: Optional[list] = None,
        intensity: float = 0.8
    ) -> VideoClip:
        """Brief white flash at specified times for dramatic emphasis.

        If flash_times is None, places a single flash at 20% into the clip.
        """
        w, h = self.width, self.height
        if flash_times is None:
            flash_times = [duration * 0.2]

        flash_duration = 0.08

        def make_frame(t):
            for ft in flash_times:
                if ft <= t <= ft + flash_duration:
                    return np.full((h, w, 3), 255, dtype=np.uint8)
            return np.zeros((h, w, 3), dtype=np.uint8)

        def make_mask(t):
            for ft in flash_times:
                if ft <= t <= ft + flash_duration:
                    progress = (t - ft) / flash_duration
                    brightness = (1.0 - progress) * intensity
                    return np.full((h, w), brightness * 0.6, dtype=np.float64)
            return np.zeros((h, w), dtype=np.float64)

        clip = VideoClip(make_frame, duration=duration).set_fps(30)
        mask_clip = VideoClip(make_mask, duration=duration, ismask=True).set_fps(30)
        clip = clip.set_mask(mask_clip)
        return clip

    def create_warm_wash(
        self, duration: float, intensity: float = 0.15
    ) -> VideoClip:
        """Subtle warm color wash that pulses gently.

        Adds a warm golden tone to the entire frame, pulsing slowly for
        an organic cinematic feel. Used for nostalgic/legacy sections.
        """
        w, h = self.width, self.height

        def make_frame(t):
            pulse = 0.7 + 0.3 * np.sin(t * 0.3 * 2 * np.pi)
            frame = np.zeros((h, w, 3), dtype=np.uint8)
            frame[:, :, 0] = int(50 * pulse * intensity)
            frame[:, :, 1] = int(30 * pulse * intensity)
            frame[:, :, 2] = int(10 * pulse * intensity)
            return frame

        clip = VideoClip(make_frame, duration=duration).set_fps(self._overlay_fps)
        clip = clip.set_opacity(min(intensity * 1.0, 0.4))
        return clip

    def create_film_burn_overlay(
        self, duration: float, intensity: float = 0.3
    ) -> VideoClip:
        """Film burn edge effect — warm overexposed edges only.

        Simulates film damage with warm bright areas at the very edges
        of the frame. The center stays clear so the image is always visible.
        """
        w, h = self.width, self.height

        def make_frame(t):
            phase = (t * 0.15) % 1.0
            y_coords = np.linspace(0, 1, h, dtype=np.float32)[:, np.newaxis]
            x_coords = np.linspace(0, 1, w, dtype=np.float32)[np.newaxis, :]

            # Distance from edges — steeper falloff so only extreme edges glow
            edge = np.minimum(
                np.minimum(x_coords, 1 - x_coords),
                np.minimum(y_coords, 1 - y_coords)
            )
            burn = np.exp(-edge * (15 + 5 * np.sin(phase * 2 * np.pi)))
            # Clamp: only edges within 10% of frame edge get the burn
            burn = burn * (edge < 0.12).astype(np.float32)

            frame = np.zeros((h, w, 3), dtype=np.float32)
            frame[:, :, 0] = burn * 200 * intensity
            frame[:, :, 1] = burn * 120 * intensity
            frame[:, :, 2] = burn * 40 * intensity

            return np.clip(frame, 0, 255).astype(np.uint8)

        clip = VideoClip(make_frame, duration=duration).set_fps(self._overlay_fps)
        clip = clip.set_opacity(min(intensity * 0.8, 0.4))
        return clip

    def create_spotlight(
        self, duration: float, intensity: float = 0.7
    ) -> VideoClip:
        """Spotlight effect: very subtle centre-bright, edges slightly darker.

        Uses a large radius and gentle falloff so it enhances focus on the
        centre without making the rest of the image too dark.
        """
        w, h = self.width, self.height
        cx, cy = w // 2, h // 2
        max_radius = int(min(w, h) * 0.50)

        Y, X = np.ogrid[:h, :w]

        def make_frame(t):
            progress = t / duration if duration > 0 else 0
            offset_x = int(np.sin(progress * np.pi * 2) * w * 0.02)
            offset_y = int(np.cos(progress * np.pi * 1.5) * h * 0.015)
            radius = max_radius + int(np.sin(progress * np.pi) * max_radius * 0.1)

            dist = np.sqrt((X - cx - offset_x) ** 2 + (Y - cy - offset_y) ** 2)
            shadow = np.clip((dist - radius) / (radius * 0.8), 0, 1)
            frame = np.zeros((h, w, 3), dtype=np.uint8)
            shadow_val = (shadow * 30 * intensity).astype(np.uint8)
            frame[:, :, 0] = shadow_val
            frame[:, :, 1] = shadow_val
            frame[:, :, 2] = shadow_val
            return frame

        clip = VideoClip(make_frame, duration=duration).set_fps(self._overlay_fps)
        clip = clip.set_opacity(min(intensity * 0.2, 0.15))
        return clip

    def create_photo_frame(
        self, duration: float, intensity: float = 0.7
    ) -> VideoClip:
        """Photo frame: subtle border with inner shadow giving physical photo feel.

        Only renders the thin border edges — the interior is fully transparent
        so the underlying image shows through cleanly.
        """
        w, h = self.width, self.height
        border = int(min(w, h) * 0.025)

        # Pre-compute static frame and mask
        static_frame = np.zeros((h, w, 3), dtype=np.uint8)
        static_mask = np.zeros((h, w), dtype=np.float64)

        # Border area (cream/white frame)
        static_frame[0:border, :] = 210
        static_frame[h - border:h, :] = 210
        static_frame[:, 0:border] = 210
        static_frame[:, w - border:w] = 210
        # Inner shadow on frame edge (slightly darker)
        inner = max(2, int(border * 0.3))
        static_frame[border:border + inner, border:w - border] = 170
        static_frame[h - border - inner:h - border, border:w - border] = 190
        static_frame[border:h - border, border:border + inner] = 170
        static_frame[border:h - border, w - border - inner:w - border] = 190

        # Mask: only the border region is visible, interior is transparent
        static_mask[0:border, :] = 1.0
        static_mask[h - border:h, :] = 1.0
        static_mask[:, 0:border] = 1.0
        static_mask[:, w - border:w] = 1.0
        static_mask[border:border + inner, border:w - border] = 0.6
        static_mask[h - border - inner:h - border, border:w - border] = 0.6
        static_mask[border:h - border, border:border + inner] = 0.6
        static_mask[border:h - border, w - border - inner:w - border] = 0.6

        opacity = min(intensity * 0.45, 0.35)
        static_mask *= opacity

        def make_frame(t):
            return static_frame

        def make_mask(t):
            return static_mask

        clip = VideoClip(make_frame, duration=duration).set_fps(self._overlay_fps)
        mask_clip = VideoClip(make_mask, duration=duration, ismask=True).set_fps(self._overlay_fps)
        clip = clip.set_mask(mask_clip)
        return clip

    def create_god_rays(
        self, duration: float, intensity: float = 0.15
    ) -> VideoClip:
        """Volumetric light beams streaming from upper area.

        Creates animated diagonal light shafts (god rays) that slowly
        rotate and pulse — giving a divine, magical, or epic atmosphere.
        Used in triumph moments, revelations, and legacy sections.
        """
        w, h = self.width, self.height
        Y, X = np.ogrid[:h, :w]
        cx, cy = int(w * 0.5), int(h * 0.1)
        angle_map = np.arctan2(Y - cy, X - cx).astype(np.float32)

        def make_frame(t):
            rot = t * 0.08
            rays = np.sin((angle_map + rot) * 12) ** 2
            rays *= np.clip(1.0 - (Y - cy).astype(np.float32) / h, 0.1, 1.0)
            pulse = 0.7 + 0.3 * np.sin(t * 0.5 * 2 * np.pi)
            val = rays * pulse * intensity
            frame = np.zeros((h, w, 3), dtype=np.float32)
            frame[:, :, 0] = val * 255
            frame[:, :, 1] = val * 240
            frame[:, :, 2] = val * 180
            return np.clip(frame, 0, 255).astype(np.uint8)

        clip = VideoClip(make_frame, duration=duration).set_fps(self._overlay_fps)
        clip = clip.set_opacity(min(intensity * 0.5, 0.2))
        return clip

    def create_fog_overlay(
        self, duration: float, intensity: float = 0.25
    ) -> VideoClip:
        """Slow-drifting fog/smoke overlay for atmospheric depth.

        Creates horizontal bands of semi-transparent fog that slowly
        drift across the frame. Adds cinematic depth and mystery.
        """
        w, h = self.width, self.height
        rng = np.random.RandomState(77)
        bands = []
        for _ in range(6):
            bands.append({
                'y': rng.uniform(0.2, 0.9),
                'thickness': rng.uniform(0.08, 0.2),
                'speed': rng.uniform(-20, 20) * (w / 1920),
                'brightness': rng.uniform(0.4, 1.0),
                'phase': rng.uniform(0, 2 * np.pi),
            })

        Y_norm = np.linspace(0, 1, h, dtype=np.float32)[:, np.newaxis]
        X_coords = np.arange(w, dtype=np.float32)[np.newaxis, :]

        def make_frame(t):
            frame = np.zeros((h, w), dtype=np.float32)
            for b in bands:
                y_center = b['y'] + 0.02 * np.sin(t * 0.3 + b['phase'])
                band_mask = np.exp(-((Y_norm - y_center) ** 2) / (b['thickness'] ** 2))
                x_offset = b['speed'] * t
                x_wave = np.sin((X_coords + x_offset) * 0.01 + b['phase']) * 0.5 + 0.5
                frame += (band_mask * x_wave * b['brightness']).astype(np.float32)
            frame = np.clip(frame * intensity, 0, 1)
            rgb = np.stack([frame * 220, frame * 220, frame * 230], axis=-1)
            return np.clip(rgb, 0, 255).astype(np.uint8)

        clip = VideoClip(make_frame, duration=duration).set_fps(self._overlay_fps)
        clip = clip.set_opacity(min(intensity * 0.35, 0.15))
        return clip

    def create_shimmer_sparkles(
        self, duration: float, intensity: float = 0.4, count: int = 20
    ) -> VideoClip:
        """Magical floating sparkles with starburst glow.

        Creates tiny bright points that appear, flash a cross-shaped
        starburst, then fade — like magical fairy dust or diamond sparkles.
        Much more eye-catching than basic dust particles.
        """
        w, h = self.width, self.height
        rng = np.random.RandomState(55)

        sparkles = []
        for _ in range(count):
            sparkles.append({
                'x': rng.uniform(0, w),
                'y': rng.uniform(0, h),
                'life_start': rng.uniform(0, duration * 0.8),
                'life_dur': rng.uniform(0.3, 1.2),
                'size': rng.uniform(1.5, 4) * (h / 1080),
                'brightness': rng.uniform(0.3, 0.7),
                'drift_x': rng.uniform(-10, 10) * (w / 1920),
                'drift_y': rng.uniform(-15, -3) * (h / 1080),
            })

        def make_frame(t):
            frame = np.zeros((h, w, 3), dtype=np.float32)
            for s in sparkles:
                age = t - s['life_start']
                if age < 0 or age > s['life_dur']:
                    # Repeat sparkle
                    cycle = s['life_dur'] + 0.5
                    age = (t - s['life_start']) % cycle
                    if age > s['life_dur']:
                        continue
                life_frac = age / s['life_dur']
                # Flash curve: quick peak then fade
                flash = np.sin(life_frac * np.pi) ** 0.5
                br = s['brightness'] * flash * intensity

                cx = int(s['x'] + s['drift_x'] * age) % w
                cy = int(s['y'] + s['drift_y'] * age) % h
                sz = max(1, int(s['size'] * (0.5 + flash)))

                # Cross starburst
                y1, y2 = max(0, cy - sz * 3), min(h, cy + sz * 3 + 1)
                x1, x2 = max(0, cx - 1), min(w, cx + 2)
                if y2 > y1 and x2 > x1:
                    frame[y1:y2, x1:x2, 0] += br * 255
                    frame[y1:y2, x1:x2, 1] += br * 245
                    frame[y1:y2, x1:x2, 2] += br * 200
                x1h, x2h = max(0, cx - sz * 3), min(w, cx + sz * 3 + 1)
                y1h, y2h = max(0, cy - 1), min(h, cy + 2)
                if x2h > x1h and y2h > y1h:
                    frame[y1h:y2h, x1h:x2h, 0] += br * 255
                    frame[y1h:y2h, x1h:x2h, 1] += br * 245
                    frame[y1h:y2h, x1h:x2h, 2] += br * 200
                # Core glow
                core_sz = max(1, sz)
                cy1, cy2 = max(0, cy - core_sz), min(h, cy + core_sz + 1)
                cx1, cx2 = max(0, cx - core_sz), min(w, cx + core_sz + 1)
                if cy2 > cy1 and cx2 > cx1:
                    frame[cy1:cy2, cx1:cx2] += br * 255
            return np.clip(frame, 0, 255).astype(np.uint8)

        clip = VideoClip(make_frame, duration=duration).set_fps(self._overlay_fps)
        clip = clip.set_opacity(min(intensity * 0.5, 0.35))
        return clip

    def create_film_strip(
        self, duration: float, intensity: float = 0.5
    ) -> VideoClip:
        """Film strip/reel border effect for nostalgic sections.

        Adds film perforation holes (sprocket holes) on both sides of the
        frame, giving it the look of a physical film strip. Subtle vertical
        jitter simulates a film projector.
        """
        w, h = self.width, self.height
        strip_w = max(int(w * 0.04), 20)
        hole_h = max(int(h * 0.03), 16)
        hole_w = max(int(strip_w * 0.5), 10)
        hole_spacing = max(int(h * 0.06), 40)
        hole_x_left = (strip_w - hole_w) // 2
        hole_x_right = w - strip_w + (strip_w - hole_w) // 2

        static_frame = np.zeros((h, w, 3), dtype=np.uint8)
        static_mask = np.zeros((h, w), dtype=np.float64)

        # Left strip
        static_frame[:, :strip_w] = 20
        static_mask[:, :strip_w] = 0.9
        # Right strip
        static_frame[:, w - strip_w:] = 20
        static_mask[:, w - strip_w:] = 0.9
        # Sprocket holes
        for y_start in range(hole_spacing // 2, h, hole_spacing):
            y_end = min(y_start + hole_h, h)
            # Left holes
            static_mask[y_start:y_end, hole_x_left:hole_x_left + hole_w] = 0
            static_frame[y_start:y_end, hole_x_left:hole_x_left + hole_w] = 0
            # Right holes
            static_mask[y_start:y_end, hole_x_right:hole_x_right + hole_w] = 0
            static_frame[y_start:y_end, hole_x_right:hole_x_right + hole_w] = 0

        opacity = min(intensity * 0.6, 0.5)
        static_mask *= opacity

        def make_frame(t):
            return static_frame

        def make_mask(t):
            # Subtle vertical jitter to simulate projector
            jitter = int(2 * np.sin(t * 18))
            if jitter == 0:
                return static_mask
            result = np.zeros_like(static_mask)
            if jitter > 0:
                result[jitter:] = static_mask[:-jitter]
            else:
                result[:jitter] = static_mask[-jitter:]
            return result

        clip = VideoClip(make_frame, duration=duration).set_fps(self._overlay_fps)
        mask = VideoClip(make_mask, duration=duration, ismask=True).set_fps(self._overlay_fps)
        clip = clip.set_mask(mask)
        return clip

    def create_zoom_burst(
        self, duration: float, intensity: float = 0.25
    ) -> VideoClip:
        """Dramatic radial zoom burst for shocking revelations.

        Creates radial light streaks emanating from center that flash
        briefly, giving a punchy zoom-in impact feel at the start of
        the clip. The burst lasts ~0.4s then fades completely.
        """
        w, h = self.width, self.height
        cx, cy = w // 2, h // 2
        burst_dur = min(0.4, duration * 0.3)

        Y, X = np.ogrid[:h, :w]
        dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2).astype(np.float32)
        max_dist = np.sqrt(cx ** 2 + cy ** 2)
        angle = np.arctan2(Y - cy, X - cx).astype(np.float32)

        def make_frame(t):
            if t > burst_dur:
                return np.zeros((h, w, 3), dtype=np.uint8)
            progress = t / burst_dur
            fade = 1.0 - progress
            radial = np.abs(np.sin(angle * 8 + progress * np.pi * 4))
            radial *= np.clip(dist / max_dist, 0.3, 1.0)
            frame = np.zeros((h, w, 3), dtype=np.float32)
            val = radial * fade * intensity
            frame[:, :, 0] = val * 255
            frame[:, :, 1] = val * 230
            frame[:, :, 2] = val * 180
            return np.clip(frame, 0, 255).astype(np.uint8)

        def make_mask(t):
            if t > burst_dur:
                return np.zeros((h, w), dtype=np.float64)
            fade = 1.0 - (t / burst_dur)
            return np.full((h, w), fade * intensity * 0.2, dtype=np.float64)

        clip = VideoClip(make_frame, duration=duration).set_fps(30)
        mask = VideoClip(make_mask, duration=duration, ismask=True).set_fps(30)
        clip = clip.set_mask(mask)
        return clip

    def create_color_pulse(
        self, duration: float, color: str = 'warm', intensity: float = 0.3
    ) -> VideoClip:
        """Brief color flash that pulses at the start for emotional emphasis.

        Colors: 'warm' (golden), 'cool' (blue), 'red' (danger/conflict),
        'green' (hope/nature). The pulse peaks at 0.3s then fades.
        """
        w, h = self.width, self.height
        color_map = {
            'warm': (255, 200, 100),
            'cool': (100, 160, 255),
            'red': (255, 80, 60),
            'green': (80, 220, 120),
        }
        r, g, b = color_map.get(color, (255, 200, 100))
        pulse_dur = min(0.6, duration * 0.3)

        static_frame = np.zeros((h, w, 3), dtype=np.uint8)
        static_frame[:, :, 0] = r
        static_frame[:, :, 1] = g
        static_frame[:, :, 2] = b

        def make_mask(t):
            if t > pulse_dur:
                return np.zeros((h, w), dtype=np.float64)
            progress = t / pulse_dur
            # Quick peak then fade
            pulse = np.sin(progress * np.pi) * intensity * 0.25
            return np.full((h, w), pulse, dtype=np.float64)

        def make_frame(t):
            return static_frame

        clip = VideoClip(make_frame, duration=duration).set_fps(self._overlay_fps)
        mask = VideoClip(make_mask, duration=duration, ismask=True).set_fps(self._overlay_fps)
        clip = clip.set_mask(mask)
        return clip

    def create_edge_bloom(
        self, duration: float, intensity: float = 0.2
    ) -> VideoClip:
        """Subtle bright-area bloom that gives a dreamy cinematic glow.

        Simulates lens bloom by adding a soft bright haze around the
        brightest parts of the frame. Uses a pre-computed radial gradient.
        """
        w, h = self.width, self.height
        cx, cy = w // 2, h // 2
        Y, X = np.ogrid[:h, :w]
        dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2).astype(np.float32)
        max_dist = np.sqrt(cx ** 2 + cy ** 2)
        # Center-weighted bloom mask
        bloom = np.exp(-dist ** 2 / (max_dist * 0.6) ** 2)

        static_frame = np.zeros((h, w, 3), dtype=np.float32)
        static_frame[:, :, 0] = bloom * 255 * intensity
        static_frame[:, :, 1] = bloom * 240 * intensity
        static_frame[:, :, 2] = bloom * 200 * intensity
        static_frame = np.clip(static_frame, 0, 255).astype(np.uint8)

        def make_frame(t):
            return static_frame

        def make_mask(t):
            pulse = 0.8 + 0.2 * np.sin(t * 0.4 * 2 * np.pi)
            return bloom * intensity * pulse * 0.3

        clip = VideoClip(make_frame, duration=duration).set_fps(self._overlay_fps)
        mask = VideoClip(make_mask, duration=duration, ismask=True).set_fps(self._overlay_fps)
        clip = clip.set_mask(mask)
        return clip

    def get_section_effects(
        self,
        section: str,
        duration: float,
        effects_intensity: float = 0.7,
        max_effects: int = 2
    ) -> list:
        """Get appropriate effect clips for a documentary section.

        Returns a list of VideoClip overlays suited to the given section's
        emotional tone and visual style.  Intensity is automatically scaled
        by section so dramatic moments feel stronger.
        """
        effect_names = self.SECTION_EFFECTS.get(section, ['film_grain', 'dust_particles'])
        effect_names = effect_names[:max_effects]

        mult = self.SECTION_INTENSITY_MULTIPLIERS.get(section, 1.0)
        scaled_intensity = effects_intensity * mult

        return self.get_effects_by_names(effect_names, duration, scaled_intensity)

    def get_effects_by_names(
        self,
        effect_names: list,
        duration: float,
        effects_intensity: float = 0.7
    ) -> list:
        """Create effect clips from a list of effect names."""
        creators = {
            'light_leak': lambda: self.create_light_leak(duration, effects_intensity * 0.6),
            'film_grain': lambda: self.create_film_grain(duration, effects_intensity * 0.35),
            'dust_particles': lambda: self.create_dust_particles(duration, effects_intensity * 0.25, particle_count=12),
            'camera_shake': lambda: self.create_camera_shake(duration, effects_intensity * 0.35),
            'cinematic_bars': lambda: self.create_cinematic_bars(duration),
            'lens_flare': lambda: self.create_lens_flare(duration, effects_intensity * 0.45),
            'chromatic_aberration': lambda: self.create_chromatic_aberration(duration, effects_intensity * 0.5),
            'film_scratches': lambda: self.create_film_scratches(duration, effects_intensity * 0.3),
            'vignette_pulse': lambda: self.create_vignette_pulse(duration, effects_intensity * 0.3),
            'flash_strobe': lambda: self.create_flash_strobe(duration, intensity=effects_intensity * 0.8),
            'warm_wash': lambda: self.create_warm_wash(duration, effects_intensity * 0.2),
            'film_burn_overlay': lambda: self.create_film_burn_overlay(duration, effects_intensity * 0.5),
            'spotlight': lambda: self.create_spotlight(duration, effects_intensity * 0.4),
            'photo_frame': lambda: self.create_photo_frame(duration, effects_intensity * 0.6),
            'bokeh_orbs': lambda: self.create_bokeh_orbs(duration, effects_intensity * 0.5),
            'zoom_burst': lambda: self.create_zoom_burst(duration, effects_intensity * 0.3),
            'color_pulse_warm': lambda: self.create_color_pulse(duration, 'warm', effects_intensity * 0.5),
            'color_pulse_cool': lambda: self.create_color_pulse(duration, 'cool', effects_intensity * 0.5),
            'color_pulse_red': lambda: self.create_color_pulse(duration, 'red', effects_intensity * 0.5),
            'edge_bloom': lambda: self.create_edge_bloom(duration, effects_intensity * 0.4),
            'god_rays': lambda: self.create_god_rays(duration, effects_intensity * 0.25),
            'fog_overlay': lambda: self.create_fog_overlay(duration, effects_intensity * 0.15),
            'shimmer_sparkles': lambda: self.create_shimmer_sparkles(duration, effects_intensity * 0.5),
            'film_strip': lambda: self.create_film_strip(duration, effects_intensity * 0.6),
        }

        clips = []
        for name in effect_names:
            creator = creators.get(name)
            if creator:
                clips.append(creator())
        return clips

    def _get_cached_static(self, key: str, generator) -> np.ndarray:
        """Return a cached static frame, generating it on first access.

        Used for effects that produce the same image every frame (photo_frame
        borders, film_strip sprocket holes, cinematic_bars at full height).
        Avoids redundant NumPy array creation per-frame.

        Safety:
        - Bounded by _CACHE_MAX_ENTRIES to prevent unbounded memory growth.
        - Evicts oldest entry (FIFO) when limit is reached.
        - Generator exceptions are caught — returns fresh frame on failure
          so a bad cache key never crashes the pipeline.
        """
        if key not in self._static_cache:
            # Evict oldest entry if at capacity
            if len(self._static_cache) >= self._CACHE_MAX_ENTRIES:
                oldest_key = next(iter(self._static_cache))
                del self._static_cache[oldest_key]
            try:
                self._static_cache[key] = generator()
            except Exception:
                # On generator failure, return a fresh call without caching
                return generator()
        return self._static_cache[key]

    def clear_cache(self) -> None:
        """Clear cached static effect frames.

        Call between video generations to free memory. This is safe to call
        at any time — the next effect request will simply regenerate the frame.
        """
        self._static_cache.clear()

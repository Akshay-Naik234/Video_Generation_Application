"""Ken Burns movement styles for dynamic image animations.

Includes humanization features that make auto-generated videos feel like
they were manually edited by a professional. Key techniques:
- Camera breathing: subtle sine-wave motion simulating a real camera operator
- Micro-shake: tiny random jitter for dramatic sections (handheld feel)
- Organic easing: varied easing curves per section instead of uniform cubic
- Movement settle: brief deceleration at start/end of motions
- Natural imperfection: slight randomness in zoom/pan targets
"""

import hashlib
from pathlib import Path
from typing import Tuple, Optional, TYPE_CHECKING

import numpy as np
from PIL import Image as PILImage

if TYPE_CHECKING:
    from .color_grading import ColorGrading
    from .ai_effects import DepthEstimator, ParallaxEngine


class MovementStyles:
    """Ken Burns movement styles with human-feel camera motion."""

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
        'static',
        'push_in',
        'pull_out',
        'zoom_pulse',
        'float_drift',
    ]

    # Easing style per section: different sections get different motion curves
    # so the video doesn't feel like one uniform robotic animation.
    SECTION_EASING = {
        'COLD_OPEN': 'ease_out_quart',     # Fast start, slow settle — grabs attention
        'EARLY_LIFE': 'ease_in_out_sine',  # Gentle, smooth — nostalgic feel
        'THE_SPARK': 'ease_in_out_cubic',  # Standard smooth
        'THE_RISE': 'ease_out_cubic',      # Confident forward motion
        'THE_CONFLICT': 'ease_in_quart',   # Slow start, accelerating — building tension
        'THE_CLIMAX': 'overshoot_settle',  # Slight overshoot then settle — dramatic impact
        'THE_FALL': 'ease_in_out_sine',    # Gentle, sad
        'LEGACY': 'ease_out_sine',         # Slow fade-away feel
        'CTA': 'ease_in_out_cubic',        # Standard
    }

    # Sections that get handheld micro-shake for dramatic feel
    SHAKE_SECTIONS = {'THE_CONFLICT', 'THE_CLIMAX', 'THE_FALL'}

    def __init__(self, resolution: Tuple[int, int]):
        self.width, self.height = resolution
        self.resolution = resolution
        self._clip_seed_counter = 0  # Deterministic seed per clip for reproducible variation

    # Section-to-vignette-intensity mapping: dramatic sections get stronger edge darkening
    SECTION_VIGNETTE_INTENSITY = {
        'COLD_OPEN': 0.5,
        'EARLY_LIFE': 0.3,
        'THE_SPARK': 0.4,
        'THE_RISE': 0.4,
        'THE_CONFLICT': 0.55,
        'THE_CLIMAX': 0.6,
        'THE_FALL': 0.5,
        'LEGACY': 0.35,
        'CTA': 0.3,
    }

    def create_animated_clip(
        self,
        image_path: Path,
        duration: float,
        movement_type: str,
        zoom_intensity: float = 1.15,
        color_grader: 'ColorGrading' = None,
        color_grade: str = None,
        enable_vignette: bool = False,
        section: str = '',
        depth_estimator: 'Optional[DepthEstimator]' = None,
        parallax_engine: 'Optional[ParallaxEngine]' = None,
        enable_parallax: bool = False,
        enable_dof: bool = False,
        subject_center: Optional[Tuple[float, float]] = None,
        enable_human_feel: bool = True,
        speed_factor: float = 1.0,
    ):
        """Create an animated clip with the specified movement style and effects.
        
        Uses a memory-efficient approach with make_frame instead of creating many sub-clips.
        Pre-scales the image larger to allow for smooth zooming without per-frame resizing.
        
        When enable_human_feel=True, adds organic camera breathing, micro-shake for
        dramatic sections, varied easing curves, and natural imperfection to make
        the video feel like it was hand-edited by a professional.
        
        When enable_parallax=True and depth_estimator is provided, uses AI depth
        estimation to create 2.5D parallax Ken Burns effects.
        
        When enable_dof=True and depth_estimator is provided, applies cinematic
        depth-of-field blur (sharp subject, blurred background).
        
        speed_factor controls animation speed within the clip's timeline slot:
        >1.0 = faster animation (e.g., 1.1 for action sections),
        <1.0 = slower animation (e.g., 0.85 for emotional peaks).
        The clip duration stays unchanged; only the Ken Burns motion speed varies.
        """
        base_img = PILImage.open(image_path)
        if base_img.mode != 'RGB':
            base_img = base_img.convert('RGB')

        # Generate a deterministic seed for this clip so randomness is reproducible
        # but each clip gets unique variation
        clip_seed = int(hashlib.md5(str(image_path).encode()).hexdigest()[:8], 16)
        self._clip_seed_counter += 1
        rng = np.random.RandomState(clip_seed + self._clip_seed_counter)

        # --- AI Parallax path: depth-aware 2.5D Ken Burns ---
        use_parallax = (
            enable_parallax
            and depth_estimator is not None
            and parallax_engine is not None
        )

        if use_parallax:
            return self._create_parallax_clip(
                base_img, duration, movement_type, zoom_intensity,
                color_grader, color_grade, enable_vignette, section,
                depth_estimator, parallax_engine, enable_dof, subject_center,
                enable_human_feel=enable_human_feel, rng=rng,
            )

        # --- Standard (flat) Ken Burns path ---
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
            vignette_intensity = self.SECTION_VIGNETTE_INTENSITY.get(section, 0.4)
            base_array = self._apply_vignette(base_array, intensity=vignette_intensity)

        scaled_img = PILImage.fromarray(base_array)
        
        center_x = scaled_width / 2.0
        center_y = scaled_height / 2.0

        # --- Humanization: pre-compute per-clip variation ---
        # Slight random offset to zoom target so each clip feels unique
        zoom_variation = rng.uniform(-0.008, 0.008) if enable_human_feel else 0.0
        # Slight random pan bias so camera doesn't always center perfectly
        pan_bias_x = rng.uniform(-0.003, 0.003) if enable_human_feel else 0.0
        pan_bias_y = rng.uniform(-0.002, 0.002) if enable_human_feel else 0.0
        # Choose easing curve based on section
        easing_fn = self._get_easing_for_section(section) if enable_human_feel else self._ease_in_out_cubic
        # Pre-generate micro-shake offsets (only for dramatic sections)
        use_shake = enable_human_feel and section in self.SHAKE_SECTIONS
        if use_shake:
            # Pre-generate smooth noise for shake (not per-frame random which looks jittery)
            num_shake_samples = max(int(duration * 30), 60)
            shake_x = self._generate_smooth_noise(num_shake_samples, 0.0008, rng)
            shake_y = self._generate_smooth_noise(num_shake_samples, 0.0006, rng)
        else:
            shake_x = shake_y = None
        # Camera breathing parameters (subtle sine wave)
        breath_freq = rng.uniform(0.3, 0.6) if enable_human_feel else 0
        breath_amp = 0.004 if enable_human_feel else 0
        
        def make_frame(t):
            progress = t / duration if duration > 0 else 0
            progress = max(0.0, min(1.0, progress))

            # Apply speed ramp: scale progress so animation moves faster/slower
            # within the same timeline slot. Clamped to [0, 1] so it never
            # overshoots the movement range.
            if speed_factor != 1.0:
                progress = min(1.0, progress * speed_factor)

            # Apply settle: brief deceleration at very start and end
            if enable_human_feel:
                settled = self._apply_settle(progress)
            else:
                settled = progress

            eased = easing_fn(settled)
            
            zoom, pan_x, pan_y = self._calculate_movement(
                movement_type, eased, zoom_intensity
            )

            # --- Humanization layers ---
            if enable_human_feel:
                # Natural zoom imperfection
                zoom += zoom_variation * eased
                # Pan bias (camera not perfectly centered)
                pan_x += pan_bias_x
                pan_y += pan_bias_y
                # Camera breathing (subtle oscillation)
                if breath_amp > 0:
                    breath = breath_amp * np.sin(2 * np.pi * breath_freq * t)
                    zoom += breath
                # Micro-shake for dramatic sections
                if use_shake:
                    idx = min(int(progress * (len(shake_x) - 1)), len(shake_x) - 1)
                    pan_x += shake_x[idx]
                    pan_y += shake_y[idx]
            
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

    def _create_parallax_clip(
        self,
        base_img: PILImage.Image,
        duration: float,
        movement_type: str,
        zoom_intensity: float,
        color_grader,
        color_grade: str,
        enable_vignette: bool,
        section: str,
        depth_estimator,
        parallax_engine,
        enable_dof: bool,
        subject_center: Optional[Tuple[float, float]],
        enable_human_feel: bool = True,
        rng: Optional[np.random.RandomState] = None,
    ):
        """Create a 2.5D parallax Ken Burns clip using AI depth estimation.

        Foreground elements move faster than background elements, creating a
        convincing 3D parallax illusion from a single 2D image. This technique
        is used by top documentary and biography YouTube creators.
        """
        from .ai_effects import DepthOfFieldEffect

        if rng is None:
            rng = np.random.RandomState(42)

        # Resize to output resolution for parallax processing
        resized = base_img.resize(self.resolution, PILImage.LANCZOS)
        img_array = np.array(resized)

        # Apply color grading before depth estimation
        if color_grader and color_grade:
            img_array = color_grader.apply_grade(img_array, color_grade)

        if enable_vignette:
            vignette_intensity = self.SECTION_VIGNETTE_INTENSITY.get(section, 0.4)
            img_array = self._apply_vignette(img_array, intensity=vignette_intensity)

        # Estimate depth map
        depth_map = depth_estimator.estimate_depth(img_array)

        # Apply depth-of-field blur if enabled
        if enable_dof:
            dof = DepthOfFieldEffect()
            img_array = dof.apply(img_array, depth_map, blur_strength=6.0, focus_point=0.7)

        src_img = img_array.copy()
        src_depth = depth_map.copy()

        # Humanization for parallax path
        easing_fn = self._get_easing_for_section(section) if enable_human_feel else self._ease_in_out_cubic
        use_shake = enable_human_feel and section in self.SHAKE_SECTIONS
        breath_freq = rng.uniform(0.3, 0.6) if enable_human_feel else 0
        breath_amp = 0.003 if enable_human_feel else 0

        def make_parallax_frame(t):
            progress = t / duration if duration > 0 else 0
            progress = max(0.0, min(1.0, progress))
            if enable_human_feel:
                progress = self._apply_settle(progress)
            eased = easing_fn(progress)

            # Camera breathing adds subtle zoom oscillation
            extra_intensity = 0.0
            if enable_human_feel and breath_amp > 0:
                extra_intensity = breath_amp * np.sin(2 * np.pi * breath_freq * t)

            frame = parallax_engine.create_parallax_frame(
                src_img, src_depth, eased, movement_type,
                zoom_intensity - 1.0 + 0.5 + extra_intensity
            )
            return frame

        from moviepy.video.VideoClip import VideoClip
        clip = VideoClip(make_parallax_frame, duration=duration)
        clip = clip.set_fps(30)
        return clip

    def _apply_vignette(self, image: np.ndarray, intensity: float = 0.4) -> np.ndarray:
        """Apply a vignette effect to the image.
        
        Args:
            image: Input image as numpy array.
            intensity: How strong the darkening is at edges (0.0-1.0).
                       0.4 = subtle cinematic, 0.6 = dramatic spotlight.
        """
        rows, cols = image.shape[:2]
        X = np.arange(0, cols)
        Y = np.arange(0, rows)
        X, Y = np.meshgrid(X, Y)
        center_x, center_y = cols / 2, rows / 2
        distance = np.sqrt((X - center_x) ** 2 + (Y - center_y) ** 2)
        max_distance = np.sqrt(center_x ** 2 + center_y ** 2)
        vignette = 1 - (distance / max_distance) * intensity
        min_brightness = max(0.3, 1.0 - intensity)
        vignette = np.clip(vignette, min_brightness, 1.0)
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

        elif movement_type == 'push_in':
            # Slow deliberate push toward subject - great for tension/revelation moments
            zoom = 1.0 + (zoom_intensity - 1.0) * 1.1 * progress
            pan_y = -0.005 * progress  # Slight upward drift adds gravitas
            return zoom, 0, pan_y

        elif movement_type == 'pull_out':
            # Slow pull away from subject - great for legacy/reflection moments
            zoom = zoom_intensity - (zoom_intensity - 1.0) * 0.9 * progress
            pan_y = 0.005 * progress  # Slight downward drift for melancholy
            return zoom, 0, pan_y

        elif movement_type == 'zoom_pulse':
            # Heartbeat-like zoom pulse for dramatic/climactic moments
            # Two quick pulses then settle - mimics a racing heartbeat
            pulse = np.sin(progress * np.pi * 4) * np.exp(-progress * 2)
            zoom = 1.0 + (zoom_intensity - 1.0) * 0.6 * progress + 0.03 * pulse
            return zoom, 0, 0

        elif movement_type == 'float_drift':
            # Shivanshu-style 3D float effect: subtle oscillating drift that
            # makes still images feel alive. Combines gentle zoom breathing with
            # slow horizontal/vertical oscillation for a parallax-like feel.
            zoom = 1.0 + 0.03 * np.sin(progress * np.pi * 1.5) + (zoom_intensity - 1.0) * 0.3 * progress
            pan_x = 0.015 * np.sin(progress * np.pi * 2.0)
            pan_y = 0.008 * np.cos(progress * np.pi * 1.3)
            return zoom, pan_x, pan_y

        elif movement_type == 'static':
            return 1.0, 0, 0

        else:
            zoom = 1.0 + (zoom_intensity - 1.0) * progress * 0.5
            return zoom, 0, 0

    # --- Humanization helpers ---

    def _get_easing_for_section(self, section: str):
        """Return the easing function appropriate for the narrative section.
        
        Different sections use different easing curves so the video doesn't
        feel like one uniform robotic animation. This mimics how human editors
        instinctively vary the "feel" of motion per scene.
        """
        easing_name = self.SECTION_EASING.get(section, 'ease_in_out_cubic')
        easing_map = {
            'ease_in_out_cubic': self._ease_in_out_cubic,
            'ease_in_out_sine': self._ease_in_out_sine,
            'ease_out_quart': self._ease_out_quart,
            'ease_out_cubic': self._ease_out_cubic,
            'ease_in_quart': self._ease_in_quart,
            'ease_out_sine': self._ease_out_sine,
            'overshoot_settle': self._ease_overshoot_settle,
        }
        return easing_map.get(easing_name, self._ease_in_out_cubic)

    @staticmethod
    def _apply_settle(progress: float) -> float:
        """Apply a subtle settle at the very start and end of motion.
        
        Human camera operators don't start/stop movement instantly — there's
        always a brief acceleration at the start and deceleration at the end.
        This maps the first/last 8% of progress through an extra ease curve
        that creates that natural settle feel.
        """
        settle_zone = 0.08
        if progress < settle_zone:
            # Slow start: ease-in within the settle zone
            local = progress / settle_zone
            return settle_zone * (local * local) 
        elif progress > (1.0 - settle_zone):
            # Slow end: ease-out within the settle zone
            local = (progress - (1.0 - settle_zone)) / settle_zone
            return (1.0 - settle_zone) + settle_zone * (1.0 - (1.0 - local) * (1.0 - local))
        return progress

    @staticmethod
    def _generate_smooth_noise(num_samples: int, amplitude: float,
                               rng: np.random.RandomState) -> np.ndarray:
        """Generate smooth random noise for micro-shake effect.
        
        Uses cumulative sum of small random steps, then normalizes and smooths
        with a running average. This produces organic-looking camera jitter
        instead of harsh per-frame randomness.
        """
        raw = rng.normal(0, 1, num_samples)
        # Cumulative sum creates a random walk
        walk = np.cumsum(raw)
        # Normalize to [-1, 1] range
        walk_range = walk.max() - walk.min()
        if walk_range > 0:
            walk = 2.0 * (walk - walk.min()) / walk_range - 1.0
        else:
            walk = np.zeros(num_samples)
        # Smooth with running average (window of 5)
        kernel = np.ones(5) / 5.0
        smoothed = np.convolve(walk, kernel, mode='same')
        return smoothed * amplitude

    # --- Easing functions ---
    # Multiple easing curves for varied motion feel per section.

    def _ease_in_out_cubic(self, t: float) -> float:
        """Standard cubic ease-in-out."""
        return 3 * t * t - 2 * t * t * t

    @staticmethod
    def _ease_in_out_sine(t: float) -> float:
        """Gentle sinusoidal ease — nostalgic, smooth feel."""
        return 0.5 * (1.0 - np.cos(np.pi * t))

    @staticmethod
    def _ease_out_quart(t: float) -> float:
        """Fast start, slow settle — attention-grabbing."""
        return 1.0 - (1.0 - t) ** 4

    @staticmethod
    def _ease_out_cubic(t: float) -> float:
        """Confident forward motion with soft landing."""
        return 1.0 - (1.0 - t) ** 3

    @staticmethod
    def _ease_in_quart(t: float) -> float:
        """Slow start, accelerating — building tension."""
        return t ** 4

    @staticmethod
    def _ease_out_sine(t: float) -> float:
        """Gentle fade-away feel."""
        return np.sin(t * np.pi * 0.5)

    @staticmethod
    def _ease_overshoot_settle(t: float) -> float:
        """Slight overshoot then settle back — dramatic impact.
        
        Like a camera that pushes slightly past its target and settles.
        Used for climactic moments to add weight and impact.
        """
        # Overshoot by ~3% at t=0.85, then settle to 1.0
        if t < 0.85:
            return 1.03 * (3 * t * t - 2 * t * t * t) / (3 * 0.85 * 0.85 - 2 * 0.85 ** 3)
        else:
            # Settle from overshoot back to 1.0
            local = (t - 0.85) / 0.15
            return 1.03 - 0.03 * local

    def _dramatic_ease(self, t: float) -> float:
        """Dramatic easing for impactful moments."""
        return 0.5 * (np.sin((t - 0.5) * np.pi) + 1)

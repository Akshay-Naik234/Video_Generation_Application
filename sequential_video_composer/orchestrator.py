"""Sequential Video Orchestrator — Main orchestration class.

Enhanced with documentary-style section-aware animation selection:
movements, transitions, and visual effects are chosen based on the
section (COLD_OPEN, EARLY_LIFE, …) and emotional_tone from the
duration config JSON.
"""

import logging
import os
import re
import json
import random
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Union

import numpy as np
from moviepy.editor import AudioFileClip, CompositeVideoClip
from moviepy.video.VideoClip import VideoClip

from .transitions import TransitionEffects
from .movements import MovementStyles
from .color_grading import ColorGrading
from .clip_factory import ClipFactory
from .text_overlays import TextOverlayEngine
from .effects import DocumentaryEffects
from .performance import RenderOptimizer

logger = logging.getLogger(__name__)


class SequentialVideoOrchestrator:
    """Orchestrates video creation from sequentially numbered images with professional effects."""

    SUPPORTED_IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp', '.tiff'}

    def __init__(
        self,
        images_root: Union[str, Path],
        output_path: Union[str, Path] = "sequential_video_output.mp4",
        resolution: Tuple[int, int] = (1920, 1080),
        fps: int = 30,
        image_duration: float = 4.0,
        crossfade_duration: float = 1.5,
        zoom_intensity: float = 1.08,
        effects_intensity: float = 0.4,
        audio_path: Optional[Union[str, Path]] = None,
        transition_style: str = "random",
        movement_style: str = "random",
        color_grade: str = "cinematic",
        enable_vignette: bool = True,
        enable_film_grain: bool = False,
        enable_text_overlays: bool = True,
        overlay_accent_color: Tuple[int, int, int] = (218, 165, 32),
        duration_config_path: Optional[Union[str, Path]] = None,
        enable_documentary_effects: bool = True,
    ):
        self.images_root = Path(images_root)
        self.output_path = Path(output_path)
        self.resolution = resolution
        self.width, self.height = resolution
        self.fps = fps
        self.image_duration = image_duration
        self.crossfade_duration = crossfade_duration
        self.zoom_intensity = zoom_intensity
        self.effects_intensity = effects_intensity
        self.audio_path = Path(audio_path) if audio_path else None
        self.transition_style = transition_style
        self.movement_style = movement_style
        self.color_grade = color_grade
        self.enable_vignette = enable_vignette
        self.enable_film_grain = enable_film_grain
        self.enable_text_overlays = enable_text_overlays
        self.overlay_accent_color = overlay_accent_color
        self.duration_config_path = Path(duration_config_path) if duration_config_path else None
        self.enable_documentary_effects = enable_documentary_effects
        self.image_durations: Dict[int, Dict] = {}
        self.overlay_texts: Dict[int, str] = {}
        self.image_sections: Dict[int, str] = {}
        self.image_tones: Dict[int, str] = {}
        self.image_shot_types: Dict[int, str] = {}
        self.image_movement_types: Dict[int, str] = {}
        self.image_transition_types: Dict[int, str] = {}
        self.image_effects: Dict[int, list] = {}
        self.image_map_labels: Dict[int, str] = {}
        self.image_text_animations: Dict[int, str] = {}
        self.total_video_duration: float = 0
        self.preview_mode: bool = False
        self.fast_mode: bool = False
        self.aspect_mode: str = 'fill'

        self._validate_config()

        if self.duration_config_path:
            self._load_duration_config()

        self.transitions = TransitionEffects(resolution)
        self.movements = MovementStyles(resolution)
        self.color_grading = ColorGrading()
        self.clip_factory = ClipFactory(self)
        self.text_overlay_engine = TextOverlayEngine(resolution, accent_color=overlay_accent_color)
        self.effects = DocumentaryEffects(resolution, fps=fps)
        self.optimizer = RenderOptimizer(resolution=resolution)

    def _validate_config(self) -> None:
        """Validate configuration values and log warnings for suspect values."""
        if self.fps < 1 or self.fps > 120:
            logger.warning("FPS %d is outside normal range (1-120), using 30", self.fps)
            self.fps = 30
        if self.image_duration <= 0:
            logger.warning("image_duration must be positive, using 4.0")
            self.image_duration = 4.0
        if self.crossfade_duration < 0:
            logger.warning("crossfade_duration must be non-negative, using 0")
            self.crossfade_duration = 0
        if not (0.0 <= self.effects_intensity <= 2.0):
            logger.warning("effects_intensity %.2f out of range [0, 2], clamping", self.effects_intensity)
            self.effects_intensity = max(0.0, min(2.0, self.effects_intensity))
        if not (1.0 <= self.zoom_intensity <= 2.0):
            logger.warning("zoom_intensity %.2f out of range [1, 2], clamping", self.zoom_intensity)
            self.zoom_intensity = max(1.0, min(2.0, self.zoom_intensity))
        w, h = self.resolution
        if w < 320 or h < 240:
            logger.warning("Resolution %dx%d is very small; results may look poor", w, h)
        if self.audio_path and not self.audio_path.exists():
            logger.warning("Audio file not found: %s — will be skipped", self.audio_path)
            self.audio_path = None

    def _load_duration_config(self) -> None:
        """Load image durations, timing, sections, and tones from JSON config."""
        if not self.duration_config_path or not self.duration_config_path.exists():
            logger.warning("Duration config file not found: %s", self.duration_config_path)
            return

        try:
            with open(self.duration_config_path, 'r') as f:
                config = json.load(f)

            metadata = config.get('video_metadata', {})
            self.total_video_duration = metadata.get('total_duration_seconds', 0)

            images_data = config.get('images', [])
            if not isinstance(images_data, list):
                logger.error("Duration config 'images' must be a list, got %s", type(images_data).__name__)
                return
            for img_data in images_data:
                if not isinstance(img_data, dict):
                    logger.warning("Skipping non-dict entry in images list: %s", type(img_data).__name__)
                    continue
                image_num = img_data.get('image')
                duration = img_data.get('duration')
                start_time = img_data.get('start_time')
                end_time = img_data.get('end_time')
                overlay_text = img_data.get('overlay_text', '')
                section = img_data.get('section', '')
                tone = img_data.get('emotional_tone', '')
                shot_type = img_data.get('shot_type', '')

                if image_num is not None:
                    self.image_durations[image_num] = {
                        'duration': float(duration) if duration is not None else self.image_duration,
                        'start_time': float(start_time) if start_time is not None else None,
                        'end_time': float(end_time) if end_time is not None else None
                    }
                    if overlay_text:
                        self.overlay_texts[image_num] = overlay_text
                    if section:
                        self.image_sections[image_num] = section
                    if tone:
                        self.image_tones[image_num] = tone
                    if shot_type:
                        self.image_shot_types[image_num] = shot_type

                    movement_type = img_data.get('movement_type', '')
                    if movement_type:
                        self.image_movement_types[image_num] = movement_type

                    transition_type = img_data.get('transition_type', '')
                    if transition_type:
                        self.image_transition_types[image_num] = transition_type

                    effects = img_data.get('effects', [])
                    if effects:
                        self.image_effects[image_num] = effects

                    map_label = img_data.get('map_label', '')
                    if map_label:
                        self.image_map_labels[image_num] = map_label

                    text_animation = img_data.get('text_animation', '')
                    if text_animation:
                        self.image_text_animations[image_num] = text_animation

            logger.info("Loaded timing data for %d images from config", len(self.image_durations))
            if self.overlay_texts:
                logger.info("Found %d overlay text entries", len(self.overlay_texts))
            if self.image_sections:
                logger.info("Found section data for %d images", len(self.image_sections))
            if self.total_video_duration:
                logger.info("Total video duration from config: %ss", self.total_video_duration)
        except (json.JSONDecodeError, KeyError) as e:
            logger.error("Error loading duration config: %s", e)

    def get_timing_for_image(self, image_number: int) -> Dict:
        """Get the full timing info for a specific image number."""
        if self.image_durations and image_number in self.image_durations:
            return self.image_durations[image_number]
        return {
            'duration': self.image_duration,
            'start_time': None,
            'end_time': None
        }

    def get_duration_for_image(self, image_number: int) -> float:
        """Get the duration for a specific image number."""
        timing = self.get_timing_for_image(image_number)
        return timing.get('duration', self.image_duration)

    # ---- Micro-pacing variation (v2.0) ----

    # Golden ratio for timing variation
    _PHI = 1.618

    def apply_micro_pacing(
        self, base_duration: float, image_index: int, section: str = ''
    ) -> float:
        """Apply subtle micro-pacing variation within sections.

        Alternates between 'punch' (shorter) and 'breathe' (longer) images
        to prevent monotonous rhythm. Variation is ±15% of base duration,
        modulated by section type.

        Only applied when no explicit per-image duration is set in JSON config.
        """
        # Sections that should have tighter variation (action scenes)
        tight_sections = {'COLD_OPEN', 'THE_CONFLICT', 'THE_CLIMAX'}
        # Sections that should have wider variation (contemplative)
        wide_sections = {'LEGACY', 'EARLY_LIFE'}

        if section in tight_sections:
            variation = 0.10
        elif section in wide_sections:
            variation = 0.18
        else:
            variation = 0.15

        # Alternate punch/breathe using golden ratio for natural feel
        if image_index % 3 == 0:
            # Punch image (shorter)
            factor = 1.0 - variation
        elif image_index % 3 == 1:
            # Normal image
            factor = 1.0
        else:
            # Breathe image (longer)
            factor = 1.0 + variation

        result = base_duration * factor
        return max(2.0, min(result, 8.0))

    def discover_numbered_images(self) -> List[Tuple[int, Path]]:
        """Discover and sort images by their numeric prefix."""
        if not self.images_root.exists():
            raise FileNotFoundError(f"Images directory not found: {self.images_root}")

        numbered_images = []
        pattern = re.compile(r'^(\d+)\.')

        for file_path in self.images_root.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in self.SUPPORTED_IMAGE_EXTENSIONS:
                match = pattern.match(file_path.name)
                if match:
                    number = int(match.group(1))
                    numbered_images.append((number, file_path))

        if not numbered_images:
            raise ValueError(
                f"No numbered images found in {self.images_root}. "
                "Images should be named like: 1.png, 2.jpg, 3.jpeg, etc."
            )

        numbered_images.sort(key=lambda x: x[0])
        logger.info("Discovered %d numbered images", len(numbered_images))

        for num, path in numbered_images:
            logger.debug("  %d: %s", num, path.name)

        return numbered_images

    def _get_movement_for_image(self, index: int, total: int, image_num: int = 0) -> str:
        """Get movement style for an image based on section, tone, and position.

        Priority: per-image movement_type from JSON > section/tone pool > legacy logic.
        """
        # 1. Per-image explicit movement from JSON config (highest priority)
        if image_num and image_num in self.image_movement_types:
            movement = self.image_movement_types[image_num]
            if movement in MovementStyles.MOVEMENT_TYPES:
                logger.debug("  Using per-image movement_type from config: %s", movement)
                return movement
            else:
                logger.warning("Unknown movement_type '{movement}' in config, falling back to section pool")

        section = self.image_sections.get(image_num, '')
        tone = self.image_tones.get(image_num, '')

        # 2. Section-aware documentary mode (fallback)
        if self.movement_style in ('documentary', 'random') and section:
            section_pool = MovementStyles.SECTION_MOVEMENTS.get(section, [])
            tone_pool = MovementStyles.TONE_MOVEMENTS.get(tone, [])

            # Build a weighted pool: section movements + tone movements
            pool = section_pool + tone_pool
            # Filter out shaky movements for calm sections
            calm_sections = {'LEGACY', 'CTA', 'EARLY_LIFE'}
            if section in calm_sections:
                pool = [m for m in pool if m not in ('handheld_drift', 'whip_pan', 'dutch_tilt')]
            if pool:
                return random.choice(pool)

        if self.movement_style == "random":
            return random.choice(MovementStyles.MOVEMENT_TYPES)
        elif self.movement_style == "sequential":
            movements = ['zoom_in', 'pan_left', 'zoom_out', 'pan_right', 'diagonal_tl_br', 'gentle_drift']
            return movements[index % len(movements)]
        elif self.movement_style == "dramatic_sequence":
            if index == 0:
                return 'dramatic_zoom'
            elif index == total - 1:
                return 'zoom_out'
            elif index < total // 3:
                return random.choice(['zoom_in', 'pan_right', 'gentle_drift'])
            elif index < 2 * total // 3:
                return random.choice(['pan_left', 'pan_right', 'breathing'])
            else:
                return random.choice(['zoom_out', 'focus_center', 'gentle_drift'])
        elif self.movement_style == "documentary":
            subtle_movements = [
                'parallax_depth', 'gentle_drift', 'push_in', 'float_up',
                'breathing', 'crane_up', 'orbit', 'zoom_in',
            ]
            return subtle_movements[index % len(subtle_movements)]
        else:
            return self.movement_style if self.movement_style in MovementStyles.MOVEMENT_TYPES else 'zoom_in'

    def _get_transition_for_image(self, index: int, total: int, image_num: int = 0) -> str:
        """Get transition style for an image based on section and position.

        Priority: per-image transition_type from JSON > section pool > legacy logic.
        """
        # 1. Per-image explicit transition from JSON config (highest priority)
        if image_num and image_num in self.image_transition_types:
            transition = self.image_transition_types[image_num]
            if transition in TransitionEffects.TRANSITION_TYPES:
                logger.debug("  Using per-image transition_type from config: %s", transition)
                return transition
            else:
                logger.warning("Unknown transition_type '{transition}' in config, falling back to section pool")

        section = self.image_sections.get(image_num, '')

        # 2. Section-aware mode (fallback)
        if self.transition_style in ('cinematic', 'random') and section:
            section_pool = TransitionEffects.SECTION_TRANSITIONS.get(section, [])
            if section_pool:
                return random.choice(section_pool)

        if self.transition_style == "random":
            return random.choice(TransitionEffects.TRANSITION_TYPES)
        elif self.transition_style == "sequential":
            transitions = ['crossfade', 'slide_left', 'fade_through_black', 'slide_right', 'zoom_in']
            return transitions[index % len(transitions)]
        elif self.transition_style == "cinematic":
            if index == 0:
                return 'fade_through_black'
            elif index == total - 2:
                return 'fade_through_black'
            else:
                return random.choice(['crossfade', 'slide_left', 'slide_right'])
        else:
            return self.transition_style if self.transition_style in TransitionEffects.TRANSITION_TYPES else 'crossfade'

    def create_image_clips(self, numbered_images: List[Tuple[int, Path]]) -> List[Dict]:
        """Create animated clips for each image with movement, effects, and timing info."""
        return self.clip_factory.create_image_clips(numbered_images)

    def create_timeline_video(self, clips_data: List[Dict]) -> CompositeVideoClip:
        """Create video with clips positioned at their exact start times."""
        return self.clip_factory.create_timeline_video(clips_data)

    def create_video(self) -> None:
        """Main method to create the sequential video with professional effects."""
        import time as _time
        t0 = _time.monotonic()

        # Propagate rendering mode to movements engine
        self.movements.aspect_mode = self.aspect_mode
        self.movements.fast_mode = self.fast_mode or self.preview_mode

        logger.info("=" * 60)
        logger.info("Sequential Video Orchestrator (Documentary Enhanced)")
        logger.info("=" * 60)
        logger.info("  Images root     : %s", self.images_root)
        logger.info("  Output path     : %s", self.output_path)
        logger.info("  Resolution      : %s", self.resolution)
        logger.info("  FPS             : %s", self.fps)
        logger.info("  Image duration  : %ss", self.image_duration)
        logger.info("  Transition style: %s", self.transition_style)
        logger.info("  Movement style  : %s", self.movement_style)
        logger.info("  Color grade     : %s", self.color_grade)
        logger.info("  Effects enabled : %s", self.enable_documentary_effects)
        if self.fast_mode:
            logger.info("  Render mode     : FAST (skip sharpening, fast resize + encode)")
        elif self.preview_mode:
            logger.info("  Render mode     : PREVIEW (480p, fast encode)")
        else:
            logger.info("  Render mode     : QUALITY")
        if self.total_video_duration:
            logger.info("  Target duration : %ss", self.total_video_duration)
        logger.info("=" * 60)

        numbered_images = self.discover_numbered_images()
        logger.info("[1/5] Creating animated clips...")
        clips_data = self.create_image_clips(numbered_images)

        if not clips_data:
            raise ValueError("No valid image clips were created")

        logger.info("[2/5] Building timeline...")
        main_video = self.create_timeline_video(clips_data)

        if main_video is None:
            raise ValueError("Failed to create timeline video")

        logger.info("  Main video duration: %ss", main_video.duration)

        overlays = [main_video]

        # Documentary effects (section-aware overlays)
        logger.info("[3/5] Applying effects...")
        if self.enable_documentary_effects:
            doc_overlays = self._create_documentary_effect_overlays(clips_data)
            overlays.extend(doc_overlays)
            logger.info("  Applied %d documentary effect overlays", len(doc_overlays))
        else:
            if self.effects_intensity > 0.3:
                particles = self.clip_factory.create_particle_overlay(
                    main_video.duration,
                    self.effects_intensity * 0.15
                )
                overlays.append(particles)

            if self.enable_film_grain and self.effects_intensity > 0.5:
                grain = self.clip_factory.create_film_grain_overlay(
                    main_video.duration,
                    self.effects_intensity * 0.1
                )
                overlays.append(grain)

        # Text overlays from duration config
        logger.info("[4/5] Adding text overlays...")
        if self.enable_text_overlays and self.overlay_texts:
            text_clips = self._create_text_overlay_clips(clips_data)
            overlays.extend(text_clips)
            logger.info("  Added %d text overlay clips", len(text_clips))
        else:
            logger.info("  No text overlays configured")

        if len(overlays) > 1:
            main_video = CompositeVideoClip(overlays)

        if self.audio_path and self.audio_path.exists():
            logger.info("  Adding audio from: %s", self.audio_path)
            audio_clip = AudioFileClip(str(self.audio_path))
            if audio_clip.duration > main_video.duration:
                audio_clip = audio_clip.subclip(0, main_video.duration)
            audio_clip = self._normalize_audio(audio_clip)
            main_video = main_video.set_audio(audio_clip)

        logger.info("[5/5] Exporting video...")
        self._export_video(main_video)

        # Release thread pool and buffers after export
        self.optimizer.cleanup()

        elapsed = _time.monotonic() - t0
        logger.info("=" * 60)
        logger.info("Video created successfully in %.1fs", elapsed)
        logger.info("  Output: %s", self.output_path)
        logger.info("  Size  : %.1f MB", self.output_path.stat().st_size / (1024 * 1024))
        logger.info("=" * 60)

    def _create_documentary_effect_overlays(self, clips_data: List[Dict]) -> List:
        """Create documentary effect overlays.

        Priority: per-image effects from JSON > section-based effects > global defaults.
        When per-image effects are specified, they are applied to each image individually.
        Otherwise, falls back to grouping by section.
        """
        effect_clips = []

        # 1. Per-image effects from JSON config (highest priority)
        max_effects_per_image = 2
        per_image_handled = set()
        if self.image_effects:
            logger.info(f"\nApplying per-image effects from JSON config...")
            for data in clips_data:
                img_num = data.get('image_num')
                if img_num and img_num in self.image_effects:
                    start = data.get('start_time', 0) or 0
                    dur = data.get('duration', 0) or 0
                    if dur <= 0:
                        continue
                    effect_names = self.image_effects[img_num][:max_effects_per_image]
                    img_fx = self.effects.get_effects_by_names(
                        effect_names, dur, self.effects_intensity
                    )
                    for fx in img_fx:
                        fx = fx.set_start(start)
                        effect_clips.append(fx)
                    per_image_handled.add(img_num)
                    logger.debug(f"  Image {img_num}: {effect_names} ({dur:.1f}s)")
            logger.info("  Total per-image effect clips: %d", len(effect_clips))

        # 2. Section-based effects for images WITHOUT per-image effects
        if self.image_sections:
            # Group remaining clips by section
            sections: Dict[str, Dict] = {}
            for data in clips_data:
                img_num = data.get('image_num')
                if img_num in per_image_handled:
                    continue
                section = self.image_sections.get(img_num, 'UNKNOWN')
                start = data.get('start_time', 0) or 0
                end = data.get('end_time') or (start + data.get('duration', 0))
                if section not in sections:
                    sections[section] = {'start': start, 'end': end}
                else:
                    sections[section]['start'] = min(sections[section]['start'], start)
                    sections[section]['end'] = max(sections[section]['end'], end)

            if sections:
                logger.info(f"\nApplying section-based effects for remaining {len(sections)} sections...")
                for section, times in sections.items():
                    sec_duration = times['end'] - times['start']
                    if sec_duration <= 0:
                        continue
                    section_fx = self.effects.get_section_effects(
                        section, sec_duration, self.effects_intensity, max_effects=2
                    )
                    for fx in section_fx:
                        fx = fx.set_start(times['start'])
                        effect_clips.append(fx)
                    logger.info(f"  {section}: {times['start']:.1f}s - {times['end']:.1f}s "
                          f"({sec_duration:.1f}s) — {len(section_fx)} effects")

        elif not per_image_handled:
            # 3. No section data and no per-image effects — global defaults
            last = clips_data[-1]
            end_time = last.get('end_time')
            if end_time is not None:
                dur = end_time
            else:
                dur = (last.get('start_time') or 0) + last.get('duration', 0)
            if dur > 0:
                effect_clips.extend([
                    self.effects.create_film_grain(dur, self.effects_intensity * 0.12),
                    self.effects.create_dust_particles(dur, self.effects_intensity * 0.15),
                ])

        return effect_clips

    def _create_text_overlay_clips(self, clips_data: List[Dict]) -> List:
        """Create MoviePy clips for all overlay text entries with animation support."""
        text_clips = []
        engine = self.text_overlay_engine

        total_overlays = len(self.overlay_texts) + len(self.image_map_labels)
        logger.info(f"\nCreating {total_overlays} text/map overlays...")

        for data in clips_data:
            image_num = data.get('image_num')
            start_time = data.get('start_time')
            duration = data.get('duration', 0)

            if start_time is None or duration <= 0:
                continue

            # --- Map label overlay (pin + location/year) ---
            if image_num in self.image_map_labels:
                map_label = self.image_map_labels[image_num]
                map_clip = self._create_map_label_clip(map_label, duration)
                if map_clip:
                    map_clip = map_clip.set_start(start_time)
                    text_clips.append(map_clip)
                    logger.debug(f"  Image {image_num}: [MAP_LABEL] \"{map_label}\" "
                          f"at {start_time:.1f}s ({duration:.1f}s)")

            # --- Text overlay ---
            if image_num not in self.overlay_texts:
                continue

            overlay_text = self.overlay_texts[image_num]
            text_anim = self.image_text_animations.get(image_num, 'fade_in')

            overlay_rgba = engine.render_overlay_text(overlay_text)
            rgb_frame = overlay_rgba[:, :, :3]
            alpha_mask = overlay_rgba[:, :, 3].astype(np.float64) / 255.0

            # Apply text animation style
            clip = self._create_animated_text_clip(
                rgb_frame, alpha_mask, duration, text_anim
            )
            clip = clip.set_start(start_time)
            text_clips.append(clip)

            overlay_type = TextOverlayEngine.classify_overlay_text(overlay_text)
            logger.debug(f"  Image {image_num}: [{overlay_type}] [{text_anim}] "
                  f"\"{overlay_text[:50]}...\" at {start_time:.1f}s ({duration:.1f}s)")

        return text_clips

    def _create_animated_text_clip(
        self, rgb_frame, alpha_mask, duration: float, animation: str
    ) -> VideoClip:
        """Create a text overlay clip with the specified animation style."""
        h, w = rgb_frame.shape[:2]

        if animation == 'bounce':
            # Text bounces in from below with overshoot
            def make_frame(t, _rgb=rgb_frame):
                return _rgb

            def make_mask(t, _alpha=alpha_mask):
                progress = min(t / max(duration * 0.3, 0.3), 1.0)
                # Bounce overshoot curve
                if progress < 0.6:
                    p = progress / 0.6
                    offset = 1.0 - p * p
                elif progress < 0.8:
                    p = (progress - 0.6) / 0.2
                    offset = -0.05 * np.sin(p * np.pi)
                else:
                    offset = 0.0
                shift = int(h * 0.15 * offset)
                if shift > 0 and shift < h:
                    result = np.zeros_like(_alpha)
                    result[:-shift] = _alpha[shift:]
                    return result
                fade_out_start = duration * 0.85
                fade = 1.0
                if t > fade_out_start:
                    fade = max(0, 1.0 - (t - fade_out_start) / (duration * 0.15))
                return _alpha * fade

            clip = VideoClip(make_frame, duration=duration).set_fps(self.fps)
            mask_clip = VideoClip(make_mask, duration=duration, ismask=True).set_fps(self.fps)
            clip = clip.set_mask(mask_clip)

        elif animation == 'typewriter':
            # Text appears column by column (left to right)
            def make_frame(t, _rgb=rgb_frame):
                return _rgb

            def make_mask(t, _alpha=alpha_mask):
                reveal_dur = min(duration * 0.6, 2.0)
                progress = min(t / reveal_dur, 1.0) if reveal_dur > 0 else 1.0
                col_cutoff = int(w * progress)
                result = np.zeros_like(_alpha)
                if col_cutoff > 0:
                    result[:, :col_cutoff] = _alpha[:, :col_cutoff]
                # Fade out at end
                fade_out_start = duration * 0.85
                fade = 1.0
                if t > fade_out_start:
                    fade = max(0, 1.0 - (t - fade_out_start) / (duration * 0.15))
                return result * fade

            clip = VideoClip(make_frame, duration=duration).set_fps(self.fps)
            mask_clip = VideoClip(make_mask, duration=duration, ismask=True).set_fps(self.fps)
            clip = clip.set_mask(mask_clip)

        elif animation == 'highlight':
            # Word-by-word highlight effect (progressive reveal with colored background)
            def make_frame(t, _rgb=rgb_frame):
                reveal_dur = min(duration * 0.7, 3.0)
                progress = min(t / reveal_dur, 1.0) if reveal_dur > 0 else 1.0
                col_cutoff = int(w * progress)
                result = _rgb.copy()
                # Highlight band: warm golden overlay on revealed portion
                if col_cutoff > 0:
                    highlight = np.zeros_like(result[:, :col_cutoff])
                    highlight[:, :, 0] = 40  # slight warm tint
                    highlight[:, :, 1] = 30
                    highlight[:, :, 2] = 10
                    result[:, :col_cutoff] = np.clip(
                        result[:, :col_cutoff].astype(np.int16) + highlight.astype(np.int16),
                        0, 255
                    ).astype(np.uint8)
                return result

            def make_mask(t, _alpha=alpha_mask):
                fade_out_start = duration * 0.85
                fade = 1.0
                if t > fade_out_start:
                    fade = max(0, 1.0 - (t - fade_out_start) / (duration * 0.15))
                fade_in = min(t / 0.3, 1.0)
                return _alpha * fade * fade_in

            clip = VideoClip(make_frame, duration=duration).set_fps(self.fps)
            mask_clip = VideoClip(make_mask, duration=duration, ismask=True).set_fps(self.fps)
            clip = clip.set_mask(mask_clip)

        elif animation == 'slide_up':
            # Text slides up from below
            def make_frame(t, _rgb=rgb_frame):
                return _rgb

            def make_mask(t, _alpha=alpha_mask):
                slide_dur = min(duration * 0.25, 0.8)
                progress = min(t / slide_dur, 1.0) if slide_dur > 0 else 1.0
                shift = int(h * 0.1 * (1.0 - progress))
                result = np.zeros_like(_alpha)
                if shift > 0 and shift < h:
                    result[:-shift] = _alpha[shift:]
                else:
                    result = _alpha.copy()
                result *= progress  # fade in with slide
                fade_out_start = duration * 0.85
                if t > fade_out_start:
                    fade = max(0, 1.0 - (t - fade_out_start) / (duration * 0.15))
                    result *= fade
                return result

            clip = VideoClip(make_frame, duration=duration).set_fps(self.fps)
            mask_clip = VideoClip(make_mask, duration=duration, ismask=True).set_fps(self.fps)
            clip = clip.set_mask(mask_clip)

        elif animation == 'glow_underline':
            # Text appears with a glowing golden underline that sweeps left-to-right
            glow_h = max(4, int(h * 0.025))

            def make_frame(t, _rgb=rgb_frame):
                result = _rgb.copy()
                sweep_dur = min(duration * 0.5, 1.5)
                progress = min(t / sweep_dur, 1.0) if sweep_dur > 0 else 1.0
                col = int(w * progress)
                if col <= 0:
                    return result
                # Find bottom of text content
                text_rows = np.where(_rgb.max(axis=-1).max(axis=1) > 0)[0]
                if len(text_rows) == 0:
                    bot = int(h * 0.85)
                else:
                    bot = min(int(text_rows[-1]) + 4, h - glow_h - 2)
                # Draw golden glow line
                for dy in range(glow_h):
                    y = bot + dy
                    if 0 <= y < h:
                        fade = 1.0 - abs(dy - glow_h / 2) / (glow_h / 2)
                        result[y, :col, 0] = min(255, int(result[y, :col, 0].mean() + 200 * fade))
                        result[y, :col, 1] = min(255, int(result[y, :col, 1].mean() + 160 * fade))
                        result[y, :col, 2] = min(255, int(result[y, :col, 2].mean() + 40 * fade))
                return result

            def make_mask(t, _alpha=alpha_mask):
                fade_in = min(t / 0.3, 1.0)
                fade_out_start = duration * 0.85
                fade = 1.0
                if t > fade_out_start:
                    fade = max(0, 1.0 - (t - fade_out_start) / (duration * 0.15))
                return _alpha * fade_in * fade

            clip = VideoClip(make_frame, duration=duration).set_fps(self.fps)
            mask_clip = VideoClip(make_mask, duration=duration, ismask=True).set_fps(self.fps)
            clip = clip.set_mask(mask_clip)

        elif animation == 'scale_impact':
            # Text scales up from small to full size with a punch for key facts/numbers
            def make_frame(t, _rgb=rgb_frame):
                return _rgb

            def make_mask(t, _alpha=alpha_mask):
                scale_dur = min(duration * 0.3, 0.6)
                if t < scale_dur and scale_dur > 0:
                    progress = t / scale_dur
                    # Back ease: overshoot then settle
                    p = progress - 1.0
                    scale = 1.0 + p * p * (2.7 * p + 1.7)
                    # Scale the alpha mask from center
                    if scale < 1.0 and scale > 0.1:
                        sh, sw = _alpha.shape
                        cy, cx = sh // 2, sw // 2
                        scaled_h = int(sh * scale)
                        scaled_w = int(sw * scale)
                        if scaled_h > 0 and scaled_w > 0:
                            result = np.zeros_like(_alpha)
                            # Simple center crop of the alpha
                            top = max(0, cy - scaled_h // 2)
                            left = max(0, cx - scaled_w // 2)
                            result[top:top + min(scaled_h, sh), left:left + min(scaled_w, sw)] = \
                                _alpha[top:top + min(scaled_h, sh), left:left + min(scaled_w, sw)]
                            return result * progress
                    return _alpha * progress
                # Sustain
                fade_out_start = duration * 0.85
                fade = 1.0
                if t > fade_out_start:
                    fade = max(0, 1.0 - (t - fade_out_start) / (duration * 0.15))
                return _alpha * fade

            clip = VideoClip(make_frame, duration=duration).set_fps(self.fps)
            mask_clip = VideoClip(make_mask, duration=duration, ismask=True).set_fps(self.fps)
            clip = clip.set_mask(mask_clip)

        elif animation == 'wave':
            # Text with a sinusoidal wave distortion that settles
            def make_frame(t, _rgb=rgb_frame):
                return _rgb

            def make_mask(t, _alpha=alpha_mask):
                settle_dur = min(duration * 0.5, 2.0)
                progress = min(t / settle_dur, 1.0) if settle_dur > 0 else 1.0
                # Decreasing wave amplitude as it settles
                amplitude = int(h * 0.03 * (1.0 - progress))
                if amplitude <= 0:
                    fade_out_start = duration * 0.85
                    fade = 1.0
                    if t > fade_out_start:
                        fade = max(0, 1.0 - (t - fade_out_start) / (duration * 0.15))
                    return _alpha * fade
                # Apply wave distortion to alpha mask
                result = np.zeros_like(_alpha)
                for row in range(_alpha.shape[0]):
                    offset = int(amplitude * np.sin(row * 0.05 + t * 4))
                    src_row = row + offset
                    if 0 <= src_row < _alpha.shape[0]:
                        result[row] = _alpha[src_row]
                fade_in = min(t / 0.3, 1.0)
                return result * fade_in

            clip = VideoClip(make_frame, duration=duration).set_fps(self.fps)
            mask_clip = VideoClip(make_mask, duration=duration, ismask=True).set_fps(self.fps)
            clip = clip.set_mask(mask_clip)

        else:
            # Default: smooth fade_in / fade_out via alpha mask
            _fade_in_dur = min(0.5, duration * 0.15)
            _fade_out_start = duration * 0.85

            def make_frame(_t, _rgb=rgb_frame):
                return _rgb

            def make_mask(t, _alpha=alpha_mask):
                opacity = 1.0
                if t < _fade_in_dur and _fade_in_dur > 0:
                    opacity = t / _fade_in_dur
                if t > _fade_out_start:
                    out_dur = duration - _fade_out_start
                    if out_dur > 0:
                        opacity *= max(0.0, 1.0 - (t - _fade_out_start) / out_dur)
                return _alpha * opacity

            clip = VideoClip(make_frame, duration=duration).set_fps(self.fps)
            mask_clip = VideoClip(make_mask, duration=duration, ismask=True).set_fps(self.fps)
            clip = clip.set_mask(mask_clip)

        return clip

    @staticmethod
    def _load_font(size: int, bold: bool = False):
        """Load a TrueType font with robust cross-platform fallbacks."""
        from PIL import ImageFont
        paths = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
            "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf" if bold else "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
            "/usr/share/fonts/truetype/ubuntu/Ubuntu-B.ttf" if bold else "/usr/share/fonts/truetype/ubuntu/Ubuntu-R.ttf",
            "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/TTF/DejaVuSans.ttf",
            "/System/Library/Fonts/Helvetica.ttc",
            "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        ]
        for path in paths:
            try:
                return ImageFont.truetype(path, size)
            except (OSError, IOError):
                continue
        try:
            return ImageFont.load_default(size=size)
        except TypeError:
            return ImageFont.load_default()

    def _create_map_label_clip(self, map_label: str, duration: float):
        """Create a map label overlay with location pin + text.

        map_label format: "Location Name | Year"
        Renders: (1) glowing location pin marker, (2) location text, (3) year text
        positioned in the lower-third area for readability.
        """
        try:
            from PIL import Image, ImageDraw, ImageFont
        except ImportError:
            logger.warning("PIL not available, skipping map label")
            return None

        w, h = self.resolution
        parts = map_label.split('|')
        location = parts[0].strip() if len(parts) > 0 else ''
        year = parts[1].strip() if len(parts) > 1 else ''

        img = Image.new('RGBA', (w, h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)

        font_size_loc = int(h * 0.038)
        font_size_year = int(h * 0.048)
        font_loc = self._load_font(font_size_loc, bold=False)
        font_year = self._load_font(font_size_year, bold=True)

        pin_x = int(w * 0.5)
        pin_y = int(h * 0.58)

        pin_glow_r = int(h * 0.028)
        pin_r = int(h * 0.018)
        for offset in range(pin_glow_r, pin_r, -1):
            alpha = int(90 * (1 - (offset - pin_r) / (pin_glow_r - pin_r)))
            draw.ellipse(
                [pin_x - offset, pin_y - offset, pin_x + offset, pin_y + offset],
                fill=(255, 180, 50, alpha)
            )
        draw.ellipse(
            [pin_x - pin_r, pin_y - pin_r, pin_x + pin_r, pin_y + pin_r],
            fill=(255, 80, 30, 240)
        )
        stem_len = int(h * 0.022)
        draw.line(
            [(pin_x, pin_y + pin_r), (pin_x, pin_y + pin_r + stem_len)],
            fill=(255, 80, 30, 200), width=max(2, int(pin_r * 0.4))
        )

        text_y = pin_y + pin_r + stem_len + int(h * 0.015)
        pad = int(h * 0.012)

        loc_bbox = draw.textbbox((0, 0), location, font=font_loc) if location else (0, 0, 0, 0)
        year_bbox = draw.textbbox((0, 0), year, font=font_year) if year else (0, 0, 0, 0)
        loc_w = loc_bbox[2] - loc_bbox[0]
        year_w = year_bbox[2] - year_bbox[0]
        loc_h = loc_bbox[3] - loc_bbox[1]
        year_h = year_bbox[3] - year_bbox[1]
        max_text_w = max(loc_w, year_w)
        total_text_h = (loc_h + int(h * 0.012) + year_h) if year else loc_h

        if location or year:
            panel_x1 = pin_x - max_text_w // 2 - pad * 2
            panel_y1 = text_y - pad
            panel_x2 = pin_x + max_text_w // 2 + pad * 2
            panel_y2 = text_y + total_text_h + pad * 2
            panel_x1 = max(0, panel_x1)
            panel_x2 = min(w, panel_x2)
            draw.rounded_rectangle(
                [panel_x1, panel_y1, panel_x2, panel_y2],
                radius=int(h * 0.008),
                fill=(0, 0, 0, 180)
            )

        if location:
            text_x = max(pad, pin_x - loc_w // 2)
            draw.text((text_x, text_y), location, fill=(255, 255, 255, 245), font=font_loc)
            text_y += loc_h + int(h * 0.012)

        if year:
            text_x = max(pad, pin_x - year_w // 2)
            draw.text((text_x, text_y), year, fill=(255, 200, 80, 250), font=font_year)

        overlay_rgba = np.array(img)
        rgb_frame = overlay_rgba[:, :, :3]
        alpha_mask = overlay_rgba[:, :, 3].astype(np.float64) / 255.0

        def make_frame(t, _rgb=rgb_frame):
            return _rgb

        def make_mask(t, _alpha=alpha_mask):
            fade_in = min(t / 0.8, 1.0)
            fade_out_start = duration * 0.85
            fade = 1.0
            if t > fade_out_start:
                fade = max(0, 1.0 - (t - fade_out_start) / (duration * 0.15))
            return _alpha * fade_in * fade

        clip = VideoClip(make_frame, duration=duration).set_fps(self.fps)
        mask_clip = VideoClip(make_mask, duration=duration, ismask=True).set_fps(self.fps)
        clip = clip.set_mask(mask_clip)
        return clip

    @staticmethod
    def _normalize_audio(audio_clip: AudioFileClip, target_db: float = -14.0) -> AudioFileClip:
        """Normalize audio to web-video loudness with peak limiting.

        Brings mean volume to *target_db* (default -14 dB for web video)
        and applies a soft limiter so peaks never exceed -1.5 dB.
        """
        try:
            samples = audio_clip.to_soundarray(fps=44100)
        except TypeError:
            # numpy 2.x requires sequences for vstack; collect chunks manually
            chunks = list(audio_clip.iter_chunks(fps=44100, quantize=False, chunksize=2000))
            samples = np.vstack(chunks) if chunks else np.array([])
        if samples.size == 0:
            return audio_clip

        rms = np.sqrt(np.mean(samples ** 2))
        if rms < 1e-10:
            return audio_clip

        current_db = 20 * np.log10(rms + 1e-10)
        gain_db = target_db - current_db
        gain = 10 ** (gain_db / 20.0)
        samples = samples * gain

        # Soft limiter: peaks above -1.5 dB (≈0.841) are compressed
        peak_limit = 10 ** (-1.5 / 20.0)  # ~0.841
        above = np.abs(samples) > peak_limit
        if np.any(above):
            samples[above] = np.sign(samples[above]) * (
                peak_limit + (np.abs(samples[above]) - peak_limit) * 0.3
            )

        samples = np.clip(samples, -1.0, 1.0)

        from moviepy.audio.AudioClip import AudioArrayClip
        return AudioArrayClip(samples, fps=44100).set_duration(audio_clip.duration)

    def _export_video(self, video: CompositeVideoClip) -> None:
        """Export the final video with professional settings.

        Encodes with H.264 High profile, BT.709 color metadata, and
        progressive-download flag.  Uses CRF-based encoding with VBV
        rate-control to prevent bitrate spikes, and `-tune film` for
        cinematic content with pan/zoom and transitions.
        """
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        temp_audio = str(self.output_path.with_suffix('.temp-audio.m4a'))
        thread_count = min(os.cpu_count() or 4, 8)

        if self.preview_mode:
            preset = 'veryfast'
            crf = '26'
        elif getattr(self, 'fast_mode', False):
            preset = 'veryfast'
            crf = '22'
        else:
            preset = 'medium'
            crf = '18'

        try:
            video.write_videofile(
                str(self.output_path),
                fps=self.fps,
                codec='libx264',
                audio_codec='aac',
                audio_bitrate='192k',
                temp_audiofile=temp_audio,
                remove_temp=True,
                threads=thread_count,
                preset=preset,
                ffmpeg_params=[
                    '-crf', crf,
                    '-tune', 'film',
                    '-maxrate', '8M',
                    '-bufsize', '16M',
                    '-g', '60',
                    '-sc_threshold', '40',
                    '-pix_fmt', 'yuv420p',
                    '-profile:v', 'high',
                    '-level', '4.1',
                    '-color_range', '1',
                    '-colorspace', 'bt709',
                    '-color_primaries', 'bt709',
                    '-color_trc', 'bt709',
                    '-movflags', '+faststart',
                ]
            )
        except Exception as e:
            logger.error("Export error with optimized settings: %s", e)
            logger.info("Retrying with basic settings...")
            video.write_videofile(
                str(self.output_path),
                fps=self.fps,
                codec='libx264',
                audio_codec='aac',
                audio_bitrate='192k',
                threads=thread_count,
                ffmpeg_params=[
                    '-pix_fmt', 'yuv420p',
                    '-color_range', '1',
                    '-colorspace', 'bt709',
                    '-color_primaries', 'bt709',
                    '-color_trc', 'bt709',
                    '-movflags', '+faststart',
                ]
            )

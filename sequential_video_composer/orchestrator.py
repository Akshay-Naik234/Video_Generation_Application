"""Sequential Video Orchestrator - Main orchestration class."""

import re
import json
import random
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Union

from moviepy.editor import AudioFileClip, CompositeVideoClip

from .transitions import TransitionEffects
from .movements import MovementStyles
from .color_grading import ColorGrading
from .clip_factory import ClipFactory
from .text_overlays import TextOverlayEngine
from .ai_effects import (
    DepthEstimator, ParallaxEngine, SubjectDetector,
    WeatherEffects, get_ai_status,
)
from .sound_design import SoundDesignEngine


class SequentialVideoOrchestrator:
    """Orchestrates video creation from sequentially numbered images with professional effects.
    
    Supports section-aware processing: when duration config JSON includes section metadata
    (section, emotional_tone, shot_type, color_temperature), the orchestrator automatically
    selects appropriate color grading, movement styles, and transitions for each image
    based on its narrative position in the story arc.
    """

    SUPPORTED_IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp', '.tiff'}

    # Section-to-movement mapping for narrative-aware Ken Burns effects
    SECTION_MOVEMENT_MAP = {
        'COLD_OPEN': ['dramatic_zoom', 'zoom_in', 'push_in'],
        'EARLY_LIFE': ['gentle_drift', 'breathing', 'pan_right', 'minimal', 'float_drift'],
        'THE_SPARK': ['zoom_in', 'pan_left', 'diagonal_tl_br', 'focus_center'],
        'THE_RISE': ['zoom_in', 'pan_right', 'diagonal_tl_br', 'dramatic_zoom'],
        'THE_CONFLICT': ['pan_left', 'pan_right', 'zoom_in', 'dramatic_zoom', 'push_in', 'zoom_pulse'],
        'THE_CLIMAX': ['dramatic_zoom', 'zoom_in', 'focus_center', 'push_in', 'zoom_pulse'],
        'THE_FALL': ['zoom_out', 'pull_out', 'gentle_drift', 'breathing'],
        'LEGACY': ['gentle_drift', 'breathing', 'zoom_out', 'pull_out', 'minimal', 'float_drift'],
        'CTA': ['zoom_out', 'gentle_drift', 'minimal', 'static'],
    }

    # Shot-type-to-movement preferences (override section map when shot_type is available)
    # Close-ups get subtle movement, wide shots get drift/pan, aerial gets slow pan
    SHOT_TYPE_MOVEMENT_MAP = {
        'extreme_closeup': ['minimal', 'breathing', 'static'],
        'closeup': ['zoom_in', 'breathing', 'minimal', 'focus_center'],
        'medium': ['gentle_drift', 'zoom_in', 'pan_left', 'pan_right'],
        'wide': ['gentle_drift', 'pan_right', 'pan_left', 'breathing', 'float_drift'],
        'aerial': ['pan_right', 'pan_left', 'gentle_drift'],
        'over_shoulder': ['push_in', 'zoom_in', 'focus_center'],
        'detail': ['focus_center', 'minimal', 'breathing'],
        'silhouette': ['zoom_out', 'pull_out', 'breathing', 'gentle_drift'],
        'crowd': ['pan_right', 'pan_left', 'gentle_drift', 'zoom_out'],
        'landscape': ['gentle_drift', 'pan_right', 'pan_left', 'zoom_out', 'float_drift'],
        'still_life': ['zoom_in', 'focus_center', 'minimal'],
    }

    # Section-to-color-grade mapping for narrative-aware visual tone
    SECTION_COLOR_MAP = {
        'COLD_OPEN': 'cool',
        'EARLY_LIFE': 'warm',
        'THE_SPARK': 'documentary',
        'THE_RISE': 'cinematic',
        'THE_CONFLICT': 'high_contrast',
        'THE_CLIMAX': 'dramatic',
        'THE_FALL': 'soft',
        'LEGACY': 'warm',
        'CTA': 'natural',
    }

    # Emotional tone to color grade overrides
    EMOTION_COLOR_MAP = {
        'tension': 'high_contrast',
        'nostalgia': 'vintage',
        'hope': 'warm',
        'darkness': 'dramatic',
        'devastation': 'cool',
        'triumph': 'cinematic',
        'bittersweet': 'soft',
    }

    def __init__(
        self,
        images_root: Union[str, Path],
        output_path: Union[str, Path] = "sequential_video_output.mp4",
        resolution: Tuple[int, int] = (1920, 1080),
        fps: int = 30,
        image_duration: float = 4.0,
        crossfade_duration: float = 1.2,
        zoom_intensity: float = 1.08,
        effects_intensity: float = 0.7,
        audio_path: Optional[Union[str, Path]] = None,
        transition_style: str = "random",
        movement_style: str = "random",
        color_grade: str = "cinematic",
        enable_vignette: bool = True,
        enable_film_grain: bool = False,
        enable_letterbox: bool = True,
        enable_chapter_cards: bool = True,
        enable_dynamic_pacing: bool = True,
        pattern_interrupt_interval: int = 75,
        enable_hook_overlay: bool = True,
        enable_cta_end_screen: bool = True,
        enable_brightness_normalization: bool = True,
        enable_color_continuity: bool = True,
        enable_speed_ramp: bool = True,
        audio_fade_in: float = 1.0,
        audio_fade_out: float = 2.0,
        channel_name: str = 'Subscribe',
        ai_animation_enabled: bool = True,
        enable_parallax: bool = True,
        enable_dof: bool = True,
        enable_weather: bool = True,
        enable_human_feel: bool = True,
        enable_sound_design: bool = True,
        sound_design_intensity: float = 0.08,
        enable_pytorch_depth: bool = True,
        duration_config_path: Optional[Union[str, Path]] = None
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
        self.enable_letterbox = enable_letterbox
        self.enable_chapter_cards = enable_chapter_cards
        self.enable_dynamic_pacing = enable_dynamic_pacing
        self.pattern_interrupt_interval = pattern_interrupt_interval
        self.duration_config_path = Path(duration_config_path) if duration_config_path else None
        self.image_durations: Dict[int, Dict] = {}
        self.total_video_duration: float = 0
        self._movement_history: List[str] = []  # Track recent movements to prevent repetition
        self.enable_hook_overlay: bool = enable_hook_overlay
        self.enable_cta_end_screen: bool = enable_cta_end_screen
        self.enable_brightness_normalization: bool = enable_brightness_normalization
        self.enable_color_continuity: bool = enable_color_continuity
        self.enable_speed_ramp: bool = enable_speed_ramp
        self.audio_fade_in: float = audio_fade_in
        self.audio_fade_out: float = audio_fade_out
        self.channel_name: str = channel_name
        self.ai_animation_enabled: bool = ai_animation_enabled
        self.enable_parallax: bool = enable_parallax and ai_animation_enabled
        self.enable_dof: bool = enable_dof and ai_animation_enabled
        self.enable_weather: bool = enable_weather and ai_animation_enabled
        self.enable_human_feel: bool = enable_human_feel
        self.enable_sound_design: bool = enable_sound_design
        self.sound_design_intensity: float = sound_design_intensity
        self.enable_pytorch_depth: bool = enable_pytorch_depth

        if self.duration_config_path:
            self._load_duration_config()

        self.transitions = TransitionEffects(resolution)
        self.movements = MovementStyles(resolution)
        self.color_grading = ColorGrading()
        self.clip_factory = ClipFactory(self)
        self.text_overlay_engine = TextOverlayEngine(resolution)

        # Initialize AI effects (optional - gracefully degrades if deps missing)
        self.depth_estimator: Optional[DepthEstimator] = None
        self.parallax_engine: Optional[ParallaxEngine] = None
        self.subject_detector: Optional[SubjectDetector] = None
        self.weather_effects: Optional[WeatherEffects] = None

        if self.enable_parallax or self.enable_dof:
            self.depth_estimator = DepthEstimator(use_ai=True)
            self.parallax_engine = ParallaxEngine(resolution)
            self.subject_detector = SubjectDetector()

        if self.enable_weather:
            self.weather_effects = WeatherEffects()

        # Initialize sound design engine
        self.sound_design_engine: Optional[SoundDesignEngine] = None
        if self.enable_sound_design:
            self.sound_design_engine = SoundDesignEngine()

    def _load_duration_config(self) -> None:
        """Load image durations, timing, and section metadata from JSON configuration file.
        
        Reads the full image prompt JSON which may include:
        - Basic timing: image, start_time, duration, end_time
        - Section metadata: section, emotional_tone, shot_type, color_temperature
        
        Section metadata enables automatic narrative-aware color grading, movement
        selection, and transition choices that match the emotional arc of the story.
        """
        if not self.duration_config_path or not self.duration_config_path.exists():
            print(f"Duration config file not found: {self.duration_config_path}")
            return

        try:
            with open(self.duration_config_path, 'r') as f:
                config = json.load(f)

            metadata = config.get('video_metadata', {})
            self.total_video_duration = metadata.get('total_duration_seconds', 0)
            
            images_data = config.get('images', [])
            for img_data in images_data:
                image_num = img_data.get('image')
                duration = img_data.get('duration')
                start_time = img_data.get('start_time')
                end_time = img_data.get('end_time')
                section = img_data.get('section', '')
                emotional_tone = img_data.get('emotional_tone', '')
                shot_type = img_data.get('shot_type', '')
                color_temperature = img_data.get('color_temperature', '')
                
                overlay_text = img_data.get('overlay_text', '')
                
                if image_num is not None:
                    self.image_durations[image_num] = {
                        'duration': float(duration) if duration is not None else self.image_duration,
                        'start_time': float(start_time) if start_time is not None else None,
                        'end_time': float(end_time) if end_time is not None else None,
                        'section': section,
                        'emotional_tone': emotional_tone,
                        'shot_type': shot_type,
                        'color_temperature': color_temperature,
                        'overlay_text': overlay_text,
                    }

            print(f"Loaded timing data for {len(self.image_durations)} images from config")
            if self.total_video_duration:
                print(f"Total video duration from config: {self.total_video_duration}s")
            
            # Report section metadata availability
            sections_found = sum(1 for d in self.image_durations.values() if d.get('section'))
            if sections_found > 0:
                print(f"Section metadata available for {sections_found} images (section-aware processing enabled)")
            overlays_found = sum(1 for d in self.image_durations.values() if d.get('overlay_text'))
            if overlays_found > 0:
                print(f"Overlay text available for {overlays_found} images (date/location stamps enabled)")
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Error loading duration config: {e}")

    def get_timing_for_image(self, image_number: int) -> Dict:
        """Get the full timing info for a specific image number."""
        if self.image_durations and image_number in self.image_durations:
            return self.image_durations[image_number]
        return {
            'duration': self.image_duration,
            'start_time': None,
            'end_time': None,
            'section': '',
            'emotional_tone': '',
            'shot_type': '',
            'color_temperature': '',
            'overlay_text': '',
        }

    def get_duration_for_image(self, image_number: int) -> float:
        """Get the duration for a specific image number."""
        timing = self.get_timing_for_image(image_number)
        return timing.get('duration', self.image_duration)

    def get_color_grade_for_image(self, image_number: int) -> str:
        """Get the appropriate color grade for an image based on section/emotional metadata.
        
        Priority: emotional_tone override > section mapping > global color_grade setting.
        This allows each image to have its own color treatment that matches the narrative arc.
        """
        if self.image_durations and image_number in self.image_durations:
            timing = self.image_durations[image_number]
            emotional_tone = timing.get('emotional_tone', '')
            section = timing.get('section', '')
            
            # Emotional tone takes priority for specific color adjustments
            if emotional_tone and emotional_tone in self.EMOTION_COLOR_MAP:
                return self.EMOTION_COLOR_MAP[emotional_tone]
            
            # Section-based color grading
            if section and section in self.SECTION_COLOR_MAP:
                return self.SECTION_COLOR_MAP[section]
        
        return self.color_grade

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
        print(f"Discovered {len(numbered_images)} numbered images")

        for num, path in numbered_images:
            print(f"  {num}: {path.name}")

        return numbered_images

    def _get_movement_for_image(self, index: int, total: int, image_number: int = None) -> str:
        """Get movement style for an image based on its position, section, and shot_type.
        
        Priority: shot_type + section intersection > section-only > fallback.
        When both shot_type and section are available, picks movements that appear
        in both preference lists (intersection), giving the most contextually
        appropriate movement. Falls back to section map if no intersection.
        
        Includes movement repetition prevention: avoids using the same movement
        type 3 times in a row, which top creators avoid to prevent visual monotony.
        """
        chosen = None

        if image_number is not None and self.image_durations:
            timing = self.image_durations.get(image_number, {})
            section = timing.get('section', '')
            shot_type = timing.get('shot_type', '')
            
            section_movements = self.SECTION_MOVEMENT_MAP.get(section, [])
            shot_movements = self.SHOT_TYPE_MOVEMENT_MAP.get(shot_type, [])
            
            # Best: intersection of section + shot_type preferences
            if section_movements and shot_movements:
                intersection = [m for m in section_movements if m in shot_movements]
                if intersection:
                    chosen = self._pick_non_repeating(intersection, index)
            
            # Fallback: shot_type preferences alone
            if chosen is None and shot_movements:
                chosen = self._pick_non_repeating(shot_movements, index)
            
            # Fallback: section preferences alone
            if chosen is None and section_movements:
                chosen = self._pick_non_repeating(section_movements, index)

        if chosen is None:
            if self.movement_style == "random":
                chosen = random.choice(MovementStyles.MOVEMENT_TYPES)
            elif self.movement_style == "sequential":
                movements = ['zoom_in', 'pan_left', 'zoom_out', 'pan_right', 'diagonal_tl_br', 'gentle_drift']
                chosen = movements[index % len(movements)]
            elif self.movement_style == "dramatic_sequence":
                if index == 0:
                    chosen = 'dramatic_zoom'
                elif index == total - 1:
                    chosen = 'zoom_out'
                elif index < total // 3:
                    chosen = random.choice(['zoom_in', 'pan_right', 'gentle_drift'])
                elif index < 2 * total // 3:
                    chosen = random.choice(['pan_left', 'pan_right', 'breathing'])
                else:
                    chosen = random.choice(['zoom_out', 'focus_center', 'gentle_drift'])
            elif self.movement_style == "documentary":
                subtle_movements = ['zoom_in', 'gentle_drift', 'zoom_out', 'focus_center', 'breathing', 'minimal']
                chosen = subtle_movements[index % len(subtle_movements)]
            else:
                chosen = self.movement_style if self.movement_style in MovementStyles.MOVEMENT_TYPES else 'zoom_in'

        # Track movement history for repetition prevention
        self._movement_history.append(chosen)
        if len(self._movement_history) > 3:
            self._movement_history = self._movement_history[-3:]
        return chosen

    def _pick_non_repeating(self, candidates: list, index: int) -> str:
        """Pick a movement from candidates while avoiding 3x repetition.

        If the last 2 movements are the same type, filter that type out of
        candidates before picking. This prevents visual monotony where the
        same Ken Burns movement repeats 3+ times in a row.
        """
        if len(self._movement_history) >= 2 and self._movement_history[-1] == self._movement_history[-2]:
            repeated = self._movement_history[-1]
            filtered = [m for m in candidates if m != repeated]
            if filtered:
                return filtered[index % len(filtered)]
        return candidates[index % len(candidates)]

    def _get_transition_for_image(self, index: int, total: int, image_number: int = None,
                                   prev_image_number: int = None) -> str:
        """Get transition style for an image based on its position and section metadata.
        
        Uses section boundaries to apply dramatic transitions (fade_through_black)
        at section changes, and smoother crossfades within sections.
        
        Args:
            prev_image_number: Actual image number of the previous image (handles gaps
                in numbering, e.g. images 4.png and 6.png with no 5.png).
        """
        # Section-aware transitions: use dramatic transitions at section boundaries
        if image_number is not None and self.image_durations:
            timing = self.image_durations.get(image_number, {})
            section = timing.get('section', '')
            
            # Use actual previous image number instead of assuming consecutive numbering
            prev_timing = {}
            if prev_image_number is not None:
                prev_timing = self.image_durations.get(prev_image_number, {})
            prev_section = prev_timing.get('section', '')
            
            if section and prev_section and section != prev_section:
                return 'fade_through_black'
            elif section:
                if section in ('COLD_OPEN', 'THE_CONFLICT', 'THE_CLIMAX'):
                    return random.choice(['crossfade', 'zoom_in'])
                elif section in ('EARLY_LIFE', 'LEGACY'):
                    return 'crossfade'
                else:
                    return random.choice(['crossfade', 'slide_left', 'slide_right'])

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
        print("=" * 60)
        print("Sequential Video Orchestrator (Enhanced - Retention Optimized)")
        print("=" * 60)
        print(f"Images root: {self.images_root}")
        print(f"Output path: {self.output_path}")
        print(f"Resolution: {self.resolution}")
        print(f"Image duration: {self.image_duration}s")
        print(f"Transition style: {self.transition_style}")
        print(f"Movement style: {self.movement_style}")
        print(f"Color grade: {self.color_grade}")
        if self.total_video_duration:
            print(f"Target video duration: {self.total_video_duration}s")

        # Report section-aware processing status
        sections_found = sum(1 for d in self.image_durations.values() if d.get('section'))
        if sections_found > 0:
            print(f"\nSection-aware processing: ENABLED ({sections_found} images with metadata)")
            print("  - Color grading: auto per section/emotion")
            print("  - Movement: narrative-matched Ken Burns (shot_type aware)")
            print("  - Transitions: section-boundary aware")
        if self.enable_human_feel:
            print(f"\nHuman-feel editing: ENABLED")
            print("  - Camera breathing + micro-shake for dramatic sections")
            print("  - Varied easing curves per section")
            print("  - Hard cuts at dramatic entrances, breathing room at emotional moments")
            print("  - Timing micro-variation for natural rhythm")
        overlays_found = sum(1 for d in self.image_durations.values() if d.get('overlay_text'))
        if overlays_found > 0:
            print(f"  - Text overlays: {overlays_found} images with date/location/name stamps")

        # Report AI effects status
        ai_status = get_ai_status()
        print(f"\nAI Effects:")
        print(f"  Depth backend: {ai_status['depth_backend']}")
        print(f"  Parallax quality: {ai_status['parallax_quality']}")
        if self.enable_parallax:
            print(f"  2.5D Parallax: ENABLED")
        if self.enable_dof:
            print(f"  Depth-of-Field: ENABLED")
        if self.enable_weather:
            print(f"  Weather effects: ENABLED")
        print()

        numbered_images = self.discover_numbered_images()
        clips_data = self.create_image_clips(numbered_images)

        if not clips_data:
            raise ValueError("No valid image clips were created")

        main_video = self.create_timeline_video(clips_data)

        if main_video is None:
            raise ValueError("Failed to create timeline video")

        print(f"Main video duration: {main_video.duration}s")

        overlays = [main_video]

        # Add cinematic letterbox bars for dramatic sections
        if self.enable_letterbox:
            letterbox_sections = self._get_letterbox_ranges(clips_data)
            for start, duration in letterbox_sections:
                letterbox = self.clip_factory.create_letterbox_overlay(duration)
                letterbox = letterbox.set_start(start).crossfadein(0.5).crossfadeout(0.5)
                overlays.append(letterbox)
                print(f"  Letterbox bars: {start:.1f}s - {start + duration:.1f}s")

        # Add text overlay chapter cards at section boundaries
        if self.enable_chapter_cards:
            chapter_overlays = self._create_chapter_card_overlays(clips_data)
            overlays.extend(chapter_overlays)

        # Add date/location/name text overlays from config metadata
        text_overlays = self._create_text_overlay_clips(clips_data)
        overlays.extend(text_overlays)

        # Add progress indicator overlays at section transitions
        progress_overlays = self._create_progress_indicator_overlays(clips_data)
        overlays.extend(progress_overlays)

        # Add light leak overlays at section transitions
        light_leak_overlays = self._create_light_leak_overlays(clips_data)
        overlays.extend(light_leak_overlays)

        # Add flash transitions at high-impact section entries (Dhruv Rathee style)
        flash_overlays = self._create_flash_transition_overlays(clips_data)
        overlays.extend(flash_overlays)

        # Add weather/atmosphere overlays for emotional sections
        if self.enable_weather and self.weather_effects is not None:
            weather_overlays = self._create_weather_overlays(clips_data)
            overlays.extend(weather_overlays)

        # Add hook overlay for the first 5 seconds (most critical retention element)
        if self.enable_hook_overlay:
            hook_overlays = self._create_hook_overlay(clips_data)
            overlays.extend(hook_overlays)

        # Add CTA end screen overlay for the last 15 seconds
        if self.enable_cta_end_screen:
            cta_overlays = self._create_cta_end_screen_overlay(main_video.duration)
            overlays.extend(cta_overlays)

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

        if len(overlays) > 1:
            main_video = CompositeVideoClip(overlays)

        if self.audio_path and self.audio_path.exists():
            print(f"Adding audio from: {self.audio_path}")
            audio_clip = AudioFileClip(str(self.audio_path))
            if audio_clip.duration > main_video.duration:
                audio_clip = audio_clip.subclip(0, main_video.duration)
            # Professional audio fade in/out
            if self.audio_fade_in > 0:
                audio_clip = audio_clip.audio_fadein(self.audio_fade_in)
            if self.audio_fade_out > 0:
                audio_clip = audio_clip.audio_fadeout(self.audio_fade_out)
                print(f"  Audio fades: {self.audio_fade_in}s in, {self.audio_fade_out}s out")

            # Mix programmatic sound effects into audio at section transitions
            if self.enable_sound_design and self.sound_design_engine is not None:
                audio_clip = self._mix_sound_effects(audio_clip, clips_data)

            main_video = main_video.set_audio(audio_clip)

        self._export_video(main_video)
        print(f"Video created successfully: {self.output_path}")

    def _mix_sound_effects(self, audio_clip, clips_data: List[Dict]):
        """Mix programmatic sound effects into the narration audio at section transitions.

        Generates whooshes, risers, bass drops, and ambient pads at section boundaries,
        then mixes them under the narration at low volume for cinematic feel.

        Args:
            audio_clip: MoviePy AudioFileClip with narration.
            clips_data: List of clip data dicts with section and timing info.

        Returns:
            New AudioFileClip with sound effects mixed in.
        """
        import tempfile
        import os

        try:
            # Extract audio as numpy array
            audio_fps = audio_clip.fps or 44100
            audio_array = audio_clip.to_soundarray(fps=audio_fps)

            # Detect section boundaries (where section changes)
            effects_to_mix = []
            prev_section = None
            for data in clips_data:
                section = data.get('section', '')
                start_time = data.get('start_time')
                if start_time is None or not section:
                    continue

                if section != prev_section:
                    # Section transition — get appropriate sound effects
                    transition_duration = data.get('crossfade_duration', 0.8)
                    section_effects = self.sound_design_engine.get_effects_for_section(
                        section, transition_duration
                    )
                    for effect_audio, volume in section_effects:
                        effects_to_mix.append((start_time, effect_audio, volume))
                    prev_section = section

            if not effects_to_mix:
                print("  Sound design: no section transitions found, skipping")
                return audio_clip

            # Convert stereo/mono to consistent format
            is_stereo = audio_array.ndim == 2 and audio_array.shape[1] >= 2

            # Mix effects into audio array
            mixed = self.sound_design_engine.mix_effects_into_audio(
                narration=audio_array,
                effects=effects_to_mix,
                master_volume=self.sound_design_intensity,
            )

            # Write mixed audio to temp file and reload as AudioFileClip
            temp_fd, temp_path = tempfile.mkstemp(suffix='.wav')
            os.close(temp_fd)
            try:
                from moviepy.audio.AudioClip import AudioArrayClip
                mixed_clip = AudioArrayClip(mixed, fps=audio_fps)
                mixed_clip = mixed_clip.set_duration(audio_clip.duration)
                print(f"  Sound design: mixed {len(effects_to_mix)} effects at section transitions")
                print(f"    Master volume: {self.sound_design_intensity} ({20 * math.log10(max(self.sound_design_intensity, 0.001)):.0f} dB)")
                return mixed_clip
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)

        except Exception as e:
            print(f"  Sound design: error mixing effects ({e}), using original audio")
            return audio_clip

    def _get_letterbox_ranges(self, clips_data: List[Dict]) -> List[Tuple[float, float]]:
        """Find time ranges where cinematic letterbox bars should appear.
        
        Returns (start_time, duration) pairs for sections that benefit from
        2.39:1 widescreen bars (CLIMAX, CONFLICT, FALL).
        """
        ranges = []
        current_start = None
        current_end = None

        for data in clips_data:
            section = data.get('section', '')
            start = data.get('start_time')
            duration = data.get('duration', 0)
            if start is None:
                continue

            if section in ClipFactory.LETTERBOX_SECTIONS:
                if current_start is None:
                    current_start = start
                current_end = start + duration
            else:
                if current_start is not None:
                    ranges.append((current_start, current_end - current_start))
                    current_start = None
                    current_end = None

        if current_start is not None:
            ranges.append((current_start, current_end - current_start))

        return ranges

    def _create_chapter_card_overlays(self, clips_data: List[Dict]) -> List:
        """Create text overlay clips for section transitions.
        
        Generates chapter title cards that appear briefly at the start of each
        new narrative section, giving viewers a sense of story progression.
        """
        from moviepy.video.VideoClip import VideoClip
        overlays = []
        seen_sections = set()

        for data in clips_data:
            section = data.get('section', '')
            start_time = data.get('start_time')
            if not section or section in seen_sections or start_time is None:
                continue

            seen_sections.add(section)
            card_array = self.text_overlay_engine.create_chapter_card(section)
            if card_array is None:
                continue

            # Create a 2.5 second overlay from the RGBA array.
            # Use the alpha channel as a proper mask so only the text band
            # is visible (transparent areas stay transparent, not black).
            card_duration = 2.5
            rgb_array = card_array[:, :, :3]
            alpha_mask = card_array[:, :, 3].astype(float) / 255.0

            def make_card_frame(t, rgb=rgb_array):
                return rgb

            def make_card_mask(t, a=alpha_mask):
                return a

            card_clip = VideoClip(make_card_frame, duration=card_duration)
            card_clip = card_clip.set_fps(self.fps)
            mask_clip = VideoClip(make_card_mask, duration=card_duration, ismask=True)
            mask_clip = mask_clip.set_fps(self.fps)
            card_clip = card_clip.set_mask(mask_clip)
            card_clip = (
                card_clip
                .set_start(start_time)
                .crossfadein(0.4)
                .crossfadeout(0.6)
            )
            overlays.append(card_clip)
            print(f"  Chapter card: '{TextOverlayEngine.SECTION_TITLES.get(section, '')}' at {start_time:.1f}s")

        return overlays

    def _create_text_overlay_clips(self, clips_data: List[Dict]) -> List:
        """Create text overlay clips for dates, locations, names from config metadata.
        
        Reads overlay_text from each image's config and renders the appropriate
        overlay type with slide-in animation (Shivanshu-style) for lower thirds
        and info cards, or fade for year/location stamps.
        
        overlay_text format examples:
          "1884"                    → year stamp (top-right)
          "New York City"           → location stamp (bottom-left)  
          "1943 | New York"         → date-location combo (bottom-right)
          "Nikola Tesla — Inventor" → lower third with slide-in (bottom-left)
          "$2.3 Million"            → animated number counter (center)
          "Born: July 10, 1856"     → info card (bottom-right)
        """
        import re as _re
        from moviepy.video.VideoClip import VideoClip
        overlays = []

        for data in clips_data:
            image_num = data.get('image_num')
            start_time = data.get('start_time')
            duration = data.get('duration', 0)
            if start_time is None or image_num is None:
                continue

            timing = self.image_durations.get(image_num, {})
            overlay_text = timing.get('overlay_text', '')
            if not overlay_text:
                continue

            text = overlay_text.strip()
            overlay_duration = min(3.0, duration * 0.6)
            overlay_start = start_time + 0.5

            # Pattern: "$NUMBER" or "NUMBER+" → animated counter
            # Requires $ prefix, comma in number, or + suffix to avoid matching bare years like "1943"
            counter_match = _re.match(r'^(\$)([\d,]+)\+?\s*(.*)$', text)
            if not counter_match:
                counter_match = _re.match(r'^()([\d,]+)\+\s*(.*)$', text)  # "50000+"
            if not counter_match:
                counter_match = _re.match(r'^()([\d]+,[\d,]+)\s*(.*)$', text)  # "2,300,000"
            if counter_match and len(counter_match.group(2).replace(',', '')) >= 4:
                prefix = counter_match.group(1)
                target = int(counter_match.group(2).replace(',', ''))
                suffix = counter_match.group(3)
                fps_for_counter = 15
                num_frames = int(overlay_duration * fps_for_counter)
                if num_frames > 2:
                    counter_frames = self.text_overlay_engine.create_animated_counter_frames(
                        target_number=target, prefix=prefix, suffix=(' ' + suffix) if suffix else '',
                        num_frames=num_frames,
                    )
                    if counter_frames:
                        def make_counter_frame(t, frames=counter_frames, fps=fps_for_counter):
                            idx = min(int(t * fps), len(frames) - 1)
                            return frames[idx][:, :, :3]

                        def make_counter_mask(t, frames=counter_frames, fps=fps_for_counter):
                            idx = min(int(t * fps), len(frames) - 1)
                            return frames[idx][:, :, 3].astype(float) / 255.0

                        counter_clip = VideoClip(make_counter_frame, duration=overlay_duration)
                        counter_clip = counter_clip.set_fps(fps_for_counter)
                        mask_clip = VideoClip(make_counter_mask, duration=overlay_duration, ismask=True)
                        mask_clip = mask_clip.set_fps(fps_for_counter)
                        counter_clip = counter_clip.set_mask(mask_clip)
                        counter_clip = (
                            counter_clip
                            .set_start(overlay_start)
                            .crossfadein(0.2)
                            .crossfadeout(0.4)
                        )
                        overlays.append(counter_clip)
                        print(f"  Animated counter: '{text}' at {overlay_start:.1f}s (image {image_num})")
                        continue

            # Pattern: "NAME — TITLE" → slide-in lower third
            if '—' in text or ' - ' in text:
                sep = '—' if '—' in text else ' - '
                parts = [p.strip() for p in text.split(sep, 1)]
                # Use slide-in animation
                slide_fps = 15
                num_frames = int(overlay_duration * slide_fps)
                if num_frames > 2:
                    full_text = f"{parts[0]} — {parts[1]}" if len(parts) > 1 else parts[0]

                    # Cache to avoid rendering the same overlay twice per frame
                    # (once for RGB, once for mask). Store as {t_key: (rgb, alpha)}.
                    _slide_cache = {}

                    def _get_slide_frame(t, txt=full_text, eng=self.text_overlay_engine, _cache=_slide_cache):
                        # Quantize t to frame boundaries to ensure cache hits
                        t_key = round(t, 4)
                        if t_key not in _cache:
                            slide_in_time = 0.4
                            if t < slide_in_time:
                                progress = t / slide_in_time
                                eased = 1.0 - (1.0 - progress) ** 3  # ease-out cubic
                            else:
                                eased = 1.0
                            frame = eng.create_slide_in_overlay(
                                txt, position='bottom_left', slide_from='left', progress=eased
                            )
                            _cache[t_key] = (frame[:, :, :3], frame[:, :, 3].astype(float) / 255.0)
                            # Keep cache small: only store last 2 frames
                            if len(_cache) > 2:
                                oldest = min(_cache.keys())
                                del _cache[oldest]
                        return _cache[t_key]

                    def make_slide_frame(t, _get=_get_slide_frame):
                        return _get(t)[0]

                    def make_slide_mask(t, _get=_get_slide_frame):
                        return _get(t)[1]

                    slide_clip = VideoClip(make_slide_frame, duration=overlay_duration)
                    slide_clip = slide_clip.set_fps(slide_fps)
                    mask_clip = VideoClip(make_slide_mask, duration=overlay_duration, ismask=True)
                    mask_clip = mask_clip.set_fps(slide_fps)
                    slide_clip = slide_clip.set_mask(mask_clip)
                    slide_clip = (
                        slide_clip
                        .set_start(overlay_start)
                        .crossfadeout(0.4)
                    )
                    overlays.append(slide_clip)
                    print(f"  Slide-in overlay: '{text}' at {overlay_start:.1f}s (image {image_num})")
                    continue

            # Static overlay types (year stamp, location stamp, etc.)
            overlay_array = None

            # Pattern: "YEAR | LOCATION" → date-location combo
            if '|' in text:
                parts = [p.strip() for p in text.split('|', 1)]
                overlay_array = self.text_overlay_engine.create_date_location_stamp(
                    date_text=parts[0], location_text=parts[1]
                )
            # Pattern: pure year (4 digits) → year stamp
            elif _re.match(r'^\d{4}$', text):
                overlay_array = self.text_overlay_engine.create_year_stamp(year=text)
            # Pattern: "YEAR, LOCATION" → year stamp with label
            elif _re.match(r'^\d{4},\s*.+$', text):
                parts = text.split(',', 1)
                overlay_array = self.text_overlay_engine.create_year_stamp(
                    year=parts[0].strip(), label=parts[1].strip()
                )
            # Pattern: starts with location-like keywords → location stamp
            elif any(text.lower().startswith(k) for k in ('new ', 'los ', 'san ', 'st.', 'mount', 'lake', 'fort',
                                                           'london', 'paris', 'chicago', 'washington', 'boston',
                                                           'mumbai', 'delhi', 'tokyo', 'berlin', 'rome')):
                overlay_array = self.text_overlay_engine.create_location_stamp(location=text)
            # Pattern: quoted text → quote card (auto-detection)
            # Matches "text in quotes" or text starting with opening quote marks
            elif (_re.match(r'^["\u201C\u201D\u2018\u2019]', text) or
                  _re.search(r'["\u201C\u201D]\s*[-\u2014]\s*\w', text)):
                # Strip outer quotes if present
                clean_quote = _re.sub(r'^["\u201C\u201D\u2018\u2019]+|["\u201C\u201D\u2018\u2019]+$', '', text).strip()
                # Try to extract attribution after dash/em-dash
                attr_match = _re.split(r'\s*[-\u2014]\s*(?=[A-Z])', clean_quote, maxsplit=1)
                if len(attr_match) == 2:
                    overlay_array = self.text_overlay_engine.create_quote_card(
                        quote=attr_match[0].strip(), attribution=attr_match[1].strip()
                    )
                else:
                    overlay_array = self.text_overlay_engine.create_quote_card(quote=clean_quote)
            # Default: info card
            else:
                overlay_array = self.text_overlay_engine.create_info_card(text=text)

            if overlay_array is None:
                continue

            rgb_array = overlay_array[:, :, :3]
            alpha_mask = overlay_array[:, :, 3].astype(float) / 255.0

            def make_overlay_frame(t, rgb=rgb_array):
                return rgb

            def make_overlay_mask(t, a=alpha_mask):
                return a

            overlay_clip = VideoClip(make_overlay_frame, duration=overlay_duration)
            overlay_clip = overlay_clip.set_fps(self.fps)
            mask_clip = VideoClip(make_overlay_mask, duration=overlay_duration, ismask=True)
            mask_clip = mask_clip.set_fps(self.fps)
            overlay_clip = overlay_clip.set_mask(mask_clip)

            overlay_clip = (
                overlay_clip
                .set_start(overlay_start)
                .crossfadein(0.3)
                .crossfadeout(0.4)
            )
            overlays.append(overlay_clip)
            print(f"  Text overlay: '{text}' at {overlay_start:.1f}s (image {image_num})")

        return overlays

    def _create_progress_indicator_overlays(self, clips_data: List[Dict]) -> List:
        """Create progress indicator overlays that show story position at section transitions.
        
        Renders a minimal timeline dot indicator at the top of the frame whenever
        a new section begins, showing the viewer where they are in the biography arc.
        """
        from moviepy.video.VideoClip import VideoClip
        overlays = []
        seen_sections = set()

        # Collect ordered unique sections from clips
        all_sections = []
        for data in clips_data:
            section = data.get('section', '')
            if section and section not in seen_sections:
                all_sections.append(section)
                seen_sections.add(section)

        if len(all_sections) < 3:
            return overlays  # Not enough sections for a meaningful indicator

        seen_sections_for_overlay = set()
        for data in clips_data:
            section = data.get('section', '')
            start_time = data.get('start_time')
            if not section or section in seen_sections_for_overlay or start_time is None:
                continue

            seen_sections_for_overlay.add(section)

            indicator_array = self.text_overlay_engine.create_progress_indicator(
                sections=all_sections,
                current_section=section,
            )
            if indicator_array is None:
                continue

            # Check if the indicator has any visible content
            alpha_channel = indicator_array[:, :, 3]
            if alpha_channel.max() == 0:
                continue

            indicator_duration = 3.0
            rgb_array = indicator_array[:, :, :3]
            alpha_mask = indicator_array[:, :, 3].astype(float) / 255.0

            def make_frame(t, rgb=rgb_array):
                return rgb

            def make_mask(t, a=alpha_mask):
                return a

            ind_clip = VideoClip(make_frame, duration=indicator_duration)
            ind_clip = ind_clip.set_fps(self.fps)
            mask_clip = VideoClip(make_mask, duration=indicator_duration, ismask=True)
            mask_clip = mask_clip.set_fps(self.fps)
            ind_clip = ind_clip.set_mask(mask_clip)
            ind_clip = (
                ind_clip
                .set_start(start_time + 0.3)
                .crossfadein(0.4)
                .crossfadeout(0.6)
            )
            overlays.append(ind_clip)

        if overlays:
            print(f"  Progress indicators: {len(overlays)} section markers")

        return overlays

    def _create_light_leak_overlays(self, clips_data: List[Dict]) -> List:
        """Create warm light leak overlays at section transitions.
        
        Adds a brief warm golden flash at section boundaries to create a cinematic
        transition effect. Uses a mask clip so the base video is only slightly
        affected (max 12% dimming at peak), avoiding the full-frame darkening bug
        that occurs with set_opacity on RGB clips.
        """
        from moviepy.video.VideoClip import VideoClip
        import numpy as np
        overlays = []
        prev_section = ''

        for data in clips_data:
            section = data.get('section', '')
            start_time = data.get('start_time')
            if not section or start_time is None or start_time <= 0:
                continue

            if prev_section and section != prev_section:
                leak_duration = 1.2
                width, height = self.resolution

                def make_leak_frame(t, w=width, h=height):
                    # Full-brightness warm golden frame (RGB only).
                    # The mask controls how much of this shows through.
                    frame = np.zeros((h, w, 3), dtype=np.uint8)
                    frame[:, :, 0] = 255  # R
                    frame[:, :, 1] = 200  # G
                    frame[:, :, 2] = 100  # B
                    return frame

                def make_leak_mask(t, dur=leak_duration, w=width, h=height):
                    # Bell-curve mask: peaks at mid-duration, max opacity 0.12
                    # so the base video is dimmed by at most ~12% at peak.
                    progress = t / dur if dur > 0 else 0
                    intensity = max(0.0, 1.0 - (2.0 * progress - 1.0) ** 2)
                    return np.full((h, w), intensity * 0.12, dtype=np.float64)

                leak_clip = VideoClip(make_leak_frame, duration=leak_duration)
                leak_clip = leak_clip.set_fps(self.fps)
                mask_clip = VideoClip(make_leak_mask, duration=leak_duration, ismask=True)
                mask_clip = mask_clip.set_fps(self.fps)
                leak_clip = leak_clip.set_mask(mask_clip)

                leak_start = max(0, start_time - 0.6)
                leak_clip = leak_clip.set_start(leak_start)
                overlays.append(leak_clip)

            prev_section = section

        if overlays:
            print(f"  Light leaks: {len(overlays)} section transition effects")

        return overlays

    def _create_flash_transition_overlays(self, clips_data: List[Dict]) -> List:
        """Create brief white flash overlays at high-impact section transitions.
        
        Dhruv Rathee-style film burn: a quick white flash that punctuates
        major story beats (CLIMAX, CONFLICT entries). Uses a mask clip with
        very low peak opacity (8%) so the base video is barely affected,
        creating a subtle "camera flash" feel rather than a full white-out.
        """
        from moviepy.video.VideoClip import VideoClip
        import numpy as np
        overlays = []
        flash_sections = {'THE_CLIMAX', 'THE_CONFLICT', 'COLD_OPEN'}
        prev_section = ''

        for data in clips_data:
            section = data.get('section', '')
            start_time = data.get('start_time')
            if not section or start_time is None or start_time <= 0:
                continue

            if prev_section and section != prev_section and section in flash_sections:
                flash_duration = 0.5
                width, height = self.resolution

                def make_flash_frame(t, w=width, h=height):
                    return np.full((h, w, 3), 255, dtype=np.uint8)

                def make_flash_mask(t, dur=flash_duration, w=width, h=height):
                    # Sharp attack, quick decay — mimics camera flash
                    progress = t / dur if dur > 0 else 0
                    # Fast rise to peak at 0.2, then exponential decay
                    if progress < 0.2:
                        intensity = progress / 0.2
                    else:
                        intensity = np.exp(-4.0 * (progress - 0.2))
                    return np.full((h, w), max(0.0, intensity * 0.08), dtype=np.float64)

                flash_clip = VideoClip(make_flash_frame, duration=flash_duration)
                flash_clip = flash_clip.set_fps(self.fps)
                mask_clip = VideoClip(make_flash_mask, duration=flash_duration, ismask=True)
                mask_clip = mask_clip.set_fps(self.fps)
                flash_clip = flash_clip.set_mask(mask_clip)
                flash_clip = flash_clip.set_start(max(0, start_time - 0.15))
                overlays.append(flash_clip)

            prev_section = section

        if overlays:
            print(f"  Flash transitions: {len(overlays)} high-impact section entries")

        return overlays

    def _create_hook_overlay(self, clips_data: List[Dict]) -> List:
        """Create a dramatic hook text overlay for the first 5 seconds.

        Top creators show the most dramatic moment as text in the first few seconds
        to instantly hook viewers. If the first image has overlay_text, use that as
        the hook. Otherwise, look for the COLD_OPEN section's first overlay_text.
        Falls back gracefully if no hook text is available.
        """
        from moviepy.video.VideoClip import VideoClip
        overlays = []

        # Find hook text from COLD_OPEN section or first image
        hook_text = ''
        for data in clips_data:
            section = data.get('section', '')
            image_num = data.get('image_num')
            if image_num is not None:
                timing = self.image_durations.get(image_num, {})
                overlay_text = timing.get('overlay_text', '')
                if overlay_text and (section == 'COLD_OPEN' or not hook_text):
                    hook_text = overlay_text
                    if section == 'COLD_OPEN':
                        break

        if not hook_text:
            return overlays

        hook_array = self.text_overlay_engine.create_hook_overlay(hook_text=hook_text)
        if hook_array is None:
            return overlays

        hook_duration = 4.0
        rgb_array = hook_array[:, :, :3]
        alpha_mask = hook_array[:, :, 3].astype(float) / 255.0

        def make_hook_frame(t, rgb=rgb_array):
            return rgb

        def make_hook_mask(t, a=alpha_mask):
            return a

        hook_clip = VideoClip(make_hook_frame, duration=hook_duration)
        hook_clip = hook_clip.set_fps(self.fps)
        mask_clip = VideoClip(make_hook_mask, duration=hook_duration, ismask=True)
        mask_clip = mask_clip.set_fps(self.fps)
        hook_clip = hook_clip.set_mask(mask_clip)
        hook_clip = (
            hook_clip
            .set_start(0.5)
            .crossfadein(0.3)
            .crossfadeout(0.8)
        )
        overlays.append(hook_clip)
        print(f"  Hook overlay: '{hook_text[:50]}...' at 0.5s-4.5s")

        return overlays

    def _create_cta_end_screen_overlay(self, video_duration: float) -> List:
        """Create a CTA end screen overlay for the last 15 seconds.

        Top creators use a clear call-to-action in the final seconds to drive
        subscriptions and engagement. This renders a subscribe button area with
        space for YouTube's built-in end screen elements.
        """
        from moviepy.video.VideoClip import VideoClip
        overlays = []

        if video_duration < 20:
            return overlays  # Video too short for end screen

        cta_array = self.text_overlay_engine.create_cta_end_screen(
            channel_name=self.channel_name
        )
        if cta_array is None:
            return overlays

        cta_duration = 15.0
        cta_start = max(0, video_duration - cta_duration)
        rgb_array = cta_array[:, :, :3]
        alpha_mask = cta_array[:, :, 3].astype(float) / 255.0

        def make_cta_frame(t, rgb=rgb_array):
            return rgb

        def make_cta_mask(t, a=alpha_mask):
            return a

        cta_clip = VideoClip(make_cta_frame, duration=cta_duration)
        cta_clip = cta_clip.set_fps(self.fps)
        mask_clip = VideoClip(make_cta_mask, duration=cta_duration, ismask=True)
        mask_clip = mask_clip.set_fps(self.fps)
        cta_clip = cta_clip.set_mask(mask_clip)
        cta_clip = (
            cta_clip
            .set_start(cta_start)
            .crossfadein(0.8)
            .crossfadeout(0.3)
        )
        overlays.append(cta_clip)
        print(f"  CTA end screen: {cta_start:.1f}s - {video_duration:.1f}s")

        return overlays

    def _create_weather_overlays(self, clips_data: List[Dict]) -> List:
        """Create section-aware weather/atmosphere particle overlays.

        Adds emotional atmosphere effects based on the narrative section:
        - Rain particles for THE_FALL (sadness, loss)
        - Dust/embers for COLD_OPEN, THE_CONFLICT (tension, drama)
        - Light particles for LEGACY, THE_SPARK (hope, remembrance)

        Weather type is determined by section and emotional_tone, with emotion
        taking priority over section defaults.
        """
        from moviepy.video.VideoClip import VideoClip
        overlays = []

        if self.weather_effects is None:
            return overlays

        # Group consecutive clips by weather type to create longer overlays
        current_weather = None
        current_start = None
        current_duration = 0.0
        current_emotion = ''

        for data in clips_data:
            section = data.get('section', '')
            emotional_tone = data.get('emotional_tone', '')
            start = data.get('start_time')
            duration = data.get('duration', 0)

            if start is None:
                continue

            weather = self.weather_effects.get_weather_for_section(section, emotional_tone)

            if weather == current_weather and weather is not None:
                current_duration = (start + duration) - current_start
            else:
                # Emit previous weather overlay if any
                if current_weather is not None and current_start is not None and current_duration > 1.0:
                    overlay = self._build_weather_clip(
                        current_weather, current_start, current_duration
                    )
                    if overlay is not None:
                        overlays.append(overlay)
                        print(f"  Weather: {current_weather} at {current_start:.1f}s-{current_start + current_duration:.1f}s")

                current_weather = weather
                current_start = start
                current_duration = duration

        # Emit last group
        if current_weather is not None and current_start is not None and current_duration > 1.0:
            overlay = self._build_weather_clip(
                current_weather, current_start, current_duration
            )
            if overlay is not None:
                overlays.append(overlay)
                print(f"  Weather: {current_weather} at {current_start:.1f}s-{current_start + current_duration:.1f}s")

        return overlays

    def _build_weather_clip(self, weather_type: str, start: float, duration: float):
        """Build a single weather overlay MoviePy clip using on-the-fly frame generation.

        Uses create_weather_frame() which computes each frame statelessly from
        time t, avoiding the O(n) memory cost of pre-generating all frames.
        A 120s section at 30fps would have needed ~28 GB; this uses O(1) memory.
        """
        from moviepy.video.VideoClip import VideoClip

        # Capture references for the closure
        weather_fx = self.weather_effects
        w, h = self.width, self.height

        # Use a small frame cache (last 2 frames) to avoid rendering twice
        # per frame (once for RGB, once for mask)
        _frame_cache = {}

        def _get_weather(t, _fx=weather_fx, _w=w, _h=h, _dur=duration,
                         _wt=weather_type, _cache=_frame_cache):
            t_key = round(t, 3)
            if t_key not in _cache:
                try:
                    frame = _fx.create_weather_frame(_w, _h, t, _dur, _wt, intensity=0.5)
                except Exception:
                    frame = np.zeros((_h, _w, 4), dtype=np.uint8)
                _cache[t_key] = frame
                if len(_cache) > 2:
                    oldest = min(_cache.keys())
                    del _cache[oldest]
            return _cache[t_key]

        def make_weather_frame(t, _get=_get_weather):
            return _get(t)[:, :, :3]

        def make_weather_mask(t, _get=_get_weather):
            return _get(t)[:, :, 3].astype(float) / 255.0

        clip = VideoClip(make_weather_frame, duration=duration)
        clip = clip.set_fps(self.fps)
        mask = VideoClip(make_weather_mask, duration=duration, ismask=True)
        mask = mask.set_fps(self.fps)
        clip = clip.set_mask(mask)
        clip = clip.set_start(start).crossfadein(0.8).crossfadeout(0.8)
        return clip

    def _export_video(self, video: CompositeVideoClip) -> None:
        """Export the final video with YouTube-optimized professional settings.
        
        Uses CRF 18 for high quality (YouTube recommends 15-18 for uploads),
        4 threads for faster encoding, and bf=2 for B-frame optimization.
        """
        print("Exporting video with YouTube-optimized settings...")

        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            video.write_videofile(
                str(self.output_path),
                fps=self.fps,
                codec='libx264',
                audio_codec='aac',
                audio_bitrate='192k',
                temp_audiofile='temp-audio.m4a',
                remove_temp=True,
                threads=4,
                preset='slow',
                ffmpeg_params=[
                    '-crf', '18',
                    '-pix_fmt', 'yuv420p',
                    '-bf', '2',
                    '-g', '60',
                    '-movflags', '+faststart',
                ]
            )
            print(f"Video exported successfully: {self.output_path}")
        except Exception as e:
            print(f"Export error with optimized settings: {e}")
            print("Retrying with standard settings...")
            video.write_videofile(
                str(self.output_path),
                fps=self.fps,
                codec='libx264',
                audio_codec='aac',
                threads=2,
                preset='medium',
                ffmpeg_params=['-crf', '23', '-pix_fmt', 'yuv420p']
            )

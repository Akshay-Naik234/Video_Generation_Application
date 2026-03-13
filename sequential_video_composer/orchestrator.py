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

        if self.duration_config_path:
            self._load_duration_config()

        self.transitions = TransitionEffects(resolution)
        self.movements = MovementStyles(resolution)
        self.color_grading = ColorGrading()
        self.clip_factory = ClipFactory(self)
        self.text_overlay_engine = TextOverlayEngine(resolution)

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
        """
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
                    return intersection[index % len(intersection)]
            
            # Fallback: shot_type preferences alone
            if shot_movements:
                return shot_movements[index % len(shot_movements)]
            
            # Fallback: section preferences alone
            if section_movements:
                return section_movements[index % len(section_movements)]

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
            subtle_movements = ['zoom_in', 'gentle_drift', 'zoom_out', 'focus_center', 'breathing', 'minimal']
            return subtle_movements[index % len(subtle_movements)]
        else:
            return self.movement_style if self.movement_style in MovementStyles.MOVEMENT_TYPES else 'zoom_in'

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
        overlays_found = sum(1 for d in self.image_durations.values() if d.get('overlay_text'))
        if overlays_found > 0:
            print(f"  - Text overlays: {overlays_found} images with date/location/name stamps")
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
            main_video = main_video.set_audio(audio_clip)

        self._export_video(main_video)
        print(f"Video created successfully: {self.output_path}")

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
            counter_match = _re.match(r'^(\$?)([\d,]+)\+?\s*(.*)$', text)
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

                    def make_slide_frame(t, txt=full_text, dur=overlay_duration, eng=self.text_overlay_engine):
                        # Slide in during first 0.5s, hold, then static
                        slide_in_time = 0.4
                        if t < slide_in_time:
                            progress = t / slide_in_time
                            eased = 1.0 - (1.0 - progress) ** 3  # ease-out cubic
                        else:
                            eased = 1.0
                        frame = eng.create_slide_in_overlay(txt, position='bottom_left', slide_from='left', progress=eased)
                        return frame[:, :, :3]

                    def make_slide_mask(t, txt=full_text, dur=overlay_duration, eng=self.text_overlay_engine):
                        slide_in_time = 0.4
                        if t < slide_in_time:
                            progress = t / slide_in_time
                            eased = 1.0 - (1.0 - progress) ** 3
                        else:
                            eased = 1.0
                        frame = eng.create_slide_in_overlay(txt, position='bottom_left', slide_from='left', progress=eased)
                        return frame[:, :, 3].astype(float) / 255.0

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

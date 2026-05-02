"""Sequential Video Orchestrator — Main orchestration class.

Enhanced with documentary-style section-aware animation selection:
movements, transitions, and visual effects are chosen based on the
section (COLD_OPEN, EARLY_LIFE, …) and emotional_tone from the
duration config JSON.
"""

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
        crossfade_duration: float = 1.2,
        zoom_intensity: float = 1.08,
        effects_intensity: float = 0.7,
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
        self.total_video_duration: float = 0

        if self.duration_config_path:
            self._load_duration_config()

        self.transitions = TransitionEffects(resolution)
        self.movements = MovementStyles(resolution)
        self.color_grading = ColorGrading()
        self.clip_factory = ClipFactory(self)
        self.text_overlay_engine = TextOverlayEngine(resolution, accent_color=overlay_accent_color)
        self.effects = DocumentaryEffects(resolution)

    def _load_duration_config(self) -> None:
        """Load image durations, timing, sections, and tones from JSON config."""
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

            print(f"Loaded timing data for {len(self.image_durations)} images from config")
            if self.overlay_texts:
                print(f"Found {len(self.overlay_texts)} overlay text entries")
            if self.image_sections:
                print(f"Found section data for {len(self.image_sections)} images")
            if self.total_video_duration:
                print(f"Total video duration from config: {self.total_video_duration}s")
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Error loading duration config: {e}")

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

    def _get_movement_for_image(self, index: int, total: int, image_num: int = 0) -> str:
        """Get movement style for an image based on section, tone, and position.

        When a duration config with section/tone data is available, the
        movement is chosen from section-appropriate and tone-appropriate
        pools. Otherwise falls back to the legacy position-based logic.
        """
        section = self.image_sections.get(image_num, '')
        tone = self.image_tones.get(image_num, '')

        # Section-aware documentary mode
        if self.movement_style in ('documentary', 'random') and section:
            section_pool = MovementStyles.SECTION_MOVEMENTS.get(section, [])
            tone_pool = MovementStyles.TONE_MOVEMENTS.get(tone, [])

            # Build a weighted pool: section movements + tone movements
            pool = section_pool + tone_pool
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

        When section data is available, uses section-appropriate transitions
        for documentary feel. Otherwise falls back to legacy logic.
        """
        section = self.image_sections.get(image_num, '')

        # Section-aware mode
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
        print("Starting Sequential Video Orchestrator (Documentary Enhanced)...")
        print(f"Images root: {self.images_root}")
        print(f"Output path: {self.output_path}")
        print(f"Resolution: {self.resolution}")
        print(f"Image duration: {self.image_duration}s")
        print(f"Transition style: {self.transition_style}")
        print(f"Movement style: {self.movement_style}")
        print(f"Color grade: {self.color_grade}")
        print(f"Documentary effects: {self.enable_documentary_effects}")
        if self.total_video_duration:
            print(f"Target video duration: {self.total_video_duration}s")

        numbered_images = self.discover_numbered_images()
        clips_data = self.create_image_clips(numbered_images)

        if not clips_data:
            raise ValueError("No valid image clips were created")

        main_video = self.create_timeline_video(clips_data)

        if main_video is None:
            raise ValueError("Failed to create timeline video")

        print(f"Main video duration: {main_video.duration}s")

        overlays = [main_video]

        # Documentary effects (section-aware overlays)
        if self.enable_documentary_effects:
            doc_overlays = self._create_documentary_effect_overlays(clips_data)
            overlays.extend(doc_overlays)
        else:
            # Legacy particle/grain overlays
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
        if self.enable_text_overlays and self.overlay_texts:
            text_clips = self._create_text_overlay_clips(clips_data)
            overlays.extend(text_clips)

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

    def _create_documentary_effect_overlays(self, clips_data: List[Dict]) -> List:
        """Create section-aware documentary effect overlays.

        Groups consecutive images by section, then applies the appropriate
        visual effects (film grain, light leaks, dust particles, etc.) as
        overlays spanning each section's time range.
        """
        if not self.image_sections:
            # No section data — apply global light effects
            dur = clips_data[-1].get('end_time') or clips_data[-1].get('start_time', 0) + clips_data[-1].get('duration', 0)
            if dur <= 0:
                return []
            return [
                self.effects.create_film_grain(dur, self.effects_intensity * 0.12),
                self.effects.create_dust_particles(dur, self.effects_intensity * 0.15),
            ]

        # Group clips by section to find time ranges
        sections: Dict[str, Dict] = {}
        for data in clips_data:
            img_num = data.get('image_num')
            section = self.image_sections.get(img_num, 'UNKNOWN')
            start = data.get('start_time', 0) or 0
            end = data.get('end_time') or (start + data.get('duration', 0))

            if section not in sections:
                sections[section] = {'start': start, 'end': end}
            else:
                sections[section]['start'] = min(sections[section]['start'], start)
                sections[section]['end'] = max(sections[section]['end'], end)

        effect_clips = []
        print(f"\nCreating documentary effect overlays for {len(sections)} sections...")

        for section, times in sections.items():
            sec_duration = times['end'] - times['start']
            if sec_duration <= 0:
                continue

            section_effects = self.effects.get_section_effects(
                section, sec_duration, self.effects_intensity, max_effects=2
            )
            for fx in section_effects:
                fx = fx.set_start(times['start'])
                effect_clips.append(fx)

            print(f"  {section}: {times['start']:.1f}s - {times['end']:.1f}s "
                  f"({sec_duration:.1f}s) — {len(section_effects)} effects")

        return effect_clips

    def _create_text_overlay_clips(self, clips_data: List[Dict]) -> List:
        """Create MoviePy clips for all overlay text entries."""
        text_clips = []
        engine = self.text_overlay_engine

        print(f"\nCreating {len(self.overlay_texts)} text overlays...")

        for data in clips_data:
            image_num = data.get('image_num')
            if image_num not in self.overlay_texts:
                continue

            overlay_text = self.overlay_texts[image_num]
            start_time = data.get('start_time')
            duration = data.get('duration', 0)

            if start_time is None or duration <= 0:
                continue

            overlay_rgba = engine.render_overlay_text(overlay_text)

            rgb_frame = overlay_rgba[:, :, :3]
            alpha_mask = overlay_rgba[:, :, 3].astype(np.float64) / 255.0

            def make_frame(_t, _rgb=rgb_frame):
                return _rgb

            def make_mask(_t, _alpha=alpha_mask):
                return _alpha

            clip = VideoClip(make_frame, duration=duration)
            clip = clip.set_fps(1)
            mask_clip = VideoClip(make_mask, duration=duration, ismask=True)
            mask_clip = mask_clip.set_fps(1)
            clip = clip.set_mask(mask_clip)

            fade_in = min(0.5, duration * 0.15)
            fade_out = min(0.4, duration * 0.12)
            clip = clip.set_start(start_time).crossfadein(fade_in).crossfadeout(fade_out)

            text_clips.append(clip)

            overlay_type = TextOverlayEngine.classify_overlay_text(overlay_text)
            print(f"  Image {image_num}: [{overlay_type}] \"{overlay_text[:50]}...\" "
                  f"at {start_time:.1f}s ({duration:.1f}s)")

        return text_clips

    def _export_video(self, video: CompositeVideoClip) -> None:
        """Export the final video with professional settings."""
        print("Exporting video...")

        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            video.write_videofile(
                str(self.output_path),
                fps=self.fps,
                codec='libx264',
                audio_codec='aac',
                temp_audiofile='temp-audio.m4a',
                remove_temp=True,
                threads=2,
                preset='medium',
                ffmpeg_params=['-crf', '23', '-pix_fmt', 'yuv420p']
            )
            print(f"Video exported successfully: {self.output_path}")
        except Exception as e:
            print(f"Export error: {e}")
            print("Retrying with basic settings...")
            video.write_videofile(
                str(self.output_path),
                fps=self.fps,
                codec='libx264',
                audio_codec='aac'
            )

"""Clip factory for creating and composing video clips."""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple, Dict, TYPE_CHECKING

import numpy as np
from moviepy.editor import CompositeVideoClip, ColorClip, concatenate_videoclips
from moviepy.video.VideoClip import VideoClip
from PIL import Image as PILImage
from tqdm import tqdm

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from .orchestrator import SequentialVideoOrchestrator


class ClipFactory:
    """Factory for creating and composing video clips."""

    # Crossfade / overlap tuning constants
    MAX_FADE_DURATION = 0.8       # seconds
    FADE_DURATION_RATIO = 0.18    # fraction of clip duration used as base fade
    MIN_FADE_DURATION = 0.25      # seconds (floor)
    MAX_FADE_RATIO = 0.35         # fade may not exceed this fraction of clip duration
    MAX_OVERLAP_RATIO = 0.25      # overlap may not exceed this fraction of clip duration
    LAST_CLIP_FADE_MULT = 1.5     # last clip gets a longer fade-out
    LAST_CLIP_MAX_FADE = 0.4      # max fraction of duration for last clip fade

    def __init__(self, orchestrator: 'SequentialVideoOrchestrator'):
        self.orchestrator = orchestrator

    def create_image_clips(self, numbered_images: List[Tuple[int, Path]]) -> List[Dict]:
        """Create animated clips for each image with movement, effects, and timing info."""
        clips_data = []
        total = len(numbered_images)
        failed = 0

        for i, (num, image_path) in enumerate(tqdm(numbered_images, desc="Processing images")):
            logger.info("Processing image %d: %s", num, image_path.name)

            if not image_path.exists():
                logger.warning("Image %d missing: %s — inserting black placeholder", num, image_path)
                failed += 1
                timing = self.orchestrator.get_timing_for_image(num)
                dur = timing.get('duration', self.orchestrator.image_duration)
                placeholder = self._create_placeholder_clip(dur, f"Image {num} missing")
                clips_data.append({
                    'clip': placeholder, 'transition': 'crossfade',
                    'image_num': num,
                    'start_time': timing.get('start_time'),
                    'end_time': timing.get('end_time'),
                    'duration': dur,
                })
                continue

            try:
                img = PILImage.open(image_path)
                img.load()
                logger.debug("  Loaded image: %s, mode: %s", img.size, img.mode)
            except Exception as e:
                logger.error("Failed to load image %d (%s): %s", num, image_path, e)
                failed += 1
                timing = self.orchestrator.get_timing_for_image(num)
                dur = timing.get('duration', self.orchestrator.image_duration)
                placeholder = self._create_placeholder_clip(dur, f"Image {num} load error")
                clips_data.append({
                    'clip': placeholder, 'transition': 'crossfade',
                    'image_num': num,
                    'start_time': timing.get('start_time'),
                    'end_time': timing.get('end_time'),
                    'duration': dur,
                })
                continue

            timing = self.orchestrator.get_timing_for_image(num)
            image_duration = timing.get('duration', self.orchestrator.image_duration)
            start_time = timing.get('start_time')
            end_time = timing.get('end_time')
            
            movement = self.orchestrator._get_movement_for_image(i, total, image_num=num)
            logger.debug("  Duration: %ss", image_duration)
            logger.debug("  Start time: %ss", start_time)
            logger.debug("  End time: %ss", end_time)
            logger.debug("  Movement style: %s", movement)

            # Section-aware color grading: use section-specific grade when available
            section = self.orchestrator.image_sections.get(num, '')
            if section:
                grade = self.orchestrator.color_grading.grade_for_section(section)
            else:
                grade = self.orchestrator.color_grade

            clip = self.orchestrator.movements.create_animated_clip(
                image_path=image_path,
                duration=image_duration,
                movement_type=movement,
                zoom_intensity=self.orchestrator.zoom_intensity,
                color_grader=self.orchestrator.color_grading,
                color_grade=grade,
                enable_vignette=self.orchestrator.enable_vignette,
                section=section,
            )

            transition = self.orchestrator._get_transition_for_image(i, total, image_num=num)
            clips_data.append({
                'clip': clip,
                'transition': transition,
                'image_num': num,
                'start_time': start_time,
                'end_time': end_time,
                'duration': image_duration
            })

        if failed:
            logger.warning("%d/%d images failed to load", failed, total)
        logger.info("Successfully created %d clips from %d images", len(clips_data), total)
        return clips_data

    # Section-aware fade durations: dramatic sections get snappier cuts,
    # emotional sections get longer, more graceful fades.
    SECTION_FADE_MULTIPLIERS = {
        'COLD_OPEN': 0.8,
        'THE_CONFLICT': 0.7,
        'THE_CLIMAX': 0.75,
        'EARLY_LIFE': 1.0,
        'LEGACY': 1.0,
        'CTA': 0.9,
        'THE_FALL': 0.9,
        'THE_SPARK': 1.0,
        'THE_RISE': 0.9,
    }

    def create_timeline_video(self, clips_data: List[Dict]) -> CompositeVideoClip:
        """Create video with clips positioned at their exact start times.

        Key design decisions for seamless playback:
        - Clips are extended and overlapped so there is NEVER a black gap
        - Section-aware fade durations (dramatic = snappy, emotional = graceful)
        - Smooth crossfade blending between consecutive clips
        - First clip fades in from black, last clip fades out to black
        """
        if not clips_data:
            return None

        has_timing = clips_data[0].get('start_time') is not None

        if has_timing:
            logger.info("Creating timeline-based video with seamless crossfade overlap...")
            positioned_clips = []
            total_clips = len(clips_data)

            for i, data in enumerate(clips_data):
                clip = data['clip']
                start_time = data['start_time']
                duration = data['duration']

                if start_time is None:
                    continue

                # Section-aware fade duration
                img_num = data.get('image_num', 0)
                section = self.orchestrator.image_sections.get(img_num, '')
                fade_mult = self.SECTION_FADE_MULTIPLIERS.get(section, 1.0)
                base_fade = min(self.MAX_FADE_DURATION, duration * self.FADE_DURATION_RATIO)
                fade_duration = base_fade * fade_mult
                fade_duration = max(self.MIN_FADE_DURATION, min(fade_duration, duration * self.MAX_FADE_RATIO))

                # Compute overlap: extend clip start earlier so it overlaps
                # with the previous clip, eliminating any black gap.
                if i > 0:
                    overlap = min(fade_duration, duration * self.MAX_OVERLAP_RATIO)
                    effective_start = max(0.0, start_time - overlap)
                else:
                    effective_start = start_time

                # Build the positioned clip with smooth fade
                positioned_clip = clip.set_start(effective_start)

                # First clip: no fade-in (avoid black screen at start)
                if i == 0:
                    positioned_clip = positioned_clip.crossfadeout(fade_duration)
                # Last clip: fade out to black only
                elif i == total_clips - 1:
                    positioned_clip = positioned_clip.crossfadein(fade_duration)
                    positioned_clip = positioned_clip.crossfadeout(
                        min(fade_duration * self.LAST_CLIP_FADE_MULT, duration * self.LAST_CLIP_MAX_FADE)
                    )
                else:
                    positioned_clip = positioned_clip.crossfadein(fade_duration)
                    positioned_clip = positioned_clip.crossfadeout(fade_duration)

                positioned_clips.append(positioned_clip)
                logger.debug("  Image %s: starts at %.2fs (orig %ss), dur %ss, fade %.2fs, transition=%s",
                             data['image_num'], effective_start, start_time, duration, fade_duration,
                             data.get('transition', 'crossfade'))

            total_duration = self.orchestrator.total_video_duration
            if not total_duration and clips_data:
                last_clip = clips_data[-1]
                total_duration = last_clip.get('end_time') or (
                    last_clip.get('start_time', 0) + last_clip.get('duration', 0)
                )

            logger.info("Total video duration: %ss", total_duration)
            logger.info("Positioned %d clips with crossfade overlap", len(positioned_clips))

            background = ColorClip(
                size=self.orchestrator.resolution,
                color=(0, 0, 0),
                duration=total_duration
            )

            final_video = CompositeVideoClip(
                [background] + positioned_clips,
                size=self.orchestrator.resolution
            ).set_duration(total_duration)

            return final_video
        else:
            logger.info("No timing data available, using sequential concatenation...")
            return self._concatenate_clips(clips_data)

    def _concatenate_clips(self, clips_data: List[Dict]) -> CompositeVideoClip:
        """Concatenate clips with overlapping crossfades to avoid black frames.

        Each consecutive pair of clips overlaps by ``fade_duration`` so that
        clip *i* fading out and clip *i+1* fading in blend together instead
        of revealing the black background.
        """
        if not clips_data:
            return None

        fade_duration = min(
            self.orchestrator.crossfade_duration * 0.3,
            clips_data[0]['duration'] * 0.15,
        )
        fade_duration = max(fade_duration, 0.2)

        positioned = []
        current_time = 0.0

        for i, data in enumerate(clips_data):
            clip = data['clip']

            if i == 0:
                placed = clip.set_start(current_time).crossfadeout(fade_duration)
            elif i == len(clips_data) - 1:
                placed = clip.set_start(current_time).crossfadein(fade_duration)
            else:
                placed = (
                    clip.set_start(current_time)
                    .crossfadein(fade_duration)
                    .crossfadeout(fade_duration)
                )

            positioned.append(placed)
            current_time += data['duration'] - fade_duration

        total_duration = current_time + fade_duration

        background = ColorClip(
            size=self.orchestrator.resolution,
            color=(0, 0, 0),
            duration=total_duration,
        )

        return CompositeVideoClip(
            [background] + positioned,
            size=self.orchestrator.resolution,
        ).set_duration(total_duration)

    def _create_placeholder_clip(self, duration: float, label: str = '') -> 'VideoClip':
        """Create a black placeholder clip with optional warning text.

        Used when an image fails to load so the timeline stays in sync
        with the audio rather than producing a gap.
        """
        w, h = self.orchestrator.resolution
        black_frame = np.zeros((h, w, 3), dtype=np.uint8)

        if label:
            try:
                from PIL import ImageDraw, ImageFont
                pil_img = PILImage.fromarray(black_frame)
                draw = ImageDraw.Draw(pil_img)
                try:
                    font = ImageFont.truetype(
                        '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 24
                    )
                except (OSError, IOError):
                    font = ImageFont.load_default()
                bbox = draw.textbbox((0, 0), label, font=font)
                tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
                draw.text(
                    ((w - tw) // 2, (h - th) // 2), label,
                    font=font, fill=(180, 180, 180)
                )
                black_frame = np.array(pil_img)
            except Exception:
                pass

        static = black_frame

        def make_frame(t):
            return static

        clip = VideoClip(make_frame, duration=duration).set_fps(
            self.orchestrator.fps
        )
        return clip

    def create_particle_overlay(self, duration: float, intensity: float = 0.3) -> ColorClip:
        """Create a subtle particle/dust overlay effect."""
        return ColorClip(
            size=self.orchestrator.resolution,
            color=(255, 255, 255)
        ).set_opacity(intensity * 0.08).set_duration(duration)

    def create_film_grain_overlay(self, duration: float, intensity: float = 0.3) -> ColorClip:
        """Create a film grain overlay effect."""
        return ColorClip(
            size=self.orchestrator.resolution,
            color=(128, 128, 128)
        ).set_opacity(intensity * 0.12).set_duration(duration)

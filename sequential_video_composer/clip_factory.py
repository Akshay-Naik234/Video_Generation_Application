"""Clip factory for creating and composing video clips."""

from pathlib import Path
from typing import List, Tuple, Dict, TYPE_CHECKING

import numpy as np
from moviepy.editor import CompositeVideoClip, ColorClip, concatenate_videoclips
from moviepy.video.VideoClip import VideoClip
from PIL import Image as PILImage
from tqdm import tqdm

if TYPE_CHECKING:
    from .orchestrator import SequentialVideoOrchestrator


class ClipFactory:
    """Factory for creating and composing video clips."""

    def __init__(self, orchestrator: 'SequentialVideoOrchestrator'):
        self.orchestrator = orchestrator

    def create_image_clips(self, numbered_images: List[Tuple[int, Path]]) -> List[Dict]:
        """Create animated clips for each image with movement, effects, and timing info."""
        clips_data = []
        total = len(numbered_images)
        failed = 0
        brightness_target = self._calculate_batch_brightness_target(numbered_images)
        if brightness_target:
            print(f"Batch brightness target luminance: {brightness_target:.1f}")

        for i, (num, image_path) in enumerate(tqdm(numbered_images, desc="Processing images")):
            print(f"Processing image {num}: {image_path.name}")

            if not image_path.exists():
                print(f"WARNING: Image file does not exist, skipping: {image_path}")
                failed += 1
                continue

            try:
                img = PILImage.open(image_path)
                img.verify()
                img = PILImage.open(image_path)
                print(f"  Loaded image: {img.size}, mode: {img.mode}")
            except Exception as e:
                print(f"ERROR: Failed to load image {image_path}: {e}")
                failed += 1
                continue

            timing = self.orchestrator.get_timing_for_image(num)
            image_duration = timing.get('duration', self.orchestrator.image_duration)
            start_time = timing.get('start_time')
            end_time = timing.get('end_time')
            
            movement = self.orchestrator._get_movement_for_image(i, total, image_num=num)
            print(f"  Duration: {image_duration}s")
            print(f"  Start time: {start_time}s")
            print(f"  End time: {end_time}s")
            print(f"  Movement style: {movement}")

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
                brightness_target=brightness_target,
            )

            transition = self.orchestrator._get_transition_for_image(i, total, image_num=num)
            clips_data.append({
                'clip': clip,
                'transition': transition,
                'image_num': num,
                'image_path': image_path,
                'start_time': start_time,
                'end_time': end_time,
                'duration': image_duration
            })

        if failed:
            print(f"\nWARNING: {failed}/{total} images failed to load")
        print(f"Successfully created {len(clips_data)} clips from {total} images")
        return clips_data

    def _calculate_batch_brightness_target(
        self, numbered_images: List[Tuple[int, Path]]
    ) -> float:
        """Analyze all input images and choose a stable luminance target.

        The target is intentionally conservative: it lifts dark batches toward
        a readable documentary baseline without forcing every image to the
        exact same flat exposure.
        """
        luminances = []
        for _, image_path in numbered_images:
            try:
                with PILImage.open(image_path) as img:
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    img.thumbnail((320, 320), PILImage.LANCZOS)
                    arr = np.array(img)
                    luminances.append(self.orchestrator.color_grading.measure_luminance(arr))
            except Exception:
                continue

        if not luminances:
            return 120.0

        median_luma = float(np.median(luminances))
        # Keep the video consistently legible, but avoid over-brightening
        # deliberately bright source batches.
        return float(np.clip(max(118.0, median_luma), 118.0, 135.0))

    # Section-aware fade durations: dramatic sections get snappier cuts,
    # emotional sections get longer, more graceful fades.
    SECTION_FADE_MULTIPLIERS = {
        'COLD_OPEN': 0.8,
        'THE_CONFLICT': 0.7,
        'THE_CLIMAX': 0.75,
        'EARLY_LIFE': 1.4,
        'LEGACY': 1.5,
        'CTA': 1.3,
        'THE_FALL': 1.2,
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
            print("Creating timeline-based video with seamless crossfade overlap...")
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
                base_fade = min(0.8, duration * 0.18)
                fade_duration = base_fade * fade_mult
                fade_duration = max(0.25, min(fade_duration, duration * 0.35))

                # Compute overlap: extend clip start earlier so it overlaps
                # with the previous clip, eliminating any black gap.
                if i > 0:
                    overlap = min(fade_duration, duration * 0.25)
                    effective_start = max(0.0, start_time - overlap)
                else:
                    effective_start = start_time

                # Build the positioned clip with smooth fade
                positioned_clip = clip.set_start(effective_start)

                # First clip: fade in from black only
                if i == 0:
                    positioned_clip = positioned_clip.crossfadein(fade_duration)
                    positioned_clip = positioned_clip.crossfadeout(fade_duration)
                # Last clip: fade out to black only
                elif i == total_clips - 1:
                    positioned_clip = positioned_clip.crossfadein(fade_duration)
                    positioned_clip = positioned_clip.crossfadeout(min(fade_duration * 1.5, duration * 0.4))
                else:
                    positioned_clip = positioned_clip.crossfadein(fade_duration)
                    positioned_clip = positioned_clip.crossfadeout(fade_duration)

                positioned_clips.append(positioned_clip)
                print(f"  Image {data['image_num']}: starts at {effective_start:.2f}s "
                      f"(orig {start_time}s), dur {duration}s, fade {fade_duration:.2f}s")

            total_duration = self.orchestrator.total_video_duration
            if not total_duration and clips_data:
                last_clip = clips_data[-1]
                total_duration = last_clip.get('end_time') or (
                    last_clip.get('start_time', 0) + last_clip.get('duration', 0)
                )

            print(f"Total video duration: {total_duration}s")
            print(f"Positioned {len(positioned_clips)} clips with crossfade overlap")

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
            print("No timing data available, using sequential concatenation...")
            return self._concatenate_clips(clips_data)

    def _concatenate_clips(self, clips_data: List[Dict]) -> CompositeVideoClip:
        """Fallback method to concatenate clips sequentially."""
        clean_clips = []
        for data in clips_data:
            clip = data['clip']
            duration = data['duration']
            fade_duration = min(self.orchestrator.crossfade_duration * 0.3, duration * 0.15)
            clean_clip = clip.fadein(fade_duration).fadeout(fade_duration)
            clean_clips.append(clean_clip)

        return concatenate_videoclips(clean_clips, method="compose")

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

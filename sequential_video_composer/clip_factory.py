"""Clip factory for creating and composing video clips."""

from pathlib import Path
from typing import List, Tuple, Dict, TYPE_CHECKING

from moviepy.editor import CompositeVideoClip, ColorClip, concatenate_videoclips
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

        for i, (num, image_path) in enumerate(tqdm(numbered_images, desc="Processing images")):
            print(f"Processing image {num}: {image_path.name}")

            if not image_path.exists():
                print(f"Warning: Image file does not exist: {image_path}")
                continue

            try:
                img = PILImage.open(image_path)
                print(f"  Loaded image: {img.size}, mode: {img.mode}")
            except Exception as e:
                print(f"Error loading image {image_path}: {e}")
                continue

            timing = self.orchestrator.get_timing_for_image(num)
            image_duration = timing.get('duration', self.orchestrator.image_duration)
            start_time = timing.get('start_time')
            end_time = timing.get('end_time')
            
            movement = self.orchestrator._get_movement_for_image(i, total)
            print(f"  Duration: {image_duration}s")
            print(f"  Start time: {start_time}s")
            print(f"  End time: {end_time}s")
            print(f"  Movement style: {movement}")

            clip = self.orchestrator.movements.create_animated_clip(
                image_path=image_path,
                duration=image_duration,
                movement_type=movement,
                zoom_intensity=self.orchestrator.zoom_intensity,
                color_grader=self.orchestrator.color_grading,
                color_grade=self.orchestrator.color_grade,
                enable_vignette=self.orchestrator.enable_vignette
            )

            transition = self.orchestrator._get_transition_for_image(i, total)
            clips_data.append({
                'clip': clip,
                'transition': transition,
                'image_num': num,
                'start_time': start_time,
                'end_time': end_time,
                'duration': image_duration
            })

        return clips_data

    def create_timeline_video(self, clips_data: List[Dict]) -> CompositeVideoClip:
        """Create video with clips positioned at their exact start times.

        Adjacent clips are overlapped and blended via ``crossfadein`` so the
        video flows continuously between images. Without this, every boundary
        briefly fades to the black background, producing a visible "blink"
        between images.
        """
        if not clips_data:
            return None

        has_timing = clips_data[0].get('start_time') is not None

        if has_timing:
            print("Creating timeline-based video with exact start/end times...")
            positioned_clips = []

            valid_clips = [d for d in clips_data if d.get('start_time') is not None]
            shortest_duration = min((d['duration'] for d in valid_clips), default=0)

            # Crossfade duration: bounded by configured value and a fraction of
            # the shortest clip so we never eat more than ~35% of any image.
            crossfade = min(
                self.orchestrator.crossfade_duration,
                max(0.25, shortest_duration * 0.35)
            )
            # Intro/outro fades from/to black are only applied to the very
            # first and last clip so the video starts and ends gracefully.
            intro_outro_fade = min(0.6, shortest_duration * 0.25) if shortest_duration else 0.3

            total_clips = len(valid_clips)
            for i, data in enumerate(valid_clips):
                clip = data['clip']
                start_time = data['start_time']
                duration = data['duration']
                is_first = (i == 0)
                is_last = (i == total_clips - 1)

                # Extend every non-last clip so it remains visible while the
                # next clip fades in on top. This gives us a real crossfade
                # rather than a fade-to-black/fade-from-black gap.
                if not is_last:
                    clip = clip.set_duration(duration + crossfade)

                positioned_clip = clip.set_start(start_time)

                if is_first:
                    positioned_clip = positioned_clip.fadein(intro_outro_fade)
                else:
                    # crossfadein blends this clip over the still-rendering
                    # previous clip for a smooth handoff.
                    positioned_clip = positioned_clip.crossfadein(crossfade)

                if is_last:
                    positioned_clip = positioned_clip.fadeout(intro_outro_fade)

                positioned_clips.append(positioned_clip)
                print(f"  Image {data['image_num']}: starts at {start_time}s, duration {duration}s")

            total_duration = self.orchestrator.total_video_duration
            if not total_duration and clips_data:
                last_clip = clips_data[-1]
                total_duration = last_clip.get('end_time') or (last_clip.get('start_time', 0) + last_clip.get('duration', 0))

            print(f"Total video duration: {total_duration}s")
            print(f"Crossfade duration between clips: {crossfade:.2f}s")

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

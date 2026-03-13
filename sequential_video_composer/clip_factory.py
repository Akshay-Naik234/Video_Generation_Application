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
        """Create animated clips for each image with movement, effects, and timing info.
        
        Uses section metadata (when available) to select appropriate color grading
        and movement styles for each image based on its narrative position.
        """
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
            section = timing.get('section', '')
            emotional_tone = timing.get('emotional_tone', '')
            
            # Section-aware movement selection
            movement = self.orchestrator._get_movement_for_image(i, total, image_number=num)
            
            # Section-aware color grading
            color_grade = self.orchestrator.get_color_grade_for_image(num)
            
            print(f"  Duration: {image_duration}s")
            print(f"  Start time: {start_time}s")
            print(f"  End time: {end_time}s")
            if section:
                print(f"  Section: {section} | Emotion: {emotional_tone}")
            print(f"  Movement: {movement} | Color grade: {color_grade}")

            clip = self.orchestrator.movements.create_animated_clip(
                image_path=image_path,
                duration=image_duration,
                movement_type=movement,
                zoom_intensity=self.orchestrator.zoom_intensity,
                color_grader=self.orchestrator.color_grading,
                color_grade=color_grade,
                enable_vignette=self.orchestrator.enable_vignette
            )

            # Section-aware transition selection
            transition = self.orchestrator._get_transition_for_image(i, total, image_number=num)
            clips_data.append({
                'clip': clip,
                'transition': transition,
                'image_num': num,
                'start_time': start_time,
                'end_time': end_time,
                'duration': image_duration,
                'section': section,
                'emotional_tone': emotional_tone,
            })

        return clips_data

    def create_timeline_video(self, clips_data: List[Dict]) -> CompositeVideoClip:
        """Create video with clips positioned at their exact start times.
        
        Uses proper crossfade overlaps between consecutive clips for smooth
        cinematic transitions instead of hard cuts. Fade durations are scaled
        based on image duration and section context for a natural feel.
        """
        if not clips_data:
            return None

        has_timing = clips_data[0].get('start_time') is not None
        
        if has_timing:
            print("Creating timeline-based video with cinematic crossfade overlaps...")
            positioned_clips = []
            
            for i, data in enumerate(clips_data):
                clip = data['clip']
                start_time = data['start_time']
                duration = data['duration']
                section = data.get('section', '')
                
                if start_time is None:
                    continue
                
                # Scale fade duration based on image duration and section
                # Longer fades for emotional sections, shorter for fast-paced ones
                if section in ('COLD_OPEN', 'THE_CONFLICT'):
                    fade_ratio = 0.10  # Faster cuts for tension
                elif section in ('THE_FALL', 'LEGACY'):
                    fade_ratio = 0.25  # Longer fades for emotional weight
                elif section in ('THE_CLIMAX',):
                    fade_ratio = 0.20  # Let the peak breathe
                else:
                    fade_ratio = 0.15  # Default cinematic fade
                
                fade_duration = min(0.8, max(0.3, duration * fade_ratio))
                
                # Apply crossfade overlap: start slightly earlier to overlap with previous clip
                overlap_offset = fade_duration * 0.5 if i > 0 else 0
                adjusted_start = max(0, start_time - overlap_offset)
                
                positioned_clip = (
                    clip
                    .set_start(adjusted_start)
                    .crossfadein(fade_duration)
                    .crossfadeout(fade_duration)
                )
                positioned_clips.append(positioned_clip)
                print(f"  Image {data['image_num']}: starts at {adjusted_start:.2f}s, "
                      f"duration {duration}s, fade {fade_duration:.2f}s"
                      f"{' [' + section + ']' if section else ''}")
            
            total_duration = self.orchestrator.total_video_duration
            if not total_duration and clips_data:
                last_clip = clips_data[-1]
                total_duration = last_clip.get('end_time') or (last_clip.get('start_time', 0) + last_clip.get('duration', 0))
            
            print(f"Total video duration: {total_duration}s")
            
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
        """Fallback method to concatenate clips sequentially with smooth crossfades."""
        clean_clips = []
        for data in clips_data:
            clip = data['clip']
            duration = data['duration']
            fade_duration = min(self.orchestrator.crossfade_duration * 0.5, duration * 0.15)
            fade_duration = max(0.3, fade_duration)
            clean_clip = clip.crossfadein(fade_duration).crossfadeout(fade_duration)
            clean_clips.append(clean_clip)

        return concatenate_videoclips(clean_clips, method="compose")

    def create_particle_overlay(self, duration: float, intensity: float = 0.3) -> VideoClip:
        """Create a subtle animated dust/particle overlay effect using random noise.
        
        Generates per-frame random bright spots that simulate floating dust particles
        in a cinematic light beam, adding depth and atmosphere to the video.
        """
        width, height = self.orchestrator.resolution
        particle_opacity = intensity * 0.06

        def make_particle_frame(t):
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            num_particles = int(width * height * 0.00008 * intensity)
            if num_particles > 0:
                ys = np.random.randint(0, height, num_particles)
                xs = np.random.randint(0, width, num_particles)
                brightness = np.random.randint(180, 255, num_particles)
                for y, x, b in zip(ys, xs, brightness):
                    y_start = max(0, y - 1)
                    y_end = min(height, y + 2)
                    x_start = max(0, x - 1)
                    x_end = min(width, x + 2)
                    frame[y_start:y_end, x_start:x_end] = [b, b, b]
            return frame

        particle_clip = VideoClip(make_particle_frame, duration=duration)
        particle_clip = particle_clip.set_fps(15)
        particle_clip = particle_clip.set_opacity(particle_opacity)
        return particle_clip

    def create_film_grain_overlay(self, duration: float, intensity: float = 0.3) -> VideoClip:
        """Create a realistic film grain overlay using per-frame random noise.
        
        Generates monochromatic noise that simulates 35mm film grain texture,
        adding a cinematic documentary aesthetic to the video.
        """
        width, height = self.orchestrator.resolution
        grain_opacity = intensity * 0.08

        def make_grain_frame(t):
            noise = np.random.randint(0, 60, (height, width), dtype=np.uint8)
            grain = np.stack([noise, noise, noise], axis=-1)
            return grain

        grain_clip = VideoClip(make_grain_frame, duration=duration)
        grain_clip = grain_clip.set_fps(24)
        grain_clip = grain_clip.set_opacity(grain_opacity)
        return grain_clip

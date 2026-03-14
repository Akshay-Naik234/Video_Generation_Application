"""Clip factory for creating and composing video clips.

Includes humanization features that make the editing rhythm feel natural:
- Timing micro-variation: slight random variation on crossfade durations
- Hard cuts for impact: dramatic section entries skip crossfade entirely
- Breathing room: longer holds between intense sections
- Editing rhythm: alternating fast/slow pacing creates visual heartbeat
"""

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
    """Factory for creating and composing video clips with human-feel editing.
    
    Includes dynamic pacing, pattern interrupts, cinematic letterboxing,
    professional particle/grain overlays, and humanization features that
    make the video feel like it was hand-edited by a professional.
    """

    # Dynamic pacing: speed multipliers per section (1.0 = normal speed)
    # Fast cuts for tension, slower for emotional breathing room
    SECTION_PACE = {
        'COLD_OPEN': 1.15,       # Slightly faster - hook the viewer
        'EARLY_LIFE': 0.95,      # Gentle pace
        'THE_SPARK': 1.05,       # Pick up energy
        'THE_RISE': 1.10,        # Building momentum
        'THE_CONFLICT': 1.20,    # Fast cuts for tension
        'THE_CLIMAX': 1.25,      # Peak intensity
        'THE_FALL': 0.85,        # Slow down for emotional weight
        'LEGACY': 0.90,          # Reflective pace
        'CTA': 1.0,              # Normal
    }

    # Sections that get cinematic letterbox bars (2.39:1 aspect ratio feel)
    LETTERBOX_SECTIONS = {'THE_CLIMAX', 'THE_CONFLICT', 'THE_FALL'}

    # Pattern interrupt interval in seconds (break viewer autopilot)
    PATTERN_INTERRUPT_INTERVAL = 75  # Every ~75 seconds

    def __init__(self, orchestrator: 'SequentialVideoOrchestrator'):
        self.orchestrator = orchestrator

    # Speed ramp multipliers: emotional peaks get slight slow-down for impact,
    # action sections get a subtle speed-up to maintain energy.
    SPEED_RAMP_MAP = {
        'COLD_OPEN': 1.0,
        'EARLY_LIFE': 1.0,
        'THE_SPARK': 1.0,
        'THE_RISE': 1.05,
        'THE_CONFLICT': 1.10,
        'THE_CLIMAX': 0.90,       # Slow down at peak emotional moments
        'THE_FALL': 0.85,         # Slow down for devastating emotional weight
        'LEGACY': 0.95,
        'CTA': 1.0,
    }

    def create_image_clips(self, numbered_images: List[Tuple[int, Path]]) -> List[Dict]:
        """Create animated clips for each image with movement, effects, and timing info.
        
        Uses section metadata (when available) to select appropriate color grading
        and movement styles for each image based on its narrative position.
        Includes brightness normalization and color continuity smoothing when enabled.
        Passes enable_human_feel through to the movement engine for organic motion.
        """
        clips_data = []
        total = len(numbered_images)
        prev_image_array = None  # For color continuity smoothing

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

            # Pre-process image for brightness normalization and color continuity
            if self.orchestrator.enable_brightness_normalization or self.orchestrator.enable_color_continuity:
                proc_img = img.convert('RGB')
                proc_array = np.array(proc_img)

                if self.orchestrator.enable_brightness_normalization:
                    proc_array = self.orchestrator.color_grading.normalize_brightness(proc_array)

                if self.orchestrator.enable_color_continuity and prev_image_array is not None:
                    proc_array = self.orchestrator.color_grading.smooth_color_transition(
                        prev_image_array, proc_array
                    )

                prev_image_array = proc_array.copy()  # Store processed version for consistent comparison

                # Save processed image to temp path for the movement engine
                proc_img = PILImage.fromarray(proc_array)
                import tempfile
                temp_fd, tmp_name = tempfile.mkstemp(suffix=image_path.suffix)
                import os as _os
                _os.close(temp_fd)
                temp_path = Path(tmp_name)
                proc_img.save(str(temp_path))
                effective_image_path = temp_path
            else:
                effective_image_path = image_path

            timing = self.orchestrator.get_timing_for_image(num)
            image_duration = timing.get('duration', self.orchestrator.image_duration)
            start_time = timing.get('start_time')
            end_time = timing.get('end_time')
            section = timing.get('section', '')
            emotional_tone = timing.get('emotional_tone', '')

            # Speed ramp: adjust image duration based on emotional section
            if self.orchestrator.enable_speed_ramp and section:
                ramp = self.SPEED_RAMP_MAP.get(section, 1.0)
                if ramp != 1.0:
                    image_duration = image_duration / ramp  # Slower ramp = longer display
                    if end_time is not None and start_time is not None:
                        end_time = start_time + image_duration
            
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

            # Determine if AI parallax/DOF should be used for this image
            ai_kwargs = {}
            if hasattr(self.orchestrator, 'depth_estimator') and self.orchestrator.depth_estimator is not None:
                ai_kwargs['depth_estimator'] = self.orchestrator.depth_estimator
                ai_kwargs['parallax_engine'] = self.orchestrator.parallax_engine
                ai_kwargs['enable_parallax'] = self.orchestrator.enable_parallax
                ai_kwargs['enable_dof'] = self.orchestrator.enable_dof

            clip = self.orchestrator.movements.create_animated_clip(
                image_path=effective_image_path,
                duration=image_duration,
                movement_type=movement,
                zoom_intensity=self.orchestrator.zoom_intensity,
                color_grader=self.orchestrator.color_grading,
                color_grade=color_grade,
                enable_vignette=self.orchestrator.enable_vignette,
                section=section,
                enable_human_feel=self.orchestrator.enable_human_feel,
                **ai_kwargs,
            )

            # Clean up temp file if we created one
            if effective_image_path != image_path and effective_image_path.exists():
                try:
                    effective_image_path.unlink()
                except OSError:
                    pass

            # Section-aware transition selection (pass actual previous image number for gap handling)
            prev_num = numbered_images[i - 1][0] if i > 0 else None
            transition = self.orchestrator._get_transition_for_image(
                i, total, image_number=num, prev_image_number=prev_num
            )
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

    def _get_fade_duration(self, duration: float, section: str) -> float:
        """Calculate fade duration based on image duration and section context.
        
        Faster cuts for high-tension sections, longer fades for emotional weight.
        """
        if section in ('COLD_OPEN', 'THE_CONFLICT'):
            fade_ratio = 0.10  # Faster cuts for tension
        elif section in ('THE_FALL', 'LEGACY'):
            fade_ratio = 0.25  # Longer fades for emotional weight
        elif section in ('THE_CLIMAX',):
            fade_ratio = 0.20  # Let the peak breathe
        else:
            fade_ratio = 0.15  # Default cinematic fade
        return min(0.8, max(0.3, duration * fade_ratio))

    # Sections where a hard cut (no crossfade) feels more impactful.
    # Human editors often use hard cuts at dramatic entrances.
    HARD_CUT_SECTIONS = {'THE_CONFLICT', 'THE_CLIMAX'}

    # Sections that benefit from extra breathing room (slightly longer crossfade)
    BREATHING_ROOM_SECTIONS = {'THE_FALL', 'LEGACY'}

    def create_timeline_video(self, clips_data: List[Dict]) -> CompositeVideoClip:
        """Create video with clips positioned at their exact start times.
        
        Each clip is extended by a crossfade overlap amount so consecutive clips
        naturally overlap, producing smooth cinematic transitions. Fade durations
        are scaled based on image duration and section context.
        
        Humanization features:
        - Timing micro-variation: +-5-10% random jitter on crossfade durations
        - Hard cuts: dramatic section entrances skip crossfade for impact
        - Breathing room: extra-long fades between intense sections
        - Editing rhythm: subtle pacing variation creates visual heartbeat
        """
        if not clips_data:
            return None

        has_timing = clips_data[0].get('start_time') is not None
        enable_human = getattr(self.orchestrator, 'enable_human_feel', True)

        # Pre-generate deterministic random jitter per clip for reproducible humanization
        seed = sum(d.get('image_num', 0) for d in clips_data)
        rng = np.random.RandomState(seed) if enable_human else None
        
        if has_timing:
            print("Creating timeline-based video with cinematic crossfade overlaps...")
            positioned_clips = []
            prev_section = ''
            
            for i, data in enumerate(clips_data):
                clip = data['clip']
                start_time = data['start_time']
                duration = data['duration']
                section = data.get('section', '')
                
                if start_time is None:
                    continue
                
                # Dynamic pacing: modulate fade based on section pace multiplier
                base_fade = self._get_fade_duration(duration, section)
                if self.orchestrator.enable_dynamic_pacing:
                    pace = self.get_pace_multiplier(section)
                    fade_duration = base_fade / pace  # Faster pace = shorter fades
                else:
                    pace = 1.0
                    fade_duration = base_fade
                fade_duration = min(0.8, max(0.2, fade_duration))
                
                # Pattern interrupt: use faster fade at interval boundaries
                is_interrupt = self.should_pattern_interrupt(start_time)
                if is_interrupt:
                    fade_duration = max(0.15, fade_duration * 0.5)

                # --- Humanization: timing micro-variation ---
                is_hard_cut = False
                if enable_human and rng is not None:
                    # Detect section boundary (entering a new section)
                    is_section_entry = (section != prev_section and prev_section != '')

                    # Hard cut at dramatic section entrances (like human editors do)
                    if (is_section_entry and section in self.HARD_CUT_SECTIONS
                            and i > 0):
                        fade_duration = 0.08  # Near-instant cut
                        is_hard_cut = True

                    # Extra breathing room at emotional transitions
                    elif is_section_entry and section in self.BREATHING_ROOM_SECTIONS:
                        fade_duration *= 1.4  # 40% longer for emotional weight

                    # Timing micro-variation: +-8% random jitter
                    if not is_hard_cut:
                        jitter = rng.uniform(-0.08, 0.08)
                        fade_duration *= (1.0 + jitter)

                    fade_duration = min(0.9, max(0.08, fade_duration))

                prev_section = section
                
                # Extend each clip's duration by the overlap amount so it bleeds
                # into the next clip's territory, creating a true crossfade overlap.
                # The last clip is NOT extended (nothing follows it).
                is_last_clip = (i == len(clips_data) - 1)
                overlap = fade_duration if not is_last_clip else 0
                extended_clip = clip.set_duration(duration + overlap)
                
                positioned_clip = (
                    extended_clip
                    .set_start(start_time)
                    .crossfadein(fade_duration)
                )
                if is_last_clip:
                    positioned_clip = positioned_clip.crossfadeout(fade_duration)
                positioned_clips.append(positioned_clip)
                extras = []
                if section:
                    extras.append(section)
                if is_interrupt:
                    extras.append('INTERRUPT')
                if is_hard_cut:
                    extras.append('HARD CUT')
                extra_str = f" [{' | '.join(extras)}]" if extras else ''
                print(f"  Image {data['image_num']}: starts at {start_time:.2f}s, "
                      f"duration {duration}s (+{overlap:.2f}s overlap), "
                      f"fade {fade_duration:.2f}s, pace {pace:.2f}x{extra_str}")
            
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
        total = len(clips_data)
        for i, data in enumerate(clips_data):
            clip = data['clip']
            duration = data['duration']
            fade_duration = min(self.orchestrator.crossfade_duration * 0.5, duration * 0.15)
            fade_duration = max(0.3, fade_duration)
            clean_clip = clip.crossfadein(fade_duration)
            # Only apply crossfadeout to the last clip to avoid brightness dips
            # at every transition midpoint (alpha compositing: 0.25*out + 0.5*in = 75%)
            if i == total - 1:
                clean_clip = clean_clip.crossfadeout(fade_duration)
            clean_clips.append(clean_clip)

        return concatenate_videoclips(clean_clips, method="compose")

    def get_pace_multiplier(self, section: str) -> float:
        """Get the dynamic pacing multiplier for a section.
        
        Returns a speed factor: >1.0 means faster cuts (shorter display),
        <1.0 means slower pace (longer display). Used to modulate the effective
        crossfade timing to create rhythmic variation that keeps viewers engaged.
        """
        return self.SECTION_PACE.get(section, 1.0)

    def should_pattern_interrupt(self, start_time: float) -> bool:
        """Check if this clip position should trigger a pattern interrupt.
        
        Pattern interrupts break viewer autopilot by inserting a visual change
        every ~75 seconds. Top creators use this to reset viewer attention.
        """
        if start_time <= 0:
            return False
        interval = self.orchestrator.pattern_interrupt_interval
        # Trigger when we cross an interval boundary
        return (int(start_time) % interval) < 8 and start_time > interval * 0.5

    def create_letterbox_overlay(self, duration: float, bar_height_ratio: float = 0.08) -> VideoClip:
        """Create cinematic letterbox bars (top and bottom black bars).
        
        Adds a 2.39:1 widescreen feel for dramatic sections. Bar height is
        ~8% of frame height on each side. Uses a mask clip so only the bar
        regions are opaque while the middle of the frame stays transparent.
        """
        width, height = self.orchestrator.resolution
        bar_height = int(height * bar_height_ratio)

        def make_letterbox_frame(t):
            # Solid black frame (RGB only)
            return np.zeros((height, width, 3), dtype=np.uint8)

        def make_mask_frame(t):
            # Mask: 1.0 for bar regions, 0.0 for middle (transparent)
            mask = np.zeros((height, width), dtype=np.float64)
            mask[:bar_height, :] = 0.85  # Top bar
            mask[height - bar_height:, :] = 0.85  # Bottom bar
            return mask

        letterbox = VideoClip(make_letterbox_frame, duration=duration)
        letterbox = letterbox.set_fps(self.orchestrator.fps)
        mask_clip = VideoClip(make_mask_frame, duration=duration, ismask=True)
        mask_clip = mask_clip.set_fps(self.orchestrator.fps)
        letterbox = letterbox.set_mask(mask_clip)
        return letterbox

    def create_particle_overlay(self, duration: float, intensity: float = 0.3) -> VideoClip:
        """Create a subtle animated dust/particle overlay using vectorized numpy.
        
        Fully vectorized: no Python for-loops per frame. Generates random bright
        spots that simulate floating dust particles in a cinematic light beam.
        """
        width, height = self.orchestrator.resolution
        particle_opacity = intensity * 0.06

        def make_particle_frame(t):
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            num_particles = int(width * height * 0.00008 * intensity)
            if num_particles > 0:
                ys = np.random.randint(1, height - 1, num_particles)
                xs = np.random.randint(1, width - 1, num_particles)
                brightness = np.random.randint(180, 255, num_particles).astype(np.uint8)
                # Vectorized 3x3 particle stamp using advanced indexing
                for dy in range(-1, 2):
                    for dx in range(-1, 2):
                        frame[ys + dy, xs + dx, 0] = brightness
                        frame[ys + dy, xs + dx, 1] = brightness
                        frame[ys + dy, xs + dx, 2] = brightness
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

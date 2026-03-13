"""Utility functions for video creation."""

import json
from pathlib import Path
from typing import Tuple, Optional, Union


def create_sequential_video(
    images_root: Union[str, Path],
    output_path: Union[str, Path] = "sequential_video_output.mp4",
    resolution: Tuple[int, int] = (1920, 1080),
    fps: int = 30,
    image_duration: float = 4.0,
    crossfade_duration: float = 1.2,
    zoom_intensity: float = 1.15,
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
    duration_config_path: Optional[Union[str, Path]] = None
) -> None:
    """Convenience function to create a sequential video from numbered images.
    
    Args:
        images_root: Path to directory containing numbered images (1.png, 2.jpg, etc.)
        output_path: Output video file path
        resolution: Video resolution as (width, height)
        fps: Frames per second
        image_duration: Default duration each image is displayed (seconds)
        crossfade_duration: Duration of transitions between images (seconds)
        zoom_intensity: Ken Burns zoom intensity (1.0 = no zoom, 1.2 = 20% zoom)
        effects_intensity: Overall effects intensity (0.0 to 1.0)
        audio_path: Optional path to audio file
        transition_style: Transition style - 'random', 'sequential', 'cinematic', or specific type
        movement_style: Movement style - 'random', 'sequential', 'dramatic_sequence', or specific type
        color_grade: Color grading style - 'cinematic', 'documentary', 'vintage', etc.
        enable_vignette: Enable vignette effect
        enable_film_grain: Enable film grain overlay
        enable_letterbox: Enable cinematic letterbox bars for dramatic sections
        enable_chapter_cards: Enable chapter title cards at section transitions
        enable_dynamic_pacing: Enable per-section speed variation
        pattern_interrupt_interval: Seconds between pattern interrupts (default 75)
        enable_hook_overlay: Enable dramatic hook text overlay in first 5 seconds
        enable_cta_end_screen: Enable subscribe CTA overlay in last 15 seconds
        enable_brightness_normalization: Normalize image brightness for consistency
        enable_color_continuity: Smooth color transitions between adjacent clips
        enable_speed_ramp: Adjust clip duration based on emotional section intensity
        audio_fade_in: Audio fade-in duration in seconds (0 to disable)
        audio_fade_out: Audio fade-out duration in seconds (0 to disable)
        channel_name: Channel name for CTA end screen subscribe button
        ai_animation_enabled: Master toggle for all AI animation effects (default True)
        enable_parallax: Enable 2.5D depth parallax Ken Burns effect
        enable_dof: Enable depth-of-field cinematic blur
        enable_weather: Enable section-aware weather/atmosphere particles
        duration_config_path: Optional path to JSON file with per-image durations
    """
    from .orchestrator import SequentialVideoOrchestrator
    
    orchestrator = SequentialVideoOrchestrator(
        images_root=images_root,
        output_path=output_path,
        resolution=resolution,
        fps=fps,
        image_duration=image_duration,
        crossfade_duration=crossfade_duration,
        zoom_intensity=zoom_intensity,
        effects_intensity=effects_intensity,
        audio_path=audio_path,
        transition_style=transition_style,
        movement_style=movement_style,
        color_grade=color_grade,
        enable_vignette=enable_vignette,
        enable_film_grain=enable_film_grain,
        enable_letterbox=enable_letterbox,
        enable_chapter_cards=enable_chapter_cards,
        enable_dynamic_pacing=enable_dynamic_pacing,
        pattern_interrupt_interval=pattern_interrupt_interval,
        enable_hook_overlay=enable_hook_overlay,
        enable_cta_end_screen=enable_cta_end_screen,
        enable_brightness_normalization=enable_brightness_normalization,
        enable_color_continuity=enable_color_continuity,
        enable_speed_ramp=enable_speed_ramp,
        audio_fade_in=audio_fade_in,
        audio_fade_out=audio_fade_out,
        channel_name=channel_name,
        ai_animation_enabled=ai_animation_enabled,
        enable_parallax=enable_parallax,
        enable_dof=enable_dof,
        enable_weather=enable_weather,
        duration_config_path=duration_config_path
    )
    orchestrator.create_video()


def load_config_and_create_video(config_path: Union[str, Path]) -> None:
    """Load configuration from JSON and create video."""
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, 'r') as f:
        config = json.load(f)

    config_dir = config_path.parent

    images_root = config_dir / config.get('images_root', 'assets/images')
    output_path = config_dir / config.get('output', 'sequential_video_output.mp4')

    res_str = config.get('res', '1920x1080')
    width, height = map(int, res_str.split('x'))
    resolution = (width, height)

    audio_path = None
    if 'audio' in config and config['audio']:
        audio_path = config_dir / config['audio']

    duration_config_path = None
    if 'duration_config' in config and config['duration_config']:
        duration_config_path = config_dir / config['duration_config']

    create_sequential_video(
        images_root=images_root,
        output_path=output_path,
        resolution=resolution,
        fps=config.get('fps', 30),
        image_duration=config.get('image_duration', 4.0),
        crossfade_duration=config.get('crossfade', 1.2),
        zoom_intensity=config.get('zoom', 1.15),
        effects_intensity=config.get('effects_intensity', 0.7),
        audio_path=audio_path,
        transition_style=config.get('transition_style', 'random'),
        movement_style=config.get('movement_style', 'random'),
        color_grade=config.get('color_grade', 'cinematic'),
        enable_vignette=config.get('enable_vignette', True),
        enable_film_grain=config.get('enable_film_grain', False),
        enable_letterbox=config.get('enable_letterbox', True),
        enable_chapter_cards=config.get('enable_chapter_cards', True),
        enable_dynamic_pacing=config.get('enable_dynamic_pacing', True),
        pattern_interrupt_interval=config.get('pattern_interrupt_interval', 75),
        enable_hook_overlay=config.get('enable_hook_overlay', True),
        enable_cta_end_screen=config.get('enable_cta_end_screen', True),
        enable_brightness_normalization=config.get('enable_brightness_normalization', True),
        enable_color_continuity=config.get('enable_color_continuity', True),
        enable_speed_ramp=config.get('enable_speed_ramp', True),
        audio_fade_in=config.get('audio_fade_in', 1.0),
        audio_fade_out=config.get('audio_fade_out', 2.0),
        channel_name=config.get('channel_name', 'Subscribe'),
        ai_animation_enabled=config.get('ai_animation_enabled', True),
        enable_parallax=config.get('enable_parallax', True),
        enable_dof=config.get('enable_dof', True),
        enable_weather=config.get('enable_weather', True),
        duration_config_path=duration_config_path
    )

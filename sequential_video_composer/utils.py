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
    zoom_intensity: float = 1.08,
    effects_intensity: float = 0.7,
    audio_path: Optional[Union[str, Path]] = None,
    transition_style: str = "random",
    movement_style: str = "random",
    color_grade: str = "cinematic",
    enable_vignette: bool = True,
    enable_film_grain: bool = False,
    enable_text_overlays: bool = True,
    duration_config_path: Optional[Union[str, Path]] = None,
    enable_documentary_effects: bool = True,
    aspect_mode: str = "fill",
    preview_mode: bool = False,
    fast_mode: bool = False,
) -> None:
    """Convenience function to create a sequential video from numbered images.

    Args:
        images_root: Path to directory containing numbered images (1.png, 2.jpg, etc.)
        output_path: Output video file path
        resolution: Video resolution as (width, height)
        fps: Frames per second
        image_duration: Default duration each image is displayed (seconds)
        crossfade_duration: Duration of transitions between images (seconds)
        zoom_intensity: Ken Burns zoom intensity (1.0 = no zoom, 1.08 = 8% zoom)
        effects_intensity: Overall effects intensity (0.0 to 1.0)
        audio_path: Optional path to audio file
        transition_style: Transition style - 'random', 'sequential', 'cinematic', or specific type
        movement_style: Movement style - 'random', 'sequential', 'dramatic_sequence',
            'documentary', or specific type
        color_grade: Color grading style - 'cinematic', 'documentary', 'vintage', etc.
        enable_vignette: Enable vignette effect
        enable_film_grain: Enable film grain overlay
        enable_text_overlays: Enable stylish text overlay rendering from duration config
        duration_config_path: Optional path to JSON file with per-image durations
        enable_documentary_effects: Enable section-aware documentary effects
            (light leaks, film grain, dust particles, camera shake, etc.)
        aspect_mode: How to fit images — 'fill' (crop), 'fit' (scale), 'letterbox' (black bars)
        preview_mode: If True, render at 480p with faster settings for quick preview
        fast_mode: If True, skip per-frame sharpening, use faster resize + encode settings
    """
    from .orchestrator import SequentialVideoOrchestrator

    if preview_mode:
        resolution = (854, 480)
        fps = 24

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
        enable_text_overlays=enable_text_overlays,
        duration_config_path=duration_config_path,
        enable_documentary_effects=enable_documentary_effects,
    )
    orchestrator.aspect_mode = aspect_mode
    orchestrator.preview_mode = preview_mode
    orchestrator.fast_mode = fast_mode
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
        zoom_intensity=config.get('zoom', 1.08),
        effects_intensity=config.get('effects_intensity', 0.7),
        audio_path=audio_path,
        transition_style=config.get('transition_style', 'random'),
        movement_style=config.get('movement_style', 'random'),
        color_grade=config.get('color_grade', 'cinematic'),
        enable_vignette=config.get('enable_vignette', True),
        enable_film_grain=config.get('enable_film_grain', False),
        enable_text_overlays=config.get('enable_text_overlays', True),
        duration_config_path=duration_config_path,
        enable_documentary_effects=config.get('enable_documentary_effects', True),
    )

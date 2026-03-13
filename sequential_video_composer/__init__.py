"""Sequential Video Composer - Creates videos from numbered images with animations."""

from .orchestrator import SequentialVideoOrchestrator
from .transitions import TransitionEffects
from .movements import MovementStyles
from .color_grading import ColorGrading
from .text_overlays import TextOverlayEngine
from .ai_effects import (
    DepthEstimator,
    ParallaxEngine,
    DepthOfFieldEffect,
    SubjectDetector,
    WeatherEffects,
    get_ai_status,
)
from .sound_design import SoundDesignEngine
from .utils import create_sequential_video, load_config_and_create_video

__version__ = "2.0.0"
__author__ = "Sequential Video Systems"

__all__ = [
    'SequentialVideoOrchestrator',
    'TransitionEffects',
    'MovementStyles',
    'ColorGrading',
    'TextOverlayEngine',
    'DepthEstimator',
    'ParallaxEngine',
    'DepthOfFieldEffect',
    'SubjectDetector',
    'WeatherEffects',
    'get_ai_status',
    'SoundDesignEngine',
    'create_sequential_video',
    'load_config_and_create_video'
]

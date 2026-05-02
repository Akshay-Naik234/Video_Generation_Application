"""Sequential Video Composer - Creates videos from numbered images with animations.

Enhanced with documentary-style animations, section-aware effects,
and cinematic transitions.
"""

from .orchestrator import SequentialVideoOrchestrator
from .transitions import TransitionEffects
from .movements import MovementStyles
from .color_grading import ColorGrading
from .text_overlays import TextOverlayEngine
from .effects import DocumentaryEffects
from .utils import create_sequential_video, load_config_and_create_video

__version__ = "2.0.0"
__author__ = "Sequential Video Systems"

__all__ = [
    'SequentialVideoOrchestrator',
    'TransitionEffects',
    'MovementStyles',
    'ColorGrading',
    'TextOverlayEngine',
    'DocumentaryEffects',
    'create_sequential_video',
    'load_config_and_create_video'
]

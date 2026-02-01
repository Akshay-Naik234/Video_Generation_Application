"""Sequential Video Orchestrator - Backward compatibility module.

This module re-exports all classes and functions from the refactored modules
for backward compatibility with existing code.
"""

from .orchestrator import SequentialVideoOrchestrator
from .transitions import TransitionEffects
from .movements import MovementStyles
from .color_grading import ColorGrading
from .utils import create_sequential_video, load_config_and_create_video

__all__ = [
    'SequentialVideoOrchestrator',
    'TransitionEffects',
    'MovementStyles',
    'ColorGrading',
    'create_sequential_video',
    'load_config_and_create_video'
]


if __name__ == "__main__":
    import sys
    from pathlib import Path

    if len(sys.argv) > 1:
        config_file = sys.argv[1]
        load_config_and_create_video(config_file)
    else:
        default_config = Path(__file__).parent / "video_config.json"
        if default_config.exists():
            load_config_and_create_video(default_config)
        else:
            print("Usage: python sequential_video_orchestrator.py [config.json]")
            print("Or place a video_config.json in the same directory")

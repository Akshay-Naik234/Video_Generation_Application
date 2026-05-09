"""Command-line interface for Sequential Video Composer."""

import argparse
import logging
import sys
from pathlib import Path

from .sequential_video_orchestrator import (
    create_sequential_video,
    load_config_and_create_video,
    TransitionEffects,
    MovementStyles,
    ColorGrading
)

logger = logging.getLogger(__name__)


def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        description="Create videos from sequentially numbered images with professional animations.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Using a config file
  python -m sequential_video_composer --config video_config.json

  # Using command-line arguments
  python -m sequential_video_composer --images ./assets/images --output video.mp4

  # With custom settings
  python -m sequential_video_composer --images ./images --output video.mp4 --duration 5 --zoom 1.2

  # With specific transition and movement styles
  python -m sequential_video_composer --images ./images --transition cinematic --movement dramatic_sequence

  # With random variety
  python -m sequential_video_composer --images ./images --transition random --movement random

Image Naming Convention:
  Images should be named with a numeric prefix followed by an extension:
  1.png, 2.jpg, 3.jpeg, 10.png, etc.
  
  The images will be processed in numeric order (1, 2, 3, ..., 10, 11, ...)

Transition Styles:
  random, sequential, cinematic, crossfade, slide_left, slide_right, slide_up, slide_down,
  zoom_in, zoom_out, fade_through_black, fade_through_white, wipe_left, wipe_right,
  whip_pan, light_leak, film_burn, glitch, blur_through, flash_cut, iris_wipe,
  morph_dissolve, zoom_blur, push_left, push_right, luma_fade, fade_through_warm,
  ink_splash, parallax_slide, zoom_dissolve, color_sweep

Movement Styles:
  random, sequential, dramatic_sequence, documentary,
  zoom_in, zoom_out, pan_left, pan_right, pan_up, pan_down,
  diagonal_tl_br, diagonal_tr_bl, breathing, dramatic_zoom, gentle_drift, focus_center,
  parallax_depth, push_in, push_out, orbit, whip_pan, dolly_zoom, handheld_drift,
  crane_up, crane_down, spiral_zoom, tilt_shift, dutch_tilt, rack_focus, bounce_zoom,
  float_up, reveal_left, reveal_right, map_zoom, map_pan, timeline_reveal

Color Grades:
  cinematic, documentary, vintage, modern, warm, cool, high_contrast, soft, dramatic,
  natural, teal_orange, noir, golden_hour
        """
    )

    parser.add_argument(
        '--config', '-c',
        type=str,
        help='Path to JSON configuration file'
    )

    parser.add_argument(
        '--images', '-i',
        type=str,
        help='Path to directory containing numbered images'
    )

    parser.add_argument(
        '--output', '-o',
        type=str,
        default='sequential_video_output.mp4',
        help='Output video file path (default: sequential_video_output.mp4)'
    )

    parser.add_argument(
        '--resolution', '-r',
        type=str,
        default='1920x1080',
        help='Video resolution in WIDTHxHEIGHT format (default: 1920x1080)'
    )

    parser.add_argument(
        '--fps',
        type=int,
        default=30,
        help='Frames per second (default: 30)'
    )

    parser.add_argument(
        '--duration', '-d',
        type=float,
        default=4.0,
        help='Duration per image in seconds (default: 4.0)'
    )

    parser.add_argument(
        '--crossfade',
        type=float,
        default=1.2,
        help='Crossfade transition duration in seconds (default: 1.2)'
    )

    parser.add_argument(
        '--zoom', '-z',
        type=float,
        default=1.15,
        help='Ken Burns zoom intensity (default: 1.15)'
    )

    parser.add_argument(
        '--effects',
        type=float,
        default=0.7,
        help='Effects intensity from 0.0 to 1.0 (default: 0.7)'
    )

    parser.add_argument(
        '--audio', '-a',
        type=str,
        help='Path to audio file to add to the video'
    )

    parser.add_argument(
        '--transition', '-t',
        type=str,
        default='random',
        help='Transition style: random, sequential, cinematic, or specific type (default: random)'
    )

    parser.add_argument(
        '--movement', '-m',
        type=str,
        default='random',
        help='Movement style: random, sequential, dramatic_sequence, documentary, or specific type (default: random)'
    )

    parser.add_argument(
        '--color-grade', '-g',
        type=str,
        default='cinematic',
        help='Color grading style: cinematic, documentary, vintage, etc. (default: cinematic)'
    )

    parser.add_argument(
        '--vignette',
        action='store_true',
        default=True,
        help='Enable vignette effect (default: enabled)'
    )

    parser.add_argument(
        '--no-vignette',
        action='store_true',
        help='Disable vignette effect'
    )

    parser.add_argument(
        '--film-grain',
        action='store_true',
        help='Enable film grain overlay effect'
    )

    parser.add_argument(
        '--duration-config',
        type=str,
        help='Path to JSON file with per-image durations (overrides --duration)'
    )

    parser.add_argument(
        '--text-overlays',
        action='store_true',
        default=True,
        help='Enable text overlay rendering from duration config (default: enabled)'
    )

    parser.add_argument(
        '--no-text-overlays',
        action='store_true',
        help='Disable text overlay rendering'
    )

    parser.add_argument(
        '--documentary-effects',
        action='store_true',
        default=True,
        help='Enable documentary effects (light leaks, film grain, dust, etc.) (default: enabled)'
    )

    parser.add_argument(
        '--no-documentary-effects',
        action='store_true',
        help='Disable documentary effects'
    )

    parser.add_argument(
        '--preview',
        action='store_true',
        help='Quick preview render at 480p with CRF 28 (much faster)'
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Validate config and images without rendering'
    )

    parser.add_argument(
        '--render-frame',
        type=int,
        default=None,
        metavar='N',
        help='Render only image N as a single PNG frame for debugging'
    )

    parser.add_argument(
        '--aspect-mode',
        type=str,
        choices=['fill', 'fit', 'letterbox'],
        default='fill',
        help='How to fit images: fill (crop to fill), fit (scale down), letterbox (black bars) (default: fill)'
    )

    parser.add_argument(
        '-v', '--verbose',
        action='count',
        default=0,
        help='Increase logging verbosity (-v for INFO, -vv for DEBUG)'
    )

    args = parser.parse_args()

    # Configure logging based on verbosity
    log_level = logging.WARNING
    if args.verbose >= 2:
        log_level = logging.DEBUG
    elif args.verbose >= 1:
        log_level = logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(levelname)s [%(name)s] %(message)s',
    )

    if args.config:
        config_path = Path(args.config)
        if not config_path.exists():
            print(f"Error: Configuration file not found: {config_path}")
            sys.exit(1)

        print(f"Loading configuration from: {config_path}")
        load_config_and_create_video(config_path)

    elif args.images:
        images_path = Path(args.images)
        if not images_path.exists():
            print(f"Error: Images directory not found: {images_path}")
            sys.exit(1)

        try:
            width, height = map(int, args.resolution.split('x'))
            resolution = (width, height)
        except ValueError:
            print(f"Error: Invalid resolution format: {args.resolution}")
            print("Expected format: WIDTHxHEIGHT (e.g., 1920x1080)")
            sys.exit(1)

        audio_path = Path(args.audio) if args.audio else None
        if audio_path and not audio_path.exists():
            print(f"Warning: Audio file not found: {audio_path}")
            audio_path = None

        duration_config_path = Path(args.duration_config) if args.duration_config else None
        if duration_config_path and not duration_config_path.exists():
            print(f"Warning: Duration config file not found: {duration_config_path}")
            duration_config_path = None

        enable_vignette = args.vignette and not args.no_vignette
        enable_text_overlays = args.text_overlays and not args.no_text_overlays
        enable_documentary_effects = args.documentary_effects and not args.no_documentary_effects

        # Dry-run: validate without rendering
        if args.dry_run:
            from .orchestrator import SequentialVideoOrchestrator
            orch = SequentialVideoOrchestrator(
                images_root=images_path,
                output_path=args.output,
                resolution=resolution,
                fps=args.fps,
                image_duration=args.duration,
                duration_config_path=duration_config_path,
            )
            images = orch.discover_numbered_images()
            print(f"Dry-run OK: {len(images)} images found, config valid.")
            return

        # Render single frame for debugging
        if args.render_frame is not None:
            from .orchestrator import SequentialVideoOrchestrator
            orch = SequentialVideoOrchestrator(
                images_root=images_path,
                output_path=args.output,
                resolution=resolution,
                fps=args.fps,
                image_duration=args.duration,
                duration_config_path=duration_config_path,
            )
            orch.aspect_mode = args.aspect_mode
            images = orch.discover_numbered_images()
            target = args.render_frame
            match = [p for n, p in images if n == target]
            if not match:
                print(f"Error: Image {target} not found among {len(images)} images")
                sys.exit(1)
            clip = orch.movements.create_animated_clip(
                match[0], args.duration, args.zoom,
                movement_type='zoom_in', section='EARLY_LIFE',
            )
            frame = clip.get_frame(0)
            from PIL import Image as PILImage
            out_path = Path(args.output).with_suffix('.png')
            PILImage.fromarray(frame).save(str(out_path))
            print(f"Frame saved to: {out_path}")
            return

        if args.preview:
            print("Preview mode: rendering at 480p with fast settings")

        print(f"Creating video from images in: {images_path}")
        print(f"  Transition style: {args.transition}")
        print(f"  Movement style: {args.movement}")
        print(f"  Color grade: {args.color_grade}")
        print(f"  Text overlays: {enable_text_overlays}")
        print(f"  Documentary effects: {enable_documentary_effects}")
        if duration_config_path:
            print(f"  Duration config: {duration_config_path}")

        create_sequential_video(
            images_root=images_path,
            output_path=args.output,
            resolution=resolution,
            fps=args.fps,
            image_duration=args.duration,
            crossfade_duration=args.crossfade,
            zoom_intensity=args.zoom,
            effects_intensity=args.effects,
            audio_path=audio_path,
            transition_style=args.transition,
            movement_style=args.movement,
            color_grade=args.color_grade,
            enable_vignette=enable_vignette,
            enable_film_grain=args.film_grain,
            enable_text_overlays=enable_text_overlays,
            duration_config_path=duration_config_path,
            enable_documentary_effects=enable_documentary_effects,
            aspect_mode=args.aspect_mode,
            preview_mode=args.preview,
        )

    else:
        project_root = Path(__file__).parent.parent
        default_config = project_root / "examples" / "input" / "video_config.json"
        if default_config.exists():
            print(f"Using default configuration: {default_config}")
            load_config_and_create_video(default_config)
        else:
            parser.print_help()
            print("\nError: Please provide either --config or --images argument")
            sys.exit(1)


if __name__ == "__main__":
    main()

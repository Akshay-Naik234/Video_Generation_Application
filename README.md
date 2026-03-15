# Video Generation Application

A professional biography video generation pipeline that creates cinematic, Netflix/HBO-grade videos from numbered images and narration audio. Built for the [LifeSpark Chronicles](https://www.youtube.com/@LifeSparkChronicles2348) YouTube channel.

## Table of Contents

- [Features](#features)
- [Prerequisites](#prerequisites)
- [Project Setup](#project-setup)
- [Project Structure](#project-structure)
- [Complete Workflow](#complete-workflow)
- [Running the Video Generator](#running-the-video-generator)
- [Configuration Reference](#configuration-reference)
- [JSON Format Reference](#json-format-reference)
- [Prompts Guide](#prompts-guide)
- [Troubleshooting](#troubleshooting)

---

## Features

- **Section-aware processing** - Automatically selects movements, color grading, transitions, and sound effects based on the narrative section (COLD_OPEN, EARLY_LIFE, THE_SPARK, THE_RISE, THE_CONFLICT, THE_CLIMAX, THE_FALL, LEGACY, CTA)
- **AI-enhanced animation** - MiDaS depth estimation, 2.5D parallax Ken Burns, depth-of-field blur, weather overlays (all free/open-source)
- **Human-feel editing** - Camera breathing, micro-shake, organic easing curves, timing jitter, hard cuts at dramatic moments
- **Text overlays** - Year stamps, location stamps, lower thirds, quote cards, animated counters, info cards, chapter cards
- **Sound design** - Programmatic whooshes, risers, bass drops, ambient pads at section transitions (no external audio files needed)
- **13 text overlay types**, **8 movement types**, **10 color grades**, **14 transition types**

---

## Prerequisites

Before setting up the project, make sure you have the following installed on your computer:

### Required Software

| Software | Version | Purpose | Install Link |
|----------|---------|---------|-------------|
| **Python** | 3.9 or higher | Runtime | [python.org/downloads](https://www.python.org/downloads/) |
| **FFmpeg** | 4.0 or higher | Video encoding/audio processing | See below |
| **Git** | Any recent version | Clone the repository | [git-scm.com](https://git-scm.com/) |

### Installing FFmpeg

**Windows:**
```bash
# Using winget (Windows 10/11)
winget install FFmpeg

# Or download from https://ffmpeg.org/download.html
# Extract and add the bin/ folder to your system PATH
```

**macOS:**
```bash
brew install ffmpeg
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt update
sudo apt install ffmpeg
```

Verify FFmpeg is installed:
```bash
ffmpeg -version
```

### Verify Python Version
```bash
python --version
# Should show Python 3.9 or higher
```

---

## Project Setup

### Step 1: Clone the Repository

```bash
git clone https://github.com/Akshay-Naik234/Video_Generation_Application.git
cd Video_Generation_Application
```

### Step 2: Create a Virtual Environment

**Windows:**
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
venv\Scripts\activate
```

**macOS / Linux:**
```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate
```

You should see `(venv)` at the beginning of your terminal prompt, confirming the virtual environment is active.

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

This installs:
- **moviepy** (1.0.3) - Video compositing engine
- **numpy** - Array operations and sound synthesis
- **Pillow** - Image processing
- **tqdm** - Progress bars
- **PyTorch CPU** (~200-300 MB) - AI depth estimation (MiDaS)
- **torchvision** - Image transforms for MiDaS
- **opencv-python-headless** - Optional, better image processing

### Step 4: Install Whisper (for timestamp extraction)

```bash
pip install openai-whisper
```

### Step 5: Verify Installation

```bash
python -c "from sequential_video_composer import SequentialVideoOrchestrator; print('Setup OK')"
```

If this prints `Setup OK`, the installation is complete.

---

## Project Structure

```
Video_Generation_Application/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── running_instructions.txt           # Quick reference commands
│
├── assets/
│   └── images/                        # Place your numbered images here (1.png, 2.png, ...)
│
├── examples/
│   ├── audio/
│   │   └── narration.mp3              # Place your narration audio here
│   ├── input/
│   │   ├── video_config.json          # Example config file
│   │   └── image_display_duration.json # Example duration config
│   ├── input_voices/                  # Individual voice segment MP3s (for audio_merger.py)
│   └── output_voice/                  # Merged audio output directory
│
├── prompts/                           # ChatGPT/AI prompts for content generation
│   ├── Devin_Biography_MasterPrompt_USA.txt         # Script generation prompt
│   ├── Devin_Image_Prompt_Generation_Master_Prompt.txt  # Image prompt + duration config generation
│   ├── YouTube_Biography_Script_Generator.txt        # Original script prompt
│   ├── Thumbnail_Creation.txt                        # Thumbnail prompt
│   ├── Youtube_MetaData_Creation_Prompt.txt          # Title/description/tags prompt
│   └── ...                            # Other prompt variants
│
├── image_display_duration.json        # Your duration config goes here (generated by ChatGPT)
│
├── audio_merger.py                    # Merge multiple audio segments into one narration file
├── timestamp_extraction.py            # Extract word-level timestamps from narration using Whisper
│
└── sequential_video_composer/         # Main video generation engine
    ├── __init__.py
    ├── __main__.py                    # Entry point for python -m
    ├── cli.py                         # Command-line interface
    ├── orchestrator.py                # Main orchestration (creates the final video)
    ├── clip_factory.py                # Creates and composes video clips
    ├── movements.py                   # Ken Burns camera movements (8 types)
    ├── transitions.py                 # Transition effects (14 types)
    ├── color_grading.py               # Color grading (10 styles)
    ├── text_overlays.py               # Text overlay engine (13 types)
    ├── ai_effects.py                  # AI depth estimation, parallax, DOF, weather
    ├── sound_design.py                # Programmatic sound effects
    ├── utils.py                       # Utility/convenience functions
    ├── video_config.json              # Default configuration file
    └── sequential_video_orchestrator.py # Backward compatibility module
```

---

## Complete Workflow

Here is the end-to-end process for creating a biography video:

### Step 1: Write the Script

1. Open ChatGPT (GPT-4 recommended)
2. Paste the contents of `prompts/Devin_Biography_MasterPrompt_USA.txt`
3. Provide the person's name and any specific details
4. ChatGPT generates a full biography script with 9 narrative sections

### Step 2: Generate Narration Audio

1. Use a text-to-speech service (e.g., ElevenLabs, Google TTS, or any TTS tool) to convert the script to audio
2. If you have multiple audio segments, place them in `examples/input_voices/` and run:
   ```bash
   python audio_merger.py
   ```
   This merges them into a single `examples/output_voice/narration.mp3`
3. Copy the final narration file to `examples/audio/narration.mp3`

### Step 3: Extract Timestamps

```bash
python timestamp_extraction.py large
```

This uses OpenAI Whisper to extract word-level timestamps from `examples/audio/narration.mp3` and saves them as `examples/audio/narration_timestamps.json`.

- Model sizes: `tiny`, `base`, `small`, `medium`, `large`
- `large` gives the best accuracy but takes more time and memory

### Step 4: Generate Image Prompts + Duration Config

1. Open ChatGPT
2. Paste the contents of `prompts/Devin_Image_Prompt_Generation_Master_Prompt.txt`
3. Provide:
   - The person's name
   - The full script (from Step 1)
   - The narration timestamps JSON (from Step 3)
4. ChatGPT generates a JSON file containing:
   - DALL-E 3 image prompts for each scene
   - Precise timing (start_time, duration, end_time)
   - Section metadata (section, emotional_tone, shot_type, color_temperature)
   - Overlay text (dates, locations, names, quotes)
5. Save this JSON as `image_display_duration.json` in the project root

### Step 5: Generate Images

1. Use DALL-E 3 (or any image generator) to create images from the prompts in the JSON
2. Name each image with its number: `1.png`, `2.png`, `3.png`, ..., `101.png`
3. Place all images in `assets/images/`

### Step 6: Generate the Video

Make sure your virtual environment is activated, then run:

```bash
python -m sequential_video_composer --config sequential_video_composer/video_config.json
```

The video will be saved to `sequential_video_output.mp4` (or whatever path is set in the config).

---

## Running the Video Generator

### Method 1: Using Config File (Recommended)

```bash
# Activate virtual environment first
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Run with the default config
python -m sequential_video_composer --config sequential_video_composer/video_config.json

# Or with the example config
python -m sequential_video_composer --config examples/input/video_config.json
```

### Method 2: Using Command-Line Arguments

```bash
python -m sequential_video_composer \
  --images assets/images \
  --output my_video.mp4 \
  --audio examples/audio/narration.mp3 \
  --duration-config image_display_duration.json \
  --transition cinematic \
  --movement documentary \
  --color-grade cinematic \
  --zoom 1.12 \
  --duration 7 \
  --crossfade 0.8 \
  --film-grain
```

### Method 3: Using Python Script

```python
from sequential_video_composer import load_config_and_create_video

load_config_and_create_video("sequential_video_composer/video_config.json")
```

Or with direct parameters:

```python
from sequential_video_composer import create_sequential_video

create_sequential_video(
    images_root="assets/images",
    output_path="my_video.mp4",
    audio_path="examples/audio/narration.mp3",
    duration_config_path="image_display_duration.json",
    resolution=(1920, 1080),
    fps=30,
    transition_style="cinematic",
    movement_style="documentary",
    color_grade="cinematic",
    ai_animation_enabled=True,
    enable_human_feel=True,
    enable_sound_design=True,
)
```

### Deactivating the Virtual Environment

When you are done, deactivate the virtual environment:

```bash
deactivate
```

---

## Configuration Reference

The `video_config.json` file controls all settings. Here is every option:

```json
{
  "images_root": "../assets/images",
  "output": "sequential_video_output.mp4",
  "res": "1920x1080",
  "fps": 30,
  "image_duration": 7.0,
  "crossfade": 0.8,
  "zoom": 1.12,
  "effects_intensity": 0.8,
  "transition_style": "cinematic",
  "movement_style": "documentary",
  "color_grade": "cinematic",
  "enable_vignette": true,
  "enable_film_grain": true,
  "enable_letterbox": true,
  "enable_chapter_cards": true,
  "enable_dynamic_pacing": true,
  "pattern_interrupt_interval": 75,
  "enable_hook_overlay": true,
  "enable_cta_end_screen": true,
  "enable_brightness_normalization": true,
  "enable_color_continuity": true,
  "enable_speed_ramp": true,
  "audio_fade_in": 1.0,
  "audio_fade_out": 2.0,
  "channel_name": "Subscribe",
  "ai_animation_enabled": true,
  "enable_parallax": true,
  "enable_dof": true,
  "enable_weather": true,
  "enable_human_feel": true,
  "enable_sound_design": true,
  "sound_design_intensity": 0.08,
  "enable_pytorch_depth": true,
  "audio": "../examples/audio/narration.mp3",
  "duration_config": "../image_display_duration.json"
}
```

### Config Options Explained

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `images_root` | string | - | Path to folder containing numbered images (1.png, 2.jpg, etc.) |
| `output` | string | `"sequential_video_output.mp4"` | Output video file path |
| `res` | string | `"1920x1080"` | Video resolution (WIDTHxHEIGHT) |
| `fps` | int | `30` | Frames per second |
| `image_duration` | float | `7.0` | Default duration per image (seconds). Overridden by duration_config. |
| `crossfade` | float | `0.8` | Crossfade transition duration (seconds) |
| `zoom` | float | `1.12` | Ken Burns zoom intensity (1.0 = no zoom) |
| `effects_intensity` | float | `0.8` | Overall visual effects intensity (0.0 - 1.0) |
| `transition_style` | string | `"cinematic"` | `random`, `sequential`, `cinematic`, or specific type |
| `movement_style` | string | `"documentary"` | `random`, `sequential`, `dramatic_sequence`, `documentary`, or specific type |
| `color_grade` | string | `"cinematic"` | `cinematic`, `documentary`, `vintage`, `warm`, `cool`, `high_contrast`, `soft`, `dramatic`, `natural`, `modern` |
| `enable_vignette` | bool | `true` | Subtle darkening at edges |
| `enable_film_grain` | bool | `true` | Film grain noise overlay |
| `enable_letterbox` | bool | `true` | Cinematic 2.39:1 bars during dramatic sections |
| `enable_chapter_cards` | bool | `true` | Section title cards at narrative transitions |
| `enable_dynamic_pacing` | bool | `true` | Per-section crossfade speed variation |
| `pattern_interrupt_interval` | int | `75` | Seconds between pattern interrupt cuts |
| `enable_hook_overlay` | bool | `true` | Dramatic text in first 5 seconds |
| `enable_cta_end_screen` | bool | `true` | Subscribe prompt in last 15 seconds |
| `enable_brightness_normalization` | bool | `true` | Prevent jarring brightness jumps |
| `enable_color_continuity` | bool | `true` | Smooth color transitions between clips |
| `enable_speed_ramp` | bool | `true` | Section-based animation speed variation |
| `audio_fade_in` | float | `1.0` | Audio fade-in duration (0 to disable) |
| `audio_fade_out` | float | `2.0` | Audio fade-out duration (0 to disable) |
| `channel_name` | string | `"Subscribe"` | Text on CTA end screen button |
| `ai_animation_enabled` | bool | `true` | Master toggle for all AI effects |
| `enable_parallax` | bool | `true` | 2.5D depth-based parallax movement |
| `enable_dof` | bool | `true` | Depth-of-field blur effect |
| `enable_weather` | bool | `true` | Atmospheric particle overlays (rain, dust, embers) |
| `enable_human_feel` | bool | `true` | Human-feel editing (breathing, shake, varied easing) |
| `enable_sound_design` | bool | `true` | Programmatic sound effects at transitions |
| `sound_design_intensity` | float | `0.08` | Sound effects volume (0.0 - 1.0). 0.08 = ~-22 dB |
| `enable_pytorch_depth` | bool | `true` | Use PyTorch MiDaS for better depth estimation |
| `audio` | string | - | Path to narration audio file |
| `duration_config` | string | - | Path to `image_display_duration.json` |

> **Note:** All paths in `video_config.json` are relative to the config file's location.

---

## JSON Format Reference

The `image_display_duration.json` file (generated by ChatGPT using the Image Prompt Generation prompt) has this structure:

```json
{
  "video_metadata": {
    "person_name": "Jack Parsons",
    "title": "The Sorcerer Who Built NASA: Jack Parsons",
    "total_duration_seconds": 694.38,
    "total_images": 101,
    "timing_mode": "CONTENT_MATCHED_TIMESTAMP_LOCKED"
  },
  "thumbnail": {
    "prompt": "...",
    "overlay_text": "THE SORCERER OF NASA",
    "text_position": "left",
    "face_percentage": 65,
    "color_scheme": "Deep Red (#8B0000) + Black (#000000)",
    "face_expression": "intense brooding stare"
  },
  "images": [
    {
      "image": 1,
      "start_time": 0.0,
      "duration": 5.43,
      "end_time": 5.43,
      "section": "COLD_OPEN",
      "emotional_tone": "tension",
      "shot_type": "detail",
      "color_temperature": "cool",
      "overlay_text": "June 17, 1952 | Pasadena, California",
      "prompt": "Extreme close-up of a home laboratory workbench..."
    }
  ]
}
```

### Fields Used by the Video Composer

| Field | Required | How It's Used |
|-------|----------|---------------|
| `image` | Yes | Matches image files (1.png, 2.jpg, etc.) |
| `start_time` | Yes | When to show this image in the video (seconds) |
| `duration` | Yes | How long to display the image (seconds) |
| `end_time` | No | Reference only (start_time + duration) |
| `section` | No | Drives movement, color grading, transitions, sound, weather, letterbox |
| `emotional_tone` | No | Overrides color grading and weather type |
| `shot_type` | No | Influences movement selection (detail->push_in, wide->pull_out, etc.) |
| `color_temperature` | No | Stored for reference (section/emotion maps take priority) |
| `overlay_text` | No | Renders text overlays (see patterns below) |
| `prompt` | No | Not used by video composer (for DALL-E 3 image generation only) |

### Overlay Text Patterns

The video composer auto-detects the overlay type from the text format:

| Pattern | Example | Overlay Type |
|---------|---------|-------------|
| `YEAR \| LOCATION` | `"1952 \| Pasadena, California"` | Date-location combo stamp |
| `NAME -- TITLE` | `"Jack Parsons -- Co-Founder, JPL"` | Slide-in lower third |
| `4-digit year` | `"1884"` | Year stamp (top-right) |
| `YEAR, LOCATION` | `"1943, New York"` | Year stamp with label |
| Location keyword | `"New York City"` | Location stamp (bottom-left) |
| `"Quote" -- Attribution` | `"Imagination..." -- Einstein` | Centered quote card |
| `$NUMBER` | `"$2,300,000"` | Animated counter |
| Anything else | `"Born: July 10, 1856"` | Info card (bottom-right) |

### Sections

| Section | Color Grade | Movement Style | Sound Effects |
|---------|------------|----------------|---------------|
| `COLD_OPEN` | cool | push_in | whoosh |
| `EARLY_LIFE` | warm/vintage | float_drift, gentle | ambient pad |
| `THE_SPARK` | documentary | zoom_in | riser |
| `THE_RISE` | cinematic | pan, zoom_in | whoosh |
| `THE_CONFLICT` | high_contrast | dramatic, zoom_pulse | bass drop + tick |
| `THE_CLIMAX` | dramatic | zoom_pulse, dramatic | bass drop + riser |
| `THE_FALL` | soft | gentle, float_drift | ambient pad |
| `LEGACY` | warm | float_drift, gentle | ambient pad |
| `CTA` | natural | zoom_out | whoosh |

---

## Prompts Guide

| Prompt File | When to Use | What It Does |
|-------------|-------------|-------------|
| `Devin_Biography_MasterPrompt_USA.txt` | Step 1 | Generates the full biography script with 9 sections |
| `Devin_Image_Prompt_Generation_Master_Prompt.txt` | Step 4 | Generates image prompts + timing + section metadata + overlay text |
| `Thumbnail_Creation.txt` | After video | Generates thumbnail image prompt |
| `Youtube_MetaData_Creation_Prompt.txt` | After video | Generates title, description, tags for YouTube |
| `YouTube_Script_Cleaner.txt` | Optional | Cleans up/refines the script |

---

## Troubleshooting

### "No numbered images found"
Images must be named with a number prefix: `1.png`, `2.jpg`, `3.jpeg`, etc. The number must be at the start of the filename.

### "Duration config file not found"
Check that the `duration_config` path in `video_config.json` correctly points to your `image_display_duration.json`. Paths are relative to the config file location.

### PyTorch installation fails
If PyTorch CPU-only fails to install, you can skip it. The AI depth estimation will fall back to a numpy heuristic (lower quality but functional):
```bash
pip install moviepy==1.0.3 numpy Pillow tqdm opencv-python-headless
```

### Video is too large
- Reduce `fps` from 30 to 24
- Reduce `res` from `"1920x1080"` to `"1280x720"`
- The export uses CRF 18 (high quality). For smaller files, this can be adjusted in the code.

### Sound effects are too loud/quiet
Adjust `sound_design_intensity` in the config:
- `0.04` = quieter (~-28 dB)
- `0.08` = default (~-22 dB)
- `0.15` = louder (~-16 dB)

### "ModuleNotFoundError: No module named 'moviepy'"
Make sure your virtual environment is activated:
```bash
# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate

# Then run again
python -m sequential_video_composer --config sequential_video_composer/video_config.json
```

### FFmpeg not found
MoviePy requires FFmpeg. Install it (see [Prerequisites](#prerequisites)) and make sure it's in your system PATH.

### Font not found warning
Text overlays use `/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf`. On Windows/macOS, a fallback font is used automatically (text may appear smaller). To fix, install DejaVu Sans font.

---

## Quick Reference

```bash
# Activate virtual environment
source venv/bin/activate          # macOS/Linux
venv\Scripts\activate             # Windows

# Merge audio segments
python audio_merger.py

# Extract timestamps
python timestamp_extraction.py large

# Generate video
python -m sequential_video_composer --config sequential_video_composer/video_config.json

# Deactivate when done
deactivate
```

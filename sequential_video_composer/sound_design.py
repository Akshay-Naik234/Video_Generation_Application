"""Programmatic sound design system for documentary-style videos.

Maps visual events (transitions, effects, section changes) to sound effects
and ambient atmosphere, creating the subliminal audio-visual sync that makes
content feel premium.

Features:
- Transition sound effects (whooshes, impacts, camera sounds)
- Section-aware ambient atmosphere (rain, crowd, nature, city)
- Dynamic audio ducking for narration/music mixing
- Foley sound mapping for visual effects
- Volume envelope system with section-aware intensity

Usage:
    sound = SoundDesignEngine(resolution=(1920, 1080))
    # Get sound effects for a transition
    sounds = sound.get_transition_sounds('whip_pan', start_time=5.0)
    # Get ambient atmosphere for a section
    atmos = sound.get_section_atmosphere('THE_CONFLICT', duration=30.0)
    # Mix all sound layers
    final_audio = sound.mix_audio_layers(layers, narration_path)
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class SoundDesignEngine:
    """Maps visual events to programmatic sound effects and atmosphere."""

    # Transition type → sound effect mapping
    TRANSITION_SOUNDS = {
        'whip_pan': 'whoosh_fast',
        'flash_cut': 'impact_boom',
        'slide_left': 'whoosh_soft',
        'slide_right': 'whoosh_soft',
        'zoom_in_transition': 'whoosh_rising',
        'zoom_out_transition': 'whoosh_falling',
        'glitch': 'glitch_digital',
        'film_burn': 'film_projector_click',
        'light_leak': 'shimmer',
        'ink_splash': 'splash_soft',
        'crossfade': None,   # silent transition
        'fade_through_black': None,
        'fade_through_warm': 'shimmer',
    }

    # Effect overlay → foley sound mapping
    EFFECT_SOUNDS = {
        'flash_strobe': 'camera_shutter',
        'film_scratches': 'film_projector_scratch',
        'film_strip': 'film_projector_whir',
        'photo_frame': 'paper_rustle',
        'zoom_burst': 'whoosh_rising',
        'camera_shake': 'rumble_low',
    }

    # Section → ambient atmosphere mapping
    SECTION_ATMOSPHERE = {
        'COLD_OPEN': 'tension_drone',
        'EARLY_LIFE': 'birds_gentle',
        'THE_SPARK': 'inspiration_shimmer',
        'THE_RISE': 'crowd_distant',
        'THE_CONFLICT': 'rain_thunder',
        'THE_CLIMAX': 'crowd_roar',
        'THE_FALL': 'wind_desolate',
        'LEGACY': 'birds_gentle',
        'CTA': 'upbeat_ambient',
    }

    # Section → atmosphere volume multiplier
    SECTION_ATMOSPHERE_VOLUME = {
        'COLD_OPEN': 0.15,
        'EARLY_LIFE': 0.10,
        'THE_SPARK': 0.12,
        'THE_RISE': 0.15,
        'THE_CONFLICT': 0.20,
        'THE_CLIMAX': 0.25,
        'THE_FALL': 0.18,
        'LEGACY': 0.08,
        'CTA': 0.10,
    }

    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate

    def generate_whoosh(
        self, duration: float = 0.4, direction: str = 'fast'
    ) -> np.ndarray:
        """Generate a procedural whoosh sound effect.

        Uses filtered noise with an amplitude envelope for a natural
        whoosh that can accompany whip_pans and slide transitions.
        """
        n_samples = int(self.sample_rate * duration)
        t = np.linspace(0, duration, n_samples)

        # White noise base
        noise = np.random.randn(n_samples).astype(np.float32)

        # Amplitude envelope: quick rise, sustained, quick fall
        if direction == 'fast':
            envelope = np.sin(np.pi * t / duration) ** 0.5
        elif direction == 'rising':
            envelope = t / duration
        elif direction == 'falling':
            envelope = 1.0 - t / duration
        else:
            envelope = np.sin(np.pi * t / duration)

        # Frequency sweep using modulated sine for tonal quality
        freq_start = 200 if direction == 'rising' else 800
        freq_end = 800 if direction == 'rising' else 200
        freqs = np.linspace(freq_start, freq_end, n_samples)
        phase = np.cumsum(freqs / self.sample_rate) * 2 * np.pi
        tone = np.sin(phase) * 0.3

        result = (noise * 0.7 + tone) * envelope * 0.5
        return np.clip(result, -1.0, 1.0).astype(np.float32)

    def generate_impact(self, duration: float = 0.3) -> np.ndarray:
        """Generate a procedural impact/boom sound for flash_cut transitions."""
        n_samples = int(self.sample_rate * duration)
        t = np.linspace(0, duration, n_samples)

        # Low-frequency boom with exponential decay
        freq = 60 + 40 * np.exp(-t * 8)
        phase = np.cumsum(freq / self.sample_rate) * 2 * np.pi
        boom = np.sin(phase) * np.exp(-t * 5)

        # Noise burst for initial transient
        noise = np.random.randn(n_samples).astype(np.float32)
        transient = noise * np.exp(-t * 20) * 0.4

        result = (boom + transient) * 0.6
        return np.clip(result, -1.0, 1.0).astype(np.float32)

    def generate_shimmer(self, duration: float = 1.0) -> np.ndarray:
        """Generate a gentle shimmer/sparkle sound for light effects."""
        n_samples = int(self.sample_rate * duration)
        t = np.linspace(0, duration, n_samples)

        # Multiple high-frequency sinusoids with slight detuning
        freqs = [3200, 4800, 6400, 8000]
        shimmer = np.zeros(n_samples, dtype=np.float32)
        for f in freqs:
            shimmer += np.sin(2 * np.pi * f * t + np.random.rand() * 2 * np.pi).astype(np.float32)

        # Gentle fade-in/fade-out envelope
        envelope = np.sin(np.pi * t / duration) ** 2
        result = shimmer / len(freqs) * envelope * 0.15
        return np.clip(result, -1.0, 1.0).astype(np.float32)

    def generate_atmosphere(
        self, atmosphere_type: str, duration: float, volume: float = 0.1
    ) -> np.ndarray:
        """Generate procedural ambient atmosphere for a section.

        Produces different noise profiles for different atmospheres:
        - tension_drone: low rumble
        - birds_gentle: filtered high noise
        - rain_thunder: broadband noise with occasional booms
        - crowd_distant/crowd_roar: filtered mid-range noise
        - wind_desolate: slowly modulated noise
        """
        n_samples = int(self.sample_rate * duration)
        t = np.linspace(0, duration, n_samples)

        if atmosphere_type in ('tension_drone',):
            # Low-frequency drone
            freq = 80 + 20 * np.sin(2 * np.pi * 0.1 * t)
            base = np.sin(2 * np.pi * freq * t / self.sample_rate).astype(np.float32)
            # Slow amplitude modulation
            mod = 0.5 + 0.5 * np.sin(2 * np.pi * 0.05 * t)
            result = base * mod.astype(np.float32)

        elif atmosphere_type in ('birds_gentle', 'inspiration_shimmer'):
            # High-passed filtered noise with chirp-like modulation
            noise = np.random.randn(n_samples).astype(np.float32) * 0.3
            # Simple high-pass via first-order difference
            result = np.diff(noise, prepend=noise[0])
            mod = 0.5 + 0.5 * np.sin(2 * np.pi * 2.0 * t)
            result = result * mod.astype(np.float32)

        elif atmosphere_type in ('rain_thunder',):
            # Broadband noise with occasional low-freq booms
            noise = np.random.randn(n_samples).astype(np.float32) * 0.5
            # Add periodic thunder rumble every ~8 seconds
            thunder_interval = 8.0
            for offset in np.arange(0, duration, thunder_interval):
                start_idx = int(offset * self.sample_rate)
                thunder_len = min(int(2.0 * self.sample_rate), n_samples - start_idx)
                if thunder_len > 0:
                    tt = np.linspace(0, 2.0, thunder_len)
                    thunder = np.sin(2 * np.pi * 50 * tt) * np.exp(-tt * 1.5)
                    noise[start_idx:start_idx + thunder_len] += thunder.astype(np.float32) * 0.4
            result = noise

        elif atmosphere_type in ('crowd_distant', 'crowd_roar', 'upbeat_ambient'):
            # Mid-range filtered noise
            noise = np.random.randn(n_samples).astype(np.float32)
            # Simple smoothing for band-pass effect
            kernel_size = 5
            kernel = np.ones(kernel_size, dtype=np.float32) / kernel_size
            result = np.convolve(noise, kernel, mode='same')
            if atmosphere_type == 'crowd_roar':
                result *= 1.5

        elif atmosphere_type in ('wind_desolate',):
            # Slowly modulated broadband noise
            noise = np.random.randn(n_samples).astype(np.float32) * 0.4
            mod = 0.3 + 0.7 * np.sin(2 * np.pi * 0.15 * t)
            result = noise * mod.astype(np.float32)

        else:
            # Default: very quiet ambience
            result = np.random.randn(n_samples).astype(np.float32) * 0.1

        # Apply volume and fade in/out
        fade_samples = min(int(0.5 * self.sample_rate), n_samples // 4)
        if fade_samples > 0:
            fade_in = np.linspace(0, 1, fade_samples, dtype=np.float32)
            fade_out = np.linspace(1, 0, fade_samples, dtype=np.float32)
            result[:fade_samples] *= fade_in
            result[-fade_samples:] *= fade_out

        return np.clip(result * volume, -1.0, 1.0).astype(np.float32)

    def get_transition_sound_type(self, transition_type: str) -> Optional[str]:
        """Return the sound effect type for a given transition."""
        return self.TRANSITION_SOUNDS.get(transition_type)

    def get_section_atmosphere_config(
        self, section: str
    ) -> Tuple[str, float]:
        """Return (atmosphere_type, volume) for a section."""
        atmos = self.SECTION_ATMOSPHERE.get(section, 'birds_gentle')
        vol = self.SECTION_ATMOSPHERE_VOLUME.get(section, 0.10)
        return atmos, vol

    def apply_ducking(
        self,
        music: np.ndarray,
        narration_active: np.ndarray,
        duck_level: float = 0.3,
        attack_samples: int = 2205,
        release_samples: int = 4410,
    ) -> np.ndarray:
        """Apply dynamic ducking to music when narration is active.

        Args:
            music: Music audio array.
            narration_active: Boolean array (same length as music) indicating
                              when narration is present.
            duck_level: Volume level during ducking (0.3 = 30% volume).
            attack_samples: Samples for duck-in ramp (~50ms at 44100Hz).
            release_samples: Samples for duck-out ramp (~100ms).

        Returns:
            Ducked music array.
        """
        envelope = np.ones(len(music), dtype=np.float32)

        # Find narration regions and apply ducking envelope
        active = narration_active.astype(np.float32)
        # Smooth the envelope for gradual transitions
        for i in range(1, len(envelope)):
            if active[i] > 0.5:
                # Attack: ramp down
                envelope[i] = max(duck_level, envelope[i - 1] - 1.0 / max(attack_samples, 1))
            else:
                # Release: ramp up
                envelope[i] = min(1.0, envelope[i - 1] + 1.0 / max(release_samples, 1))

        return (music * envelope).astype(np.float32)

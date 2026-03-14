"""Programmatic sound design engine for cinematic video generation.

Generates transition sound effects entirely from numpy synthesis — no external
audio files or paid tools required. All effects are deterministic and
section-aware.

Available effects:
    - Whoosh: frequency sweep for transitions (like Dhruv Rathee style)
    - Riser: ascending tone building tension before dramatic moments
    - Bass drop: low-frequency impact hit for climax/conflict entries
    - Ambient pad: warm background tone for emotional sections
    - Tick: subtle clock-like pulse for tension building

Section mapping:
    COLD_OPEN    → whoosh (attention grab)
    EARLY_LIFE   → ambient pad (warm nostalgia)
    THE_SPARK    → riser (building curiosity)
    THE_RISE     → whoosh (momentum)
    THE_CONFLICT → bass drop (impact)
    THE_CLIMAX   → bass drop + riser (maximum drama)
    THE_FALL     → ambient pad (emotional weight)
    LEGACY       → ambient pad (reflection)
    CTA          → whoosh (energy for subscribe prompt)

All audio is generated at 44100 Hz, mono float32, normalized to [-1, 1].
The orchestrator mixes these at low volume (-18 to -24 dB) under narration.
"""

import math
from typing import Optional, Dict, List, Tuple

import numpy as np


# Standard audio sample rate
SAMPLE_RATE = 44100


class SoundDesignEngine:
    """Generates cinematic sound effects programmatically using numpy synthesis.

    All effects are generated as float32 numpy arrays at 44100 Hz.
    Effects are designed to sit under narration at low volume.
    """

    # Section → list of (effect_type, relative_volume) to play at entry
    SECTION_SOUND_MAP: Dict[str, List[Tuple[str, float]]] = {
        'COLD_OPEN': [('whoosh', 0.7)],
        'EARLY_LIFE': [('ambient_pad', 0.4)],
        'THE_SPARK': [('riser', 0.5)],
        'THE_RISE': [('whoosh', 0.5)],
        'THE_CONFLICT': [('bass_drop', 0.7), ('tick', 0.3)],
        'THE_CLIMAX': [('bass_drop', 0.8), ('riser', 0.6)],
        'THE_FALL': [('ambient_pad', 0.5)],
        'LEGACY': [('ambient_pad', 0.4)],
        'CTA': [('whoosh', 0.5)],
    }

    def __init__(self, sample_rate: int = SAMPLE_RATE):
        self.sample_rate = sample_rate

    def create_whoosh(self, duration: float = 0.6, direction: str = 'up') -> np.ndarray:
        """Generate a whoosh/sweep sound effect.

        A frequency sweep with noise texture, used at transitions.
        Sounds like air rushing past — the signature sound of fast editing.

        Args:
            duration: Length in seconds (default 0.6s).
            direction: 'up' (low→high) or 'down' (high→low).

        Returns:
            Float32 array of audio samples, normalized to [-1, 1].
        """
        n_samples = int(self.sample_rate * duration)
        t = np.linspace(0, duration, n_samples, endpoint=False)

        # Frequency sweep range
        if direction == 'up':
            freq = np.linspace(200, 2000, n_samples)
        else:
            freq = np.linspace(2000, 200, n_samples)

        # Cumulative phase for smooth sweep
        phase = 2 * np.pi * np.cumsum(freq) / self.sample_rate

        # Sine sweep + filtered noise for texture
        sweep = np.sin(phase) * 0.6
        rng = np.random.RandomState(42)
        noise = rng.randn(n_samples) * 0.15

        # Apply bandpass-like effect on noise using rolling average
        kernel_size = max(1, int(self.sample_rate * 0.002))
        if kernel_size > 1 and len(noise) > kernel_size:
            kernel = np.ones(kernel_size) / kernel_size
            noise = np.convolve(noise, kernel, mode='same')

        signal = sweep + noise

        # Envelope: fast attack, smooth decay
        envelope = np.ones(n_samples)
        attack = int(n_samples * 0.1)
        decay_start = int(n_samples * 0.5)
        if attack > 0:
            envelope[:attack] = np.linspace(0, 1, attack)
        if decay_start < n_samples:
            decay_len = n_samples - decay_start
            envelope[decay_start:] = np.linspace(1, 0, decay_len) ** 2

        signal *= envelope
        return self._normalize(signal)

    def create_riser(self, duration: float = 2.0) -> np.ndarray:
        """Generate a tension-building riser sound.

        An ascending tone that builds anticipation before a dramatic moment.
        Uses layered harmonics with increasing volume for cinematic tension.

        Args:
            duration: Length in seconds (default 2.0s).

        Returns:
            Float32 array of audio samples, normalized to [-1, 1].
        """
        n_samples = int(self.sample_rate * duration)
        t = np.linspace(0, duration, n_samples, endpoint=False)

        # Ascending frequency from 100 Hz to 800 Hz (exponential curve)
        freq = 100 * (8 ** (t / duration))

        # Cumulative phase
        phase = 2 * np.pi * np.cumsum(freq) / self.sample_rate

        # Layer fundamental + harmonics
        signal = np.sin(phase) * 0.5
        signal += np.sin(phase * 2) * 0.25  # 2nd harmonic
        signal += np.sin(phase * 3) * 0.12  # 3rd harmonic

        # Volume rises exponentially
        volume_curve = (t / duration) ** 2
        signal *= volume_curve

        # Smooth fade-in and abrupt end (the cut creates impact)
        fade_in = int(n_samples * 0.05)
        if fade_in > 0:
            signal[:fade_in] *= np.linspace(0, 1, fade_in)

        # Brief fade-out to avoid click
        fade_out = int(n_samples * 0.02)
        if fade_out > 0:
            signal[-fade_out:] *= np.linspace(1, 0, fade_out)

        return self._normalize(signal)

    def create_bass_drop(self, duration: float = 0.8) -> np.ndarray:
        """Generate a bass drop impact sound.

        A low-frequency hit that creates visceral impact at dramatic moments.
        Combines a sub-bass sine with noise burst for punch.

        Args:
            duration: Length in seconds (default 0.8s).

        Returns:
            Float32 array of audio samples, normalized to [-1, 1].
        """
        n_samples = int(self.sample_rate * duration)
        t = np.linspace(0, duration, n_samples, endpoint=False)

        # Sub-bass: starts at 80 Hz, drops to 30 Hz
        freq = 80 * np.exp(-2.0 * t / duration)
        phase = 2 * np.pi * np.cumsum(freq) / self.sample_rate
        sub_bass = np.sin(phase) * 0.7

        # Impact noise burst (first 50ms)
        rng = np.random.RandomState(123)
        noise = rng.randn(n_samples) * 0.4
        noise_env = np.zeros(n_samples)
        impact_samples = min(int(self.sample_rate * 0.05), n_samples)
        if impact_samples > 0:
            noise_env[:impact_samples] = np.exp(-np.linspace(0, 8, impact_samples))
        noise *= noise_env

        signal = sub_bass + noise

        # Envelope: instant attack, exponential decay
        envelope = np.exp(-3.0 * t / duration)
        signal *= envelope

        # Fade out last 5% to avoid click
        fade_out = int(n_samples * 0.05)
        if fade_out > 0:
            signal[-fade_out:] *= np.linspace(1, 0, fade_out)

        return self._normalize(signal)

    def create_ambient_pad(self, duration: float = 3.0) -> np.ndarray:
        """Generate a warm ambient pad sound.

        A soft, evolving tone that adds emotional weight to reflective sections.
        Uses layered detuned sine waves for a warm, choir-like texture.

        Args:
            duration: Length in seconds (default 3.0s).

        Returns:
            Float32 array of audio samples, normalized to [-1, 1].
        """
        n_samples = int(self.sample_rate * duration)
        t = np.linspace(0, duration, n_samples, endpoint=False)

        # Base chord: C minor (C3, Eb3, G3) with slight detuning
        freqs = [130.81, 155.56, 196.00]  # C3, Eb3, G3
        signal = np.zeros(n_samples, dtype=np.float64)

        for i, base_freq in enumerate(freqs):
            # Slight detune for warmth (±1-2 Hz)
            detune = [0, 1.2, -0.8][i]
            freq = base_freq + detune

            # Slow vibrato (0.5-1 Hz LFO)
            vibrato = 2.0 * np.sin(2 * np.pi * (0.5 + i * 0.2) * t)
            phase = 2 * np.pi * np.cumsum(freq + vibrato) / self.sample_rate
            signal += np.sin(phase) * 0.3

        # Gentle fade in and out (30% each)
        fade_in = int(n_samples * 0.3)
        fade_out = int(n_samples * 0.3)
        envelope = np.ones(n_samples)
        if fade_in > 0:
            envelope[:fade_in] = np.linspace(0, 1, fade_in)
        if fade_out > 0:
            envelope[-fade_out:] = np.linspace(1, 0, fade_out)

        signal *= envelope
        return self._normalize(signal)

    def create_tick(self, duration: float = 0.15) -> np.ndarray:
        """Generate a subtle tick/click sound.

        A brief percussive pulse used for tension building (like a clock ticking).
        Used in CONFLICT sections to create subconscious urgency.

        Args:
            duration: Length in seconds (default 0.15s).

        Returns:
            Float32 array of audio samples, normalized to [-1, 1].
        """
        n_samples = int(self.sample_rate * duration)
        t = np.linspace(0, duration, n_samples, endpoint=False)

        # Sharp attack sine at 1000 Hz
        signal = np.sin(2 * np.pi * 1000 * t) * 0.5
        # Add click transient
        signal += np.sin(2 * np.pi * 4000 * t) * 0.2

        # Very fast decay
        envelope = np.exp(-20 * t / duration)
        signal *= envelope

        return self._normalize(signal)

    def get_effects_for_section(
        self, section: str, transition_duration: float = 0.8
    ) -> List[Tuple[np.ndarray, float]]:
        """Get appropriate sound effects for a section transition.

        Returns a list of (audio_array, relative_volume) tuples.

        Args:
            section: Section name (e.g., 'THE_CLIMAX').
            transition_duration: Duration available for the transition.

        Returns:
            List of (audio_samples, volume_multiplier) tuples.
        """
        effects_spec = self.SECTION_SOUND_MAP.get(section, [])
        results = []

        for effect_type, volume in effects_spec:
            audio = self._generate_effect(effect_type, transition_duration)
            if audio is not None:
                results.append((audio, volume))

        return results

    def _generate_effect(
        self, effect_type: str, transition_duration: float
    ) -> Optional[np.ndarray]:
        """Generate a single effect by type name."""
        if effect_type == 'whoosh':
            return self.create_whoosh(duration=min(0.6, transition_duration))
        elif effect_type == 'riser':
            return self.create_riser(duration=min(2.0, transition_duration * 2))
        elif effect_type == 'bass_drop':
            return self.create_bass_drop(duration=min(0.8, transition_duration))
        elif effect_type == 'ambient_pad':
            return self.create_ambient_pad(duration=min(3.0, transition_duration * 3))
        elif effect_type == 'tick':
            return self.create_tick(duration=0.15)
        return None

    def mix_effects_into_audio(
        self,
        narration: np.ndarray,
        effects: List[Tuple[float, np.ndarray, float]],
        master_volume: float = 0.08,
    ) -> np.ndarray:
        """Mix sound effects into the narration audio track.

        Effects are mixed at very low volume (-22 to -28 dB below narration)
        to add cinematic feel without overpowering the voice.

        Args:
            narration: The narration audio as float32 array (mono or stereo).
            effects: List of (start_time_seconds, audio_array, volume) tuples.
            master_volume: Master volume for all effects (default 0.08 = -22 dB).

        Returns:
            Mixed audio as float32 array (same shape as narration).
        """
        if not effects:
            return narration.copy().astype(np.float32)

        # MoviePy's to_soundarray() returns (N, nchannels) for both mono and stereo.
        # Mono = (N, 1), stereo = (N, 2). A true 1D array is also possible.
        is_2d = narration.ndim == 2
        n_channels = narration.shape[1] if is_2d else 1
        mixed = narration.copy().astype(np.float64)

        for start_time, effect_audio, volume in effects:
            start_sample = int(start_time * self.sample_rate)
            end_sample = start_sample + len(effect_audio)

            num_samples = mixed.shape[0]
            if start_sample >= num_samples:
                continue

            actual_end = min(end_sample, num_samples)
            length = actual_end - start_sample
            effect_slice = effect_audio[:length] * volume * master_volume

            if is_2d:
                # Mix into all channels (works for both mono (N,1) and stereo (N,2))
                for ch in range(n_channels):
                    mixed[start_sample:actual_end, ch] += effect_slice
            else:
                mixed[start_sample:actual_end] += effect_slice

        # Soft clip to prevent distortion
        peak = np.abs(mixed).max()
        if peak > 1.0:
            mixed = mixed / peak

        return mixed.astype(np.float32)

    @staticmethod
    def _normalize(signal: np.ndarray) -> np.ndarray:
        """Normalize audio signal to [-1, 1] range."""
        peak = np.abs(signal).max()
        if peak > 0:
            signal = signal / peak
        return signal.astype(np.float32)

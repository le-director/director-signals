#!/usr/bin/env python3
"""Generate deterministic synthetic audio fixtures.

Creates WAV files identical to the synthetic generators in golden_reference.py,
but materialized as stable files for Phase-2 C++ parity testing.

Format: WAV IEEE float32, mono, 22050 Hz, 60.0s exactly
"""

import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.io import wavfile

SAMPLE_RATE = 22050
DURATION_SEC = 60.0
OUTPUT_DIR = Path(__file__).parent / "synthetic_audio"


def generate_repetitive_loop() -> np.ndarray:
    """2-second A major triad loop tiled to 60s.

    Signal: 440 Hz (0.4) + 554 Hz (0.3) + 659 Hz (0.2), peak normalized.
    """
    samples = int(DURATION_SEC * SAMPLE_RATE)
    loop_duration = 2.0
    loop_samples = int(loop_duration * SAMPLE_RATE)

    loop = np.zeros(loop_samples, dtype=np.float32)
    for i in range(loop_samples):
        t = i / SAMPLE_RATE
        loop[i] = (
            0.4 * np.sin(2 * np.pi * 440 * t) +
            0.3 * np.sin(2 * np.pi * 554 * t) +
            0.2 * np.sin(2 * np.pi * 659 * t)
        )

    n_loops = int(np.ceil(samples / loop_samples))
    audio = np.tile(loop, n_loops)[:samples]
    audio = audio / np.max(np.abs(audio))
    return audio.astype(np.float32)


def generate_build_drop() -> np.ndarray:
    """Rising build (30s) then drop section (30s), scaled to 60s total.

    Build: freq sweep 200->500 Hz, amp 0.1->0.5, increasing transients.
    Drop: 600 Hz base + 120 BPM kick (60 Hz, exp decay).
    """
    samples = int(DURATION_SEC * SAMPLE_RATE)
    half_point = samples // 2
    audio = np.zeros(samples, dtype=np.float32)
    sr = SAMPLE_RATE

    # Build section (first 30s)
    for i in range(half_point):
        t = i / sr
        progress = i / half_point
        freq = 200 + progress * 300
        amplitude = 0.1 + progress * 0.4
        audio[i] = amplitude * np.sin(2 * np.pi * freq * t)
        if i % int(sr / (2 + progress * 8)) == 0:
            transient_amp = 0.2 + progress * 0.3
            audio[i] += transient_amp

    # Drop section (last 30s)
    for i in range(half_point, samples):
        t = (i - half_point) / sr
        idx = i - half_point
        audio[i] = 0.7 * np.sin(2 * np.pi * 600 * t)
        beat_interval = int(sr * 0.5)
        if idx % beat_interval < sr * 0.1:
            kick_t = (idx % beat_interval) / sr
            envelope = np.exp(-kick_t * 20)
            audio[i] += 0.8 * envelope * np.sin(2 * np.pi * 60 * kick_t)

    audio = audio / np.max(np.abs(audio))
    return audio.astype(np.float32)


def generate_contrast() -> np.ndarray:
    """Quiet verse (30s) then loud chorus (30s), scaled to 60s total.

    Verse: 220 Hz @ 0.15.
    Chorus: 440 + 880 + 1760 Hz with rhythmic impulses every 0.25s.
    """
    samples = int(DURATION_SEC * SAMPLE_RATE)
    transition_point = samples // 2
    audio = np.zeros(samples, dtype=np.float32)
    sr = SAMPLE_RATE

    # Quiet verse (first 30s)
    for i in range(transition_point):
        t = i / sr
        audio[i] = 0.15 * np.sin(2 * np.pi * 220 * t)

    # Loud chorus (last 30s)
    for i in range(transition_point, samples):
        t = (i - transition_point) / sr
        audio[i] = (
            0.5 * np.sin(2 * np.pi * 440 * t) +
            0.3 * np.sin(2 * np.pi * 880 * t) +
            0.2 * np.sin(2 * np.pi * 1760 * t)
        )
        if i % int(sr * 0.25) < sr * 0.05:
            audio[i] += 0.4

    audio = audio / np.max(np.abs(audio))
    return audio.astype(np.float32)


def write_wav(filepath: Path, audio: np.ndarray) -> str:
    """Write WAV (IEEE float32) and return SHA256 of file bytes."""
    wavfile.write(filepath, SAMPLE_RATE, audio)
    with open(filepath, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fixtures = [
        ("repetitive_loop", generate_repetitive_loop),
        ("build_drop", generate_build_drop),
        ("contrast", generate_contrast),
    ]

    manifest_entries = []

    for name, generator in fixtures:
        audio = generator()
        filepath = OUTPUT_DIR / f"{name}.wav"
        sha256 = write_wav(filepath, audio)

        print(f"{name}.wav: {sha256}")

        manifest_entries.append({
            "name": name,
            "filename": f"{name}.wav",
            "duration_sec": DURATION_SEC,
            "sample_rate_hz": SAMPLE_RATE,
            "channels": 1,
            "sha256_bytes": sha256,
        })

    manifest = {
        "version": "1.0",
        "fixtures": manifest_entries,
    }

    manifest_path = OUTPUT_DIR / "fixtures_manifest.json"
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    print(f"\nManifest written to: {manifest_path}")


if __name__ == "__main__":
    main()

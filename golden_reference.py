"""
Golden Reference Generator

Generates deterministic reference outputs from the DSP kernel for C++ validation.

Usage:
    # Generate synthetic golden outputs
    python golden_reference.py --output golden_outputs/

    # Generate from manifest (real audio)
    python golden_reference.py --manifest reference_manifest.json --output golden_outputs/

    # Validate existing golden outputs
    python golden_reference.py --validate --output golden_outputs/

Directory structure:
    golden_outputs/
    ├── kernel_v{KERNEL_VERSION}/
    │   ├── timebase_v{TIMEBASE_VERSION}/
    │   │   ├── synthetic/
    │   │   │   ├── build_drop/
    │   │   │   │   ├── metrics.json
    │   │   │   │   ├── segments.json
    │   │   │   │   └── summary.json
    │   │   │   ├── repetitive_loop/
    │   │   │   └── contrast/
    │   │   │
    │   │   └── real_local/          # not committed
    │   │       ├── haunted/
    │   │       └── kendrick/
    │   │
    │   └── README.md
"""

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

import config
from src.audio_io import load_audio, normalize_audio
from src.kernel import (
    compute_rms_energy,
    compute_stft,
    compute_spectral_features,
    compute_spectral_flux,
    compute_zcr,
    compute_onset_strength,
    compute_mfcc_stats,
    frames_to_blocks,
    normalize_block_features,
    compute_tension_curve,
    compute_novelty_curve,
    compute_fatigue_curve,
)
from src.kernel_params import KernelConfig, DEFAULT_CONFIG
from src.timebase import (
    TIMEBASE_VERSION,
    compute_canonical_block_count,
    compute_canonical_time_axis,
)
from src.events import detect_candidate_drops, detect_stagnant_segments, detect_section_boundaries
from src.export import NumpyEncoder


# =============================================================================
# SYNTHETIC AUDIO GENERATORS
# =============================================================================

def generate_build_then_drop(duration: int = 30, sr: int = 22050) -> Tuple[np.ndarray, float]:
    """
    Generate synthetic audio with build-then-drop pattern.

    First half: rising amplitude + increasing transient density
    Second half: sudden high amplitude + steady kick pattern

    Parameters:
        duration: Total duration in seconds
        sr: Sample rate

    Returns:
        Tuple of (audio, drop_time_expected)
    """
    samples = int(duration * sr)
    half_point = samples // 2
    audio = np.zeros(samples, dtype=np.float32)

    # Build section (first half): rising amplitude + transients
    for i in range(half_point):
        t = i / sr
        progress = i / half_point

        freq = 200 + progress * 300
        amplitude = 0.1 + progress * 0.4
        audio[i] = amplitude * np.sin(2 * np.pi * freq * t)

        if i % int(sr / (2 + progress * 8)) == 0:
            transient_amp = 0.2 + progress * 0.3
            audio[i] += transient_amp

    # Drop section (second half): high amplitude + kick pattern
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
    drop_time = half_point / sr

    return audio, drop_time


def generate_repetitive_loop(duration: int = 60, sr: int = 22050) -> np.ndarray:
    """
    Generate audio with constant repetitive pattern.

    Parameters:
        duration: Duration in seconds
        sr: Sample rate

    Returns:
        Audio array
    """
    samples = int(duration * sr)
    loop_duration = 2.0
    loop_samples = int(loop_duration * sr)

    loop = np.zeros(loop_samples, dtype=np.float32)
    for i in range(loop_samples):
        t = i / sr
        loop[i] = (
            0.4 * np.sin(2 * np.pi * 440 * t) +
            0.3 * np.sin(2 * np.pi * 554 * t) +
            0.2 * np.sin(2 * np.pi * 659 * t)
        )

    n_loops = int(np.ceil(samples / loop_samples))
    audio = np.tile(loop, n_loops)[:samples]
    audio = audio / np.max(np.abs(audio))

    return audio


def generate_contrast(duration: int = 40, sr: int = 22050) -> Tuple[np.ndarray, float]:
    """
    Generate audio with quiet verse then loud chorus transition.

    Parameters:
        duration: Total duration in seconds
        sr: Sample rate

    Returns:
        Tuple of (audio, transition_time_expected)
    """
    samples = int(duration * sr)
    transition_point = samples // 2
    audio = np.zeros(samples, dtype=np.float32)

    # Quiet verse section (first half)
    for i in range(transition_point):
        t = i / sr
        audio[i] = 0.15 * np.sin(2 * np.pi * 220 * t)

    # Loud chorus section (second half)
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
    transition_time = transition_point / sr

    return audio, transition_time


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def compute_sha256(data: np.ndarray) -> str:
    """Compute SHA256 checksum of numpy array."""
    return hashlib.sha256(data.tobytes()).hexdigest()


def compute_wav_file_sha256(wav_path: Path) -> str:
    """Compute SHA256 of WAV file bytes (not float buffer)."""
    with open(wav_path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()


# =============================================================================
# FIXTURE LOADING
# =============================================================================

def load_fixture_manifest(fixtures_dir: Path) -> Dict:
    """Load fixtures_manifest.json from fixtures directory."""
    manifest_path = fixtures_dir / 'fixtures_manifest.json'
    with open(manifest_path, 'r') as f:
        return json.load(f)


def load_synthetic_fixture(
    name: str,
    fixtures_dir: Path,
    expected_sha256: str = None
) -> Tuple[np.ndarray, int, str]:
    """
    Load synthetic fixture WAV file.

    Parameters:
        name: Fixture name (without .wav extension)
        fixtures_dir: Directory containing fixture WAV files
        expected_sha256: Optional expected SHA256 of WAV file bytes

    Returns:
        Tuple of (audio, sample_rate, wav_sha256)

    Raises:
        ValueError: If SHA256 doesn't match expected value
        FileNotFoundError: If fixture file doesn't exist
    """
    from scipy.io import wavfile

    wav_path = fixtures_dir / f"{name}.wav"
    wav_sha256 = compute_wav_file_sha256(wav_path)

    if expected_sha256 and wav_sha256 != expected_sha256:
        raise ValueError(
            f"Fixture {name} SHA256 mismatch:\n"
            f"  Expected: {expected_sha256}\n"
            f"  Got:      {wav_sha256}"
        )

    sr, audio = wavfile.read(wav_path)

    # Ensure float32 format
    if audio.dtype != np.float32:
        audio = audio.astype(np.float32)

    return audio, sr, wav_sha256


def get_versioned_output_path(
    base_dir: str,
    kernel_version: str = None,
    timebase_version: str = None
) -> Path:
    """
    Build versioned output path.

    Parameters:
        base_dir: Base output directory
        kernel_version: Kernel version (default: config.KERNEL_VERSION)
        timebase_version: Timebase version (default: TIMEBASE_VERSION)

    Returns:
        Path like golden_outputs/kernel_v1.1.1/timebase_v1/
    """
    if kernel_version is None:
        kernel_version = config.KERNEL_VERSION
    if timebase_version is None:
        timebase_version = TIMEBASE_VERSION

    return Path(base_dir) / f"kernel_v{kernel_version}" / f"timebase_v{timebase_version}"


# =============================================================================
# ANALYSIS PIPELINE
# =============================================================================

def run_full_analysis(
    audio: np.ndarray,
    sr: int,
    cfg: KernelConfig
) -> Dict:
    """
    Run complete analysis pipeline on audio.

    Parameters:
        audio: Audio array (mono, float32)
        sr: Sample rate
        cfg: Kernel configuration

    Returns:
        Dictionary with all analysis results
    """
    frame_length = cfg.frame.frame_length
    hop_length = cfg.frame.hop_length
    block_duration = cfg.block.block_duration_sec

    # Frame-level features
    rms = compute_rms_energy(audio, frame_length, hop_length)
    spectral = compute_spectral_features(
        audio, sr, frame_length, hop_length,
        rolloff_percent=cfg.spectral.rolloff_percent
    )
    onset = compute_onset_strength(audio, sr, frame_length, hop_length)

    # Ensure all frame arrays have same length
    min_frames = min(
        len(rms), len(onset),
        len(spectral['spectral_centroid']),
        len(spectral['spectral_bandwidth'])
    )
    rms = rms[:min_frames]
    onset = onset[:min_frames]
    centroid = spectral['spectral_centroid'][:min_frames]
    bandwidth = spectral['spectral_bandwidth'][:min_frames]

    # Block aggregation
    rms_blocks, _ = frames_to_blocks(rms, sr, hop_length, block_duration, 'mean')
    onset_blocks, _ = frames_to_blocks(onset, sr, hop_length, block_duration, 'mean')
    centroid_blocks, _ = frames_to_blocks(centroid, sr, hop_length, block_duration, 'mean')
    bandwidth_blocks, _ = frames_to_blocks(bandwidth, sr, hop_length, block_duration, 'mean')

    # Ensure all block arrays have same length
    min_blocks = min(len(rms_blocks), len(onset_blocks), len(centroid_blocks), len(bandwidth_blocks))
    rms_blocks = rms_blocks[:min_blocks]
    onset_blocks = onset_blocks[:min_blocks]
    centroid_blocks = centroid_blocks[:min_blocks]
    bandwidth_blocks = bandwidth_blocks[:min_blocks]

    # Build block feature matrix
    block_features = np.column_stack([rms_blocks, onset_blocks, centroid_blocks, bandwidth_blocks])
    block_features_norm, _ = normalize_block_features(block_features, 'robust')

    # Compute canonical time axis
    duration_sec = len(audio) / sr
    n_blocks = compute_canonical_block_count(duration_sec, block_duration)
    n_blocks = min(n_blocks, min_blocks)  # Ensure we don't exceed computed blocks
    block_times = compute_canonical_time_axis(n_blocks, block_duration, duration_sec=duration_sec)

    # Trim to canonical block count
    rms_blocks = rms_blocks[:n_blocks]
    onset_blocks = onset_blocks[:n_blocks]
    centroid_blocks = centroid_blocks[:n_blocks]
    bandwidth_blocks = bandwidth_blocks[:n_blocks]
    block_features_norm = block_features_norm[:n_blocks]

    # Tension curve
    tension_raw, tension_smooth, tension_components, tension_norm_info = compute_tension_curve(
        rms_blocks=rms_blocks,
        onset_blocks=onset_blocks,
        centroid_blocks=centroid_blocks,
        bandwidth_blocks=bandwidth_blocks,
        weights=cfg.tension.get_weights(),
        normalization_mode=cfg.tension.normalization_mode,
        sr=sr,
        smooth_alpha=cfg.tension.smooth_alpha,
        percentile_lower=cfg.tension.percentile_lower,
        percentile_upper=cfg.tension.percentile_upper
    )

    # Novelty curve
    novelty_smooth, _ = compute_novelty_curve(
        block_features_norm,
        lookback_blocks=cfg.novelty.lookback_blocks,
        smooth_window=cfg.novelty.smooth_window
    )

    # Fatigue curve
    fatigue_smooth, fatigue_components = compute_fatigue_curve(
        block_features_norm,
        novelty_smooth,
        boundary_blocks=None,
        weights=cfg.fatigue.get_boring_weights(),
        window_size=cfg.fatigue.window_blocks,
        smooth_window=cfg.fatigue.smooth_window,
        use_leaky_integrator=cfg.fatigue.use_leaky_integrator,
        gain_up=cfg.fatigue.gain_up,
        gain_down=cfg.fatigue.gain_down,
        novelty_spike_threshold=cfg.fatigue.novelty_spike_threshold
    )

    # Curves dict
    curves = {
        'tension_raw': tension_raw,
        'tension_smooth': tension_smooth,
        'novelty': novelty_smooth,
        'fatigue': fatigue_smooth,
        'tension_components': tension_components,
        'tension_normalization': tension_norm_info,
        'fatigue_components': fatigue_components,
    }

    # Event detection
    candidate_drops = detect_candidate_drops(
        tension_smooth, block_times, duration_sec
    )
    stagnant_segments = detect_stagnant_segments(
        novelty_smooth, fatigue_smooth, block_times, duration_sec
    )
    boundaries = detect_section_boundaries(
        novelty_smooth, tension_smooth, block_times, duration_sec
    )

    events = {
        'candidate_drops': candidate_drops,
        'ranked_drops': candidate_drops,  # Already ranked by score
        'stagnant_segments': stagnant_segments,
        'boundaries': boundaries,
    }

    return {
        'curves': curves,
        'events': events,
        'block_times': block_times,
        'duration_sec': duration_sec,
        'n_blocks': n_blocks,
        'sample_rate': sr,
    }


# =============================================================================
# OUTPUT GENERATION
# =============================================================================

def generate_track_outputs(
    name: str,
    audio: np.ndarray,
    sr: int,
    cfg: KernelConfig,
    output_dir: Path,
    source_info: Dict = None,
    wav_sha256: str = None
) -> List[Path]:
    """
    Generate all outputs for a single track.

    Parameters:
        name: Track name (used for output directory)
        audio: Audio array
        sr: Sample rate
        cfg: Kernel configuration
        output_dir: Base output directory (e.g., golden_outputs/kernel_v1.1.1/timebase_v1/synthetic/)
        source_info: Optional source metadata
        wav_sha256: Optional pre-computed SHA256 of WAV file bytes (for fixtures)

    Returns:
        List of created file paths
    """
    track_dir = output_dir / name
    track_dir.mkdir(parents=True, exist_ok=True)

    # Run analysis
    results = run_full_analysis(audio, sr, cfg)

    # Create metadata
    audio_metadata = {
        'duration': results['duration_sec'],
        'sample_rate': results['sample_rate'],
        'preprocessing': {
            'normalization_method': 'peak',
            'normalization_factor': 1.0,
        }
    }

    # Parameters dict
    params = cfg.to_dict()
    params['kernel_version'] = config.KERNEL_VERSION
    params['timebase_version'] = TIMEBASE_VERSION
    if source_info:
        params['source'] = source_info

    # Block times info
    block_times = results['block_times']
    sampling_interval = float(block_times[1] - block_times[0]) if len(block_times) > 1 else cfg.block.block_duration_sec

    created_files = []

    # metrics.json
    metrics = {
        'schema_version': config.SCHEMA_VERSION,
        'kernel_version': config.KERNEL_VERSION,
        'timebase_version': TIMEBASE_VERSION,
        'generated_at': datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
        'track_metadata': {
            'name': name,
            'duration': audio_metadata['duration'],
            'sample_rate': audio_metadata['sample_rate'],
            'preprocessing': audio_metadata['preprocessing'],
            'audio_checksum': wav_sha256 if wav_sha256 else compute_sha256(audio.astype(np.float32)),
        },
        'params': params,
        'curves': {
            'tension_raw': {
                'values': results['curves']['tension_raw'].tolist(),
                'sampling_interval_sec': sampling_interval,
                'length': len(results['curves']['tension_raw']),
            },
            'tension_smooth': {
                'values': results['curves']['tension_smooth'].tolist(),
                'sampling_interval_sec': sampling_interval,
                'length': len(results['curves']['tension_smooth']),
            },
            'novelty': {
                'values': results['curves']['novelty'].tolist(),
                'sampling_interval_sec': sampling_interval,
                'length': len(results['curves']['novelty']),
            },
            'fatigue': {
                'values': results['curves']['fatigue'].tolist(),
                'sampling_interval_sec': sampling_interval,
                'length': len(results['curves']['fatigue']),
            },
        },
        'block_times': block_times.tolist(),
    }

    metrics_path = track_dir / 'metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2, cls=NumpyEncoder)
    created_files.append(metrics_path)

    # segments.json
    segments = {
        'schema_version': config.SCHEMA_VERSION,
        'kernel_version': config.KERNEL_VERSION,
        'timebase_version': TIMEBASE_VERSION,
        'track_name': name,
        'candidate_drops': results['events']['candidate_drops'],
        'ranked_drops': results['events']['ranked_drops'],
        'stagnant_segments': results['events']['stagnant_segments'],
        'section_boundaries': results['events']['boundaries'],
    }

    segments_path = track_dir / 'segments.json'
    with open(segments_path, 'w') as f:
        json.dump(segments, f, indent=2, cls=NumpyEncoder)
    created_files.append(segments_path)

    # summary.json
    tension_values = results['curves']['tension_smooth']
    top_tension_idx = int(np.argmax(tension_values)) if len(tension_values) > 0 else 0
    top_tension_time = float(block_times[top_tension_idx]) if len(block_times) > top_tension_idx else 0.0
    top_tension_value = float(tension_values[top_tension_idx]) if len(tension_values) > top_tension_idx else 0.0

    stagnant_segments = results['events']['stagnant_segments']
    longest_stagnant = max((s['duration'] for s in stagnant_segments), default=0.0)

    summary = {
        'schema_version': config.SCHEMA_VERSION,
        'kernel_version': config.KERNEL_VERSION,
        'timebase_version': TIMEBASE_VERSION,
        'duration_sec': results['duration_sec'],
        'n_blocks': results['n_blocks'],
        'top_tension_peak': {
            'time_sec': top_tension_time,
            'value': top_tension_value,
        },
        'num_candidate_drops': len(results['events']['candidate_drops']),
        'num_stagnant_segments': len(stagnant_segments),
        'longest_stagnant_duration_sec': longest_stagnant,
        'top_drops': [
            {'time': d['time'], 'score': d['score']}
            for d in results['events']['ranked_drops'][:5]
        ],
    }

    summary_path = track_dir / 'summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, cls=NumpyEncoder)
    created_files.append(summary_path)

    return created_files


# =============================================================================
# MANIFEST HANDLING
# =============================================================================

def load_manifest(manifest_path: str) -> Dict:
    """Load manifest JSON file."""
    with open(manifest_path, 'r') as f:
        return json.load(f)


def load_audio_from_manifest_entry(
    entry: Dict,
    target_sr: int = 22050
) -> Tuple[np.ndarray, int, Dict]:
    """
    Load audio from a manifest entry.

    Parameters:
        entry: Manifest entry dict with 'path', optional 'start_sec', 'end_sec'
        target_sr: Target sample rate

    Returns:
        Tuple of (audio, sample_rate, source_info)
    """
    path = entry['path']
    audio, sr = load_audio(path, target_sr=target_sr)

    # Optional cropping
    start_sec = entry.get('start_sec')
    end_sec = entry.get('end_sec')

    if start_sec is not None or end_sec is not None:
        start_sample = int(start_sec * sr) if start_sec else 0
        end_sample = int(end_sec * sr) if end_sec else len(audio)
        audio = audio[start_sample:end_sample]

    # Normalize
    audio, _ = normalize_audio(audio, method='peak')

    source_info = {
        'type': 'manifest',
        'path': path,
        'crop': {
            'start_sec': start_sec,
            'end_sec': end_sec,
        },
        'bpm': entry.get('bpm'),
        'tags': entry.get('tags', []),
    }

    return audio, sr, source_info


# =============================================================================
# GOLDEN REFERENCE GENERATION
# =============================================================================

def generate_synthetic_references(
    output_dir: str,
    fixtures_dir: str = 'fixtures/synthetic_audio',
    cfg: KernelConfig = None
) -> List[Path]:
    """
    Generate golden references for all synthetic test tracks from committed fixtures.

    Parameters:
        output_dir: Base output directory
        fixtures_dir: Directory containing synthetic audio fixtures
        cfg: Kernel configuration

    Returns:
        List of generated file paths
    """
    if cfg is None:
        cfg = DEFAULT_CONFIG

    fixtures_path = Path(fixtures_dir)
    manifest = load_fixture_manifest(fixtures_path)
    fixture_info = {f['name']: f for f in manifest['fixtures']}

    versioned_path = get_versioned_output_path(output_dir)
    synthetic_path = versioned_path / 'synthetic'
    synthetic_path.mkdir(parents=True, exist_ok=True)

    all_files = []

    # Synthetic cases with their source metadata
    # Duration is now 60s for all fixtures (from committed WAVs)
    synthetic_cases = [
        ('build_drop', {
            'type': 'synthetic',
            'generator': 'build_then_drop',
            'fixture_source': 'fixtures/synthetic_audio/build_drop.wav',
            'expected_events': {'drop_time': 30.0},  # 60s track, drop at midpoint
        }),
        ('repetitive_loop', {
            'type': 'synthetic',
            'generator': 'repetitive_loop',
            'fixture_source': 'fixtures/synthetic_audio/repetitive_loop.wav',
            'expected_events': {'expected_high_fatigue': True},
        }),
        ('contrast', {
            'type': 'synthetic',
            'generator': 'section_contrast',
            'fixture_source': 'fixtures/synthetic_audio/contrast.wav',
            'expected_events': {'transition_time': 30.0},  # 60s track, transition at midpoint
        }),
    ]

    for name, source_info in synthetic_cases:
        print(f"Generating reference: {name}...")
        info = fixture_info[name]
        audio, sr, wav_sha256 = load_synthetic_fixture(
            name, fixtures_path, expected_sha256=info['sha256_bytes']
        )
        source_info['wav_sha256'] = wav_sha256
        files = generate_track_outputs(
            name, audio, sr, cfg, synthetic_path, source_info, wav_sha256=wav_sha256
        )
        all_files.extend(files)
        print(f"  Created {len(files)} files in {synthetic_path / name}")
        print(f"  audio_checksum (WAV SHA256): {wav_sha256}")

    # Generate README
    readme_path = versioned_path.parent / 'README.md'
    readme_content = f"""# Golden Reference Outputs

## Kernel Version: {config.KERNEL_VERSION}
## Timebase Version: {TIMEBASE_VERSION}
## Schema Version: {config.SCHEMA_VERSION}

Generated: {datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')}

## Audio Source

Synthetic tracks are loaded from committed fixtures in `fixtures/synthetic_audio/`.
The `audio_checksum` in each metrics.json is the SHA256 of the WAV file bytes.

## Directory Structure

```
kernel_v{config.KERNEL_VERSION}/
├── timebase_v{TIMEBASE_VERSION}/
│   ├── synthetic/
│   │   ├── build_drop/      # Rising build then high-energy drop (60s)
│   │   ├── repetitive_loop/ # Constant repeating pattern (60s)
│   │   └── contrast/        # Quiet verse to loud chorus (60s)
│   │
│   └── real_local/          # Real audio (not committed)
```

## Synthetic Test Tracks

All synthetic tracks are 60 seconds, loaded from `fixtures/synthetic_audio/`:

1. **build_drop**: Rising amplitude/transients (0-30s) then sudden drop (30-60s)
2. **repetitive_loop**: Constant 2-second A-major triad loop
3. **contrast**: Quiet 220Hz verse (0-30s) then loud multi-harmonic chorus (30-60s)

## Per-Track Outputs

- `metrics.json`: All curves, parameters, and intermediate values
- `segments.json`: Detected events (drops, stagnant segments, boundaries)
- `summary.json`: High-level summary statistics

## Validation

Run `python golden_reference.py --validate` to verify:
- audio_checksum matches fixture WAV SHA256
- time axis max <= duration
- curve lengths == canonical block count
"""

    with open(readme_path, 'w') as f:
        f.write(readme_content)
    all_files.append(readme_path)

    return all_files


def generate_manifest_references(
    manifest_path: str,
    output_dir: str,
    cfg: KernelConfig = None
) -> List[Path]:
    """
    Generate golden references from manifest file.

    Parameters:
        manifest_path: Path to manifest JSON
        output_dir: Base output directory
        cfg: Kernel configuration

    Returns:
        List of generated file paths
    """
    if cfg is None:
        cfg = DEFAULT_CONFIG

    manifest = load_manifest(manifest_path)
    versioned_path = get_versioned_output_path(output_dir)
    real_local_path = versioned_path / 'real_local'
    real_local_path.mkdir(parents=True, exist_ok=True)

    all_files = []

    for entry in manifest['tracks']:
        name = entry['name']
        print(f"Generating reference: {name}...")

        try:
            audio, sr, source_info = load_audio_from_manifest_entry(entry, cfg.frame.sample_rate)
            files = generate_track_outputs(name, audio, sr, cfg, real_local_path, source_info)
            all_files.extend(files)
            print(f"  Created {len(files)} files in {real_local_path / name}")
        except FileNotFoundError as e:
            print(f"  ERROR: Could not load audio from {entry['path']}: {e}")
        except Exception as e:
            print(f"  ERROR processing {name}: {e}")

    return all_files


def validate_references(
    output_dir: str,
    fixtures_dir: str = 'fixtures/synthetic_audio',
    cfg: KernelConfig = None
) -> bool:
    """
    Validate all golden references against committed fixtures.

    Validates:
    1. audio_checksum in golden matches WAV file SHA256
    2. WAV SHA256 matches fixture manifest
    3. time axis max <= duration + epsilon
    4. curve lengths == canonical n_blocks
    5. Curve values match re-analysis

    Parameters:
        output_dir: Base output directory
        fixtures_dir: Directory containing synthetic audio fixtures
        cfg: Kernel configuration

    Returns:
        True if all validations pass
    """
    if cfg is None:
        cfg = DEFAULT_CONFIG

    fixtures_path = Path(fixtures_dir)
    manifest = load_fixture_manifest(fixtures_path)
    fixture_info = {f['name']: f for f in manifest['fixtures']}

    versioned_path = get_versioned_output_path(output_dir)
    synthetic_path = versioned_path / 'synthetic'

    if not synthetic_path.exists():
        print(f"No golden outputs found at {synthetic_path}")
        return False

    all_passed = True
    epsilon = 1e-6

    for name in ['build_drop', 'repetitive_loop', 'contrast']:
        track_dir = synthetic_path / name
        metrics_path = track_dir / 'metrics.json'

        if not metrics_path.exists():
            print(f"SKIP: {name} - metrics.json not found")
            continue

        print(f"Validating: {name}...")

        # Load reference metrics
        with open(metrics_path, 'r') as f:
            ref_metrics = json.load(f)

        # Load fixture
        info = fixture_info[name]
        audio, sr, wav_sha256 = load_synthetic_fixture(name, fixtures_path)
        duration_sec = len(audio) / sr

        # Check 1: WAV SHA256 matches manifest
        if wav_sha256 != info['sha256_bytes']:
            print(f"  FAIL: WAV SHA256 doesn't match manifest")
            print(f"    Manifest: {info['sha256_bytes']}")
            print(f"    Computed: {wav_sha256}")
            all_passed = False
            continue
        print(f"  PASS: WAV SHA256 matches manifest")

        # Check 2: audio_checksum in golden matches WAV SHA256
        ref_checksum = ref_metrics['track_metadata']['audio_checksum']
        if ref_checksum != wav_sha256:
            print(f"  FAIL: Golden audio_checksum doesn't match WAV SHA256")
            print(f"    Golden:   {ref_checksum}")
            print(f"    WAV SHA256: {wav_sha256}")
            all_passed = False
            continue
        print(f"  PASS: audio_checksum matches WAV SHA256")

        # Check 3: time axis max <= duration + epsilon
        block_times = ref_metrics['block_times']
        if len(block_times) > 0:
            max_time = max(block_times)
            if max_time > duration_sec + epsilon:
                print(f"  FAIL: time axis max ({max_time:.6f}) > duration ({duration_sec:.6f})")
                all_passed = False
            else:
                print(f"  PASS: time axis max ({max_time:.2f}) <= duration ({duration_sec:.2f})")

        # Check 4: curve lengths == canonical n_blocks
        canonical_n_blocks = compute_canonical_block_count(
            duration_sec, cfg.block.block_duration_sec
        )
        for curve_name in ['tension_raw', 'tension_smooth', 'novelty', 'fatigue']:
            curve_length = ref_metrics['curves'][curve_name]['length']
            if curve_length != canonical_n_blocks:
                print(f"  FAIL: {curve_name} length ({curve_length}) != canonical ({canonical_n_blocks})")
                all_passed = False
            else:
                print(f"  PASS: {curve_name} length == {canonical_n_blocks}")

        # Check 5: Re-run analysis and compare curve values
        results = run_full_analysis(audio, sr, cfg)
        for curve_name in ['tension_raw', 'tension_smooth', 'novelty', 'fatigue']:
            ref_values = np.array(ref_metrics['curves'][curve_name]['values'], dtype=np.float32)
            new_values = results['curves'][curve_name].astype(np.float32)

            if len(ref_values) != len(new_values):
                print(f"  FAIL: {curve_name} re-analysis length mismatch ({len(ref_values)} vs {len(new_values)})")
                all_passed = False
                continue

            if not np.allclose(ref_values, new_values, rtol=1e-5, atol=1e-6):
                max_diff = np.max(np.abs(ref_values - new_values))
                print(f"  FAIL: {curve_name} re-analysis values differ (max diff: {max_diff})")
                all_passed = False
            else:
                print(f"  PASS: {curve_name} re-analysis matches")

    return all_passed


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Generate golden reference outputs for DSP kernel validation'
    )
    parser.add_argument(
        '--output', '-o',
        default='golden_outputs',
        help='Base output directory for golden reference files'
    )
    parser.add_argument(
        '--fixtures-dir', '-f',
        default='fixtures/synthetic_audio',
        help='Directory containing synthetic audio fixtures (default: fixtures/synthetic_audio)'
    )
    parser.add_argument(
        '--manifest', '-m',
        help='Path to manifest JSON for real audio files'
    )
    parser.add_argument(
        '--validate', '-v',
        action='store_true',
        help='Validate existing golden references instead of generating'
    )
    parser.add_argument(
        '--export-handoff',
        metavar='DIR',
        help='Also export kernel handoff package to specified directory'
    )

    args = parser.parse_args()

    if args.validate:
        print(f"Validating golden references in {args.output}/")
        print(f"Using fixtures from: {args.fixtures_dir}/")
        success = validate_references(args.output, args.fixtures_dir)
        return 0 if success else 1

    if args.manifest:
        print(f"Generating golden references from manifest: {args.manifest}")
        files = generate_manifest_references(args.manifest, args.output)
        print(f"\nGenerated {len(files)} files from manifest.")
    else:
        print(f"Generating synthetic golden references to {args.output}/")
        print(f"Loading fixtures from: {args.fixtures_dir}/")
        files = generate_synthetic_references(args.output, args.fixtures_dir)
        print(f"\nGenerated {len(files)} files.")

    # Export handoff package if requested
    if args.export_handoff:
        print(f"\nExporting kernel handoff package to {args.export_handoff}/...")
        from tools.export_kernel_handoff import export_kernel_handoff

        golden_root = get_versioned_output_path(args.output)
        handoff_out = Path(args.export_handoff) / f"kernel_v{config.KERNEL_VERSION}" / f"timebase_v{TIMEBASE_VERSION}"

        export_kernel_handoff(
            kernel_version=config.KERNEL_VERSION,
            timebase_version=TIMEBASE_VERSION,
            golden_root=golden_root,
            fixtures_dir=Path(args.fixtures_dir),
            out_dir=handoff_out,
        )
        print(f"\nHandoff package exported to: {handoff_out}")

    return 0


if __name__ == '__main__':
    exit(main())

#!/usr/bin/env python3
"""
director-signals - Command Line Interface

Main entry point for running analysis on audio tracks.
Uses src/kernel.py for all DSP operations (same as golden_reference.py).
"""

import argparse
import sys
from pathlib import Path
from typing import Optional, List
import numpy as np

import config
from src import audio_io, events, export
from src.kernel import (
    compute_rms_energy,
    compute_spectral_features,
    compute_onset_strength,
    frames_to_blocks,
    normalize_block_features,
    compute_tension_curve,
    compute_novelty_curve,
    compute_fatigue_curve,
)
from src.kernel_params import KernelConfig, DEFAULT_CONFIG
from src.timebase import (
    compute_canonical_block_count,
    compute_canonical_time_axis,
)


def process_single_track(
    file_path: Path,
    output_dir: Path,
    params: dict,
    verbose: bool = False
) -> bool:
    """
    Process a single audio track through the full pipeline.

    Uses kernel.py functions for C++ parity with golden_reference.py.

    Parameters:
        file_path: Path to audio file
        output_dir: Output directory for results
        params: Parameters dict (from config or overrides)
        verbose: Print verbose progress messages

    Returns:
        True if successful, False otherwise
    """
    track_name = file_path.stem

    try:
        if verbose:
            print(f"\nProcessing: {file_path.name}")
            print("-" * 60)

        # Get kernel config
        cfg = DEFAULT_CONFIG

        # Step 1: Load and preprocess audio
        if verbose:
            print("1. Loading and preprocessing audio...")

        audio_data = audio_io.preprocess_audio(
            str(file_path),
            target_sr=params.get('target_sr', config.TARGET_SAMPLE_RATE),
            normalize_method=params.get('normalize_method', config.NORMALIZATION_METHOD),
            trim=params.get('trim_silence', config.TRIM_SILENCE)
        )

        audio = audio_data['audio']
        sr = audio_data['sample_rate']
        duration = audio_data['duration']

        if verbose:
            print(f"   Duration: {duration:.2f}s, Sample rate: {sr} Hz")

        # Validate audio
        audio_io.validate_audio(audio, sr)

        # Get frame/block parameters
        frame_length = params.get('frame_length', config.FRAME_LENGTH)
        hop_length = params.get('hop_length', config.HOP_LENGTH)
        block_duration = params.get('block_duration', config.BLOCK_DURATION_SEC)

        # Step 2: Extract frame-level features using kernel functions
        if verbose:
            print("2. Extracting frame-level features...")

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

        if verbose:
            print(f"   Extracted {min_frames} frames")

        # Step 3: Aggregate to blocks using kernel functions
        if verbose:
            print("3. Aggregating features to blocks...")

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

        # Build block feature matrix and normalize
        block_features = np.column_stack([rms_blocks, onset_blocks, centroid_blocks, bandwidth_blocks])
        block_features_norm, _ = normalize_block_features(block_features, 'robust')

        # Compute canonical time axis
        n_blocks = compute_canonical_block_count(duration, block_duration)
        n_blocks = min(n_blocks, min_blocks)  # Ensure we don't exceed computed blocks
        block_times = compute_canonical_time_axis(n_blocks, block_duration, duration_sec=duration)

        # Trim to canonical block count
        rms_blocks = rms_blocks[:n_blocks]
        onset_blocks = onset_blocks[:n_blocks]
        centroid_blocks = centroid_blocks[:n_blocks]
        bandwidth_blocks = bandwidth_blocks[:n_blocks]
        block_features_norm = block_features_norm[:n_blocks]

        if verbose:
            print(f"   Created {n_blocks} blocks")

        # Step 4: Compute curves using kernel functions
        if verbose:
            print("4. Computing long-horizon curves...")

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
        novelty_smooth, novelty_raw = compute_novelty_curve(
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

        # Build curves dict (matching golden_reference.py format)
        curves = {
            'tension_raw': tension_raw,
            'tension_smooth': tension_smooth,
            'novelty': novelty_smooth,
            'fatigue': fatigue_smooth,
            'tension_components': tension_components,
            'tension_normalization': tension_norm_info,
            'fatigue_components': fatigue_components,
        }

        if verbose:
            print(f"   Computed tension, novelty, and fatigue curves")

        # Step 5: Detect events
        if verbose:
            print("5. Detecting events...")

        candidate_drops = events.detect_candidate_drops(
            tension_smooth, block_times, duration
        )
        stagnant_segments = events.detect_stagnant_segments(
            novelty_smooth, fatigue_smooth, block_times, duration
        )
        boundaries = events.detect_section_boundaries(
            novelty_smooth, tension_smooth, block_times, duration
        )

        detected_events = {
            'candidate_drops': candidate_drops,
            'ranked_drops': candidate_drops,  # Already ranked by score
            'stagnant_segments': stagnant_segments,
            'boundaries': boundaries,
        }

        if verbose:
            n_drops = len(detected_events['candidate_drops'])
            n_stagnant = len(detected_events['stagnant_segments'])
            n_boundaries = len(detected_events['boundaries'])
            print(f"   Found {n_drops} candidate drops, {n_stagnant} stagnant segments, "
                  f"{n_boundaries} boundaries")

        # Step 6: Export results
        if verbose:
            print("6. Exporting results...")

        analysis_results = {
            'audio_metadata': audio_data,
            'params': params,
            'curves': curves,
            'events': detected_events,
            'block_times': block_times
        }

        created_files = export.export_all_outputs(
            analysis_results,
            output_dir,
            track_name,
            generate_plots=params.get('generate_plots', True)
        )

        if verbose:
            print(f"   Created {len(created_files)} output files")

        # Print summary
        summary_path = output_dir / f"{track_name}_summary.json"
        if summary_path.exists():
            import json
            with open(summary_path) as f:
                summary = json.load(f)
            export.print_analysis_summary(summary, track_name)

        return True

    except Exception as e:
        print(f"ERROR processing {file_path.name}: {e}", file=sys.stderr)
        if verbose:
            import traceback
            traceback.print_exc()
        return False


def process_audio_array(
    audio: np.ndarray,
    sr: int,
    params: dict,
    audio_metadata: dict,
    verbose: bool = False
) -> dict:
    """
    Process an audio array through the kernel pipeline.

    Used by both process_single_track and run_demo_mode.

    Parameters:
        audio: Audio array
        sr: Sample rate
        params: Parameters dict
        audio_metadata: Metadata dict for the audio
        verbose: Print verbose messages

    Returns:
        Analysis results dict
    """
    cfg = DEFAULT_CONFIG
    duration = len(audio) / sr

    frame_length = params.get('frame_length', config.FRAME_LENGTH)
    hop_length = params.get('hop_length', config.HOP_LENGTH)
    block_duration = params.get('block_duration', config.BLOCK_DURATION_SEC)

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

    # Build block feature matrix and normalize
    block_features = np.column_stack([rms_blocks, onset_blocks, centroid_blocks, bandwidth_blocks])
    block_features_norm, _ = normalize_block_features(block_features, 'robust')

    # Compute canonical time axis
    n_blocks = compute_canonical_block_count(duration, block_duration)
    n_blocks = min(n_blocks, min_blocks)
    block_times = compute_canonical_time_axis(n_blocks, block_duration, duration_sec=duration)

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
    novelty_smooth, novelty_raw = compute_novelty_curve(
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

    # Build curves dict
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
    candidate_drops = events.detect_candidate_drops(
        tension_smooth, block_times, duration
    )
    stagnant_segments = events.detect_stagnant_segments(
        novelty_smooth, fatigue_smooth, block_times, duration
    )
    boundaries = events.detect_section_boundaries(
        novelty_smooth, tension_smooth, block_times, duration
    )

    detected_events = {
        'candidate_drops': candidate_drops,
        'ranked_drops': candidate_drops,
        'stagnant_segments': stagnant_segments,
        'boundaries': boundaries,
    }

    return {
        'audio_metadata': audio_metadata,
        'params': params,
        'curves': curves,
        'events': detected_events,
        'block_times': block_times
    }


def process_directory(
    input_dir: Path,
    output_dir: Path,
    params: dict,
    verbose: bool = False
) -> dict:
    """
    Process all audio files in a directory.

    Parameters:
        input_dir: Input directory containing audio files
        output_dir: Output directory for results
        params: Parameters dict
        verbose: Print verbose messages

    Returns:
        Dict with success/failure counts
    """
    # Find audio files
    audio_extensions = ['.wav', '.mp3', '.flac', '.ogg', '.m4a']
    audio_files = []

    for ext in audio_extensions:
        audio_files.extend(input_dir.glob(f'*{ext}'))
        audio_files.extend(input_dir.glob(f'*{ext.upper()}'))

    if not audio_files:
        print(f"No audio files found in {input_dir}")
        return {'success': 0, 'failed': 0}

    print(f"Found {len(audio_files)} audio files")

    success_count = 0
    failed_count = 0

    for audio_file in sorted(audio_files):
        # Create per-track output directory
        track_output_dir = output_dir / audio_file.stem

        success = process_single_track(audio_file, track_output_dir, params, verbose)

        if success:
            success_count += 1
        else:
            failed_count += 1

    print(f"\nProcessing complete: {success_count} successful, {failed_count} failed")

    return {'success': success_count, 'failed': failed_count}


def run_demo_mode(output_dir: Path, params: dict, verbose: bool = False) -> bool:
    """
    Run demo mode using synthetic test tracks.

    Parameters:
        output_dir: Output directory for demo results
        params: Parameters dict
        verbose: Print verbose messages

    Returns:
        True if successful
    """
    print("Running demo mode with synthetic audio...")

    # Import synthetic audio generators
    from tests.test_synthetic import (
        generate_build_then_drop,
        generate_repetitive_loop,
        generate_section_contrast
    )

    sr = params.get('target_sr', config.TARGET_SAMPLE_RATE)

    # Generate test tracks
    test_tracks = [
        {
            'name': 'demo_build_drop',
            'audio': generate_build_then_drop(duration=30, sr=sr)[0],
            'description': 'Build-then-drop pattern'
        },
        {
            'name': 'demo_repetitive',
            'audio': generate_repetitive_loop(duration=45, sr=sr),
            'description': 'Repetitive loop pattern'
        },
        {
            'name': 'demo_contrast',
            'audio': generate_section_contrast(duration=40, sr=sr)[0],
            'description': 'Section contrast pattern'
        }
    ]

    print(f"Generated {len(test_tracks)} synthetic test tracks")

    for track_info in test_tracks:
        print(f"\nProcessing: {track_info['name']} ({track_info['description']})")
        print("-" * 60)

        try:
            # Create audio metadata
            audio_metadata = {
                'audio': track_info['audio'],
                'sample_rate': sr,
                'duration': len(track_info['audio']) / sr,
                'preprocessing': {
                    'original_sr': sr,
                    'resampled': False,
                    'normalization_method': 'peak',
                    'normalization_factor': 1.0,
                    'trimmed': False,
                    'trim_samples': None
                }
            }

            audio = track_info['audio']

            # Process using kernel functions
            if verbose:
                print("Processing with kernel functions...")

            analysis_results = process_audio_array(
                audio, sr, params, audio_metadata, verbose
            )

            # Export
            if verbose:
                print("Exporting results...")

            track_output_dir = output_dir / track_info['name']
            created_files = export.export_all_outputs(
                analysis_results,
                track_output_dir,
                track_info['name'],
                generate_plots=True
            )

            print(f"Created {len(created_files)} output files in {track_output_dir}")

            # Print summary
            import json
            summary_path = track_output_dir / f"{track_info['name']}_summary.json"
            with open(summary_path) as f:
                summary = json.load(f)
            export.print_analysis_summary(summary, track_info['name'])

        except Exception as e:
            print(f"ERROR: {e}", file=sys.stderr)
            if verbose:
                import traceback
                traceback.print_exc()
            return False

    print(f"\nDemo complete! Results saved to {output_dir}")
    return True


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='director-signals - Offline perceptual analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze single file
  %(prog)s track.wav --output results/

  # Analyze directory
  %(prog)s tracks/ --output results/

  # Run demo mode
  %(prog)s --demo --output demo_results/

  # Verbose output
  %(prog)s track.wav --output results/ --verbose
        """
    )

    parser.add_argument(
        'input',
        nargs='?',
        type=str,
        help='Input audio file or directory (not needed for --demo)'
    )

    parser.add_argument(
        '--output', '-o',
        type=str,
        required=True,
        help='Output directory for results'
    )

    parser.add_argument(
        '--demo',
        action='store_true',
        help='Run demo mode with synthetic test tracks (no input file needed)'
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Print verbose progress messages'
    )

    parser.add_argument(
        '--no-plots',
        action='store_true',
        help='Skip plot generation'
    )

    # Parameter overrides
    parser.add_argument(
        '--frame-length',
        type=int,
        help=f'Frame length in samples (default: {config.FRAME_LENGTH})'
    )

    parser.add_argument(
        '--hop-length',
        type=int,
        help=f'Hop length in samples (default: {config.HOP_LENGTH})'
    )

    parser.add_argument(
        '--block-duration',
        type=float,
        help=f'Block duration in seconds (default: {config.BLOCK_DURATION_SEC})'
    )

    parser.add_argument(
        '--target-sr',
        type=int,
        help=f'Target sample rate (default: {config.TARGET_SAMPLE_RATE})'
    )

    args = parser.parse_args()

    # Validate arguments
    if not args.demo and not args.input:
        parser.error("Either provide an input file/directory or use --demo")

    # Build parameters dict
    params = {
        'target_sr': args.target_sr or config.TARGET_SAMPLE_RATE,
        'frame_length': args.frame_length or config.FRAME_LENGTH,
        'hop_length': args.hop_length or config.HOP_LENGTH,
        'block_duration': args.block_duration or config.BLOCK_DURATION_SEC,
        'normalize_method': config.NORMALIZATION_METHOD,
        'block_normalize_method': config.BLOCK_NORMALIZE_METHOD,
        'trim_silence': config.TRIM_SILENCE,
        'generate_plots': not args.no_plots
    }

    output_dir = Path(args.output)

    # Run appropriate mode
    if args.demo:
        success = run_demo_mode(output_dir, params, args.verbose)
        sys.exit(0 if success else 1)

    else:
        input_path = Path(args.input)

        if not input_path.exists():
            print(f"ERROR: Input path does not exist: {input_path}", file=sys.stderr)
            sys.exit(1)

        if input_path.is_file():
            # Single file
            success = process_single_track(input_path, output_dir, params, args.verbose)
            sys.exit(0 if success else 1)

        elif input_path.is_dir():
            # Directory
            results = process_directory(input_path, output_dir, params, args.verbose)
            sys.exit(0 if results['failed'] == 0 else 1)

        else:
            print(f"ERROR: Invalid input path: {input_path}", file=sys.stderr)
            sys.exit(1)


if __name__ == '__main__':
    main()

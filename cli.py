#!/usr/bin/env python3
"""
director-signals - Command Line Interface

Main entry point for running analysis on audio tracks.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional, List
import numpy as np

import config
from src import audio_io, features, aggregation, metrics, events, export


def process_single_track(
    file_path: Path,
    output_dir: Path,
    params: dict,
    verbose: bool = False
) -> bool:
    """
    Process a single audio track through the full pipeline.

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

        # Step 2: Extract frame-level features
        if verbose:
            print("2. Extracting frame-level features...")

        frame_features = features.extract_all_features(
            audio, sr,
            frame_length=params.get('frame_length', config.FRAME_LENGTH),
            hop_length=params.get('hop_length', config.HOP_LENGTH)
        )

        if verbose:
            print(f"   Extracted {frame_features['metadata']['n_frames']} frames")

        # Step 3: Aggregate to blocks
        if verbose:
            print("3. Aggregating features to blocks...")

        block_features, feature_names, block_times = aggregation.aggregate_frame_features(
            frame_features, sr,
            frame_hop=params.get('hop_length', config.HOP_LENGTH),
            block_duration_sec=params.get('block_duration', config.BLOCK_DURATION_SEC),
            duration_sec=duration  # Pass track duration for canonical timebase
        )

        if verbose:
            print(f"   Created {len(block_times)} blocks")

        # Normalize block features
        block_features_norm, norm_params = aggregation.normalize_block_features(
            block_features,
            method=params.get('block_normalize_method', config.BLOCK_NORMALIZE_METHOD)
        )

        # Step 4: Compute curves
        if verbose:
            print("4. Computing long-horizon curves...")

        curves = metrics.compute_all_curves(
            block_features_norm,
            feature_names,
            audio=audio,
            sr=sr
        )

        if verbose:
            print(f"   Computed tension, novelty, and fatigue curves")

        # Step 5: Detect events
        if verbose:
            print("5. Detecting events...")

        detected_events = events.detect_all_events(
            curves['tension_smooth'],
            curves['novelty'],
            curves['fatigue'],
            block_times,
            audio=audio,
            sr=sr,
            duration_sec=duration  # Pass track duration for event validation
        )

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
            audio_data = {
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

            audio = audio_data['audio']

            # Extract features
            if verbose:
                print("Extracting features...")

            frame_features = features.extract_all_features(audio, sr)

            # Aggregate to blocks
            block_features, feature_names, block_times = aggregation.aggregate_frame_features(
                frame_features, sr
            )

            block_features_norm, _ = aggregation.normalize_block_features(block_features)

            # Compute curves
            if verbose:
                print("Computing curves...")

            curves = metrics.compute_all_curves(
                block_features_norm, feature_names, audio=audio, sr=sr
            )

            # Detect events
            if verbose:
                print("Detecting events...")

            detected_events = events.detect_all_events(
                curves['tension_smooth'],
                curves['novelty'],
                curves['fatigue'],
                block_times,
                audio=audio,
                sr=sr
            )

            # Export
            if verbose:
                print("Exporting results...")

            analysis_results = {
                'audio_metadata': audio_data,
                'params': params,
                'curves': curves,
                'events': detected_events,
                'block_times': block_times
            }

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

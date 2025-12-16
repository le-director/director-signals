"""
Export Module

Generate JSON outputs and plots for analysis results.
All outputs follow versioned schema for consistency.
"""

import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import config
from src import timebase


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


def create_metrics_json(
    audio_metadata: Dict,
    params: Dict,
    curves_dict: Dict,
    events_dict: Dict,
    block_times: np.ndarray
) -> Dict:
    """
    Create complete metrics JSON following schema.

    Parameters:
        audio_metadata: Dict from preprocess_audio with duration, sr, etc.
        params: Dict of all parameters used
        curves_dict: Dict from compute_all_curves
        events_dict: Dict from detect_all_events
        block_times: Array of block center times

    Returns:
        Complete metrics dict ready for JSON serialization
    """
    # Calculate sampling interval
    sampling_interval_sec = float(block_times[1] - block_times[0]) if len(block_times) > 1 else config.BLOCK_DURATION_SEC

    # Calculate time axis validation info
    duration_sec = audio_metadata['duration']
    max_block_time = float(np.max(block_times)) if len(block_times) > 0 else 0.0
    time_axis_valid = max_block_time <= duration_sec + timebase.EPSILON_SEC

    metrics = {
        'schema_version': config.SCHEMA_VERSION,

        'track_metadata': {
            'duration': audio_metadata['duration'],
            'sample_rate': audio_metadata['sample_rate'],
            'preprocessing': audio_metadata['preprocessing'],
            'time_axis_info': {
                'max_block_time': max_block_time,
                'duration_sec': duration_sec,
                'time_axis_valid': time_axis_valid,
                'n_blocks': len(block_times),
                'block_duration_sec': sampling_interval_sec
            }
        },

        'params': params,

        'curves': {
            'tension_raw': {
                'values': curves_dict['tension_raw'],
                'sampling_interval_sec': sampling_interval_sec,
                'start_time_sec': float(block_times[0]),
                'length': len(curves_dict['tension_raw']),
                'description': 'Raw tension/energy curve before smoothing',
                'range': [0.0, 1.0],
                'normalization': curves_dict.get('tension_normalization', {})
            },
            'tension_smooth': {
                'values': curves_dict['tension_smooth'],
                'sampling_interval_sec': sampling_interval_sec,
                'start_time_sec': float(block_times[0]),
                'length': len(curves_dict['tension_smooth']),
                'description': 'Smoothed tension/energy curve',
                'range': [0.0, 1.0]
            },
            'novelty': {
                'values': curves_dict['novelty'],
                'sampling_interval_sec': sampling_interval_sec,
                'start_time_sec': float(block_times[0]),
                'length': len(curves_dict['novelty']),
                'description': 'Novelty curve (degree of change)',
                'range': [0.0, 1.0]
            },
            'fatigue': {
                'values': curves_dict['fatigue'],
                'sampling_interval_sec': sampling_interval_sec,
                'start_time_sec': float(block_times[0]),
                'length': len(curves_dict['fatigue']),
                'description': 'Fatigue curve (repetition/stagnation)',
                'range': [0.0, 1.0],
                'computation_mode': curves_dict.get('fatigue_components', {}).get('computation_mode', 'weighted_average')
            }
        },

        'events': {
            'candidate_drops': events_dict.get('candidate_drops', []),
            'ranked_drops': events_dict.get('ranked_drops', []),
            'stagnant_segments': events_dict.get('stagnant_segments', []),
            'boundaries': events_dict.get('boundaries', [])
        }
    }

    return metrics


def create_summary_json(metrics_json: Dict) -> Dict:
    """
    Create summary JSON with key statistics.

    Parameters:
        metrics_json: Full metrics JSON from create_metrics_json

    Returns:
        Summary dict with top-level stats
    """
    # Extract key information
    duration = metrics_json['track_metadata']['duration']

    # Find top tension peak
    tension_values = np.array(metrics_json['curves']['tension_smooth']['values'])
    if len(tension_values) > 0:
        top_tension_idx = np.argmax(tension_values)
        sampling_interval = metrics_json['curves']['tension_smooth']['sampling_interval_sec']
        start_time = metrics_json['curves']['tension_smooth']['start_time_sec']
        top_tension_time = start_time + top_tension_idx * sampling_interval
        # Clamp to track duration to ensure validity
        top_tension_time = min(top_tension_time, duration)
        top_tension_value = float(tension_values[top_tension_idx])
    else:
        top_tension_time = 0.0
        top_tension_value = 0.0

    # Count candidate drops
    n_drops = len(metrics_json['events']['candidate_drops'])

    # Find longest stagnant segment
    stagnant_segments = metrics_json['events']['stagnant_segments']
    if stagnant_segments:
        longest_stagnant = max(stagnant_segments, key=lambda x: x['duration'])
        longest_stagnant_duration = longest_stagnant['duration']
    else:
        longest_stagnant_duration = 0.0

    # Top ranked drops
    ranked_drops = metrics_json['events'].get('ranked_drops', [])
    top_drops = []
    for drop in ranked_drops[:5]:
        top_drops.append({
            'time': drop['time'],
            'combined_score': drop.get('combined_score', drop['score'])
        })

    summary = {
        'schema_version': config.SCHEMA_VERSION,
        'duration_sec': duration,
        'top_tension_peak': {
            'time_sec': top_tension_time,
            'value': top_tension_value
        },
        'num_candidate_drops': n_drops,
        'longest_stagnant_duration_sec': longest_stagnant_duration,
        'top_drops': top_drops,
        'num_stagnant_segments': len(stagnant_segments)
    }

    return summary


def save_json(data: Dict, output_path: Path) -> None:
    """
    Save data as JSON with pretty printing.

    Parameters:
        data: Dictionary to save
        output_path: Path to output file
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2, cls=NumpyEncoder)


def plot_curves_and_events(
    curves_dict: Dict,
    events_dict: Dict,
    block_times: np.ndarray,
    output_path: Path,
    title: str = "Audio Analysis"
) -> None:
    """
    Plot curves with event markers.

    Parameters:
        curves_dict: Dict from compute_all_curves
        events_dict: Dict from detect_all_events
        block_times: Array of block times
        output_path: Path to save plot
        title: Plot title
    """
    fig, axes = plt.subplots(3, 1, figsize=config.PLOT_FIGSIZE, sharex=True)

    # Plot 1: Tension curves
    ax1 = axes[0]
    ax1.plot(block_times, curves_dict['tension_raw'],
             label='Tension (raw)', alpha=0.3, color='red', linewidth=1)
    ax1.plot(block_times, curves_dict['tension_smooth'],
             label='Tension (smooth)', color='red', linewidth=2)

    # Mark candidate drops
    candidate_drops = events_dict.get('candidate_drops', [])
    for drop in candidate_drops[:config.TOP_N_DROPS]:
        ax1.axvline(drop['time'], color='purple', alpha=0.5,
                   linestyle='--', linewidth=1)

    ax1.set_ylabel('Tension', fontsize=10)
    ax1.set_title(title, fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-0.05, 1.05)

    # Plot 2: Novelty curve
    ax2 = axes[1]
    ax2.plot(block_times, curves_dict['novelty'],
             label='Novelty', color='blue', linewidth=2)

    # Mark section boundaries
    boundaries = events_dict.get('boundaries', [])
    for boundary in boundaries:
        ax2.axvline(boundary['time'], color='green', alpha=0.5,
                   linestyle=':', linewidth=1)

    ax2.set_ylabel('Novelty', fontsize=10)
    ax2.legend(loc='upper right', fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-0.05, 1.05)

    # Plot 3: Fatigue curve with stagnant segments
    ax3 = axes[2]
    ax3.plot(block_times, curves_dict['fatigue'],
             label='Fatigue', color='orange', linewidth=2)

    # Shade stagnant segments
    stagnant_segments = events_dict.get('stagnant_segments', [])
    for segment in stagnant_segments:
        ax3.axvspan(segment['start_time'], segment['end_time'],
                   alpha=0.2, color='red', label='_nolegend_')

    ax3.set_xlabel('Time (seconds)', fontsize=10)
    ax3.set_ylabel('Fatigue', fontsize=10)
    ax3.legend(loc='upper right', fontsize=8)
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(-0.05, 1.05)

    plt.tight_layout()

    # Save plot
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=config.PLOT_DPI, bbox_inches='tight')
    plt.close(fig)


def plot_tension_components(
    component_contributions: Dict[str, np.ndarray],
    block_times: np.ndarray,
    output_path: Path,
    title: str = "Tension Components"
) -> None:
    """
    Plot individual components contributing to tension.

    Parameters:
        component_contributions: Dict of component curves
        block_times: Array of block times
        output_path: Path to save plot
        title: Plot title
    """
    fig, ax = plt.subplots(figsize=(config.PLOT_FIGSIZE[0], 6))

    # Plot each component
    colors = {'rms': 'red', 'onset_density': 'orange',
              'spectral_centroid': 'blue', 'spectral_bandwidth': 'green'}

    for comp_name, curve in component_contributions.items():
        color = colors.get(comp_name, 'gray')
        weight = config.TENSION_WEIGHTS.get(comp_name, 0.0)
        label = f"{comp_name} (w={weight:.2f})"
        ax.plot(block_times, curve, label=label, color=color, linewidth=1.5, alpha=0.7)

    ax.set_xlabel('Time (seconds)', fontsize=10)
    ax.set_ylabel('Normalized Component Value', fontsize=10)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)

    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=config.PLOT_DPI, bbox_inches='tight')
    plt.close(fig)


def export_all_outputs(
    analysis_results: Dict,
    output_dir: Path,
    track_name: str,
    generate_plots: bool = True
) -> List[Path]:
    """
    Export all outputs: JSON files and plots.

    Parameters:
        analysis_results: Dict containing all analysis data:
            - 'audio_metadata': from preprocess_audio
            - 'params': parameters used
            - 'curves': from compute_all_curves
            - 'events': from detect_all_events
            - 'block_times': array of block times
        output_dir: Output directory path
        track_name: Name of track (for filenames)
        generate_plots: Whether to generate plot files

    Returns:
        List of paths to created files
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    created_files = []

    # Create metrics JSON
    metrics_json = create_metrics_json(
        analysis_results['audio_metadata'],
        analysis_results['params'],
        analysis_results['curves'],
        analysis_results['events'],
        analysis_results['block_times']
    )

    metrics_path = output_dir / f"{track_name}_metrics.json"
    save_json(metrics_json, metrics_path)
    created_files.append(metrics_path)

    # Create summary JSON
    summary_json = create_summary_json(metrics_json)
    summary_path = output_dir / f"{track_name}_summary.json"
    save_json(summary_json, summary_path)
    created_files.append(summary_path)

    # Create segments JSON (detailed event info)
    segments_json = {
        'schema_version': config.SCHEMA_VERSION,
        'track_name': track_name,
        'stagnant_segments': analysis_results['events'].get('stagnant_segments', []),
        'candidate_drops': analysis_results['events'].get('candidate_drops', []),
        'ranked_drops': analysis_results['events'].get('ranked_drops', []),
        'section_boundaries': analysis_results['events'].get('boundaries', [])
    }
    segments_path = output_dir / f"{track_name}_segments.json"
    save_json(segments_json, segments_path)
    created_files.append(segments_path)

    # Generate plots
    if generate_plots:
        # Main curves and events plot
        plots_path = output_dir / f"{track_name}_plots.png"
        plot_curves_and_events(
            analysis_results['curves'],
            analysis_results['events'],
            analysis_results['block_times'],
            plots_path,
            title=f"Audio Analysis: {track_name}"
        )
        created_files.append(plots_path)

        # Tension components plot
        if 'tension_components' in analysis_results['curves']:
            components_path = output_dir / f"{track_name}_tension_components.png"
            plot_tension_components(
                analysis_results['curves']['tension_components'],
                analysis_results['block_times'],
                components_path,
                title=f"Tension Components: {track_name}"
            )
            created_files.append(components_path)

    return created_files


def print_analysis_summary(summary_json: Dict, track_name: str) -> None:
    """
    Print concise analysis summary to console.

    Parameters:
        summary_json: Summary JSON dict
        track_name: Track name
    """
    print(f"\n{'='*60}")
    print(f"Analysis Summary: {track_name}")
    print(f"{'='*60}")
    print(f"Duration: {summary_json['duration_sec']:.2f} seconds")
    print(f"Top tension peak: {summary_json['top_tension_peak']['time_sec']:.2f}s "
          f"(value: {summary_json['top_tension_peak']['value']:.3f})")
    print(f"Candidate drops detected: {summary_json['num_candidate_drops']}")

    if summary_json['top_drops']:
        print(f"\nTop {len(summary_json['top_drops'])} drops:")
        for i, drop in enumerate(summary_json['top_drops'], 1):
            print(f"  {i}. Time: {drop['time']:.2f}s, Score: {drop['combined_score']:.3f}")

    print(f"\nStagnant segments: {summary_json['num_stagnant_segments']}")
    if summary_json['longest_stagnant_duration_sec'] > 0:
        print(f"Longest stagnant segment: {summary_json['longest_stagnant_duration_sec']:.2f}s")

    print(f"{'='*60}\n")

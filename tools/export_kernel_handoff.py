#!/usr/bin/env python3
"""
Export kernel handoff package for Phase-2 C++ development.

Generates a package containing all parameters, golden manifests, and fixture info
needed for C++ implementation parity testing.

Usage:
    python tools/export_kernel_handoff.py \
        --kernel-version 1.1.1 \
        --timebase-version 1 \
        --golden-root golden_outputs/kernel_v1.1.1/timebase_v1/ \
        --fixtures-dir fixtures/synthetic_audio/ \
        --out kernel_handoff/kernel_v1.1.1/timebase_v1/

Output files:
    - kernel_contract.json: All parameters C++ needs to replicate
    - golden_manifest.json: List of golden metrics.json files (synthetic only)
    - fixtures_manifest.json: SHA256 of each fixture WAV file
    - diff_summary.md: Compare current vs previous version
"""

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def compute_wav_file_sha256(wav_path: Path) -> str:
    """Compute SHA256 of WAV file bytes."""
    with open(wav_path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()


def deep_diff(old: Any, new: Any, path: str = "") -> List[Dict]:
    """
    Recursively compute differences between two structures.

    Returns list of {path, old_value, new_value} dicts.
    """
    diffs = []

    if type(old) != type(new):
        diffs.append({"path": path, "old": old, "new": new})
        return diffs

    if isinstance(old, dict):
        all_keys = set(old.keys()) | set(new.keys())
        for key in sorted(all_keys):
            new_path = f"{path}.{key}" if path else key
            old_val = old.get(key)
            new_val = new.get(key)
            if key not in old:
                diffs.append({"path": new_path, "old": None, "new": new_val, "type": "added"})
            elif key not in new:
                diffs.append({"path": new_path, "old": old_val, "new": None, "type": "removed"})
            else:
                diffs.extend(deep_diff(old_val, new_val, new_path))
    elif isinstance(old, list):
        if old != new:
            diffs.append({"path": path, "old": old, "new": new})
    else:
        if old != new:
            diffs.append({"path": path, "old": old, "new": new})

    return diffs


# =============================================================================
# GOLDEN MANIFEST LOADING
# =============================================================================

def load_golden_metrics(golden_root: Path) -> List[Dict]:
    """
    Load all synthetic golden metrics.json files.

    Parameters:
        golden_root: Path to versioned golden output (e.g., golden_outputs/kernel_v1.1.1/timebase_v1/)

    Returns:
        List of dicts with track info and metrics
    """
    synthetic_path = golden_root / 'synthetic'
    if not synthetic_path.exists():
        return []

    tracks = []
    for track_dir in sorted(synthetic_path.iterdir()):
        if not track_dir.is_dir():
            continue

        metrics_path = track_dir / 'metrics.json'
        if not metrics_path.exists():
            continue

        with open(metrics_path, 'r') as f:
            metrics = json.load(f)

        tracks.append({
            'name': track_dir.name,
            'metrics_path': str(metrics_path.relative_to(golden_root)),
            'metrics': metrics,
        })

    return tracks


# =============================================================================
# CONTRACT GENERATION
# =============================================================================

def extract_kernel_contract(
    metrics: Dict,
    kernel_version: str,
    timebase_version: str
) -> Dict:
    """
    Extract kernel contract params from a metrics.json file.

    Parameters:
        metrics: Loaded metrics.json dict
        kernel_version: Kernel version string
        timebase_version: Timebase version string

    Returns:
        Structured kernel contract dict
    """
    params = metrics.get('params', {})

    contract = {
        "kernel_version": kernel_version,
        "timebase_version": timebase_version,
        "schema_version": metrics.get('schema_version', 'unknown'),
        "generated_at": datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),

        "timebase": {
            "block_duration_sec": params.get('block_duration_sec', 0.5),
            "start_time_sec": 0.0,
        },

        "frame_params": {
            "frame_length": params.get('frame_length', 2048),
            "hop_length": params.get('hop_length', 512),
            "sample_rate": params.get('sample_rate', 22050),
        },

        "curves": {
            "names": ["tension_raw", "tension_smooth", "novelty", "fatigue"],
            "expected_range": [0.0, 1.0],
            "sampling_interval_sec": params.get('block_duration_sec', 0.5),
        },

        "tension": {
            "weights": params.get('tension_weights', {
                "rms": 0.4,
                "onset_density": 0.3,
                "spectral_centroid": 0.2,
                "spectral_bandwidth": 0.1,
            }),
            "smooth_alpha": params.get('tension_smooth_alpha', 0.3),
            "percentile_lower": params.get('tension_percentile_lower', 5.0),
            "percentile_upper": params.get('tension_percentile_upper', 95.0),
            "normalization_mode": params.get('tension_normalization_mode', 'robust_track'),
        },

        "novelty": {
            "lookback_blocks": params.get('novelty_lookback_blocks', 16),
            "smooth_window": params.get('novelty_smooth_window', 3),
        },

        "fatigue": {
            "window_blocks": params.get('fatigue_window_blocks', 32),
            "smooth_window": params.get('fatigue_smooth_window', 5),
            "gain_up": params.get('fatigue_gain_up', 0.02),
            "gain_down": params.get('fatigue_gain_down', 0.08),
            "novelty_spike_threshold": params.get('fatigue_novelty_spike_threshold', 0.5),
            "boring_weights": params.get('fatigue_boring_weights', {
                "self_similarity": 0.5,
                "inverse_novelty": 0.3,
                "inverse_variance": 0.2,
            }),
            "use_leaky_integrator": params.get('fatigue_use_leaky_integrator', True),
        },

        "spectral": {
            "rolloff_percent": params.get('spectral_rolloff_percent', 0.85),
            "n_mfcc": params.get('n_mfcc', 13),
            "n_mels": params.get('n_mels', 40),
        },

        "anchored_normalization": {
            "rms_max_dbfs": params.get('anchored_rms_max_dbfs', -6.0),
            "onset_max": params.get('anchored_onset_max', 50.0),
            "centroid_max_hz": params.get('anchored_centroid_max_hz', 8000.0),
            "bandwidth_max_hz": params.get('anchored_bandwidth_max_hz', 6000.0),
        },

        "normalization": {
            "percentile_lower": params.get('normalization_percentile_lower', 1.0),
            "percentile_upper": params.get('normalization_percentile_upper', 99.0),
        },
    }

    return contract


def generate_golden_manifest(
    tracks: List[Dict],
    kernel_version: str,
    timebase_version: str,
    golden_root: str
) -> Dict:
    """
    Generate golden manifest from loaded tracks.

    Parameters:
        tracks: List of track dicts from load_golden_metrics
        kernel_version: Kernel version string
        timebase_version: Timebase version string
        golden_root: Path string to golden root

    Returns:
        Golden manifest dict
    """
    track_entries = []

    for track in tracks:
        metrics = track['metrics']
        track_meta = metrics.get('track_metadata', {})
        curves = metrics.get('curves', {})

        curve_names = list(curves.keys())
        # Filter to main curves only (exclude components/normalization info)
        main_curves = [c for c in curve_names if c in ['tension_raw', 'tension_smooth', 'novelty', 'fatigue']]

        # Get curve length from first curve
        n_blocks = 0
        if main_curves and main_curves[0] in curves:
            n_blocks = curves[main_curves[0]].get('length', 0)

        track_entries.append({
            "name": track['name'],
            "metrics_path": track['metrics_path'],
            "duration_sec": track_meta.get('duration', 0),
            "audio_checksum": track_meta.get('audio_checksum', ''),
            "n_blocks": n_blocks,
            "curves": main_curves,
        })

    return {
        "kernel_version": kernel_version,
        "timebase_version": timebase_version,
        "golden_root": golden_root,
        "generated_at": datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
        "tracks": track_entries,
    }


def generate_fixtures_manifest(fixtures_dir: Path) -> Dict:
    """
    Generate fixtures manifest with SHA256 hashes.

    Parameters:
        fixtures_dir: Path to fixtures directory

    Returns:
        Fixtures manifest dict
    """
    # Check for existing manifest
    existing_manifest_path = fixtures_dir / 'fixtures_manifest.json'
    if existing_manifest_path.exists():
        with open(existing_manifest_path, 'r') as f:
            existing = json.load(f)
        # Add fixtures_dir path to the copy
        existing['fixtures_dir'] = str(fixtures_dir)
        return existing

    # Generate fresh if no existing manifest
    fixtures = []
    for wav_file in sorted(fixtures_dir.glob('*.wav')):
        sha256 = compute_wav_file_sha256(wav_file)
        # Get basic info - duration would require scipy, so we skip it
        fixtures.append({
            "name": wav_file.stem,
            "filename": wav_file.name,
            "sha256_bytes": sha256,
        })

    return {
        "version": "1.0",
        "fixtures_dir": str(fixtures_dir),
        "fixtures": fixtures,
    }


# =============================================================================
# DIFF GENERATION
# =============================================================================

def load_previous_contract(out_dir: Path) -> Optional[Dict]:
    """
    Load previous kernel_contract.json if exists.

    Parameters:
        out_dir: Output directory to check

    Returns:
        Previous contract dict or None
    """
    contract_path = out_dir / 'kernel_contract.json'
    if not contract_path.exists():
        return None

    with open(contract_path, 'r') as f:
        return json.load(f)


def generate_diff_summary(
    old_contract: Optional[Dict],
    new_contract: Dict
) -> str:
    """
    Generate markdown diff summary between contracts.

    Parameters:
        old_contract: Previous contract or None
        new_contract: New contract

    Returns:
        Markdown string
    """
    if old_contract is None:
        return f"""# Kernel Contract: v{new_contract['kernel_version']}

## Initial Version

This is the first kernel contract for this version. No previous version to compare against.

### Versions
- kernel_version: {new_contract['kernel_version']}
- timebase_version: {new_contract['timebase_version']}
- schema_version: {new_contract['schema_version']}

### Curves
- Names: {', '.join(new_contract['curves']['names'])}
- Expected range: {new_contract['curves']['expected_range']}
- Sampling interval: {new_contract['curves']['sampling_interval_sec']}s
"""

    old_version = old_contract.get('kernel_version', 'unknown')
    new_version = new_contract.get('kernel_version', 'unknown')

    # Compute diffs, excluding generated_at timestamps
    old_copy = {k: v for k, v in old_contract.items() if k != 'generated_at'}
    new_copy = {k: v for k, v in new_contract.items() if k != 'generated_at'}
    diffs = deep_diff(old_copy, new_copy)

    lines = [
        f"# Kernel Contract Diff: v{old_version} -> v{new_version}",
        "",
        f"Generated: {new_contract['generated_at']}",
        "",
        "## Version Changes",
    ]

    # Version changes
    version_keys = ['kernel_version', 'timebase_version', 'schema_version']
    version_diffs = [d for d in diffs if d['path'] in version_keys]
    if version_diffs:
        for d in version_diffs:
            lines.append(f"- {d['path']}: {d['old']} -> {d['new']}")
    else:
        lines.append("- (no version changes)")

    # Group diffs by section
    sections = ['timebase', 'frame_params', 'curves', 'tension', 'novelty', 'fatigue', 'spectral', 'anchored_normalization', 'normalization']

    lines.append("")
    lines.append("## Parameter Changes")

    param_diffs = [d for d in diffs if d['path'] not in version_keys]

    if not param_diffs:
        lines.append("")
        lines.append("No parameter changes detected.")
    else:
        for section in sections:
            section_diffs = [d for d in param_diffs if d['path'].startswith(section)]
            if section_diffs:
                lines.append("")
                lines.append(f"### {section.replace('_', ' ').title()}")
                for d in section_diffs:
                    param_name = d['path'].replace(f"{section}.", "")
                    if d.get('type') == 'added':
                        lines.append(f"- {param_name}: (new) {d['new']}")
                    elif d.get('type') == 'removed':
                        lines.append(f"- {param_name}: {d['old']} (removed)")
                    else:
                        lines.append(f"- {param_name}: {d['old']} -> {d['new']}")

    lines.append("")
    return "\n".join(lines)


# =============================================================================
# MAIN EXPORT FUNCTION
# =============================================================================

def export_kernel_handoff(
    kernel_version: str,
    timebase_version: str,
    golden_root: Path,
    fixtures_dir: Path,
    out_dir: Path
) -> Dict[str, Path]:
    """
    Export complete kernel handoff package.

    Parameters:
        kernel_version: Kernel version string (e.g., "1.1.1")
        timebase_version: Timebase version string (e.g., "1")
        golden_root: Path to versioned golden outputs
        fixtures_dir: Path to fixtures directory
        out_dir: Output directory for handoff package

    Returns:
        Dict mapping output name to file path
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs = {}

    # Load golden metrics
    print(f"Loading golden metrics from {golden_root}...")
    tracks = load_golden_metrics(golden_root)
    if not tracks:
        print(f"WARNING: No synthetic golden tracks found in {golden_root}")

    # Load previous contract for diff
    print(f"Checking for previous contract in {out_dir}...")
    old_contract = load_previous_contract(out_dir)
    if old_contract:
        print(f"  Found previous contract v{old_contract.get('kernel_version', 'unknown')}")

    # Generate kernel contract from first track (all should have same params)
    print("Generating kernel_contract.json...")
    if tracks:
        contract = extract_kernel_contract(tracks[0]['metrics'], kernel_version, timebase_version)
    else:
        # Fallback: create minimal contract
        contract = {
            "kernel_version": kernel_version,
            "timebase_version": timebase_version,
            "schema_version": "unknown",
            "generated_at": datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
            "curves": {"names": ["tension_raw", "tension_smooth", "novelty", "fatigue"]},
        }

    contract_path = out_dir / 'kernel_contract.json'
    with open(contract_path, 'w') as f:
        json.dump(contract, f, indent=2)
    outputs['kernel_contract'] = contract_path
    print(f"  Written: {contract_path}")

    # Generate golden manifest
    print("Generating golden_manifest.json...")
    golden_manifest = generate_golden_manifest(tracks, kernel_version, timebase_version, str(golden_root))
    manifest_path = out_dir / 'golden_manifest.json'
    with open(manifest_path, 'w') as f:
        json.dump(golden_manifest, f, indent=2)
    outputs['golden_manifest'] = manifest_path
    print(f"  Written: {manifest_path}")

    # Generate fixtures manifest
    print(f"Generating fixtures_manifest.json from {fixtures_dir}...")
    fixtures_manifest = generate_fixtures_manifest(fixtures_dir)
    fixtures_path = out_dir / 'fixtures_manifest.json'
    with open(fixtures_path, 'w') as f:
        json.dump(fixtures_manifest, f, indent=2)
    outputs['fixtures_manifest'] = fixtures_path
    print(f"  Written: {fixtures_path}")

    # Generate diff summary
    print("Generating diff_summary.md...")
    diff_md = generate_diff_summary(old_contract, contract)
    diff_path = out_dir / 'diff_summary.md'
    with open(diff_path, 'w') as f:
        f.write(diff_md)
    outputs['diff_summary'] = diff_path
    print(f"  Written: {diff_path}")

    return outputs


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Export kernel handoff package for Phase-2 C++ development',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Export handoff package
    python tools/export_kernel_handoff.py \\
        --kernel-version 1.1.1 \\
        --timebase-version 1 \\
        --golden-root golden_outputs/kernel_v1.1.1/timebase_v1/ \\
        --fixtures-dir fixtures/synthetic_audio/ \\
        --out kernel_handoff/kernel_v1.1.1/timebase_v1/
"""
    )

    parser.add_argument(
        '--kernel-version', '-k',
        required=True,
        help='Kernel version (e.g., 1.1.1)'
    )
    parser.add_argument(
        '--timebase-version', '-t',
        required=True,
        help='Timebase version (e.g., 1)'
    )
    parser.add_argument(
        '--golden-root', '-g',
        required=True,
        help='Path to versioned golden outputs (e.g., golden_outputs/kernel_v1.1.1/timebase_v1/)'
    )
    parser.add_argument(
        '--fixtures-dir', '-f',
        default='fixtures/synthetic_audio',
        help='Path to fixtures directory (default: fixtures/synthetic_audio)'
    )
    parser.add_argument(
        '--out', '-o',
        required=True,
        help='Output directory for handoff package'
    )

    args = parser.parse_args()

    golden_root = Path(args.golden_root)
    fixtures_dir = Path(args.fixtures_dir)
    out_dir = Path(args.out)

    if not golden_root.exists():
        print(f"ERROR: Golden root not found: {golden_root}")
        return 1

    if not fixtures_dir.exists():
        print(f"ERROR: Fixtures directory not found: {fixtures_dir}")
        return 1

    print(f"Exporting kernel handoff package...")
    print(f"  Kernel version: {args.kernel_version}")
    print(f"  Timebase version: {args.timebase_version}")
    print(f"  Golden root: {golden_root}")
    print(f"  Fixtures dir: {fixtures_dir}")
    print(f"  Output dir: {out_dir}")
    print()

    outputs = export_kernel_handoff(
        kernel_version=args.kernel_version,
        timebase_version=args.timebase_version,
        golden_root=golden_root,
        fixtures_dir=fixtures_dir,
        out_dir=out_dir,
    )

    print()
    print("Handoff package exported successfully!")
    print(f"  Files: {len(outputs)}")
    for name, path in outputs.items():
        print(f"    - {name}: {path}")

    return 0


if __name__ == '__main__':
    sys.exit(main())

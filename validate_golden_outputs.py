#!/usr/bin/env python3
"""
Golden Output Validation Script
Validates director-signals kernel outputs for Phase-2 C++ port reference use.
"""

import json
import math
from pathlib import Path

# Track directories to validate
TRACKS = [
    ("synthetic/build_drop", "Synthetic Build-Drop", "electronic/synthetic", 30.0),
    ("synthetic/repetitive_loop", "Synthetic Repetitive Loop", "electronic/synthetic", 60.0),
    ("synthetic/contrast", "Synthetic Contrast", "electronic/synthetic", 40.0),
    ("real_local/haunted", "Haunted", "industrial/experimental pop", 372.88),
    ("real_local/kendrick", "Kendrick", "hip-hop", 223.19),
]

BASE_PATH = Path(__file__).parent / "golden_outputs/kernel_v1.1.1/timebase_v1"

class ValidationReport:
    def __init__(self, track_name: str, genre: str):
        self.track_name = track_name
        self.genre = genre
        self.structural_status = "PASS"
        self.consistency_status = "PASS"
        self.musical_status = "PASS"
        self.findings = []
        self.anomalies = []
        self.errors = []

    def add_finding(self, msg: str):
        self.findings.append(msg)

    def add_anomaly(self, timestamp: float, msg: str):
        self.anomalies.append((timestamp, msg))

    def add_error(self, msg: str):
        self.errors.append(msg)

    def fail_structural(self, msg: str):
        self.structural_status = "FAIL"
        self.add_error(msg)

    def warn_consistency(self, msg: str):
        if self.consistency_status == "PASS":
            self.consistency_status = "WARN"
        self.add_finding(msg)

    def warn_musical(self, msg: str):
        if self.musical_status == "PASS":
            self.musical_status = "WARN"
        self.add_finding(msg)

    def fail_musical(self, msg: str):
        self.musical_status = "FAIL"
        self.add_error(msg)


def validate_track(track_dir: str, track_name: str, genre: str, expected_duration: float) -> ValidationReport:
    """Validate a single track's golden outputs."""
    report = ValidationReport(track_name, genre)
    track_path = BASE_PATH / track_dir

    # Read files
    try:
        with open(track_path / "metrics.json") as f:
            metrics = json.load(f)
        report.add_finding(f"Read metrics.json from {track_path}")
    except Exception as e:
        report.fail_structural(f"Failed to read metrics.json: {e}")
        return report

    try:
        with open(track_path / "summary.json") as f:
            summary = json.load(f)
        report.add_finding(f"Read summary.json")
    except Exception as e:
        report.add_finding(f"Warning: Failed to read summary.json: {e}")
        summary = None

    try:
        with open(track_path / "segments.json") as f:
            segments = json.load(f)
        report.add_finding(f"Read segments.json")
    except Exception as e:
        report.add_finding(f"Warning: Failed to read segments.json: {e}")
        segments = None

    # STEP 1: STRUCTURAL VALIDATION
    report.add_finding("=== STEP 1: Structural Validation ===")

    # 1.1 Schema Integrity
    if "schema_version" not in metrics or not metrics["schema_version"]:
        report.fail_structural("Missing or empty schema_version")
        return report

    required_keys = ["track_metadata", "params", "curves", "block_times"]
    for key in required_keys:
        if key not in metrics:
            report.fail_structural(f"Missing required key: {key}")
            return report

    # Check for null values in critical fields
    if metrics["track_metadata"]["duration"] is None:
        report.fail_structural("track_metadata.duration is null")
        return report

    duration = metrics["track_metadata"]["duration"]

    # 1.2 Timebase Integrity
    block_times = metrics["block_times"]
    curves = metrics["curves"]

    # Check strictly increasing
    for i in range(1, len(block_times)):
        if block_times[i] <= block_times[i-1]:
            report.fail_structural(f"block_times not strictly increasing at index {i}: {block_times[i-1]} -> {block_times[i]}")
            return report

    # Check even spacing
    if len(block_times) > 1:
        interval = block_times[1] - block_times[0]
        for i in range(1, len(block_times)):
            actual_interval = block_times[i] - block_times[i-1]
            if abs(actual_interval - interval) > 1e-6:
                report.fail_structural(f"block_times not evenly spaced at index {i}: expected {interval}, got {actual_interval}")
                return report

    # Check final timestamp
    if block_times[-1] > duration + 0.1:
        report.fail_structural(f"Final block_time {block_times[-1]} exceeds duration {duration} (tolerance 0.1s)")
        return report

    # Check all curves have same length
    curve_names = ["tension_raw", "tension_smooth", "novelty", "fatigue"]
    expected_length = len(block_times)

    for curve_name in curve_names:
        if curve_name not in curves:
            report.fail_structural(f"Missing curve: {curve_name}")
            return report

        curve_data = curves[curve_name]
        if "values" not in curve_data:
            report.fail_structural(f"Curve {curve_name} missing 'values' array")
            return report

        actual_length = len(curve_data["values"])
        if actual_length != expected_length:
            report.fail_structural(f"Curve {curve_name} length mismatch: expected {expected_length}, got {actual_length}")
            return report

    report.add_finding(f"Timebase validation passed: {len(block_times)} blocks, interval={block_times[1]-block_times[0]:.2f}s")

    # 1.3 Event Validity
    if segments and "stagnant_segments" in segments:
        for seg in segments["stagnant_segments"]:
            if seg["start_time"] < 0:
                report.fail_structural(f"Stagnant segment has negative start_time: {seg['start_time']}")
                return report
            if seg["end_time"] > duration:
                report.fail_structural(f"Stagnant segment end_time {seg['end_time']} exceeds duration {duration}")
                return report
            if seg["end_time"] < seg["start_time"]:
                report.fail_structural(f"Stagnant segment end_time < start_time: {seg}")
                return report

    # 1.4 Numerical Sanity
    for curve_name in curve_names:
        values = curves[curve_name]["values"]
        for i, val in enumerate(values):
            if math.isnan(val):
                report.fail_structural(f"NaN found in {curve_name} at index {i}")
                return report
            if math.isinf(val):
                report.fail_structural(f"Infinity found in {curve_name} at index {i}")
                return report
            # Check typical ranges (tension/novelty/fatigue should be 0-1)
            if curve_name in ["tension_smooth", "novelty", "fatigue"]:
                if val < -0.01 or val > 1.01:
                    report.fail_structural(f"{curve_name} value out of range [0,1] at index {i}: {val}")
                    return report

    report.add_finding("All structural checks passed")

    # STEP 2: INTERNAL SIGNAL CONSISTENCY
    report.add_finding("\n=== STEP 2: Internal Signal Consistency ===")

    tension_smooth = curves["tension_smooth"]["values"]
    novelty = curves["novelty"]["values"]
    fatigue = curves["fatigue"]["values"]

    # Check top tension peak alignment
    if summary and "top_tension_peak" in summary:
        peak_time = summary["top_tension_peak"]["time_sec"]
        peak_value = summary["top_tension_peak"]["value"]

        # Find this time in block_times
        peak_idx = None
        for i, t in enumerate(block_times):
            if abs(t - peak_time) < 0.01:
                peak_idx = i
                break

        if peak_idx is not None:
            actual_value = tension_smooth[peak_idx]
            if abs(actual_value - peak_value) > 0.001:
                report.warn_consistency(f"Top tension peak value mismatch: summary says {peak_value}, curve has {actual_value} at {peak_time}s")

            # Check it's actually a local maximum
            is_local_max = True
            window = 3
            for j in range(max(0, peak_idx-window), min(len(tension_smooth), peak_idx+window+1)):
                if j != peak_idx and tension_smooth[j] > actual_value:
                    is_local_max = False
                    break

            if not is_local_max:
                report.warn_consistency(f"Top tension peak at {peak_time}s is not a local maximum in curve")
        else:
            report.warn_consistency(f"Could not find top tension peak time {peak_time}s in block_times")

    # Check fatigue/novelty correlation
    # Sustained low novelty should correlate with rising fatigue
    low_novelty_count = 0
    for i in range(len(novelty)):
        if novelty[i] < 0.1:
            low_novelty_count += 1
            # Check if fatigue is increasing or sustained
            if i > 0 and fatigue[i] < fatigue[i-1] - 0.1:
                report.add_anomaly(block_times[i], f"Low novelty ({novelty[i]:.3f}) but fatigue decreasing from {fatigue[i-1]:.3f} to {fatigue[i]:.3f}")

    report.add_finding(f"Found {low_novelty_count} blocks with low novelty (<0.1)")

    # STEP 3: MUSICAL PLAUSIBILITY
    report.add_finding("\n=== STEP 3: Musical Plausibility ===")

    # Check for overly normalized tension (everything near 1.0 or 0.5)
    tension_variance = sum((x - sum(tension_smooth)/len(tension_smooth))**2 for x in tension_smooth) / len(tension_smooth)
    if tension_variance < 0.01:
        report.warn_musical(f"Tension curve has very low variance ({tension_variance:.4f}), may be overly normalized")

    # Check for appropriate dynamic range
    tension_range = max(tension_smooth) - min(tension_smooth)
    if tension_range < 0.2:
        report.warn_musical(f"Tension curve has narrow dynamic range ({tension_range:.3f}), track may lack contrast")

    # Genre-specific checks
    if "hip-hop" in genre.lower():
        report.add_finding("Hip-hop track: expecting sustained energy, fewer dramatic drops")
        # Hip-hop should have relatively high average tension
        avg_tension = sum(tension_smooth) / len(tension_smooth)
        if avg_tension < 0.3:
            report.warn_musical(f"Hip-hop track has low average tension ({avg_tension:.3f}), unexpected for genre")

    elif "industrial" in genre.lower() or "experimental" in genre.lower():
        report.add_finding("Industrial/experimental track: long high-tension plateaus and intentional fatigue expected")
        # High fatigue is intentional, not a problem
        avg_fatigue = sum(fatigue) / len(fatigue)
        report.add_finding(f"Average fatigue: {avg_fatigue:.3f} (intentional pressure is valid for this genre)")

    elif "synthetic" in genre.lower():
        report.add_finding("Synthetic track: checking against expected ground truth")
        # Check expected events
        if "build_drop" in track_name.lower():
            expected_drop_time = 15.0
            # Should see tension peak near expected drop time
            peak_idx = tension_smooth.index(max(tension_smooth))
            peak_time = block_times[peak_idx]
            if abs(peak_time - expected_drop_time) > 2.0:
                report.warn_musical(f"Build-drop track: tension peak at {peak_time}s, expected near {expected_drop_time}s")
            else:
                report.add_finding(f"Build-drop: tension peak at {peak_time}s (expected ~{expected_drop_time}s) - GOOD")

        elif "repetitive" in track_name.lower():
            # Should have high average fatigue
            avg_fatigue = sum(fatigue) / len(fatigue)
            if avg_fatigue < 0.1:
                report.warn_musical(f"Repetitive track has low average fatigue ({avg_fatigue:.3f})")
            else:
                report.add_finding(f"Repetitive loop: average fatigue {avg_fatigue:.3f} - GOOD")

        elif "contrast" in track_name.lower():
            expected_transition = 20.0
            # Should see novelty spike near transition
            novelty_peak_idx = novelty.index(max(novelty))
            novelty_peak_time = block_times[novelty_peak_idx]
            if abs(novelty_peak_time - expected_transition) > 2.0:
                report.warn_musical(f"Contrast track: novelty peak at {novelty_peak_time}s, expected near {expected_transition}s")
            else:
                report.add_finding(f"Contrast: novelty peak at {novelty_peak_time}s (expected ~{expected_transition}s) - GOOD")

    return report


def main():
    print("=" * 80)
    print("GOLDEN OUTPUT VALIDATION REPORT")
    print("Kernel Version: 1.1.1 | Timebase Version: 1")
    print("=" * 80)
    print()

    all_reports = []

    for track_dir, track_name, genre, expected_duration in TRACKS:
        print(f"\n{'=' * 80}")
        print(f"TRACK: {track_name}")
        print(f"GENRE: {genre}")
        print(f"PATH: {track_dir}")
        print(f"{'=' * 80}\n")

        report = validate_track(track_dir, track_name, genre, expected_duration)

        print(f"STRUCTURAL_STATUS: {report.structural_status}")
        print(f"CONSISTENCY_STATUS: {report.consistency_status}")
        print(f"MUSICAL_STATUS: {report.musical_status}")
        print()

        print("KEY FINDINGS:")
        for finding in report.findings:
            print(f"  - {finding}")
        print()

        if report.anomalies:
            print("ANOMALIES:")
            for timestamp, msg in report.anomalies:
                print(f"  - {timestamp:.2f}s — {msg}")
            print()

        if report.errors:
            print("ERRORS:")
            for error in report.errors:
                print(f"  - {error}")
            print()

        # Final verdict
        print("FINAL VERDICT:")
        if report.structural_status == "PASS" and report.musical_status != "FAIL":
            print("  ✓ ACCEPTABLE AS KERNEL REFERENCE")
            if report.consistency_status == "WARN" or report.musical_status == "WARN":
                print("    (with minor warnings noted above)")
        else:
            print("  ✗ REQUIRES FIX:")
            for error in report.errors:
                print(f"      - {error}")

        print()
        all_reports.append(report)

    # Summary
    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)

    passed = sum(1 for r in all_reports if r.structural_status == "PASS" and r.musical_status != "FAIL")
    failed = len(all_reports) - passed

    print(f"\nTotal tracks validated: {len(all_reports)}")
    print(f"Acceptable as reference: {passed}")
    print(f"Require fixes: {failed}")

    if failed == 0:
        print("\n✓ ALL GOLDEN OUTPUTS ARE ACCEPTABLE FOR PHASE-2 PARITY WORK")
    else:
        print(f"\n✗ {failed} TRACK(S) REQUIRE FIXES BEFORE PHASE-2 WORK")


if __name__ == "__main__":
    main()

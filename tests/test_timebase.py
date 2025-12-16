"""
Timebase Module Tests

Tests for canonical time axis computation and event clamping.
Ensures timebase guarantees are upheld for Phase 1/2 parity.
"""

import pytest
import numpy as np

from src import timebase
from src import kernel


class TestCanonicalBlockCount:
    """Tests for compute_canonical_block_count."""

    def test_basic_calculation(self):
        """Test basic block count calculation."""
        # 10 seconds with 0.5s blocks = 20 blocks
        n = timebase.compute_canonical_block_count(10.0, 0.5)
        assert n == 20

    def test_final_block_within_duration(self):
        """Test that final block center <= duration."""
        duration = 10.0
        block_dur = 0.5
        n = timebase.compute_canonical_block_count(duration, block_dur)

        # Final block center = (n - 0.5) * block_dur
        final_center = (n - 0.5) * block_dur
        assert final_center <= duration, f"Final center {final_center} exceeds duration {duration}"

    def test_various_durations(self):
        """Test with various duration values."""
        test_cases = [
            (5.0, 0.5, 10),    # 5s / 0.5s = 10 blocks
            (30.0, 0.5, 60),   # 30s / 0.5s = 60 blocks
            (60.0, 1.0, 60),   # 60s / 1.0s = 60 blocks
            (3.25, 0.5, 7),    # 3.25s / 0.5s ~ 6.5 -> rounds to 7
        ]
        for duration, block_dur, expected in test_cases:
            n = timebase.compute_canonical_block_count(duration, block_dur)
            # Verify final block center is within duration
            final_center = (n - 0.5) * block_dur
            assert final_center <= duration + timebase.EPSILON_SEC

    def test_edge_case_short_duration(self):
        """Test with duration shorter than half block duration."""
        # Duration 0.3 < 0.5 * 0.5 = 0.25, so can fit 1 block
        n = timebase.compute_canonical_block_count(0.3, 0.5)
        assert n == 1  # Can fit 1 block (center at 0.25 <= 0.3)

        # Duration 0.2 < 0.5 * 0.5 = 0.25, cannot fit even 1 block
        n = timebase.compute_canonical_block_count(0.2, 0.5)
        assert n == 0  # Cannot fit any block

    def test_zero_duration(self):
        """Test with zero duration."""
        n = timebase.compute_canonical_block_count(0.0, 0.5)
        assert n == 0

    def test_negative_duration(self):
        """Test with negative duration."""
        n = timebase.compute_canonical_block_count(-5.0, 0.5)
        assert n == 0

    def test_zero_block_duration(self):
        """Test with zero block duration."""
        n = timebase.compute_canonical_block_count(10.0, 0.0)
        assert n == 0

    def test_with_start_time_offset(self):
        """Test with non-zero start time."""
        # Duration 10s, start at 2s -> effective duration = 8s
        n = timebase.compute_canonical_block_count(10.0, 0.5, start_time_sec=2.0)
        assert n == 16  # 8s / 0.5s = 16 blocks


class TestCanonicalTimeAxis:
    """Tests for compute_canonical_time_axis."""

    def test_basic_time_axis(self):
        """Test basic time axis generation."""
        times = timebase.compute_canonical_time_axis(4, 0.5)
        expected = np.array([0.25, 0.75, 1.25, 1.75], dtype=np.float32)
        np.testing.assert_array_almost_equal(times, expected)

    def test_time_axis_formula(self):
        """Test that time axis follows the formula t[i] = (i + 0.5) * block_dur."""
        n_blocks = 10
        block_dur = 0.5
        times = timebase.compute_canonical_time_axis(n_blocks, block_dur)

        for i in range(n_blocks):
            expected = (i + 0.5) * block_dur
            assert abs(times[i] - expected) < 1e-6

    def test_max_time_within_duration(self):
        """Test that max time <= duration when clamped."""
        duration = 10.0
        times = timebase.compute_canonical_time_axis(
            n_blocks=25,  # Would normally exceed 10s
            block_duration_sec=0.5,
            duration_sec=duration
        )
        assert np.max(times) <= duration

    def test_monotonic(self):
        """Test that time axis is monotonically increasing."""
        times = timebase.compute_canonical_time_axis(100, 0.5)
        assert np.all(np.diff(times) >= 0)

    def test_empty_for_zero_blocks(self):
        """Test empty array for zero blocks."""
        times = timebase.compute_canonical_time_axis(0, 0.5)
        assert len(times) == 0

    def test_negative_blocks(self):
        """Test empty array for negative blocks."""
        times = timebase.compute_canonical_time_axis(-5, 0.5)
        assert len(times) == 0

    def test_dtype_is_float32(self):
        """Test that output dtype is float32."""
        times = timebase.compute_canonical_time_axis(10, 0.5)
        assert times.dtype == np.float32


class TestBlockIndexToTime:
    """Tests for block_index_to_time."""

    def test_basic_conversion(self):
        """Test basic block index to time conversion."""
        # Block 0 center at 0.25s for 0.5s blocks
        time = timebase.block_index_to_time(0, 0.5)
        assert abs(time - 0.25) < 1e-6

        # Block 5 center at 2.75s
        time = timebase.block_index_to_time(5, 0.5)
        assert abs(time - 2.75) < 1e-6

    def test_with_duration_clamping(self):
        """Test that time is clamped to duration."""
        time = timebase.block_index_to_time(100, 0.5, duration_sec=10.0)
        assert time == 10.0


class TestClampPointEvent:
    """Tests for clamp_point_event."""

    def test_within_bounds(self):
        """Test event within bounds is not clamped."""
        time, clamped = timebase.clamp_point_event(5.0, 10.0)
        assert time == 5.0
        assert not clamped

    def test_exceeds_duration(self):
        """Test event exceeding duration is clamped."""
        time, clamped = timebase.clamp_point_event(15.0, 10.0)
        assert time == 10.0
        assert clamped

    def test_negative(self):
        """Test negative event time is clamped to 0."""
        time, clamped = timebase.clamp_point_event(-2.0, 10.0)
        assert time == 0.0
        assert clamped

    def test_exactly_at_duration(self):
        """Test event exactly at duration is not clamped."""
        time, clamped = timebase.clamp_point_event(10.0, 10.0)
        assert time == 10.0
        assert not clamped

    def test_epsilon_tolerance(self):
        """Test that events just slightly beyond duration are not clamped."""
        epsilon = timebase.EPSILON_SEC
        # Just within epsilon should not be clamped
        time, clamped = timebase.clamp_point_event(10.0 + epsilon / 2, 10.0)
        assert time == 10.0 + epsilon / 2
        assert not clamped


class TestClampSegmentEvent:
    """Tests for clamp_segment_event."""

    def test_within_bounds(self):
        """Test segment within bounds."""
        start, end, clamped = timebase.clamp_segment_event(2.0, 8.0, 10.0)
        assert start == 2.0
        assert end == 8.0
        assert not clamped

    def test_end_exceeds(self):
        """Test segment end exceeding duration."""
        start, end, clamped = timebase.clamp_segment_event(5.0, 15.0, 10.0)
        assert start == 5.0
        assert end == 10.0
        assert clamped

    def test_start_negative(self):
        """Test segment with negative start."""
        start, end, clamped = timebase.clamp_segment_event(-2.0, 5.0, 10.0)
        assert start == 0.0
        assert end == 5.0
        assert clamped

    def test_both_exceed(self):
        """Test segment where both bounds need adjustment."""
        start, end, clamped = timebase.clamp_segment_event(-1.0, 15.0, 10.0)
        assert start == 0.0
        assert end == 10.0
        assert clamped


class TestIsSegmentValid:
    """Tests for is_segment_valid."""

    def test_valid_segment(self):
        """Test valid segment."""
        assert timebase.is_segment_valid(2.0, 8.0)

    def test_invalid_reversed(self):
        """Test invalid segment where end < start."""
        assert not timebase.is_segment_valid(8.0, 2.0)

    def test_invalid_zero_duration(self):
        """Test invalid segment with zero duration."""
        assert not timebase.is_segment_valid(5.0, 5.0, min_duration_sec=0.1)

    def test_below_min_duration(self):
        """Test segment below minimum duration."""
        assert not timebase.is_segment_valid(5.0, 5.5, min_duration_sec=1.0)


class TestValidatePointEvents:
    """Tests for validate_point_events."""

    def test_valid_events(self):
        """Test validation of valid events."""
        events = [
            {'time': 5.0, 'score': 0.8},
            {'time': 8.0, 'score': 0.6}
        ]
        validated = timebase.validate_point_events(events, 10.0)

        assert len(validated) == 2
        assert validated[0]['time'] == 5.0
        assert validated[1]['time'] == 8.0

    def test_clamps_invalid_events(self):
        """Test that invalid events are clamped."""
        events = [
            {'time': 5.0, 'score': 0.8},
            {'time': 15.0, 'score': 0.6}  # Exceeds duration
        ]
        validated = timebase.validate_point_events(events, 10.0, drop_invalid=False)

        assert len(validated) == 2
        assert validated[1]['time'] == 10.0  # Clamped

    def test_drops_invalid_events(self):
        """Test dropping invalid events."""
        events = [
            {'time': 5.0, 'score': 0.8},
            {'time': 15.0, 'score': 0.6}  # Exceeds duration
        ]
        validated = timebase.validate_point_events(events, 10.0, drop_invalid=True)

        assert len(validated) == 1
        assert validated[0]['time'] == 5.0


class TestValidateSegmentEvents:
    """Tests for validate_segment_events."""

    def test_valid_segments(self):
        """Test validation of valid segments."""
        segments = [
            {'start_time': 2.0, 'end_time': 5.0, 'duration': 3.0}
        ]
        validated = timebase.validate_segment_events(segments, 10.0)

        assert len(validated) == 1
        assert validated[0]['start_time'] == 2.0
        assert validated[0]['end_time'] == 5.0

    def test_clamps_segment_end(self):
        """Test clamping of segment end time."""
        segments = [
            {'start_time': 5.0, 'end_time': 15.0, 'duration': 10.0}
        ]
        validated = timebase.validate_segment_events(segments, 10.0)

        assert len(validated) == 1
        assert validated[0]['end_time'] == 10.0
        assert validated[0]['duration'] == 5.0  # Updated

    def test_drops_too_short_segments(self):
        """Test dropping segments that become too short after clamping."""
        segments = [
            {'start_time': 9.5, 'end_time': 15.0, 'duration': 5.5}
        ]
        validated = timebase.validate_segment_events(
            segments, 10.0, min_duration_sec=1.0, drop_invalid=True
        )

        # After clamping: start=9.5, end=10.0, duration=0.5 < 1.0 -> dropped
        assert len(validated) == 0


class TestKernelParity:
    """Tests for kernel/timebase parity."""

    def test_block_count_matches(self):
        """Test that kernel and timebase compute same block count."""
        test_cases = [
            (10.0, 0.5),
            (30.0, 0.5),
            (60.0, 1.0),
            (3.25, 0.5),
        ]
        for duration, block_dur in test_cases:
            n_timebase = timebase.compute_canonical_block_count(duration, block_dur)
            n_kernel = kernel.compute_canonical_block_count(duration, block_dur)
            assert n_timebase == n_kernel, \
                f"Mismatch for duration={duration}, block_dur={block_dur}"

    def test_time_axis_matches(self):
        """Test that kernel and timebase compute same time axis."""
        n_blocks = 20
        block_dur = 0.5

        times_timebase = timebase.compute_canonical_time_axis(n_blocks, block_dur)
        times_kernel = kernel.compute_canonical_time_axis(n_blocks, block_dur)

        np.testing.assert_array_almost_equal(times_timebase, times_kernel)


class TestGuarantees:
    """Test the core guarantees of the timebase module."""

    def test_final_block_center_guarantee(self):
        """Test that final block center is always <= duration."""
        test_durations = [0.5, 1.0, 5.0, 10.0, 30.0, 60.0, 180.0]
        test_block_durs = [0.25, 0.5, 1.0, 2.0]

        for duration in test_durations:
            for block_dur in test_block_durs:
                n_blocks = timebase.compute_canonical_block_count(duration, block_dur)
                if n_blocks == 0:
                    continue

                times = timebase.compute_canonical_time_axis(n_blocks, block_dur)
                max_time = np.max(times)

                assert max_time <= duration + timebase.EPSILON_SEC, \
                    f"Guarantee violated: max_time={max_time}, duration={duration}, " \
                    f"block_dur={block_dur}, n_blocks={n_blocks}"

    def test_monotonicity_guarantee(self):
        """Test that time axis is always monotonically increasing."""
        for n_blocks in [1, 2, 10, 100, 1000]:
            times = timebase.compute_canonical_time_axis(n_blocks, 0.5)
            if len(times) > 1:
                diffs = np.diff(times)
                assert np.all(diffs > 0), "Time axis not monotonically increasing"

    def test_determinism_guarantee(self):
        """Test that same inputs produce same outputs."""
        for _ in range(10):
            n1 = timebase.compute_canonical_block_count(30.0, 0.5)
            n2 = timebase.compute_canonical_block_count(30.0, 0.5)
            assert n1 == n2

            t1 = timebase.compute_canonical_time_axis(60, 0.5)
            t2 = timebase.compute_canonical_time_axis(60, 0.5)
            np.testing.assert_array_equal(t1, t2)


class TestTimebaseAuditVerification:
    """
    Verification tests for timebase audit compliance.

    These tests ensure that all output-producing paths correctly use
    the canonical timebase contract. They serve as regression tests
    against future violations.
    """

    def test_golden_generator_block_count_60s(self):
        """Test that golden generator produces canonical block count for 60s track."""
        from golden_reference import generate_repetitive_loop, run_full_analysis
        from src.kernel_params import DEFAULT_CONFIG

        # Generate 60s synthetic track
        audio = generate_repetitive_loop(duration=60, sr=DEFAULT_CONFIG.frame.sample_rate)
        sr = DEFAULT_CONFIG.frame.sample_rate
        duration_sec = len(audio) / sr

        # Run analysis
        results = run_full_analysis(audio, sr, DEFAULT_CONFIG)

        # Verify canonical block count
        canonical_blocks = timebase.compute_canonical_block_count(
            duration_sec, DEFAULT_CONFIG.block.block_duration_sec
        )

        assert results['n_blocks'] == canonical_blocks, \
            f"Golden generator block count mismatch: got {results['n_blocks']}, expected {canonical_blocks}"

    def test_curve_length_equals_canonical_blocks(self):
        """Test that all curves have length equal to canonical block count."""
        from golden_reference import generate_build_then_drop, run_full_analysis
        from src.kernel_params import DEFAULT_CONFIG

        # Generate synthetic track
        audio, _ = generate_build_then_drop(duration=30, sr=DEFAULT_CONFIG.frame.sample_rate)
        sr = DEFAULT_CONFIG.frame.sample_rate
        duration_sec = len(audio) / sr

        # Run analysis
        results = run_full_analysis(audio, sr, DEFAULT_CONFIG)

        canonical_blocks = timebase.compute_canonical_block_count(
            duration_sec, DEFAULT_CONFIG.block.block_duration_sec
        )

        # Verify all curves have correct length
        for curve_name in ['tension_raw', 'tension_smooth', 'novelty', 'fatigue']:
            curve = results['curves'][curve_name]
            assert len(curve) == canonical_blocks, \
                f"Curve {curve_name} length {len(curve)} != canonical {canonical_blocks}"

    def test_max_time_within_duration(self):
        """Test that max curve time <= duration for synthetic 60s track."""
        from golden_reference import generate_repetitive_loop, run_full_analysis
        from src.kernel_params import DEFAULT_CONFIG

        # Generate 60s synthetic track
        audio = generate_repetitive_loop(duration=60, sr=DEFAULT_CONFIG.frame.sample_rate)
        sr = DEFAULT_CONFIG.frame.sample_rate
        duration_sec = len(audio) / sr

        # Run analysis
        results = run_full_analysis(audio, sr, DEFAULT_CONFIG)

        # Verify max time <= duration
        max_time = np.max(results['block_times'])
        assert max_time <= duration_sec + timebase.EPSILON_SEC, \
            f"Max block time {max_time} exceeds duration {duration_sec}"

    def test_no_segment_exceeds_duration(self):
        """Test that no segment end time exceeds duration."""
        from golden_reference import generate_repetitive_loop, run_full_analysis
        from src.kernel_params import DEFAULT_CONFIG

        # Generate 60s synthetic track (repetitive loop tends to generate stagnant segments)
        audio = generate_repetitive_loop(duration=60, sr=DEFAULT_CONFIG.frame.sample_rate)
        sr = DEFAULT_CONFIG.frame.sample_rate
        duration_sec = len(audio) / sr

        # Run analysis
        results = run_full_analysis(audio, sr, DEFAULT_CONFIG)

        # Check stagnant segments
        for segment in results['events'].get('stagnant_segments', []):
            assert segment['end_time'] <= duration_sec + timebase.EPSILON_SEC, \
                f"Segment end_time {segment['end_time']} exceeds duration {duration_sec}"

        # Check candidate drops
        for drop in results['events'].get('candidate_drops', []):
            assert drop['time'] <= duration_sec + timebase.EPSILON_SEC, \
                f"Drop time {drop['time']} exceeds duration {duration_sec}"

    def test_canonical_60s_is_120_blocks(self):
        """Test the canonical example: 60s @ 0.5s/block = 120 blocks."""
        n_blocks = timebase.compute_canonical_block_count(60.0, 0.5)
        assert n_blocks == 120, f"60s @ 0.5s/block should be 120 blocks, got {n_blocks}"

        times = timebase.compute_canonical_time_axis(n_blocks, 0.5, duration_sec=60.0)
        assert len(times) == 120, f"Time axis should have 120 points, got {len(times)}"
        assert times[-1] <= 60.0, f"Last block center {times[-1]} should be <= 60s"

    def test_aggregation_uses_canonical_timebase(self):
        """Test that aggregate_frame_features uses canonical timebase when duration provided."""
        from src.aggregation import aggregate_frame_features, frames_to_blocks
        from src.features import extract_all_features
        import config

        # Create synthetic audio: 60s at 22050 Hz
        sr = 22050
        duration_sec = 60.0
        audio = np.random.randn(int(duration_sec * sr)).astype(np.float32)
        audio = audio / np.max(np.abs(audio))

        # Extract features
        frame_features = extract_all_features(audio, sr)

        # Aggregate WITH duration_sec (should use canonical timebase)
        block_features, _, block_times = aggregate_frame_features(
            frame_features, sr,
            frame_hop=config.HOP_LENGTH,
            block_duration_sec=config.BLOCK_DURATION_SEC,
            duration_sec=duration_sec
        )

        canonical_blocks = timebase.compute_canonical_block_count(
            duration_sec, config.BLOCK_DURATION_SEC
        )

        # Verify canonical block count is used
        assert len(block_times) == canonical_blocks, \
            f"aggregate_frame_features should use canonical count {canonical_blocks}, got {len(block_times)}"
        assert np.max(block_times) <= duration_sec + timebase.EPSILON_SEC, \
            f"Max block time should be <= duration"

    def test_ad_hoc_produces_more_blocks(self):
        """
        Verify that ad-hoc n_frames // frames_per_block produces more blocks
        than the canonical count, confirming the mitigation is necessary.
        """
        # Simulate 60s of audio at 22050 Hz
        sr = 22050
        duration_sec = 60.0
        hop = 512
        block_dur = 0.5

        total_samples = int(duration_sec * sr)
        n_frames = total_samples // hop

        # Ad-hoc calculation (as in frames_to_blocks)
        samples_per_block = int(block_dur * sr)
        frames_per_block = samples_per_block // hop
        ad_hoc_blocks = n_frames // frames_per_block

        # Canonical calculation
        canonical_blocks = timebase.compute_canonical_block_count(duration_sec, block_dur)

        # The ad-hoc method should produce MORE blocks (this is why we need canonical correction)
        assert ad_hoc_blocks >= canonical_blocks, \
            "Ad-hoc should produce at least as many blocks as canonical"

        # For 60s @ 0.5s blocks, ad-hoc produces 123, canonical produces 120
        # This verifies the known discrepancy that the audit identified
        if duration_sec == 60.0 and block_dur == 0.5:
            assert canonical_blocks == 120, f"Expected 120 canonical blocks, got {canonical_blocks}"
            # Ad-hoc varies slightly by sample rate but should be > 120
            assert ad_hoc_blocks > canonical_blocks, \
                f"Ad-hoc ({ad_hoc_blocks}) should exceed canonical ({canonical_blocks}) for 60s track"

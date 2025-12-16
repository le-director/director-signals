"""
Timebase Module - Canonical Time Axis Utilities

Provides deterministic block count and time axis computation that guarantees
all timestamps stay within track duration bounds.

DESIGN CONSTRAINTS:
- track_metadata.duration_sec is the source of truth
- All block center times <= duration_sec
- Deterministic: same inputs -> same outputs
- No external config imports (explicit parameters for C++ portability)

SHARED TIMEBASE SPEC (Phase 1/2 Alignment):
- Block count: n = max(1, floor((duration - start) / block_dur + 0.5))
- Block center: t[i] = start + (i + 0.5) * block_dur
- Final block center <= duration (guaranteed by block count formula)
- Phase-2 C++ should port these formulas verbatim
"""

import numpy as np
from typing import Tuple, List, Dict, Optional


# =============================================================================
# VERSION
# =============================================================================

# Timebase version - bump when block/time calculation logic changes
# This version is independent of config.TIMEBASE_VERSION for module isolation
TIMEBASE_VERSION: str = "1"


# =============================================================================
# CONSTANTS
# =============================================================================

DEFAULT_BLOCK_DURATION_SEC: float = 0.5
DEFAULT_START_TIME_SEC: float = 0.0
EPSILON_SEC: float = 1e-6  # Floating point tolerance for comparisons


# =============================================================================
# CORE FUNCTIONS
# =============================================================================

def compute_canonical_block_count(
    duration_sec: float,
    block_duration_sec: float = DEFAULT_BLOCK_DURATION_SEC,
    start_time_sec: float = DEFAULT_START_TIME_SEC
) -> int:
    """
    Compute canonical block count where final block center <= duration_sec.

    CONTRACT:
    - Input: duration_sec (positive float), block_duration_sec (positive float)
    - Output: n_blocks (non-negative int)
    - Guarantee: block_center(n_blocks-1) <= duration_sec
    - Returns 0 if duration is too short to fit even one block center

    FORMULA DERIVATION:
    Block center time for block i: t[i] = start + (i + 0.5) * block_dur
    For last block (i = n-1): t[n-1] = start + (n - 0.5) * block_dur

    We need: start + (n - 0.5) * block_dur <= duration
             (n - 0.5) * block_dur <= duration - start
             n - 0.5 <= (duration - start) / block_dur
             n <= (duration - start) / block_dur + 0.5
             n = floor((duration - start) / block_dur + 0.5)

    For n=1: block center = start + 0.5 * block_dur
    So we need at least: duration >= start + 0.5 * block_dur

    Parameters:
        duration_sec: Track duration in seconds (source of truth)
        block_duration_sec: Block duration in seconds
        start_time_sec: Start time offset (usually 0)

    Returns:
        Number of blocks (0 if invalid inputs or duration too short)
    """
    if duration_sec <= 0 or block_duration_sec <= 0:
        return 0

    effective_duration = duration_sec - start_time_sec
    if effective_duration <= 0:
        return 0

    # Check if we can fit at least one block center within duration
    # First block center = start + 0.5 * block_dur
    # So we need: start + 0.5 * block_dur <= duration
    # i.e., effective_duration >= 0.5 * block_dur
    if effective_duration < 0.5 * block_duration_sec:
        return 0

    # Calculate n such that center of last block <= duration
    n_blocks = int((effective_duration / block_duration_sec) + 0.5)

    # Ensure at least 1 block (should be guaranteed by the check above)
    return max(1, n_blocks)


def compute_canonical_time_axis(
    n_blocks: int,
    block_duration_sec: float = DEFAULT_BLOCK_DURATION_SEC,
    start_time_sec: float = DEFAULT_START_TIME_SEC,
    duration_sec: Optional[float] = None
) -> np.ndarray:
    """
    Compute canonical time axis array with guaranteed bounds.

    CONTRACT:
    - Output: monotonically increasing array of block center times
    - If duration_sec provided: max(output) <= duration_sec
    - times[i] = start_time_sec + (i + 0.5) * block_duration_sec

    Parameters:
        n_blocks: Number of blocks
        block_duration_sec: Block duration in seconds
        start_time_sec: Start time offset
        duration_sec: Optional maximum duration for final clamping check

    Returns:
        Array of block center times (n_blocks,), dtype float32
    """
    if n_blocks <= 0:
        return np.array([], dtype=np.float32)

    # Calculate block center times: start + (i + 0.5) * block_dur
    times = start_time_sec + (np.arange(n_blocks) + 0.5) * block_duration_sec
    times = times.astype(np.float32)

    # Final safety clamp if duration provided (should not trigger if block count is correct)
    if duration_sec is not None:
        times = np.minimum(times, np.float32(duration_sec))

    return times


def block_index_to_time(
    block_idx: int,
    block_duration_sec: float = DEFAULT_BLOCK_DURATION_SEC,
    start_time_sec: float = DEFAULT_START_TIME_SEC,
    duration_sec: Optional[float] = None
) -> float:
    """
    Convert a single block index to time, optionally clamping to duration.

    Parameters:
        block_idx: Block index (0-based)
        block_duration_sec: Block duration in seconds
        start_time_sec: Start time offset
        duration_sec: Optional maximum duration for clamping

    Returns:
        Block center time in seconds
    """
    time = start_time_sec + (block_idx + 0.5) * block_duration_sec
    if duration_sec is not None:
        time = min(time, duration_sec)
    return float(time)


# =============================================================================
# CLAMPING HELPERS
# =============================================================================

def clamp_point_event(
    event_time: float,
    duration_sec: float,
    epsilon: float = EPSILON_SEC
) -> Tuple[float, bool]:
    """
    Clamp a point event time to valid range [0, duration_sec].

    Parameters:
        event_time: Event timestamp in seconds
        duration_sec: Track duration in seconds
        epsilon: Tolerance for out-of-bounds detection

    Returns:
        Tuple of (clamped_time, was_clamped)
        - clamped_time: Time clamped to [0, duration_sec]
        - was_clamped: True if the time was modified
    """
    was_clamped = False
    clamped_time = event_time

    if event_time < 0:
        clamped_time = 0.0
        was_clamped = True
    elif event_time > duration_sec + epsilon:
        clamped_time = duration_sec
        was_clamped = True

    return float(clamped_time), was_clamped


def clamp_segment_event(
    start_time: float,
    end_time: float,
    duration_sec: float,
    epsilon: float = EPSILON_SEC
) -> Tuple[float, float, bool]:
    """
    Clamp a segment (start_time, end_time) to valid range.

    Parameters:
        start_time: Segment start in seconds
        end_time: Segment end in seconds
        duration_sec: Track duration in seconds
        epsilon: Tolerance for out-of-bounds detection

    Returns:
        Tuple of (clamped_start, clamped_end, was_clamped)
    """
    was_clamped = False

    clamped_start = start_time
    if start_time < 0:
        clamped_start = 0.0
        was_clamped = True

    clamped_end = end_time
    if end_time > duration_sec + epsilon:
        clamped_end = duration_sec
        was_clamped = True

    return float(clamped_start), float(clamped_end), was_clamped


def is_segment_valid(
    start_time: float,
    end_time: float,
    min_duration_sec: float = 0.0
) -> bool:
    """
    Check if a segment is valid after clamping.

    A segment is invalid if:
    - end_time <= start_time
    - duration < min_duration_sec

    Parameters:
        start_time: Segment start in seconds
        end_time: Segment end in seconds
        min_duration_sec: Minimum required duration

    Returns:
        True if segment is valid
    """
    duration = end_time - start_time
    return duration >= min_duration_sec and end_time > start_time


# =============================================================================
# EVENT VALIDATION
# =============================================================================

def validate_point_events(
    events: List[Dict],
    duration_sec: float,
    time_key: str = 'time',
    drop_invalid: bool = False,
    epsilon: float = EPSILON_SEC
) -> List[Dict]:
    """
    Validate and optionally clamp point events with a time field.

    Parameters:
        events: List of event dicts with time_key field
        duration_sec: Track duration in seconds
        time_key: Key name for the time field (default 'time')
        drop_invalid: If True, drop events beyond duration instead of clamping
        epsilon: Tolerance for out-of-bounds detection

    Returns:
        List of events with clamped times (or filtered if drop_invalid=True)
    """
    validated = []

    for event in events:
        if time_key not in event:
            validated.append(event.copy())
            continue

        event_copy = event.copy()
        original_time = event_copy[time_key]
        clamped_time, was_clamped = clamp_point_event(original_time, duration_sec, epsilon)

        if was_clamped and drop_invalid:
            # Skip this event entirely
            continue

        if was_clamped:
            event_copy[time_key] = clamped_time

        validated.append(event_copy)

    return validated


def validate_segment_events(
    segments: List[Dict],
    duration_sec: float,
    start_key: str = 'start_time',
    end_key: str = 'end_time',
    duration_key: str = 'duration',
    min_duration_sec: float = 0.0,
    drop_invalid: bool = False,
    epsilon: float = EPSILON_SEC
) -> List[Dict]:
    """
    Validate and clamp segment events with start/end time fields.

    Parameters:
        segments: List of segment dicts
        duration_sec: Track duration in seconds
        start_key: Key name for start time (default 'start_time')
        end_key: Key name for end time (default 'end_time')
        duration_key: Key name for duration (default 'duration')
        min_duration_sec: Minimum segment duration after clamping
        drop_invalid: If True, drop segments that become too short
        epsilon: Tolerance for out-of-bounds detection

    Returns:
        List of segments with clamped times (or filtered if drop_invalid=True)
    """
    validated = []

    for segment in segments:
        if start_key not in segment or end_key not in segment:
            validated.append(segment.copy())
            continue

        seg_copy = segment.copy()
        original_start = seg_copy[start_key]
        original_end = seg_copy[end_key]

        clamped_start, clamped_end, was_clamped = clamp_segment_event(
            original_start, original_end, duration_sec, epsilon
        )

        # Check if segment is still valid after clamping
        if not is_segment_valid(clamped_start, clamped_end, min_duration_sec):
            if drop_invalid:
                continue

        if was_clamped:
            seg_copy[start_key] = clamped_start
            seg_copy[end_key] = clamped_end
            # Update duration if present
            if duration_key in seg_copy:
                seg_copy[duration_key] = clamped_end - clamped_start

        validated.append(seg_copy)

    return validated

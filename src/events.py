"""
Event Detection Module

Detect candidate drops, stagnant segments, and section boundaries
using deterministic heuristic rules.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from scipy import signal as scipy_signal

import config
from src import timebase


def detect_peaks_with_prominence(
    curve: np.ndarray,
    prominence_threshold: float,
    min_distance: int = config.PEAK_MIN_DISTANCE_BLOCKS
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Detect peaks in a curve with prominence filtering.

    Parameters:
        curve: 1D array
        prominence_threshold: Minimum prominence for peaks
        min_distance: Minimum distance between peaks (blocks)

    Returns:
        Tuple of (peak_indices, peak_values, prominences)
    """
    # Find peaks with prominence
    peaks, properties = scipy_signal.find_peaks(
        curve,
        prominence=prominence_threshold,
        distance=min_distance
    )

    prominences = properties['prominences']
    peak_values = curve[peaks]

    return peaks, peak_values, prominences


def check_build_pattern(
    tension_curve: np.ndarray,
    peak_idx: int,
    min_duration_blocks: int,
    min_slope: float
) -> Tuple[bool, float, int]:
    """
    Check if there's a rising tension pattern before a peak.

    Parameters:
        tension_curve: Tension curve array
        peak_idx: Index of peak to check
        min_duration_blocks: Minimum duration of build
        min_slope: Minimum slope (units per block)

    Returns:
        Tuple of (has_build, avg_slope, build_duration_blocks)
    """
    # Look back from peak
    search_start = max(0, peak_idx - min_duration_blocks * 2)
    segment = tension_curve[search_start:peak_idx]

    if len(segment) < min_duration_blocks:
        return False, 0.0, 0

    # Find local minimum before peak
    min_idx_rel = np.argmin(segment)
    min_idx = search_start + min_idx_rel

    build_duration = peak_idx - min_idx

    if build_duration < min_duration_blocks:
        return False, 0.0, build_duration

    # Calculate average slope
    tension_delta = tension_curve[peak_idx] - tension_curve[min_idx]
    avg_slope = tension_delta / build_duration

    has_build = avg_slope >= min_slope

    return has_build, avg_slope, build_duration


def check_drop_pattern(
    tension_curve: np.ndarray,
    peak_idx: int,
    post_window_blocks: int
) -> Tuple[bool, float]:
    """
    Check for post-drop pattern (stabilization or sustained energy).

    Parameters:
        tension_curve: Tension curve array
        peak_idx: Index of peak
        post_window_blocks: Number of blocks to check after peak

    Returns:
        Tuple of (has_drop_pattern, post_drop_mean)
    """
    end_idx = min(len(tension_curve), peak_idx + post_window_blocks)
    post_segment = tension_curve[peak_idx:end_idx]

    if len(post_segment) < 2:
        return False, 0.0

    post_mean = np.mean(post_segment)
    peak_value = tension_curve[peak_idx]

    # Drop pattern: sustained high energy after peak
    # or slight decay but still elevated
    has_pattern = post_mean >= peak_value * 0.6

    return has_pattern, float(post_mean)


def detect_plateau_regions(
    tension_curve: np.ndarray,
    block_times: np.ndarray,
    tension_threshold: float = config.PLATEAU_TENSION_THRESHOLD,
    min_duration_sec: float = config.PLATEAU_MIN_DURATION_SEC
) -> List[Dict]:
    """
    Detect sustained high-tension regions (plateaus).

    A plateau is a contiguous region where tension stays above threshold
    for at least the minimum duration.

    Parameters:
        tension_curve: Tension curve [0, 1]
        block_times: Array of block center times (seconds)
        tension_threshold: Minimum tension to be considered "high"
        min_duration_sec: Minimum duration for plateau

    Returns:
        List of plateau dicts with:
            - 'start_time': start time in seconds
            - 'end_time': end time in seconds
            - 'mean_tension': average tension in region
            - 'start_block': start block index
            - 'end_block': end block index
    """
    is_high = tension_curve >= tension_threshold

    plateaus = []
    in_plateau = False
    plateau_start = 0

    for i in range(len(is_high)):
        if is_high[i] and not in_plateau:
            in_plateau = True
            plateau_start = i
        elif not is_high[i] and in_plateau:
            plateau_end = i - 1
            duration = block_times[plateau_end] - block_times[plateau_start]

            if duration >= min_duration_sec:
                plateaus.append({
                    'start_time': float(block_times[plateau_start]),
                    'end_time': float(block_times[plateau_end]),
                    'mean_tension': float(np.mean(tension_curve[plateau_start:plateau_end+1])),
                    'start_block': int(plateau_start),
                    'end_block': int(plateau_end)
                })

            in_plateau = False

    # Handle plateau extending to end
    if in_plateau:
        plateau_end = len(is_high) - 1
        duration = block_times[plateau_end] - block_times[plateau_start]
        if duration >= min_duration_sec:
            plateaus.append({
                'start_time': float(block_times[plateau_start]),
                'end_time': float(block_times[plateau_end]),
                'mean_tension': float(np.mean(tension_curve[plateau_start:plateau_end+1])),
                'start_block': int(plateau_start),
                'end_block': int(plateau_end)
            })

    return plateaus


def check_post_persistence(
    tension_curve: np.ndarray,
    peak_idx: int,
    block_times: np.ndarray,
    persistence_sec: float = config.DROP_POST_PERSISTENCE_SEC,
    threshold_ratio: float = config.DROP_PERSISTENCE_THRESHOLD
) -> Tuple[bool, float]:
    """
    Check if the regime change after a peak persists for minimum duration.

    Parameters:
        tension_curve: Tension curve array
        peak_idx: Index of peak
        block_times: Block times array
        persistence_sec: Minimum duration to persist
        threshold_ratio: Fraction of peak value to maintain

    Returns:
        Tuple of (has_persistence, persistence_duration_sec)
    """
    if peak_idx >= len(tension_curve) - 1:
        return False, 0.0

    block_duration = block_times[1] - block_times[0] if len(block_times) > 1 else config.BLOCK_DURATION_SEC
    persistence_blocks = int(persistence_sec / block_duration)

    end_idx = min(len(tension_curve), peak_idx + persistence_blocks)
    post_segment = tension_curve[peak_idx:end_idx]

    if len(post_segment) < 2:
        return False, 0.0

    peak_value = tension_curve[peak_idx]
    threshold = peak_value * threshold_ratio

    # Count how many consecutive blocks stay above threshold
    consecutive_above = 0
    for val in post_segment:
        if val >= threshold:
            consecutive_above += 1
        else:
            break

    persistence_duration = consecutive_above * block_duration

    # Need at least 75% of target duration
    has_persistence = persistence_duration >= persistence_sec * 0.75

    return has_persistence, float(persistence_duration)


def count_significant_contrast(
    impact_components: Optional[Dict[str, float]],
    threshold: float = config.DROP_COMPONENT_CONTRAST_THRESHOLD
) -> int:
    """
    Count how many components show meaningful contrast.

    Parameters:
        impact_components: Dict with component deltas (rms_delta, onset_delta, etc.)
        threshold: Minimum delta to count as significant

    Returns:
        Number of components with significant positive contrast
    """
    if impact_components is None:
        return 0

    count = 0
    for key, delta in impact_components.items():
        if key.endswith('_delta') and delta >= threshold:
            count += 1

    return count


def classify_tension_peak(
    peak_idx: int,
    tension_curve: np.ndarray,
    block_times: np.ndarray,
    has_build: bool,
    build_slope: float,
    plateau_regions: List[Dict],
    impact_components: Optional[Dict[str, float]] = None
) -> str:
    """
    Classify a tension peak into event type.

    Classification rules:
    - 'drop': has_build=true + multi-component contrast + persistence
    - 'structural_shock': strong contrast but no build pattern
    - 'plateau_peak': local maximum within sustained high-tension region
    - 'minor_peak': doesn't meet any major criteria

    Parameters:
        peak_idx: Index of peak in tension curve
        tension_curve: Tension curve array
        block_times: Block times array
        has_build: Whether there's a build pattern before peak
        build_slope: Slope of the build
        plateau_regions: List of detected plateau regions
        impact_components: Dict with component deltas (if computed)

    Returns:
        Event type string
    """
    peak_time = block_times[peak_idx]

    # Check if peak is within a plateau region
    in_plateau = False
    for plateau in plateau_regions:
        if plateau['start_time'] <= peak_time <= plateau['end_time']:
            in_plateau = True
            break

    # Count significant contrast components
    n_contrast = count_significant_contrast(impact_components)

    # Check post-drop persistence
    has_persistence, persistence_duration = check_post_persistence(
        tension_curve, peak_idx, block_times
    )

    # Check pre-peak dip (for plateau peaks)
    lookback = min(10, peak_idx)
    if lookback > 0:
        pre_peak_min = np.min(tension_curve[peak_idx - lookback:peak_idx])
        pre_peak_contrast = tension_curve[peak_idx] - pre_peak_min
    else:
        pre_peak_contrast = 0.0

    # Classification logic with guard clauses

    # Guard: plateau peaks
    if in_plateau:
        if pre_peak_contrast >= config.PLATEAU_DROP_MIN_CONTRAST and has_build:
            return 'drop'
        return 'plateau_peak'

    # Guard: no build pattern means cannot be a drop
    if not has_build:
        if n_contrast >= config.DROP_MIN_CONTRAST_COMPONENTS:
            return 'structural_shock'
        return 'minor_peak'

    # Has build pattern - check full criteria
    if n_contrast >= config.DROP_MIN_CONTRAST_COMPONENTS and has_persistence:
        return 'drop'

    if n_contrast >= 1 or has_persistence:
        return 'structural_shock'

    return 'minor_peak'


def detect_candidate_drops(
    tension_curve: np.ndarray,
    block_times: np.ndarray,
    prominence_threshold: float = config.DROP_PROMINENCE_THRESHOLD,
    min_build_duration_sec: float = config.DROP_MIN_BUILD_DURATION_SEC,
    min_build_slope: float = config.DROP_MIN_BUILD_SLOPE,
    min_distance_blocks: int = config.PEAK_MIN_DISTANCE_BLOCKS,
    audio: Optional[np.ndarray] = None,
    sr: Optional[int] = None,
    duration_sec: Optional[float] = None
) -> List[Dict]:
    """
    Detect candidate drop events based on tension curve patterns.

    A drop candidate must:
    1. Be a tension peak with sufficient prominence
    2. Have a rising tension build before it
    3. Have sustained energy after it

    Events are classified into types:
    - 'drop': True drop (has_build + multi-component contrast + persistence)
    - 'structural_shock': Strong contrast without build pattern
    - 'plateau_peak': Local maximum within sustained high-tension region
    - 'minor_peak': Doesn't meet major criteria

    Parameters:
        tension_curve: Smoothed tension curve [0, 1]
        block_times: Array of block center times (seconds)
        prominence_threshold: Minimum peak prominence
        min_build_duration_sec: Minimum build duration
        min_build_slope: Minimum build slope (units per block)
        min_distance_blocks: Minimum distance between peaks
        audio: Audio array (optional, for impact scoring)
        sr: Sample rate (optional, for impact scoring)
        duration_sec: Track duration in seconds for time validation/clamping

    Returns:
        List of drop candidate dicts with:
            - 'time': drop time in seconds
            - 'block_idx': block index
            - 'tension': tension value at drop
            - 'prominence': peak prominence
            - 'score': confidence score [0, 1]
            - 'event_type': classification ('drop', 'structural_shock', 'plateau_peak', 'minor_peak')
            - 'rule_breakdown': dict of which rules passed
    """
    # Detect peaks
    peaks, peak_values, prominences = detect_peaks_with_prominence(
        tension_curve, prominence_threshold, min_distance_blocks
    )

    # Detect plateau regions for classification
    plateau_regions = detect_plateau_regions(tension_curve, block_times)

    # Convert durations to blocks
    block_duration_sec = block_times[1] - block_times[0] if len(block_times) > 1 else config.BLOCK_DURATION_SEC
    min_build_blocks = int(min_build_duration_sec / block_duration_sec)
    post_window_blocks = int(config.DROP_POST_WINDOW_SEC / block_duration_sec)

    # Optionally compute impact scores for all candidates
    impact_scores = {}
    if audio is not None and sr is not None:
        from src import metrics as metrics_module
        # Create temporary candidates for impact scoring
        temp_candidates = [
            {'time': float(block_times[peak_idx])}
            for peak_idx in peaks
        ]
        impact_scores = metrics_module.compute_drop_impact_scores(
            audio, sr, temp_candidates
        )

    candidates = []

    for i, peak_idx in enumerate(peaks):
        # Check build pattern
        has_build, build_slope, build_duration = check_build_pattern(
            tension_curve, peak_idx, min_build_blocks, min_build_slope
        )

        # Check drop pattern
        has_drop_pattern, post_mean = check_drop_pattern(
            tension_curve, peak_idx, post_window_blocks
        )

        # Check post-drop persistence
        has_persistence, persistence_duration = check_post_persistence(
            tension_curve, peak_idx, block_times
        )

        # Get impact components for this peak (if available)
        impact_components = None
        if i in impact_scores:
            impact_components = {
                'rms_delta': impact_scores[i].get('rms_delta', 0.0),
                'onset_delta': impact_scores[i].get('onset_delta', 0.0),
                'centroid_delta': impact_scores[i].get('centroid_delta', 0.0),
                'bandwidth_delta': impact_scores[i].get('bandwidth_delta', 0.0)
            }

        # Count significant contrast
        n_contrast = count_significant_contrast(impact_components)

        # Classify the event
        event_type = classify_tension_peak(
            peak_idx, tension_curve, block_times,
            has_build, build_slope, plateau_regions, impact_components
        )

        # Check if peak is in plateau
        in_plateau = False
        for plateau in plateau_regions:
            if plateau['start_time'] <= block_times[peak_idx] <= plateau['end_time']:
                in_plateau = True
                break

        # Calculate confidence score based on rule satisfaction
        score = 0.0

        # Base score from prominence (0.0 to 0.4)
        score += min(prominences[i] / 1.0, 0.4)

        # Build presence (0.0 or 0.3)
        if has_build:
            score += 0.3
            # Bonus for strong build (up to 0.1)
            score += min(build_slope / 0.5, 0.1)

        # Drop pattern (0.0 or 0.2)
        if has_drop_pattern:
            score += 0.2

        # Normalize to [0, 1]
        score = min(score, 1.0)

        candidates.append({
            'time': float(block_times[peak_idx]),
            'block_idx': int(peak_idx),
            'tension': float(peak_values[i]),
            'prominence': float(prominences[i]),
            'score': float(score),
            'event_type': event_type,
            'rule_breakdown': {
                'has_prominence': True,  # Already filtered
                'has_build': bool(has_build),
                'build_slope': float(build_slope),
                'build_duration_blocks': int(build_duration),
                'has_drop_pattern': bool(has_drop_pattern),
                'post_drop_mean': float(post_mean),
                'has_persistence': bool(has_persistence),
                'persistence_duration_sec': float(persistence_duration),
                'n_contrast_components': int(n_contrast),
                'in_plateau': bool(in_plateau)
            }
        })

    # Sort by score (highest first)
    candidates.sort(key=lambda x: x['score'], reverse=True)

    # Validate and clamp event times if duration provided
    if duration_sec is not None:
        candidates = timebase.validate_point_events(
            candidates, duration_sec, time_key='time', drop_invalid=True
        )

    return candidates


def detect_stagnant_segments(
    novelty_curve: np.ndarray,
    fatigue_curve: np.ndarray,
    block_times: np.ndarray,
    novelty_threshold: float = config.STAGNANT_NOVELTY_THRESHOLD,
    fatigue_threshold: float = config.STAGNANT_FATIGUE_THRESHOLD,
    min_duration_sec: float = config.STAGNANT_MIN_DURATION_SEC,
    duration_sec: Optional[float] = None
) -> List[Dict]:
    """
    Detect stagnant/over-looped sections.

    A stagnant segment is a contiguous region where:
    - Novelty < threshold (low change)
    - Fatigue > threshold (high repetition)
    - Duration >= minimum duration

    Parameters:
        novelty_curve: Novelty curve [0, 1]
        fatigue_curve: Fatigue curve [0, 1]
        block_times: Array of block center times (seconds)
        novelty_threshold: Maximum novelty for stagnation
        fatigue_threshold: Minimum fatigue for stagnation
        min_duration_sec: Minimum segment duration
        duration_sec: Track duration in seconds for time validation/clamping

    Returns:
        List of segment dicts with:
            - 'start_time': start time in seconds
            - 'end_time': end time in seconds
            - 'duration': duration in seconds
            - 'avg_novelty': average novelty in segment
            - 'avg_fatigue': average fatigue in segment
            - 'label': descriptive label
    """
    # Find blocks that meet criteria
    is_stagnant = (novelty_curve < novelty_threshold) & (fatigue_curve > fatigue_threshold)

    # Find contiguous regions
    segments = []
    in_segment = False
    segment_start = 0

    for i in range(len(is_stagnant)):
        if is_stagnant[i] and not in_segment:
            # Start new segment
            in_segment = True
            segment_start = i
        elif not is_stagnant[i] and in_segment:
            # End segment
            segment_end = i - 1
            segment_duration = block_times[segment_end] - block_times[segment_start]

            if segment_duration >= min_duration_sec:
                segments.append({
                    'start_block': segment_start,
                    'end_block': segment_end
                })

            in_segment = False

    # Handle segment extending to end
    if in_segment:
        segment_end = len(is_stagnant) - 1
        segment_duration = block_times[segment_end] - block_times[segment_start]
        if segment_duration >= min_duration_sec:
            segments.append({
                'start_block': segment_start,
                'end_block': segment_end
            })

    # Format output
    results = []
    for seg in segments:
        start_idx = seg['start_block']
        end_idx = seg['end_block']

        avg_novelty = float(np.mean(novelty_curve[start_idx:end_idx+1]))
        avg_fatigue = float(np.mean(fatigue_curve[start_idx:end_idx+1]))

        duration = float(block_times[end_idx] - block_times[start_idx])

        # Generate label
        if avg_fatigue > 0.8 and avg_novelty < 0.2:
            label = "Highly repetitive section"
        elif avg_fatigue > 0.7:
            label = "Repetitive section"
        else:
            label = "Stagnant section"

        results.append({
            'start_time': float(block_times[start_idx]),
            'end_time': float(block_times[end_idx]),
            'duration': duration,
            'avg_novelty': avg_novelty,
            'avg_fatigue': avg_fatigue,
            'label': label
        })

    # Validate and clamp segment times if duration provided
    if duration_sec is not None:
        results = timebase.validate_segment_events(
            results, duration_sec,
            start_key='start_time', end_key='end_time', duration_key='duration',
            min_duration_sec=min_duration_sec, drop_invalid=True
        )

    return results


def detect_section_boundaries(
    novelty_curve: np.ndarray,
    tension_curve: np.ndarray,
    block_times: np.ndarray,
    confidence_threshold: float = config.BOUNDARY_CONFIDENCE_THRESHOLD,
    duration_sec: Optional[float] = None
) -> List[Dict]:
    """
    Detect section boundaries based on novelty peaks and tension changes.

    Parameters:
        novelty_curve: Novelty curve [0, 1]
        tension_curve: Tension curve [0, 1]
        block_times: Array of block center times (seconds)
        confidence_threshold: Minimum confidence to report
        duration_sec: Track duration in seconds for time validation/clamping

    Returns:
        List of boundary dicts with:
            - 'time': boundary time in seconds
            - 'confidence': confidence score [0, 1]
            - 'reason': explanation string
    """
    # Find novelty peaks
    novelty_peaks, _, novelty_proms = detect_peaks_with_prominence(
        novelty_curve,
        prominence_threshold=0.3,  # Lower threshold for boundaries
        min_distance=int(10 / config.BLOCK_DURATION_SEC)  # 10 seconds minimum
    )

    boundaries = []

    for peak_idx in novelty_peaks:
        # Check tension change around this point
        window = 5  # blocks
        start_idx = max(0, peak_idx - window)
        end_idx = min(len(tension_curve), peak_idx + window)

        tension_before = np.mean(tension_curve[start_idx:peak_idx])
        tension_after = np.mean(tension_curve[peak_idx:end_idx])
        tension_change = abs(tension_after - tension_before)

        # Calculate confidence
        novelty_score = novelty_curve[peak_idx]
        tension_score = tension_change

        confidence = 0.6 * novelty_score + 0.4 * tension_score

        if confidence >= confidence_threshold:
            # Determine reason
            if tension_change > 0.3:
                if tension_after > tension_before:
                    reason = "Section transition to higher energy"
                else:
                    reason = "Section transition to lower energy"
            else:
                reason = "Section transition (novelty peak)"

            boundaries.append({
                'time': float(block_times[peak_idx]),
                'confidence': float(confidence),
                'reason': reason
            })

    # Validate and clamp boundary times if duration provided
    if duration_sec is not None:
        boundaries = timebase.validate_point_events(
            boundaries, duration_sec, time_key='time', drop_invalid=True
        )

    return boundaries


def rank_drop_candidates(
    drop_candidates: List[Dict],
    drop_impact_scores: Dict[int, Dict],
    top_n: int = config.TOP_N_DROPS
) -> List[Dict]:
    """
    Rank and filter drop candidates combining detection confidence and impact.

    Parameters:
        drop_candidates: List of candidate dicts from detect_candidate_drops
        drop_impact_scores: Dict from compute_drop_impact_scores
        top_n: Number of top candidates to return

    Returns:
        List of top N ranked drop candidates with impact scores added
    """
    ranked = []

    for idx, candidate in enumerate(drop_candidates):
        # Get impact score if available
        if idx in drop_impact_scores:
            impact = drop_impact_scores[idx]
            total_impact = impact['total_impact']

            # Combined score: 50% detection confidence, 50% impact
            combined_score = 0.5 * candidate['score'] + 0.5 * min(total_impact, 1.0)

            # Add impact to candidate
            candidate_with_impact = candidate.copy()
            candidate_with_impact['impact_score'] = total_impact
            candidate_with_impact['impact_components'] = {
                'rms_delta': impact['rms_delta'],
                'onset_delta': impact['onset_delta'],
                'centroid_delta': impact['centroid_delta'],
                'bandwidth_delta': impact['bandwidth_delta']
            }
            candidate_with_impact['combined_score'] = float(combined_score)

            ranked.append(candidate_with_impact)
        else:
            # No impact score available
            candidate_with_impact = candidate.copy()
            candidate_with_impact['combined_score'] = candidate['score']
            ranked.append(candidate_with_impact)

    # Sort by combined score
    ranked.sort(key=lambda x: x['combined_score'], reverse=True)

    return ranked[:top_n]


def detect_all_events(
    tension_curve: np.ndarray,
    novelty_curve: np.ndarray,
    fatigue_curve: np.ndarray,
    block_times: np.ndarray,
    audio: Optional[np.ndarray] = None,
    sr: Optional[int] = None,
    duration_sec: Optional[float] = None
) -> Dict:
    """
    Detect all events: drops, stagnant segments, boundaries.

    This is the main entry point for event detection.

    Parameters:
        tension_curve: Smoothed tension curve
        novelty_curve: Novelty curve
        fatigue_curve: Fatigue curve
        block_times: Array of block times (seconds)
        audio: Audio array (optional, for drop impact computation)
        sr: Sample rate (optional, for drop impact computation)
        duration_sec: Track duration in seconds for time validation/clamping

    Returns:
        Dictionary containing:
            - 'candidate_drops': list of drop candidates
            - 'stagnant_segments': list of stagnant segments
            - 'boundaries': list of section boundaries
            - 'ranked_drops': top N ranked drops (if audio provided)
    """
    # Detect candidate drops (pass duration for validation)
    candidate_drops = detect_candidate_drops(
        tension_curve, block_times, duration_sec=duration_sec
    )

    # Detect stagnant segments (pass duration for validation)
    stagnant_segments = detect_stagnant_segments(
        novelty_curve, fatigue_curve, block_times, duration_sec=duration_sec
    )

    # Detect section boundaries (pass duration for validation)
    boundaries = detect_section_boundaries(
        novelty_curve, tension_curve, block_times, duration_sec=duration_sec
    )

    result = {
        'candidate_drops': candidate_drops,
        'stagnant_segments': stagnant_segments,
        'boundaries': boundaries
    }

    # Optionally compute drop impact and rank
    if audio is not None and sr is not None:
        from src import metrics
        drop_impact_scores = metrics.compute_drop_impact_scores(
            audio, sr, candidate_drops
        )
        ranked_drops = rank_drop_candidates(
            candidate_drops, drop_impact_scores
        )
        result['ranked_drops'] = ranked_drops
        result['drop_impact_scores'] = drop_impact_scores

    return result

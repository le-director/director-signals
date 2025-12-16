"""
Metrics Module

Compute deterministic long-horizon curves: tension, novelty, fatigue, drop impact.
All algorithms are explainable and parameter-driven.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy.spatial.distance import cosine as cosine_distance

import config
from src import aggregation


def compute_tension_curve(
    block_features: np.ndarray,
    feature_names: List[str],
    weights: Optional[Dict[str, float]] = None,
    normalization_mode: Optional[str] = None,
    sr: int = config.TARGET_SAMPLE_RATE
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray], Dict]:
    """
    Compute tension/energy curve from block features.

    Tension is a weighted combination of:
    - RMS energy (loudness proxy)
    - Onset density (rhythmic drive/impact)
    - Spectral centroid (brightness/aggression)
    - Spectral bandwidth (fullness/richness)

    IMPORTANT: This function applies normalization ONCE at the end (not per-component)
    to preserve internal contrast in high-energy tracks.

    Parameters:
        block_features: (n_blocks, n_features) normalized feature matrix
        feature_names: List of feature names corresponding to columns
        weights: Component weights (None = use config default)
        normalization_mode: 'robust_track' or 'anchored' (None = use config default)
        sr: Sample rate (used for anchored mode)

    Returns:
        Tuple of (tension_raw, tension_smooth, component_contributions, normalization_info)
        tension_raw: unsmoothed tension curve [0, 1]
        tension_smooth: EWMA smoothed tension curve [0, 1]
        component_contributions: dict mapping component names to normalized curves
        normalization_info: dict with pre/post normalization values for export
    """
    if weights is None:
        weights = config.TENSION_WEIGHTS

    if normalization_mode is None:
        normalization_mode = config.TENSION_NORMALIZATION_MODE

    n_blocks = block_features.shape[0]

    # Extract raw (unnormalized) component values
    components_raw = {}

    for comp_name in weights.keys():
        # Find matching feature columns
        if comp_name == 'rms':
            indices = [i for i, name in enumerate(feature_names)
                      if name.startswith('rms')]
        elif comp_name == 'onset_density':
            indices = [i for i, name in enumerate(feature_names)
                      if name.startswith('onset_strength')]
        elif comp_name == 'spectral_centroid':
            indices = [i for i, name in enumerate(feature_names)
                      if name.startswith('spectral_centroid')]
        elif comp_name == 'spectral_bandwidth':
            indices = [i for i, name in enumerate(feature_names)
                      if name.startswith('spectral_bandwidth')]
        else:
            continue

        if len(indices) > 0:
            # Use first matching column (typically the mean)
            feat_col = block_features[:, indices[0]]
            components_raw[comp_name] = feat_col.copy()
        else:
            # Feature not found, use zeros
            components_raw[comp_name] = np.zeros(n_blocks, dtype=np.float32)

    # Apply normalization based on mode
    if normalization_mode == 'anchored':
        # Anchored mode: use physically meaningful reference values
        tension_raw, components_norm, normalization_info = aggregation.normalize_tension_anchored(
            rms_values=components_raw.get('rms', np.zeros(n_blocks)),
            onset_values=components_raw.get('onset_density', np.zeros(n_blocks)),
            centroid_values=components_raw.get('spectral_centroid', np.zeros(n_blocks)),
            bandwidth_values=components_raw.get('spectral_bandwidth', np.zeros(n_blocks)),
            weights=weights,
            sr=sr
        )
    else:
        # Robust track mode: weighted combination THEN single normalization
        # This preserves internal contrast better than normalizing each component

        # First, normalize each component individually for the weighted sum
        # BUT use a consistent approach (percentile-based within each component)
        components_norm = {}
        for comp_name, comp_raw in components_raw.items():
            # Use percentile normalization for each component
            p_low = np.percentile(comp_raw, config.TENSION_PERCENTILE_LOWER)
            p_high = np.percentile(comp_raw, config.TENSION_PERCENTILE_UPPER)
            if p_high - p_low > 1e-8:
                comp_norm = (comp_raw - p_low) / (p_high - p_low)
                comp_norm = np.clip(comp_norm, 0.0, 1.0)
            else:
                comp_norm = np.full_like(comp_raw, 0.5)
            components_norm[comp_name] = comp_norm.astype(np.float32)

        # Weighted combination (already in ~[0,1] range from component normalization)
        tension_combined = np.zeros(n_blocks, dtype=np.float32)
        for comp_name, weight in weights.items():
            if comp_name in components_norm:
                tension_combined += weight * components_norm[comp_name]

        # Final robust normalization using configurable percentiles
        # This is the ONLY normalization of the final curve (no double normalization)
        tension_raw, normalization_info = aggregation.normalize_tension_robust(
            tension_combined,
            percentile_lower=config.TENSION_PERCENTILE_LOWER,
            percentile_upper=config.TENSION_PERCENTILE_UPPER
        )

        # Add component raw stats to normalization info
        normalization_info['component_stats'] = {
            comp_name: {
                'raw_min': float(np.min(comp_raw)),
                'raw_max': float(np.max(comp_raw)),
                'raw_mean': float(np.mean(comp_raw))
            }
            for comp_name, comp_raw in components_raw.items()
        }

    # Smooth with EWMA
    tension_smooth = aggregation.smooth_curve(
        tension_raw,
        method='ewma',
        alpha=config.TENSION_SMOOTH_ALPHA
    )

    return tension_raw, tension_smooth, components_norm, normalization_info


def compute_novelty_curve(
    block_features: np.ndarray,
    lookback_blocks: int = config.NOVELTY_LOOKBACK_BLOCKS,
    smooth_window: int = config.NOVELTY_SMOOTH_WINDOW
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute novelty curve: degree of change relative to recent context.

    For each block, computes distance to mean of recent past blocks.
    Uses cosine distance on z-scored features.

    Parameters:
        block_features: (n_blocks, n_features) normalized feature matrix
        lookback_blocks: Number of past blocks to compare against
        smooth_window: Moving average window for smoothing

    Returns:
        Tuple of (novelty_curve, distances)
        novelty_curve: smoothed novelty in [0, 1]
        distances: raw distance values before smoothing
    """
    n_blocks = block_features.shape[0]
    distances = np.zeros(n_blocks, dtype=np.float32)

    # Z-score the features for better distance computation
    mean = np.mean(block_features, axis=0)
    std = np.std(block_features, axis=0)
    std[std < 1e-8] = 1.0  # Avoid division by zero
    features_zscore = (block_features - mean) / std

    for i in range(n_blocks):
        if i < lookback_blocks:
            # Not enough history, use what we have
            start_idx = 0
        else:
            start_idx = i - lookback_blocks

        # Recent context (exclude current block)
        if i == 0:
            # First block, no history
            distances[i] = 0.0
            continue

        context_features = features_zscore[start_idx:i, :]
        current_features = features_zscore[i, :]

        # Compute mean of context
        context_mean = np.mean(context_features, axis=0)

        # Cosine distance (1 - cosine_similarity)
        # Handle zero vectors
        if np.linalg.norm(current_features) < 1e-8 or np.linalg.norm(context_mean) < 1e-8:
            distances[i] = 0.0
        else:
            distances[i] = cosine_distance(current_features, context_mean)

    # Normalize to [0, 1]
    novelty_raw, _ = aggregation.normalize_curve_to_01(distances)

    # Smooth
    novelty_smooth = aggregation.smooth_curve(
        novelty_raw,
        method='moving_average',
        window=smooth_window
    )

    return novelty_smooth, distances


def compute_self_similarity_matrix(
    block_features: np.ndarray,
    window_size: int = config.FATIGUE_WINDOW_BLOCKS
) -> np.ndarray:
    """
    Compute rolling self-similarity scores.

    For each block, computes max/mean cosine similarity to blocks
    within the lookback window.

    Parameters:
        block_features: (n_blocks, n_features) normalized feature matrix
        window_size: Size of rolling window

    Returns:
        Array of self-similarity scores (n_blocks,) in [0, 1]
        Higher values = more similar to recent past = more repetitive
    """
    n_blocks = block_features.shape[0]
    similarity = np.zeros(n_blocks, dtype=np.float32)

    # Z-score features
    mean = np.mean(block_features, axis=0)
    std = np.std(block_features, axis=0)
    std[std < 1e-8] = 1.0
    features_zscore = (block_features - mean) / std

    for i in range(n_blocks):
        if i < 2:
            # Not enough history
            similarity[i] = 0.0
            continue

        # Define window
        start_idx = max(0, i - window_size)
        window_features = features_zscore[start_idx:i, :]
        current_features = features_zscore[i, :]

        # Compute cosine similarities to all blocks in window
        similarities = []
        for j in range(window_features.shape[0]):
            past_features = window_features[j, :]

            # Cosine similarity = 1 - cosine_distance
            if np.linalg.norm(current_features) < 1e-8 or np.linalg.norm(past_features) < 1e-8:
                sim = 0.0
            else:
                sim = 1.0 - cosine_distance(current_features, past_features)

            similarities.append(sim)

        # Use max similarity as indicator of repetition
        if len(similarities) > 0:
            similarity[i] = max(similarities)
        else:
            similarity[i] = 0.0

    # Clip to [0, 1]
    similarity = np.clip(similarity, 0.0, 1.0)

    return similarity


def compute_feature_variance(
    block_features: np.ndarray,
    window_size: int = config.FATIGUE_WINDOW_BLOCKS
) -> np.ndarray:
    """
    Compute rolling feature variance as indicator of change/stagnation.

    Parameters:
        block_features: (n_blocks, n_features) normalized feature matrix
        window_size: Size of rolling window

    Returns:
        Array of variance scores (n_blocks,) in [0, 1]
        Lower values = less change = more stagnant
    """
    n_blocks = block_features.shape[0]
    variance_scores = np.zeros(n_blocks, dtype=np.float32)

    for i in range(n_blocks):
        if i < window_size:
            # Use available history
            window = block_features[:i+1, :]
        else:
            window = block_features[i-window_size+1:i+1, :]

        if window.shape[0] < 2:
            variance_scores[i] = 0.0
            continue

        # Compute variance across time for each feature
        feature_variances = np.var(window, axis=0)

        # Average variance across features
        mean_variance = np.mean(feature_variances)
        variance_scores[i] = mean_variance

    # Normalize to [0, 1]
    variance_norm, _ = aggregation.normalize_curve_to_01(variance_scores)

    return variance_norm


def compute_fatigue_leaky_integrator(
    self_similarity: np.ndarray,
    novelty_curve: np.ndarray,
    inverse_variance: np.ndarray,
    boundaries: Optional[List[Dict]] = None,
    gain_up: float = config.FATIGUE_GAIN_UP,
    gain_down: float = config.FATIGUE_GAIN_DOWN,
    novelty_spike_threshold: float = config.FATIGUE_NOVELTY_SPIKE_THRESHOLD,
    boring_weights: Optional[Dict[str, float]] = None
) -> np.ndarray:
    """
    Compute fatigue using leaky integrator model.

    Fatigue accumulates slowly when content is boring (high similarity, low novelty)
    and recovers quickly when content is interesting (novelty spikes, boundaries).

    Model: fatigue[t] = clamp(fatigue[t-1] + delta)
    - When boring: delta = +gain_up * boring_score
    - When interesting: delta = -gain_down * (1 + novelty)

    Parameters:
        self_similarity: Self-similarity curve [0, 1]
        novelty_curve: Novelty curve [0, 1]
        inverse_variance: Inverse variance curve [0, 1]
        boundaries: Optional list of detected boundaries for accelerated recovery
        gain_up: Rate of fatigue increase (default 0.02)
        gain_down: Rate of fatigue decrease (default 0.08, 4x faster)
        novelty_spike_threshold: Novelty threshold for "interesting" classification
        boring_weights: Weights for boring signal components

    Returns:
        Fatigue curve [0, 1]
    """
    if boring_weights is None:
        boring_weights = config.FATIGUE_BORING_WEIGHTS

    n_blocks = len(self_similarity)
    fatigue = np.zeros(n_blocks, dtype=np.float32)

    # Invert novelty for boring score
    inverse_novelty = 1.0 - novelty_curve

    # Convert boundaries to block indices for fast lookup
    boundary_blocks = set()
    if boundaries:
        block_duration = config.BLOCK_DURATION_SEC
        for b in boundaries:
            boundary_time = b.get('time', 0)
            block_idx = int(boundary_time / block_duration)
            boundary_blocks.add(block_idx)
            # Also add nearby blocks for tolerance
            boundary_blocks.add(max(0, block_idx - 1))
            boundary_blocks.add(min(n_blocks - 1, block_idx + 1))

    for t in range(1, n_blocks):
        # Compute "boring" score: weighted combination of repetition indicators
        boring = (
            boring_weights.get('self_similarity', 0.5) * self_similarity[t] +
            boring_weights.get('inverse_novelty', 0.3) * inverse_novelty[t] +
            boring_weights.get('inverse_variance', 0.2) * inverse_variance[t]
        )

        # Check if current block is "interesting"
        is_novelty_spike = novelty_curve[t] > novelty_spike_threshold
        is_boundary = t in boundary_blocks

        if is_novelty_spike or is_boundary:
            # Interesting: decay fatigue quickly
            # Extra decay proportional to novelty level
            delta = -gain_down * (1.0 + novelty_curve[t])
        else:
            # Boring: accumulate fatigue slowly
            delta = gain_up * boring

        # Leaky integrator update
        fatigue[t] = fatigue[t-1] + delta

        # Clamp to [0, 1]
        fatigue[t] = np.clip(fatigue[t], 0.0, 1.0)

    return fatigue


def compute_fatigue_curve(
    block_features: np.ndarray,
    novelty_curve: np.ndarray,
    boundaries: Optional[List[Dict]] = None,
    weights: Optional[Dict[str, float]] = None,
    window_size: int = config.FATIGUE_WINDOW_BLOCKS,
    smooth_window: int = config.FATIGUE_SMOOTH_WINDOW,
    use_leaky_integrator: bool = config.FATIGUE_USE_LEAKY_INTEGRATOR
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Compute repetition/fatigue curve.

    Supports two modes:
    - Leaky integrator (default): Fatigue accumulates slowly when boring,
      recovers quickly when interesting (novelty spikes, boundaries)
    - Weighted average (legacy): Simple weighted combination of components

    Fatigue combines:
    - High self-similarity (repetitive patterns)
    - Low novelty (lack of change)
    - Low variance (sustained predictability)

    Parameters:
        block_features: (n_blocks, n_features) normalized feature matrix
        novelty_curve: Pre-computed novelty curve
        boundaries: Optional list of detected boundaries (for leaky integrator recovery)
        weights: Component weights (None = use config default)
        window_size: Rolling window size
        smooth_window: Smoothing window
        use_leaky_integrator: Use leaky integrator model (default True)

    Returns:
        Tuple of (fatigue_curve, intermediate_signals)
        fatigue_curve: smoothed fatigue in [0, 1]
        intermediate_signals: dict with 'self_similarity', 'inverse_novelty', 'inverse_variance',
                             and 'computation_mode'
    """
    if weights is None:
        weights = config.FATIGUE_WEIGHTS

    # Compute components
    self_similarity = compute_self_similarity_matrix(block_features, window_size)
    variance = compute_feature_variance(block_features, window_size)

    # Invert novelty and variance (high fatigue = low novelty/variance)
    inverse_novelty = 1.0 - novelty_curve
    inverse_variance = 1.0 - variance

    if use_leaky_integrator:
        # Use leaky integrator model
        fatigue_raw = compute_fatigue_leaky_integrator(
            self_similarity,
            novelty_curve,
            inverse_variance,
            boundaries,
            gain_up=config.FATIGUE_GAIN_UP,
            gain_down=config.FATIGUE_GAIN_DOWN,
            novelty_spike_threshold=config.FATIGUE_NOVELTY_SPIKE_THRESHOLD
        )
        computation_mode = 'leaky_integrator'
    else:
        # Original weighted combination (for backward compatibility)
        fatigue_raw = (
            weights['self_similarity'] * self_similarity +
            weights['inverse_novelty'] * inverse_novelty +
            weights['inverse_variance'] * inverse_variance
        )
        # Normalize to [0, 1]
        fatigue_raw, _ = aggregation.normalize_curve_to_01(fatigue_raw)
        computation_mode = 'weighted_average'

    # Smooth
    fatigue_smooth = aggregation.smooth_curve(
        fatigue_raw,
        method='moving_average',
        window=smooth_window
    )

    intermediate_signals = {
        'self_similarity': self_similarity,
        'inverse_novelty': inverse_novelty,
        'inverse_variance': inverse_variance,
        'computation_mode': computation_mode
    }

    return fatigue_smooth, intermediate_signals


def compute_drop_impact_scores(
    audio: np.ndarray,
    sr: int,
    drop_candidates: List[Dict],
    pre_window_sec: float = config.DROP_PRE_WINDOW_SEC,
    post_window_sec: float = config.DROP_POST_WINDOW_SEC
) -> Dict[int, Dict]:
    """
    Compute drop impact scores for candidate drops.

    Measures contrast between pre-drop and post-drop windows using:
    - RMS delta (loudness change)
    - Onset strength delta (impact increase)
    - Spectral centroid delta (brightness change)
    - Spectral bandwidth delta (fullness change)

    Parameters:
        audio: Audio array
        sr: Sample rate (Hz)
        drop_candidates: List of drop dicts with 'time' key (seconds)
        pre_window_sec: Duration of pre-drop window
        post_window_sec: Duration of post-drop window

    Returns:
        Dict mapping candidate index to impact score dict:
            - 'total_impact': combined impact score
            - 'rms_delta': RMS change
            - 'onset_delta': onset strength change
            - 'centroid_delta': spectral centroid change
            - 'bandwidth_delta': spectral bandwidth change
    """
    from src import features as feat_module

    impact_scores = {}

    for idx, candidate in enumerate(drop_candidates):
        drop_time = candidate['time']

        # Calculate sample indices
        drop_sample = int(drop_time * sr)
        pre_start_sample = max(0, drop_sample - int(pre_window_sec * sr))
        post_end_sample = min(len(audio), drop_sample + int(post_window_sec * sr))

        # Extract windows
        pre_audio = audio[pre_start_sample:drop_sample]
        post_audio = audio[drop_sample:post_end_sample]

        if len(pre_audio) < sr * 0.5 or len(post_audio) < sr * 0.5:
            # Windows too short
            impact_scores[idx] = {
                'total_impact': 0.0,
                'rms_delta': 0.0,
                'onset_delta': 0.0,
                'centroid_delta': 0.0,
                'bandwidth_delta': 0.0
            }
            continue

        # Compute features for both windows
        pre_rms = np.mean(feat_module.compute_rms_energy(pre_audio))
        post_rms = np.mean(feat_module.compute_rms_energy(post_audio))

        pre_onset = np.mean(feat_module.compute_onset_strength(pre_audio, sr))
        post_onset = np.mean(feat_module.compute_onset_strength(post_audio, sr))

        pre_spectral = feat_module.compute_spectral_features(pre_audio, sr)
        post_spectral = feat_module.compute_spectral_features(post_audio, sr)

        pre_centroid = np.mean(pre_spectral['spectral_centroid'])
        post_centroid = np.mean(post_spectral['spectral_centroid'])

        pre_bandwidth = np.mean(pre_spectral['spectral_bandwidth'])
        post_bandwidth = np.mean(post_spectral['spectral_bandwidth'])

        # Compute deltas (normalized)
        rms_delta = (post_rms - pre_rms) / (pre_rms + 1e-8)
        onset_delta = (post_onset - pre_onset) / (pre_onset + 1e-8)
        centroid_delta = (post_centroid - pre_centroid) / (pre_centroid + 1e-8)
        bandwidth_delta = (post_bandwidth - pre_bandwidth) / (pre_bandwidth + 1e-8)

        # Combined impact (weighted sum of positive deltas)
        # Drops typically have increases in all metrics
        total_impact = (
            max(0, rms_delta) * 0.4 +
            max(0, onset_delta) * 0.3 +
            max(0, centroid_delta) * 0.2 +
            max(0, bandwidth_delta) * 0.1
        )

        impact_scores[idx] = {
            'total_impact': float(total_impact),
            'rms_delta': float(rms_delta),
            'onset_delta': float(onset_delta),
            'centroid_delta': float(centroid_delta),
            'bandwidth_delta': float(bandwidth_delta)
        }

    return impact_scores


def compute_all_curves(
    block_features: np.ndarray,
    feature_names: List[str],
    audio: Optional[np.ndarray] = None,
    sr: Optional[int] = None,
    normalization_mode: Optional[str] = None
) -> Dict:
    """
    Compute all long-horizon curves.

    This is the main entry point for metrics computation.

    Parameters:
        block_features: (n_blocks, n_features) normalized feature matrix
        feature_names: List of feature names
        audio: Audio array (optional, needed for drop impact computation later)
        sr: Sample rate (optional, needed for drop impact computation later)
        normalization_mode: 'robust_track' or 'anchored' (None = use config default)

    Returns:
        Dictionary containing:
            - 'tension_raw': raw tension curve
            - 'tension_smooth': smoothed tension curve
            - 'tension_components': dict of component contributions
            - 'tension_normalization': dict with pre/post normalization info
            - 'novelty': novelty curve
            - 'novelty_distances': raw distances before smoothing
            - 'fatigue': fatigue curve
            - 'fatigue_components': dict of intermediate signals
            - 'audio': audio array (if provided)
            - 'sample_rate': sample rate (if provided)
    """
    # Compute tension (with new normalization)
    tension_raw, tension_smooth, tension_components, tension_normalization = compute_tension_curve(
        block_features, feature_names,
        normalization_mode=normalization_mode,
        sr=sr if sr else config.TARGET_SAMPLE_RATE
    )

    # Compute novelty
    novelty, novelty_distances = compute_novelty_curve(block_features)

    # Compute fatigue
    fatigue, fatigue_components = compute_fatigue_curve(
        block_features, novelty
    )

    return {
        'tension_raw': tension_raw,
        'tension_smooth': tension_smooth,
        'tension_components': tension_components,
        'tension_normalization': tension_normalization,
        'novelty': novelty,
        'novelty_distances': novelty_distances,
        'fatigue': fatigue,
        'fatigue_components': fatigue_components,
        'audio': audio,
        'sample_rate': sr
    }

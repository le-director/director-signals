"""
Aggregation Module

Convert frame-level features to block-level features and long-horizon curves.
Handles temporal aggregation, normalization, and smoothing.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy import signal as scipy_signal

import config
from src import timebase


def frames_to_blocks(
    feature_array: np.ndarray,
    sr: int,
    frame_hop: int,
    block_duration_sec: float = config.BLOCK_DURATION_SEC,
    agg_stats: Optional[List[str]] = None
) -> Tuple[np.ndarray, float]:
    """
    Aggregate frame-level features into block-level features.

    Parameters:
        feature_array: 1D array of frame-level feature values
        sr: Sample rate (Hz)
        frame_hop: Hop length in samples
        block_duration_sec: Block duration in seconds
        agg_stats: List of statistics to compute ['mean', 'median', 'std', 'p25', 'p75']
                   None = use config default

    Returns:
        Tuple of (block_features, time_per_block)
        block_features: 2D array (n_blocks, n_stats) or 1D if single stat
        time_per_block: duration of each block in seconds
    """
    if agg_stats is None:
        agg_stats = config.BLOCK_AGGREGATION_STATS

    # Calculate frames per block
    samples_per_block = int(block_duration_sec * sr)
    frames_per_block = max(1, samples_per_block // frame_hop)

    n_frames = len(feature_array)
    n_blocks = max(1, n_frames // frames_per_block)

    # Aggregate into blocks
    stats_list = []
    for stat in agg_stats:
        block_stat = np.zeros(n_blocks, dtype=np.float32)

        for i in range(n_blocks):
            start_frame = i * frames_per_block
            end_frame = min((i + 1) * frames_per_block, n_frames)
            block_frames = feature_array[start_frame:end_frame]

            if len(block_frames) == 0:
                block_stat[i] = 0.0
                continue

            if stat == 'mean':
                block_stat[i] = np.mean(block_frames)
            elif stat == 'median':
                block_stat[i] = np.median(block_frames)
            elif stat == 'std':
                block_stat[i] = np.std(block_frames)
            elif stat == 'p25':
                block_stat[i] = np.percentile(block_frames, 25)
            elif stat == 'p75':
                block_stat[i] = np.percentile(block_frames, 75)
            elif stat == 'min':
                block_stat[i] = np.min(block_frames)
            elif stat == 'max':
                block_stat[i] = np.max(block_frames)
            else:
                raise ValueError(f"Unknown aggregation statistic: {stat}")

        stats_list.append(block_stat)

    # Stack into 2D array if multiple stats, otherwise return 1D
    if len(stats_list) == 1:
        block_features = stats_list[0]
    else:
        block_features = np.column_stack(stats_list)

    return block_features, block_duration_sec


def aggregate_frame_features(
    frame_features_dict: Dict,
    sr: int,
    frame_hop: int = config.HOP_LENGTH,
    block_duration_sec: float = config.BLOCK_DURATION_SEC,
    agg_stats: Optional[List[str]] = None,
    duration_sec: Optional[float] = None
) -> Tuple[np.ndarray, List[str], np.ndarray]:
    """
    Aggregate all frame features into block feature matrix.

    Parameters:
        frame_features_dict: Dictionary from extract_all_features()
        sr: Sample rate (Hz)
        frame_hop: Hop length in samples
        block_duration_sec: Block duration in seconds
        agg_stats: Statistics to compute per block
        duration_sec: Track duration in seconds (source of truth for timebase).
                      If provided, ensures block count and times respect duration.

    Returns:
        Tuple of (block_feature_matrix, feature_names, block_times)
        block_feature_matrix: (n_blocks, n_features) array
        feature_names: list of feature names for each column
        block_times: array of block center times in seconds (guaranteed <= duration_sec)
    """
    if agg_stats is None:
        agg_stats = config.BLOCK_AGGREGATION_STATS

    # Features to aggregate (1D arrays)
    features_1d = [
        'rms', 'spectral_centroid', 'spectral_bandwidth',
        'spectral_rolloff', 'spectral_flux', 'zcr', 'onset_strength'
    ]

    block_features_list = []
    feature_names = []

    # Aggregate 1D features
    for feat_name in features_1d:
        if feat_name in frame_features_dict:
            feat_array = frame_features_dict[feat_name]
            block_feat, _ = frames_to_blocks(
                feat_array, sr, frame_hop, block_duration_sec, agg_stats
            )

            # Handle both 1D and 2D outputs
            if block_feat.ndim == 1:
                block_feat = block_feat[:, np.newaxis]

            block_features_list.append(block_feat)

            # Generate feature names with stat suffixes
            if block_feat.shape[1] == 1:
                feature_names.append(feat_name)
            else:
                for stat in agg_stats:
                    feature_names.append(f"{feat_name}_{stat}")

    # Aggregate MFCCs (2D array)
    if 'mfcc' in frame_features_dict:
        mfcc = frame_features_dict['mfcc']  # (n_mfcc, n_frames)

        # Use only mean of each MFCC coefficient per block to keep dimensionality manageable
        for mfcc_idx in range(mfcc.shape[0]):
            mfcc_coef = mfcc[mfcc_idx, :]
            block_mfcc, _ = frames_to_blocks(
                mfcc_coef, sr, frame_hop, block_duration_sec, ['mean']
            )
            block_features_list.append(block_mfcc[:, np.newaxis])
            feature_names.append(f"mfcc_{mfcc_idx}_mean")

    # Stack all features
    block_feature_matrix = np.hstack(block_features_list)
    n_blocks_available = block_feature_matrix.shape[0]

    # Use canonical block count if duration is provided
    if duration_sec is not None:
        n_blocks_canonical = timebase.compute_canonical_block_count(
            duration_sec, block_duration_sec
        )
        # Use the minimum of canonical and available blocks
        n_blocks = min(n_blocks_canonical, n_blocks_available)
        # Truncate feature matrix if needed
        if n_blocks < n_blocks_available:
            block_feature_matrix = block_feature_matrix[:n_blocks]
    else:
        n_blocks = n_blocks_available

    # Calculate block times using canonical time axis
    block_times = timebase.compute_canonical_time_axis(
        n_blocks, block_duration_sec, start_time_sec=0.0, duration_sec=duration_sec
    )

    return block_feature_matrix, feature_names, block_times


def normalize_block_features(
    block_features: np.ndarray,
    method: str = 'robust'
) -> Tuple[np.ndarray, Dict]:
    """
    Normalize block features to common scale.

    Parameters:
        block_features: (n_blocks, n_features) array
        method: Normalization method
            - 'robust': (x - median) / IQR
            - 'percentile': scale to [0, 1] using percentiles
            - 'zscore': (x - mean) / std

    Returns:
        Tuple of (normalized_features, normalization_params)
        normalization_params: dict with parameters used for each feature
    """
    if block_features.ndim == 1:
        block_features = block_features[:, np.newaxis]

    n_blocks, n_features = block_features.shape
    normalized = np.zeros_like(block_features, dtype=np.float32)
    params = {}

    for i in range(n_features):
        feature_col = block_features[:, i]

        if method == 'robust':
            # Robust scaling using median and IQR
            median = np.median(feature_col)
            q75 = np.percentile(feature_col, 75)
            q25 = np.percentile(feature_col, 25)
            iqr = q75 - q25

            if iqr > 1e-8:
                normalized[:, i] = (feature_col - median) / iqr
            else:
                normalized[:, i] = 0.0

            params[i] = {'method': 'robust', 'median': median, 'iqr': iqr}

        elif method == 'percentile':
            # Scale to [0, 1] using percentiles
            p_low = np.percentile(feature_col, config.PERCENTILE_LOWER)
            p_high = np.percentile(feature_col, config.PERCENTILE_UPPER)

            if p_high - p_low > 1e-8:
                normalized[:, i] = (feature_col - p_low) / (p_high - p_low)
                normalized[:, i] = np.clip(normalized[:, i], 0.0, 1.0)
            else:
                normalized[:, i] = 0.0

            params[i] = {
                'method': 'percentile',
                'p_low': p_low,
                'p_high': p_high
            }

        elif method == 'zscore':
            # Z-score normalization
            mean = np.mean(feature_col)
            std = np.std(feature_col)

            if std > 1e-8:
                normalized[:, i] = (feature_col - mean) / std
            else:
                normalized[:, i] = 0.0

            params[i] = {'method': 'zscore', 'mean': mean, 'std': std}

        else:
            raise ValueError(f"Unknown normalization method: {method}")

    return normalized, params


def smooth_curve(
    curve: np.ndarray,
    method: str = 'ewma',
    **params
) -> np.ndarray:
    """
    Smooth a curve using specified method.

    Parameters:
        curve: 1D array to smooth
        method: Smoothing method
            - 'ewma': Exponentially Weighted Moving Average
            - 'moving_average': Simple moving average
            - 'savgol': Savitzky-Golay filter
        **params: Method-specific parameters
            - For 'ewma': alpha (default from config)
            - For 'moving_average': window (default 3)
            - For 'savgol': window_length, polyorder

    Returns:
        Smoothed curve (same length as input)
    """
    if len(curve) == 0:
        return curve

    if method == 'ewma':
        # Exponentially Weighted Moving Average
        alpha = params.get('alpha', config.TENSION_SMOOTH_ALPHA)

        smoothed = np.zeros_like(curve, dtype=np.float32)
        smoothed[0] = curve[0]

        for i in range(1, len(curve)):
            smoothed[i] = alpha * curve[i] + (1 - alpha) * smoothed[i-1]

        return smoothed

    elif method == 'moving_average':
        # Simple moving average
        window = params.get('window', 3)

        if window <= 1:
            return curve

        # Use convolution for efficient moving average
        kernel = np.ones(window) / window
        # Pad to preserve length
        pad_width = window // 2
        padded = np.pad(curve, pad_width, mode='edge')
        smoothed = np.convolve(padded, kernel, mode='valid')

        # Ensure output length matches input
        if len(smoothed) > len(curve):
            smoothed = smoothed[:len(curve)]
        elif len(smoothed) < len(curve):
            smoothed = np.pad(smoothed, (0, len(curve) - len(smoothed)), mode='edge')

        return smoothed.astype(np.float32)

    elif method == 'savgol':
        # Savitzky-Golay filter
        window_length = params.get('window_length', 5)
        polyorder = params.get('polyorder', 2)

        if window_length > len(curve):
            window_length = len(curve) if len(curve) % 2 == 1 else len(curve) - 1

        if window_length < polyorder + 2:
            window_length = polyorder + 2
            if window_length % 2 == 0:
                window_length += 1

        if window_length > len(curve):
            return curve

        smoothed = scipy_signal.savgol_filter(
            curve, window_length, polyorder, mode='nearest'
        )
        return smoothed.astype(np.float32)

    else:
        raise ValueError(f"Unknown smoothing method: {method}")


def normalize_curve_to_01(curve: np.ndarray) -> Tuple[np.ndarray, Dict]:
    """
    Normalize curve to [0, 1] range using percentile scaling.

    Parameters:
        curve: 1D array

    Returns:
        Tuple of (normalized_curve, params)
        params: dict with 'min' and 'max' values used
    """
    p_low = np.percentile(curve, config.PERCENTILE_LOWER)
    p_high = np.percentile(curve, config.PERCENTILE_UPPER)

    if p_high - p_low > 1e-8:
        normalized = (curve - p_low) / (p_high - p_low)
        normalized = np.clip(normalized, 0.0, 1.0)
    else:
        normalized = np.zeros_like(curve)

    params = {'p_low': float(p_low), 'p_high': float(p_high)}

    return normalized.astype(np.float32), params


def normalize_tension_robust(
    curve: np.ndarray,
    percentile_lower: float = config.TENSION_PERCENTILE_LOWER,
    percentile_upper: float = config.TENSION_PERCENTILE_UPPER
) -> Tuple[np.ndarray, Dict]:
    """
    Robust normalization for tension curves using configurable percentiles.

    Uses wider percentiles (5th-95th by default) than normalize_curve_to_01
    to better preserve contrast in high-energy tracks.

    Parameters:
        curve: 1D array (raw tension values)
        percentile_lower: Lower percentile for scaling (default 5.0)
        percentile_upper: Upper percentile for scaling (default 95.0)

    Returns:
        Tuple of (normalized_curve, normalization_info)
        normalization_info: dict with pre/post values and parameters
    """
    # Store pre-normalization stats
    pre_min = float(np.min(curve))
    pre_max = float(np.max(curve))
    pre_mean = float(np.mean(curve))
    pre_std = float(np.std(curve))

    p_low = np.percentile(curve, percentile_lower)
    p_high = np.percentile(curve, percentile_upper)

    if p_high - p_low > 1e-8:
        normalized = (curve - p_low) / (p_high - p_low)
        normalized = np.clip(normalized, 0.0, 1.0)
    else:
        # No variation - set to middle value
        normalized = np.full_like(curve, 0.5)

    normalization_info = {
        'mode': 'robust_track',
        'percentile_lower': percentile_lower,
        'percentile_upper': percentile_upper,
        'p_low': float(p_low),
        'p_high': float(p_high),
        'pre_norm': {
            'min': pre_min,
            'max': pre_max,
            'mean': pre_mean,
            'std': pre_std
        },
        'post_norm': {
            'min': float(np.min(normalized)),
            'max': float(np.max(normalized)),
            'mean': float(np.mean(normalized)),
            'std': float(np.std(normalized))
        }
    }

    return normalized.astype(np.float32), normalization_info


def normalize_tension_anchored(
    rms_values: np.ndarray,
    onset_values: np.ndarray,
    centroid_values: np.ndarray,
    bandwidth_values: np.ndarray,
    weights: Dict[str, float],
    sr: int = config.TARGET_SAMPLE_RATE
) -> Tuple[np.ndarray, Dict[str, np.ndarray], Dict]:
    """
    Anchored normalization using physically meaningful reference values.

    Each component is scaled by absolute reference values rather than
    track-relative percentiles. This preserves cross-track comparability.

    Parameters:
        rms_values: Raw RMS energy values (linear amplitude)
        onset_values: Raw onset strength values
        centroid_values: Raw spectral centroid values (Hz)
        bandwidth_values: Raw spectral bandwidth values (Hz)
        weights: Component weights for combining
        sr: Sample rate (for Nyquist calculation)

    Returns:
        Tuple of (tension_curve, normalized_components, normalization_info)
    """
    # Normalize RMS using dBFS reference
    # Convert linear RMS to dB
    rms_db = 20 * np.log10(rms_values + 1e-8)
    # Scale: -60 dBFS -> 0.0, reference_dbfs -> 1.0
    reference_dbfs = config.ANCHORED_RMS_MAX_DBFS
    rms_norm = (rms_db - (-60)) / (reference_dbfs - (-60))
    rms_norm = np.clip(rms_norm, 0.0, 1.0)

    # Normalize onset strength using reference maximum
    onset_norm = onset_values / config.ANCHORED_ONSET_MAX
    onset_norm = np.clip(onset_norm, 0.0, 1.0)

    # Normalize spectral centroid using reference maximum
    # Also scale by Nyquist to account for sample rate
    nyquist = sr / 2.0
    centroid_ref = min(config.ANCHORED_CENTROID_MAX_HZ, nyquist)
    centroid_norm = centroid_values / centroid_ref
    centroid_norm = np.clip(centroid_norm, 0.0, 1.0)

    # Normalize spectral bandwidth using reference maximum
    bandwidth_ref = min(config.ANCHORED_BANDWIDTH_MAX_HZ, nyquist)
    bandwidth_norm = bandwidth_values / bandwidth_ref
    bandwidth_norm = np.clip(bandwidth_norm, 0.0, 1.0)

    # Store normalized components
    normalized_components = {
        'rms': rms_norm.astype(np.float32),
        'onset_density': onset_norm.astype(np.float32),
        'spectral_centroid': centroid_norm.astype(np.float32),
        'spectral_bandwidth': bandwidth_norm.astype(np.float32)
    }

    # Weighted combination
    tension = np.zeros(len(rms_values), dtype=np.float32)
    for comp_name, comp_values in normalized_components.items():
        if comp_name in weights:
            tension += weights[comp_name] * comp_values

    # Final clip (should already be in [0,1] but ensure)
    tension = np.clip(tension, 0.0, 1.0)

    normalization_info = {
        'mode': 'anchored',
        'references': {
            'rms_max_dbfs': config.ANCHORED_RMS_MAX_DBFS,
            'onset_max': config.ANCHORED_ONSET_MAX,
            'centroid_max_hz': centroid_ref,
            'bandwidth_max_hz': bandwidth_ref
        },
        'pre_norm': {
            'rms_db_mean': float(np.mean(rms_db)),
            'onset_mean': float(np.mean(onset_values)),
            'centroid_mean': float(np.mean(centroid_values)),
            'bandwidth_mean': float(np.mean(bandwidth_values))
        },
        'post_norm': {
            'min': float(np.min(tension)),
            'max': float(np.max(tension)),
            'mean': float(np.mean(tension)),
            'std': float(np.std(tension))
        }
    }

    return tension, normalized_components, normalization_info


def compute_delta_features(
    features: np.ndarray,
    order: int = 1
) -> np.ndarray:
    """
    Compute delta (derivative) features.

    Parameters:
        features: (n_frames, n_features) or (n_features,) array
        order: Delta order (1 = first derivative, 2 = second derivative)

    Returns:
        Delta features with same shape as input
    """
    if features.ndim == 1:
        features = features[:, np.newaxis]
        squeeze = True
    else:
        squeeze = False

    delta = np.zeros_like(features)

    # Compute simple finite difference
    for i in range(order):
        if i == 0:
            delta[1:] = np.diff(features, axis=0)
            delta[0] = delta[1]  # Pad first frame
        else:
            delta[1:] = np.diff(delta, axis=0)
            delta[0] = delta[1]

    if squeeze:
        delta = delta.squeeze()

    return delta.astype(np.float32)

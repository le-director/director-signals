"""
DSP Kernel Module - Core Signal Processing Functions

This module contains the deterministic DSP kernel for audio analysis.
All functions are designed for line-by-line C++ portability.

DESIGN CONSTRAINTS:
- No I/O operations (no file reading/writing)
- No plotting or visualization
- No event detection heuristics
- Explicit state management (no hidden globals)
- No config module imports - all parameters are explicit
- Only numpy and scipy dependencies (no librosa)

PROCESSING PIPELINE:
1. Frame-level feature extraction (audio -> frame features)
2. Block aggregation (frames -> blocks)
3. Feature normalization (blocks -> normalized blocks)
4. Curve computation (blocks -> tension, novelty, fatigue)

This file is designed to be ported verbatim to C++ for Phase 2.

SHARED TIMEBASE SPEC (Phase 1/2 Alignment):
- All block timing is governed by the shared timebase utility (see src/timebase.py)
- Block count formula: n = max(1, floor((duration - start) / block_dur + 0.5))
- Block center time: t[i] = start + (i + 0.5) * block_dur
- Final block center <= duration (guaranteed by block count formula)
- C++ port should use identical formulas for deterministic parity
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy import signal as scipy_signal
from scipy.fft import dct as scipy_dct
from scipy.spatial.distance import cosine as cosine_distance


# =============================================================================
# DEFAULT PARAMETERS (Explicit - No Config Imports)
# =============================================================================

# Frame-level defaults
DEFAULT_FRAME_LENGTH: int = 2048
DEFAULT_HOP_LENGTH: int = 512
DEFAULT_SAMPLE_RATE: int = 22050

# Block-level defaults
DEFAULT_BLOCK_DURATION_SEC: float = 0.5

# Tension curve defaults
DEFAULT_TENSION_WEIGHTS: Dict[str, float] = {
    'rms': 0.4,
    'onset_density': 0.3,
    'spectral_centroid': 0.2,
    'spectral_bandwidth': 0.1
}
DEFAULT_TENSION_SMOOTH_ALPHA: float = 0.3
DEFAULT_TENSION_PERCENTILE_LOWER: float = 5.0
DEFAULT_TENSION_PERCENTILE_UPPER: float = 95.0

# Novelty curve defaults
DEFAULT_NOVELTY_LOOKBACK_BLOCKS: int = 16
DEFAULT_NOVELTY_SMOOTH_WINDOW: int = 3

# Fatigue curve defaults
DEFAULT_FATIGUE_WINDOW_BLOCKS: int = 32
DEFAULT_FATIGUE_SMOOTH_WINDOW: int = 5
DEFAULT_FATIGUE_GAIN_UP: float = 0.02
DEFAULT_FATIGUE_GAIN_DOWN: float = 0.08
DEFAULT_FATIGUE_NOVELTY_SPIKE_THRESHOLD: float = 0.5
DEFAULT_FATIGUE_BORING_WEIGHTS: Dict[str, float] = {
    'self_similarity': 0.5,
    'inverse_novelty': 0.3,
    'inverse_variance': 0.2
}

# Normalization defaults
DEFAULT_PERCENTILE_LOWER: float = 1.0
DEFAULT_PERCENTILE_UPPER: float = 99.0
DEFAULT_SPECTRAL_ROLLOFF_PERCENT: float = 0.85

# Anchored normalization references
DEFAULT_ANCHORED_RMS_MAX_DBFS: float = -6.0
DEFAULT_ANCHORED_ONSET_MAX: float = 50.0
DEFAULT_ANCHORED_CENTROID_MAX_HZ: float = 8000.0
DEFAULT_ANCHORED_BANDWIDTH_MAX_HZ: float = 6000.0

# MFCC defaults
DEFAULT_N_MFCC: int = 13
DEFAULT_N_MELS: int = 40


# =============================================================================
# STATEFUL CLASS FOR REAL-TIME FATIGUE COMPUTATION
# =============================================================================

class FatigueState:
    """
    Explicit state container for fatigue leaky integrator.

    For C++ port: This becomes a struct with reset() method.

    CONTRACT:
    - fatigue_value is always in [0.0, 1.0]
    - Call reset() before processing a new track
    - Thread-safe if accessed from single thread
    """

    def __init__(self) -> None:
        self.fatigue_value: float = 0.0

    def reset(self) -> None:
        """Reset state to initial values."""
        self.fatigue_value = 0.0


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _create_mel_filterbank(
    n_bins: int,
    n_mels: int = DEFAULT_N_MELS,
    sr: int = DEFAULT_SAMPLE_RATE,
    fmin: float = 0.0,
    fmax: Optional[float] = None
) -> np.ndarray:
    """
    Create mel filterbank matrix.

    CONTRACT:
    - Input: n_bins (positive int), n_mels (positive int), sr (positive int)
    - Output: (n_mels, n_bins) float32 array
    - Each row is a triangular filter in the mel scale
    - Rows sum to values <= 1.0 (triangular filters)
    - Deterministic: same input -> same output

    C++ PORT NOTES:
    - Pre-compute at initialization for fixed sr/n_fft
    - Store as 2D array, use matrix multiplication with magnitude spectrum

    Parameters:
        n_bins: Number of FFT frequency bins
        n_mels: Number of mel bands
        sr: Sample rate (Hz)
        fmin: Minimum frequency (Hz)
        fmax: Maximum frequency (Hz, None = sr/2)

    Returns:
        Mel filterbank matrix (n_mels, n_bins)
    """
    if fmax is None:
        fmax = sr / 2.0

    # Convert Hz to mel scale
    # mel = 2595 * log10(1 + hz / 700)
    mel_min = 2595.0 * np.log10(1.0 + fmin / 700.0)
    mel_max = 2595.0 * np.log10(1.0 + fmax / 700.0)

    # Create mel-spaced points
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)

    # Convert back to Hz
    # hz = 700 * (10^(mel / 2595) - 1)
    hz_points = 700.0 * (np.power(10.0, mel_points / 2595.0) - 1.0)

    # Convert to FFT bin indices
    bin_points = np.floor((n_bins - 1) * hz_points / fmax).astype(np.int32)

    # Create filterbank
    filterbank = np.zeros((n_mels, n_bins), dtype=np.float32)

    for i in range(n_mels):
        left = bin_points[i]
        center = bin_points[i + 1]
        right = bin_points[i + 2]

        # Rising slope
        if center > left:
            for j in range(left, center):
                filterbank[i, j] = (j - left) / (center - left)

        # Falling slope
        if right > center:
            for j in range(center, right):
                filterbank[i, j] = (right - j) / (right - center)

    return filterbank


# =============================================================================
# FRAME-LEVEL FEATURE EXTRACTION
# =============================================================================

def compute_rms_energy(
    audio: np.ndarray,
    frame_length: int = DEFAULT_FRAME_LENGTH,
    hop_length: int = DEFAULT_HOP_LENGTH
) -> np.ndarray:
    """
    Compute RMS energy per frame.

    CONTRACT:
    - Input: audio (1D float32/float64, normalized to [-1.0, 1.0])
    - Input: frame_length (positive int, typically 2048)
    - Input: hop_length (positive int, typically 512)
    - Output: (n_frames,) float32 array, values >= 0
    - n_frames = 1 + (len(audio) - frame_length) // hop_length
    - Deterministic: same input -> same output

    C++ PORT NOTES:
    - Simple loop, easily vectorized with SIMD
    - No external dependencies

    Parameters:
        audio: Audio array (1D)
        frame_length: Frame size in samples
        hop_length: Hop size in samples

    Returns:
        Array of RMS values (length n_frames)
    """
    num_frames = 1 + (len(audio) - frame_length) // hop_length
    rms = np.zeros(num_frames, dtype=np.float32)

    for i in range(num_frames):
        start = i * hop_length
        end = start + frame_length
        frame = audio[start:end]
        rms[i] = np.sqrt(np.mean(frame ** 2))

    return rms


def compute_stft(
    audio: np.ndarray,
    frame_length: int = DEFAULT_FRAME_LENGTH,
    hop_length: int = DEFAULT_HOP_LENGTH
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Short-Time Fourier Transform.

    CONTRACT:
    - Input: audio (1D float array)
    - Output: (magnitude, phase) tuple
    - magnitude: (n_bins, n_frames) float32, values >= 0
    - phase: (n_bins, n_frames) float32, values in [-pi, pi]
    - n_bins = frame_length // 2 + 1
    - Deterministic: same input -> same output

    C++ PORT NOTES:
    - Use JUCE FFT or FFTW for efficiency
    - Hann window applied implicitly by scipy.signal.stft

    Parameters:
        audio: Audio array (1D)
        frame_length: Frame size in samples (n_fft)
        hop_length: Hop size in samples

    Returns:
        Tuple of (magnitude_spectrogram, phase_spectrogram)
    """
    # Use scipy for STFT
    f, t, stft_complex = scipy_signal.stft(
        audio,
        nperseg=frame_length,
        noverlap=frame_length - hop_length,
        nfft=frame_length
    )

    magnitude = np.abs(stft_complex).astype(np.float32)
    phase = np.angle(stft_complex).astype(np.float32)

    return magnitude, phase


def compute_spectral_features(
    audio: np.ndarray,
    sr: int = DEFAULT_SAMPLE_RATE,
    frame_length: int = DEFAULT_FRAME_LENGTH,
    hop_length: int = DEFAULT_HOP_LENGTH,
    rolloff_percent: float = DEFAULT_SPECTRAL_ROLLOFF_PERCENT
) -> Dict[str, np.ndarray]:
    """
    Compute spectral features: centroid, bandwidth, rolloff.

    CONTRACT:
    - Input: audio (1D float array), sr (positive int)
    - Output: dict with 'spectral_centroid', 'spectral_bandwidth', 'spectral_rolloff'
    - All output arrays: (n_frames,) float32
    - centroid: Hz, typically in [0, sr/2]
    - bandwidth: Hz, typically in [0, sr/2]
    - rolloff: Hz, typically in [0, sr/2]
    - Deterministic: same input -> same output

    C++ PORT NOTES:
    - Requires STFT magnitude spectrum
    - All operations are vectorizable

    Parameters:
        audio: Audio array (1D)
        sr: Sample rate (Hz)
        frame_length: Frame size in samples
        hop_length: Hop size in samples
        rolloff_percent: Percentage for rolloff (0.0 to 1.0)

    Returns:
        Dictionary with spectral features
    """
    magnitude, _ = compute_stft(audio, frame_length, hop_length)
    n_bins, n_frames = magnitude.shape

    # Frequency bins
    freqs = np.fft.rfftfreq(frame_length, 1.0 / sr)

    # Spectral centroid: weighted mean frequency
    # centroid = sum(freq * mag) / sum(mag)
    mag_sum = np.sum(magnitude, axis=0) + 1e-8
    centroid = np.sum(freqs[:, np.newaxis] * magnitude, axis=0) / mag_sum

    # Spectral bandwidth: weighted std of frequencies around centroid
    # bandwidth = sqrt(sum((freq - centroid)^2 * mag) / sum(mag))
    freq_diff_sq = (freqs[:, np.newaxis] - centroid[np.newaxis, :]) ** 2
    bandwidth = np.sqrt(
        np.sum(freq_diff_sq * magnitude, axis=0) / mag_sum
    )

    # Spectral rolloff: frequency below which X% of energy is contained
    cumsum_mag = np.cumsum(magnitude, axis=0)
    total_energy = cumsum_mag[-1, :]
    threshold = rolloff_percent * total_energy

    rolloff = np.zeros(n_frames, dtype=np.float32)
    for i in range(n_frames):
        # Find first bin where cumsum exceeds threshold
        idx = np.where(cumsum_mag[:, i] >= threshold[i])[0]
        if len(idx) > 0:
            rolloff[i] = freqs[idx[0]]
        else:
            rolloff[i] = freqs[-1]

    return {
        'spectral_centroid': centroid.astype(np.float32),
        'spectral_bandwidth': bandwidth.astype(np.float32),
        'spectral_rolloff': rolloff.astype(np.float32)
    }


def compute_spectral_flux(
    audio: np.ndarray,
    frame_length: int = DEFAULT_FRAME_LENGTH,
    hop_length: int = DEFAULT_HOP_LENGTH
) -> np.ndarray:
    """
    Compute spectral flux: frame-to-frame change in magnitude spectrum.

    CONTRACT:
    - Input: audio (1D float array)
    - Output: (n_frames,) float32 array, values >= 0
    - First frame has flux = 0 (no previous frame)
    - Uses half-wave rectification (only increases count)
    - Deterministic: same input -> same output

    C++ PORT NOTES:
    - Requires STFT magnitude
    - Simple loop with max operation

    Parameters:
        audio: Audio array (1D)
        frame_length: Frame size in samples
        hop_length: Hop size in samples

    Returns:
        Array of spectral flux values (length n_frames)
    """
    magnitude, _ = compute_stft(audio, frame_length, hop_length)
    n_frames = magnitude.shape[1]

    flux = np.zeros(n_frames, dtype=np.float32)

    for i in range(1, n_frames):
        diff = magnitude[:, i] - magnitude[:, i - 1]
        # Half-wave rectification: only count increases
        diff = np.maximum(diff, 0.0)
        flux[i] = np.sum(diff)

    return flux


def compute_zcr(
    audio: np.ndarray,
    frame_length: int = DEFAULT_FRAME_LENGTH,
    hop_length: int = DEFAULT_HOP_LENGTH
) -> np.ndarray:
    """
    Compute zero crossing rate per frame.

    CONTRACT:
    - Input: audio (1D float array)
    - Output: (n_frames,) float32 array, values in [0, 1]
    - ZCR = (number of sign changes) / frame_length
    - Deterministic: same input -> same output

    C++ PORT NOTES:
    - Simple loop counting sign changes
    - Treat zero as positive for consistency

    Parameters:
        audio: Audio array (1D)
        frame_length: Frame size in samples
        hop_length: Hop size in samples

    Returns:
        Array of ZCR values (length n_frames)
    """
    num_frames = 1 + (len(audio) - frame_length) // hop_length
    zcr = np.zeros(num_frames, dtype=np.float32)

    for i in range(num_frames):
        start = i * hop_length
        end = start + frame_length
        frame = audio[start:end]

        # Count sign changes
        signs = np.sign(frame)
        signs[signs == 0] = 1  # Treat zero as positive
        crossings = np.sum(np.abs(np.diff(signs))) / 2.0
        zcr[i] = crossings / frame_length

    return zcr


def compute_onset_strength(
    audio: np.ndarray,
    sr: int = DEFAULT_SAMPLE_RATE,
    frame_length: int = DEFAULT_FRAME_LENGTH,
    hop_length: int = DEFAULT_HOP_LENGTH
) -> np.ndarray:
    """
    Compute onset strength envelope (transient density proxy).

    CONTRACT:
    - Input: audio (1D float array), sr (positive int)
    - Output: (n_frames,) float32 array, values >= 0
    - Uses spectral flux as onset proxy
    - Deterministic: same input -> same output

    C++ PORT NOTES:
    - Implemented as spectral flux (no librosa dependency)
    - Can be enhanced with mel-band weighting if needed

    Parameters:
        audio: Audio array (1D)
        sr: Sample rate (Hz) - not used but kept for interface consistency
        frame_length: Frame size in samples
        hop_length: Hop size in samples

    Returns:
        Array of onset strength values (length n_frames)
    """
    # Use spectral flux as onset strength proxy
    # This is the librosa fallback behavior
    return compute_spectral_flux(audio, frame_length, hop_length)


def compute_mfcc_stats(
    audio: np.ndarray,
    sr: int = DEFAULT_SAMPLE_RATE,
    frame_length: int = DEFAULT_FRAME_LENGTH,
    hop_length: int = DEFAULT_HOP_LENGTH,
    n_mfcc: int = DEFAULT_N_MFCC,
    n_mels: int = DEFAULT_N_MELS
) -> np.ndarray:
    """
    Compute MFCC coefficients per frame.

    CONTRACT:
    - Input: audio (1D float array), sr (positive int)
    - Output: (n_mfcc, n_frames) float32 array
    - First coefficient (c0) relates to energy
    - Remaining coefficients capture timbre
    - Deterministic: same input -> same output

    C++ PORT NOTES:
    - Requires mel filterbank (pre-compute at init)
    - Uses DCT-II for cepstral transformation
    - scipy.fft.dct with type=2, norm='ortho'

    Parameters:
        audio: Audio array (1D)
        sr: Sample rate (Hz)
        frame_length: Frame size in samples
        hop_length: Hop size in samples
        n_mfcc: Number of MFCC coefficients
        n_mels: Number of mel bands

    Returns:
        Array of shape (n_mfcc, n_frames)
    """
    # Compute magnitude spectrogram
    magnitude, _ = compute_stft(audio, frame_length, hop_length)
    n_bins = magnitude.shape[0]

    # Create mel filterbank
    mel_filters = _create_mel_filterbank(n_bins, n_mels, sr)

    # Apply mel filters
    mel_spec = mel_filters @ magnitude

    # Log compression
    log_mel = np.log(mel_spec + 1e-8)

    # DCT to get MFCCs
    mfccs = scipy_dct(log_mel, axis=0, type=2, norm='ortho')[:n_mfcc]

    return mfccs.astype(np.float32)


# =============================================================================
# BLOCK AGGREGATION
# =============================================================================

def frames_to_blocks(
    feature_array: np.ndarray,
    sr: int = DEFAULT_SAMPLE_RATE,
    frame_hop: int = DEFAULT_HOP_LENGTH,
    block_duration_sec: float = DEFAULT_BLOCK_DURATION_SEC,
    agg_stat: str = 'mean'
) -> Tuple[np.ndarray, float]:
    """
    Aggregate frame-level features into block-level features.

    CONTRACT:
    - Input: feature_array (1D float array)
    - Input: agg_stat one of: 'mean', 'median', 'std', 'p25', 'p75', 'min', 'max'
    - Output: (block_features, block_duration_sec)
    - block_features: (n_blocks,) float32 array
    - n_blocks = n_frames // frames_per_block
    - Deterministic: same input -> same output

    C++ PORT NOTES:
    - Simple loop with statistic computation
    - For multiple stats, call multiple times or use aggregate_to_blocks()

    Parameters:
        feature_array: 1D array of frame-level feature values
        sr: Sample rate (Hz)
        frame_hop: Hop length in samples
        block_duration_sec: Block duration in seconds
        agg_stat: Statistic to compute ('mean', 'median', 'std', 'p25', 'p75', 'min', 'max')

    Returns:
        Tuple of (block_features, block_duration_sec)
    """
    # Calculate frames per block
    samples_per_block = int(block_duration_sec * sr)
    frames_per_block = max(1, samples_per_block // frame_hop)

    n_frames = len(feature_array)
    n_blocks = max(1, n_frames // frames_per_block)

    block_features = np.zeros(n_blocks, dtype=np.float32)

    for i in range(n_blocks):
        start_frame = i * frames_per_block
        end_frame = min((i + 1) * frames_per_block, n_frames)
        block_frames = feature_array[start_frame:end_frame]

        if len(block_frames) == 0:
            block_features[i] = 0.0
            continue

        if agg_stat == 'mean':
            block_features[i] = np.mean(block_frames)
        elif agg_stat == 'median':
            block_features[i] = np.median(block_frames)
        elif agg_stat == 'std':
            block_features[i] = np.std(block_frames)
        elif agg_stat == 'p25':
            block_features[i] = np.percentile(block_frames, 25)
        elif agg_stat == 'p75':
            block_features[i] = np.percentile(block_frames, 75)
        elif agg_stat == 'min':
            block_features[i] = np.min(block_frames)
        elif agg_stat == 'max':
            block_features[i] = np.max(block_frames)
        else:
            raise ValueError(f"Unknown aggregation statistic: {agg_stat}")

    return block_features, block_duration_sec


def compute_canonical_block_count(
    duration_sec: float,
    block_duration_sec: float = DEFAULT_BLOCK_DURATION_SEC,
    start_time_sec: float = 0.0
) -> int:
    """
    Compute canonical block count where final block center <= duration_sec.

    CONTRACT:
    - Input: duration_sec (positive float), block_duration_sec (positive float)
    - Output: n_blocks (non-negative int)
    - Guarantee: block_center(n_blocks-1) <= duration_sec
    - Returns 0 if duration is too short to fit even one block center
    - Same semantics as src/timebase.compute_canonical_block_count

    FORMULA:
    Block center: t[i] = start + (i + 0.5) * block_dur
    For last block (i = n-1): t[n-1] = start + (n - 0.5) * block_dur
    We need: start + (n - 0.5) * block_dur <= duration
    Therefore: n = floor((duration - start) / block_dur + 0.5)

    For n=1: block center = start + 0.5 * block_dur
    So we need at least: duration >= start + 0.5 * block_dur

    C++ PORT NOTES:
    - Simple arithmetic, no dependencies
    - Matches src/timebase.py for Phase 1/2 parity

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
    if effective_duration < 0.5 * block_duration_sec:
        return 0

    n_blocks = int((effective_duration / block_duration_sec) + 0.5)
    return max(1, n_blocks)


def compute_canonical_time_axis(
    n_blocks: int,
    block_duration_sec: float = DEFAULT_BLOCK_DURATION_SEC,
    start_time_sec: float = 0.0
) -> np.ndarray:
    """
    Compute canonical time axis array.

    CONTRACT:
    - Output: monotonically increasing array of block center times
    - times[i] = start_time_sec + (i + 0.5) * block_duration_sec
    - Same semantics as src/timebase.compute_canonical_time_axis

    C++ PORT NOTES:
    - Simple arithmetic array generation
    - Matches src/timebase.py for Phase 1/2 parity

    Parameters:
        n_blocks: Number of blocks
        block_duration_sec: Block duration in seconds
        start_time_sec: Start time offset

    Returns:
        Array of block center times (n_blocks,), dtype float32
    """
    if n_blocks <= 0:
        return np.array([], dtype=np.float32)

    times = start_time_sec + (np.arange(n_blocks) + 0.5) * block_duration_sec
    return times.astype(np.float32)


# =============================================================================
# NORMALIZATION
# =============================================================================

def normalize_block_features(
    block_features: np.ndarray,
    method: str = 'robust'
) -> Tuple[np.ndarray, Dict]:
    """
    Normalize block features to common scale.

    CONTRACT:
    - Input: block_features (n_blocks, n_features) or (n_blocks,) float array
    - Input: method one of: 'robust', 'percentile', 'zscore'
    - Output: (normalized_features, normalization_params)
    - normalized_features: same shape as input, float32
    - 'robust': centered around 0, scaled by IQR
    - 'percentile': values in [0, 1]
    - 'zscore': centered around 0, std=1
    - Deterministic: same input -> same output

    C++ PORT NOTES:
    - Per-column normalization
    - Store params for denormalization if needed

    Parameters:
        block_features: (n_blocks, n_features) or (n_blocks,) array
        method: 'robust', 'percentile', or 'zscore'

    Returns:
        Tuple of (normalized_features, normalization_params)
    """
    if block_features.ndim == 1:
        block_features = block_features[:, np.newaxis]
        squeeze = True
    else:
        squeeze = False

    n_blocks, n_features = block_features.shape
    normalized = np.zeros_like(block_features, dtype=np.float32)
    params = {}

    for i in range(n_features):
        feature_col = block_features[:, i]

        if method == 'robust':
            median = np.median(feature_col)
            q75 = np.percentile(feature_col, 75)
            q25 = np.percentile(feature_col, 25)
            iqr = q75 - q25

            if iqr > 1e-8:
                normalized[:, i] = (feature_col - median) / iqr
            else:
                normalized[:, i] = 0.0

            params[i] = {'method': 'robust', 'median': float(median), 'iqr': float(iqr)}

        elif method == 'percentile':
            p_low = np.percentile(feature_col, DEFAULT_PERCENTILE_LOWER)
            p_high = np.percentile(feature_col, DEFAULT_PERCENTILE_UPPER)

            if p_high - p_low > 1e-8:
                normalized[:, i] = (feature_col - p_low) / (p_high - p_low)
                normalized[:, i] = np.clip(normalized[:, i], 0.0, 1.0)
            else:
                normalized[:, i] = 0.0

            params[i] = {'method': 'percentile', 'p_low': float(p_low), 'p_high': float(p_high)}

        elif method == 'zscore':
            mean = np.mean(feature_col)
            std = np.std(feature_col)

            if std > 1e-8:
                normalized[:, i] = (feature_col - mean) / std
            else:
                normalized[:, i] = 0.0

            params[i] = {'method': 'zscore', 'mean': float(mean), 'std': float(std)}

        else:
            raise ValueError(f"Unknown normalization method: {method}")

    if squeeze:
        normalized = normalized.squeeze()

    return normalized, params


def smooth_curve(
    curve: np.ndarray,
    method: str = 'ewma',
    alpha: float = DEFAULT_TENSION_SMOOTH_ALPHA,
    window: int = 3,
    window_length: int = 5,
    polyorder: int = 2
) -> np.ndarray:
    """
    Smooth a curve using specified method.

    CONTRACT:
    - Input: curve (1D float array)
    - Input: method one of: 'ewma', 'moving_average', 'savgol'
    - Output: (n_samples,) float32 array, same length as input
    - Deterministic: same input -> same output

    EWMA Equation:
        smoothed[0] = curve[0]
        smoothed[i] = alpha * curve[i] + (1 - alpha) * smoothed[i-1]

    C++ PORT NOTES:
    - EWMA: simple loop with state
    - Moving average: convolution with ones kernel
    - Savgol: requires polynomial fitting (optional for C++)

    Parameters:
        curve: 1D array to smooth
        method: 'ewma', 'moving_average', or 'savgol'
        alpha: EWMA smoothing factor (0 < alpha <= 1), lower = more smoothing
        window: Moving average window size (odd integer preferred)
        window_length: Savgol window length (must be odd)
        polyorder: Savgol polynomial order

    Returns:
        Smoothed curve (same length as input)
    """
    if len(curve) == 0:
        return curve.astype(np.float32)

    if method == 'ewma':
        smoothed = np.zeros(len(curve), dtype=np.float32)
        smoothed[0] = curve[0]

        for i in range(1, len(curve)):
            smoothed[i] = alpha * curve[i] + (1.0 - alpha) * smoothed[i - 1]

        return smoothed

    elif method == 'moving_average':
        if window <= 1:
            return curve.astype(np.float32)

        kernel = np.ones(window, dtype=np.float32) / window
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
        # Adjust window_length if needed
        if window_length > len(curve):
            window_length = len(curve) if len(curve) % 2 == 1 else len(curve) - 1

        if window_length < polyorder + 2:
            window_length = polyorder + 2
            if window_length % 2 == 0:
                window_length += 1

        if window_length > len(curve):
            return curve.astype(np.float32)

        smoothed = scipy_signal.savgol_filter(
            curve, window_length, polyorder, mode='nearest'
        )
        return smoothed.astype(np.float32)

    else:
        raise ValueError(f"Unknown smoothing method: {method}")


def normalize_curve_to_01(
    curve: np.ndarray,
    percentile_lower: float = DEFAULT_PERCENTILE_LOWER,
    percentile_upper: float = DEFAULT_PERCENTILE_UPPER
) -> Tuple[np.ndarray, Dict]:
    """
    Normalize curve to [0, 1] range using percentile scaling.

    CONTRACT:
    - Input: curve (1D float array)
    - Output: (normalized_curve, params)
    - normalized_curve: (n_samples,) float32, values in [0, 1]
    - Deterministic: same input -> same output

    Parameters:
        curve: 1D array
        percentile_lower: Lower percentile (default 1.0)
        percentile_upper: Upper percentile (default 99.0)

    Returns:
        Tuple of (normalized_curve, params)
    """
    p_low = np.percentile(curve, percentile_lower)
    p_high = np.percentile(curve, percentile_upper)

    if p_high - p_low > 1e-8:
        normalized = (curve - p_low) / (p_high - p_low)
        normalized = np.clip(normalized, 0.0, 1.0)
    else:
        normalized = np.zeros_like(curve)

    params = {'p_low': float(p_low), 'p_high': float(p_high)}

    return normalized.astype(np.float32), params


def normalize_tension_robust(
    curve: np.ndarray,
    percentile_lower: float = DEFAULT_TENSION_PERCENTILE_LOWER,
    percentile_upper: float = DEFAULT_TENSION_PERCENTILE_UPPER
) -> Tuple[np.ndarray, Dict]:
    """
    Robust normalization for tension curves using configurable percentiles.

    Uses wider percentiles (5th-95th by default) than normalize_curve_to_01
    to better preserve contrast in high-energy tracks.

    CONTRACT:
    - Input: curve (1D float array)
    - Output: (normalized_curve, normalization_info)
    - normalized_curve: (n_samples,) float32, values in [0, 1]
    - normalization_info includes pre/post stats for debugging
    - Deterministic: same input -> same output

    Parameters:
        curve: 1D array (raw tension values)
        percentile_lower: Lower percentile for scaling (default 5.0)
        percentile_upper: Upper percentile for scaling (default 95.0)

    Returns:
        Tuple of (normalized_curve, normalization_info)
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
    sr: int = DEFAULT_SAMPLE_RATE,
    rms_max_dbfs: float = DEFAULT_ANCHORED_RMS_MAX_DBFS,
    onset_max: float = DEFAULT_ANCHORED_ONSET_MAX,
    centroid_max_hz: float = DEFAULT_ANCHORED_CENTROID_MAX_HZ,
    bandwidth_max_hz: float = DEFAULT_ANCHORED_BANDWIDTH_MAX_HZ
) -> Tuple[np.ndarray, Dict[str, np.ndarray], Dict]:
    """
    Anchored normalization using physically meaningful reference values.

    Each component is scaled by absolute reference values rather than
    track-relative percentiles. This preserves cross-track comparability.

    CONTRACT:
    - Input: component arrays (1D float arrays of same length)
    - Input: weights dict with keys matching component names
    - Output: (tension_curve, normalized_components, normalization_info)
    - tension_curve: (n_blocks,) float32, values in [0, 1]
    - normalized_components: dict of (n_blocks,) float32 arrays
    - Deterministic: same input -> same output

    C++ PORT NOTES:
    - Reference values can be adjusted per use case
    - dB conversion: 20 * log10(rms + 1e-8)

    Parameters:
        rms_values: Raw RMS energy values (linear amplitude)
        onset_values: Raw onset strength values
        centroid_values: Raw spectral centroid values (Hz)
        bandwidth_values: Raw spectral bandwidth values (Hz)
        weights: Component weights for combining
        sr: Sample rate (for Nyquist calculation)
        rms_max_dbfs: Reference maximum RMS in dBFS
        onset_max: Reference maximum onset strength
        centroid_max_hz: Reference maximum centroid in Hz
        bandwidth_max_hz: Reference maximum bandwidth in Hz

    Returns:
        Tuple of (tension_curve, normalized_components, normalization_info)
    """
    # Normalize RMS using dBFS reference
    rms_db = 20.0 * np.log10(rms_values + 1e-8)
    rms_norm = (rms_db - (-60.0)) / (rms_max_dbfs - (-60.0))
    rms_norm = np.clip(rms_norm, 0.0, 1.0)

    # Normalize onset strength using reference maximum
    onset_norm = onset_values / onset_max
    onset_norm = np.clip(onset_norm, 0.0, 1.0)

    # Normalize spectral centroid using reference maximum
    nyquist = sr / 2.0
    centroid_ref = min(centroid_max_hz, nyquist)
    centroid_norm = centroid_values / centroid_ref
    centroid_norm = np.clip(centroid_norm, 0.0, 1.0)

    # Normalize spectral bandwidth using reference maximum
    bandwidth_ref = min(bandwidth_max_hz, nyquist)
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

    # Final clip
    tension = np.clip(tension, 0.0, 1.0)

    normalization_info = {
        'mode': 'anchored',
        'references': {
            'rms_max_dbfs': rms_max_dbfs,
            'onset_max': onset_max,
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

    CONTRACT:
    - Input: features (n_frames, n_features) or (n_frames,) float array
    - Input: order (1 = first derivative, 2 = second derivative)
    - Output: same shape as input, float32
    - First frame padded with second frame value
    - Deterministic: same input -> same output

    C++ PORT NOTES:
    - Simple finite difference: delta[i] = features[i] - features[i-1]
    - For order > 1, apply recursively

    Parameters:
        features: (n_frames, n_features) or (n_frames,) array
        order: Delta order (1 = first derivative)

    Returns:
        Delta features with same shape as input
    """
    if features.ndim == 1:
        features = features[:, np.newaxis]
        squeeze = True
    else:
        squeeze = False

    delta = np.zeros_like(features, dtype=np.float32)

    for i in range(order):
        if i == 0:
            delta[1:] = np.diff(features, axis=0)
            delta[0] = delta[1]  # Pad first frame
        else:
            prev_delta = delta.copy()
            delta[1:] = np.diff(prev_delta, axis=0)
            delta[0] = delta[1]

    if squeeze:
        delta = delta.squeeze()

    return delta


# =============================================================================
# CURVE COMPUTATION
# =============================================================================

def compute_tension_curve(
    rms_blocks: np.ndarray,
    onset_blocks: np.ndarray,
    centroid_blocks: np.ndarray,
    bandwidth_blocks: np.ndarray,
    weights: Optional[Dict[str, float]] = None,
    normalization_mode: str = 'robust_track',
    sr: int = DEFAULT_SAMPLE_RATE,
    smooth_alpha: float = DEFAULT_TENSION_SMOOTH_ALPHA,
    percentile_lower: float = DEFAULT_TENSION_PERCENTILE_LOWER,
    percentile_upper: float = DEFAULT_TENSION_PERCENTILE_UPPER
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray], Dict]:
    """
    Compute tension/energy curve from block features.

    Tension is a weighted combination of:
    - RMS energy (loudness proxy)
    - Onset density (rhythmic drive/impact)
    - Spectral centroid (brightness/aggression)
    - Spectral bandwidth (fullness/richness)

    CONTRACT:
    - Input: component block arrays (1D float arrays of same length)
    - Input: normalization_mode 'robust_track' or 'anchored'
    - Output: (tension_raw, tension_smooth, components_norm, normalization_info)
    - tension_raw: (n_blocks,) float32, values in [0, 1]
    - tension_smooth: (n_blocks,) float32, values in [0, 1], EWMA smoothed
    - Deterministic: same input -> same output

    C++ PORT NOTES:
    - Main curve computation function
    - Two normalization modes available

    Parameters:
        rms_blocks: Block-level RMS values
        onset_blocks: Block-level onset strength values
        centroid_blocks: Block-level spectral centroid values
        bandwidth_blocks: Block-level spectral bandwidth values
        weights: Component weights (None = use defaults)
        normalization_mode: 'robust_track' or 'anchored'
        sr: Sample rate (used for anchored mode)
        smooth_alpha: EWMA smoothing factor
        percentile_lower: Lower percentile for robust normalization
        percentile_upper: Upper percentile for robust normalization

    Returns:
        Tuple of (tension_raw, tension_smooth, component_contributions, normalization_info)
    """
    if weights is None:
        weights = DEFAULT_TENSION_WEIGHTS.copy()

    n_blocks = len(rms_blocks)

    if normalization_mode == 'anchored':
        tension_raw, components_norm, normalization_info = normalize_tension_anchored(
            rms_values=rms_blocks,
            onset_values=onset_blocks,
            centroid_values=centroid_blocks,
            bandwidth_values=bandwidth_blocks,
            weights=weights,
            sr=sr
        )
    else:
        # Robust track mode: normalize each component, then combine
        components_norm = {}

        # Normalize each component individually
        for comp_name, comp_values in [
            ('rms', rms_blocks),
            ('onset_density', onset_blocks),
            ('spectral_centroid', centroid_blocks),
            ('spectral_bandwidth', bandwidth_blocks)
        ]:
            p_low = np.percentile(comp_values, percentile_lower)
            p_high = np.percentile(comp_values, percentile_upper)
            if p_high - p_low > 1e-8:
                comp_norm = (comp_values - p_low) / (p_high - p_low)
                comp_norm = np.clip(comp_norm, 0.0, 1.0)
            else:
                comp_norm = np.full_like(comp_values, 0.5)
            components_norm[comp_name] = comp_norm.astype(np.float32)

        # Weighted combination
        tension_combined = np.zeros(n_blocks, dtype=np.float32)
        for comp_name, weight in weights.items():
            if comp_name in components_norm:
                tension_combined += weight * components_norm[comp_name]

        # Final robust normalization
        tension_raw, normalization_info = normalize_tension_robust(
            tension_combined,
            percentile_lower=percentile_lower,
            percentile_upper=percentile_upper
        )

        # Add component raw stats
        normalization_info['component_stats'] = {
            'rms': {'raw_mean': float(np.mean(rms_blocks))},
            'onset_density': {'raw_mean': float(np.mean(onset_blocks))},
            'spectral_centroid': {'raw_mean': float(np.mean(centroid_blocks))},
            'spectral_bandwidth': {'raw_mean': float(np.mean(bandwidth_blocks))}
        }

    # Smooth with EWMA
    tension_smooth = smooth_curve(tension_raw, method='ewma', alpha=smooth_alpha)

    return tension_raw, tension_smooth, components_norm, normalization_info


def compute_novelty_curve(
    block_features: np.ndarray,
    lookback_blocks: int = DEFAULT_NOVELTY_LOOKBACK_BLOCKS,
    smooth_window: int = DEFAULT_NOVELTY_SMOOTH_WINDOW
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute novelty curve: degree of change relative to recent context.

    For each block, computes cosine distance to mean of recent past blocks.

    CONTRACT:
    - Input: block_features (n_blocks, n_features) float array
    - Output: (novelty_curve, distances)
    - novelty_curve: (n_blocks,) float32, values in [0, 1], smoothed
    - distances: (n_blocks,) float32, raw distances before smoothing
    - First block distance = 0 (no history)
    - Deterministic: same input -> same output

    ALGORITHM:
    1. Z-score normalize features (per-column)
    2. For each block: compute cosine distance to mean of lookback window
    3. Normalize distances to [0, 1]
    4. Smooth with moving average

    C++ PORT NOTES:
    - Uses cosine distance: 1 - (a·b) / (|a| * |b|)
    - Z-scoring requires computing mean/std per column

    Parameters:
        block_features: (n_blocks, n_features) normalized feature matrix
        lookback_blocks: Number of past blocks to compare against
        smooth_window: Moving average window for smoothing

    Returns:
        Tuple of (novelty_curve, distances)
    """
    n_blocks = block_features.shape[0]
    distances = np.zeros(n_blocks, dtype=np.float32)

    # Z-score the features
    mean = np.mean(block_features, axis=0)
    std = np.std(block_features, axis=0)
    std[std < 1e-8] = 1.0
    features_zscore = (block_features - mean) / std

    for i in range(n_blocks):
        if i == 0:
            distances[i] = 0.0
            continue

        # Define lookback window
        start_idx = max(0, i - lookback_blocks)
        context_features = features_zscore[start_idx:i, :]
        current_features = features_zscore[i, :]

        # Compute mean of context
        context_mean = np.mean(context_features, axis=0)

        # Cosine distance
        norm_current = np.linalg.norm(current_features)
        norm_context = np.linalg.norm(context_mean)

        if norm_current < 1e-8 or norm_context < 1e-8:
            distances[i] = 0.0
        else:
            distances[i] = cosine_distance(current_features, context_mean)

    # Normalize to [0, 1]
    novelty_raw, _ = normalize_curve_to_01(distances)

    # Smooth
    novelty_smooth = smooth_curve(novelty_raw, method='moving_average', window=smooth_window)

    return novelty_smooth, distances


def compute_self_similarity_matrix(
    block_features: np.ndarray,
    window_size: int = DEFAULT_FATIGUE_WINDOW_BLOCKS
) -> np.ndarray:
    """
    Compute rolling self-similarity scores.

    For each block, computes max cosine similarity to blocks within lookback window.
    Higher values = more similar to recent past = more repetitive.

    CONTRACT:
    - Input: block_features (n_blocks, n_features) float array
    - Output: (n_blocks,) float32, values in [0, 1]
    - First 2 blocks = 0 (not enough history)
    - Deterministic: same input -> same output

    C++ PORT NOTES:
    - Uses max similarity (not mean) for repetition detection
    - Cosine similarity = 1 - cosine_distance

    Parameters:
        block_features: (n_blocks, n_features) normalized feature matrix
        window_size: Size of rolling window

    Returns:
        Array of self-similarity scores (n_blocks,)
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
            similarity[i] = 0.0
            continue

        # Define window
        start_idx = max(0, i - window_size)
        window_features = features_zscore[start_idx:i, :]
        current_features = features_zscore[i, :]

        # Compute similarities to all blocks in window
        max_sim = 0.0
        for j in range(window_features.shape[0]):
            past_features = window_features[j, :]

            norm_current = np.linalg.norm(current_features)
            norm_past = np.linalg.norm(past_features)

            if norm_current < 1e-8 or norm_past < 1e-8:
                sim = 0.0
            else:
                sim = 1.0 - cosine_distance(current_features, past_features)

            if sim > max_sim:
                max_sim = sim

        similarity[i] = max_sim

    # Clip to [0, 1]
    similarity = np.clip(similarity, 0.0, 1.0)

    return similarity


def compute_feature_variance(
    block_features: np.ndarray,
    window_size: int = DEFAULT_FATIGUE_WINDOW_BLOCKS
) -> np.ndarray:
    """
    Compute rolling feature variance as indicator of change/stagnation.

    CONTRACT:
    - Input: block_features (n_blocks, n_features) float array
    - Output: (n_blocks,) float32, values in [0, 1]
    - Lower values = less change = more stagnant
    - Deterministic: same input -> same output

    C++ PORT NOTES:
    - Compute variance for each feature in window, then average
    - Normalize to [0, 1] at the end

    Parameters:
        block_features: (n_blocks, n_features) normalized feature matrix
        window_size: Size of rolling window

    Returns:
        Array of variance scores (n_blocks,)
    """
    n_blocks = block_features.shape[0]
    variance_scores = np.zeros(n_blocks, dtype=np.float32)

    for i in range(n_blocks):
        if i < window_size:
            window = block_features[:i + 1, :]
        else:
            window = block_features[i - window_size + 1:i + 1, :]

        if window.shape[0] < 2:
            variance_scores[i] = 0.0
            continue

        # Compute variance across time for each feature
        feature_variances = np.var(window, axis=0)

        # Average variance across features
        variance_scores[i] = np.mean(feature_variances)

    # Normalize to [0, 1]
    variance_norm, _ = normalize_curve_to_01(variance_scores)

    return variance_norm


def fatigue_leaky_integrator_step(
    state: FatigueState,
    self_similarity: float,
    novelty: float,
    inverse_variance: float,
    is_boundary: bool,
    gain_up: float = DEFAULT_FATIGUE_GAIN_UP,
    gain_down: float = DEFAULT_FATIGUE_GAIN_DOWN,
    novelty_spike_threshold: float = DEFAULT_FATIGUE_NOVELTY_SPIKE_THRESHOLD,
    boring_weights: Optional[Dict[str, float]] = None
) -> float:
    """
    Single-step fatigue update for real-time use.

    CONTRACT:
    - Input: state (FatigueState with current fatigue_value)
    - Input: current block values (all in [0, 1])
    - Output: updated fatigue value in [0, 1]
    - Modifies state.fatigue_value in place
    - Deterministic: same inputs -> same output

    MODEL:
    - When boring: delta = +gain_up * boring_score (slow accumulation)
    - When interesting: delta = -gain_down * (1 + novelty) (fast recovery)
    - "Interesting" = novelty spike OR section boundary

    C++ PORT NOTES:
    - Single-step function for real-time block-by-block processing
    - State persists between calls

    Parameters:
        state: FatigueState object to update
        self_similarity: Current block self-similarity [0, 1]
        novelty: Current block novelty [0, 1]
        inverse_variance: Current block inverse variance [0, 1]
        is_boundary: Whether current block is a section boundary
        gain_up: Rate of fatigue increase
        gain_down: Rate of fatigue decrease
        novelty_spike_threshold: Threshold for "interesting" classification
        boring_weights: Weights for boring score components

    Returns:
        Updated fatigue value [0, 1]
    """
    if boring_weights is None:
        boring_weights = DEFAULT_FATIGUE_BORING_WEIGHTS.copy()

    # Compute "boring" score
    inverse_novelty = 1.0 - novelty
    boring = (
        boring_weights.get('self_similarity', 0.5) * self_similarity +
        boring_weights.get('inverse_novelty', 0.3) * inverse_novelty +
        boring_weights.get('inverse_variance', 0.2) * inverse_variance
    )

    # Check if current block is "interesting"
    is_novelty_spike = novelty > novelty_spike_threshold

    if is_novelty_spike or is_boundary:
        # Interesting: decay fatigue quickly
        delta = -gain_down * (1.0 + novelty)
    else:
        # Boring: accumulate fatigue slowly
        delta = gain_up * boring

    # Update state
    state.fatigue_value = float(np.clip(state.fatigue_value + delta, 0.0, 1.0))

    return state.fatigue_value


def compute_fatigue_leaky_integrator(
    self_similarity: np.ndarray,
    novelty_curve: np.ndarray,
    inverse_variance: np.ndarray,
    boundary_blocks: Optional[List[int]] = None,
    gain_up: float = DEFAULT_FATIGUE_GAIN_UP,
    gain_down: float = DEFAULT_FATIGUE_GAIN_DOWN,
    novelty_spike_threshold: float = DEFAULT_FATIGUE_NOVELTY_SPIKE_THRESHOLD,
    boring_weights: Optional[Dict[str, float]] = None
) -> np.ndarray:
    """
    Compute fatigue using leaky integrator model (batch version).

    CONTRACT:
    - Input: component arrays (1D float arrays of same length)
    - Output: (n_blocks,) float32, values in [0, 1]
    - fatigue[0] = 0.0 (starts fresh)
    - Deterministic: same input -> same output

    ALGORITHM:
    For each block t > 0:
        boring = w1*similarity + w2*(1-novelty) + w3*(1-variance)
        if novelty > threshold or is_boundary:
            delta = -gain_down * (1 + novelty)  # Fast recovery
        else:
            delta = +gain_up * boring  # Slow accumulation
        fatigue[t] = clamp(fatigue[t-1] + delta, 0, 1)

    C++ PORT NOTES:
    - For real-time use, call fatigue_leaky_integrator_step() instead
    - This batch version processes entire track at once

    Parameters:
        self_similarity: Self-similarity curve [0, 1]
        novelty_curve: Novelty curve [0, 1]
        inverse_variance: Inverse variance curve [0, 1]
        boundary_blocks: List of block indices that are section boundaries
        gain_up: Rate of fatigue increase (default 0.02)
        gain_down: Rate of fatigue decrease (default 0.08)
        novelty_spike_threshold: Novelty threshold for "interesting"
        boring_weights: Weights for boring signal components

    Returns:
        Fatigue curve [0, 1]
    """
    if boring_weights is None:
        boring_weights = DEFAULT_FATIGUE_BORING_WEIGHTS.copy()

    n_blocks = len(self_similarity)
    fatigue = np.zeros(n_blocks, dtype=np.float32)

    # Invert novelty for boring score
    inverse_novelty = 1.0 - novelty_curve

    # Convert boundary list to set for fast lookup
    boundary_set = set(boundary_blocks) if boundary_blocks else set()

    for t in range(1, n_blocks):
        # Compute "boring" score
        boring = (
            boring_weights.get('self_similarity', 0.5) * self_similarity[t] +
            boring_weights.get('inverse_novelty', 0.3) * inverse_novelty[t] +
            boring_weights.get('inverse_variance', 0.2) * inverse_variance[t]
        )

        # Check if current block is "interesting"
        is_novelty_spike = novelty_curve[t] > novelty_spike_threshold
        is_boundary = t in boundary_set

        if is_novelty_spike or is_boundary:
            delta = -gain_down * (1.0 + novelty_curve[t])
        else:
            delta = gain_up * boring

        fatigue[t] = np.clip(fatigue[t - 1] + delta, 0.0, 1.0)

    return fatigue


def compute_fatigue_curve(
    block_features: np.ndarray,
    novelty_curve: np.ndarray,
    boundary_blocks: Optional[List[int]] = None,
    weights: Optional[Dict[str, float]] = None,
    window_size: int = DEFAULT_FATIGUE_WINDOW_BLOCKS,
    smooth_window: int = DEFAULT_FATIGUE_SMOOTH_WINDOW,
    use_leaky_integrator: bool = True,
    gain_up: float = DEFAULT_FATIGUE_GAIN_UP,
    gain_down: float = DEFAULT_FATIGUE_GAIN_DOWN,
    novelty_spike_threshold: float = DEFAULT_FATIGUE_NOVELTY_SPIKE_THRESHOLD
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Compute repetition/fatigue curve.

    Fatigue combines:
    - High self-similarity (repetitive patterns)
    - Low novelty (lack of change)
    - Low variance (sustained predictability)

    CONTRACT:
    - Input: block_features (n_blocks, n_features) float array
    - Input: novelty_curve (n_blocks,) float array in [0, 1]
    - Output: (fatigue_smooth, intermediate_signals)
    - fatigue_smooth: (n_blocks,) float32, values in [0, 1]
    - intermediate_signals includes component curves
    - Deterministic: same input -> same output

    C++ PORT NOTES:
    - Two modes: leaky_integrator (recommended) or weighted_average
    - Leaky integrator has better perceptual modeling

    Parameters:
        block_features: (n_blocks, n_features) normalized feature matrix
        novelty_curve: Pre-computed novelty curve
        boundary_blocks: List of boundary block indices (for leaky integrator)
        weights: Component weights (None = use defaults)
        window_size: Rolling window size
        smooth_window: Smoothing window
        use_leaky_integrator: Use leaky integrator model (default True)
        gain_up: Leaky integrator accumulation rate
        gain_down: Leaky integrator recovery rate
        novelty_spike_threshold: Threshold for novelty spike detection

    Returns:
        Tuple of (fatigue_curve, intermediate_signals)
    """
    if weights is None:
        weights = DEFAULT_FATIGUE_BORING_WEIGHTS.copy()

    # Compute components
    self_similarity = compute_self_similarity_matrix(block_features, window_size)
    variance = compute_feature_variance(block_features, window_size)

    # Invert novelty and variance
    inverse_novelty = 1.0 - novelty_curve
    inverse_variance = 1.0 - variance

    if use_leaky_integrator:
        fatigue_raw = compute_fatigue_leaky_integrator(
            self_similarity,
            novelty_curve,
            inverse_variance,
            boundary_blocks,
            gain_up=gain_up,
            gain_down=gain_down,
            novelty_spike_threshold=novelty_spike_threshold,
            boring_weights=weights
        )
        computation_mode = 'leaky_integrator'
    else:
        # Original weighted combination
        fatigue_raw = (
            weights.get('self_similarity', 0.5) * self_similarity +
            weights.get('inverse_novelty', 0.3) * inverse_novelty +
            weights.get('inverse_variance', 0.2) * inverse_variance
        )
        fatigue_raw, _ = normalize_curve_to_01(fatigue_raw)
        computation_mode = 'weighted_average'

    # Smooth
    fatigue_smooth = smooth_curve(fatigue_raw, method='moving_average', window=smooth_window)

    intermediate_signals = {
        'self_similarity': self_similarity,
        'inverse_novelty': inverse_novelty,
        'inverse_variance': inverse_variance,
        'computation_mode': computation_mode
    }

    return fatigue_smooth, intermediate_signals

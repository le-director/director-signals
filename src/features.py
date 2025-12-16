"""
Feature Extraction Module

Extract frame-level spectral and temporal features from audio.
All features are vectorized and aligned to same frame grid.
"""

import numpy as np
from typing import Dict, Tuple, Optional
import warnings

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

from scipy import signal as scipy_signal

import config


def compute_rms_energy(
    audio: np.ndarray,
    frame_length: int = config.FRAME_LENGTH,
    hop_length: int = config.HOP_LENGTH
) -> np.ndarray:
    """
    Compute RMS energy per frame.

    Parameters:
        audio: Audio array
        frame_length: Frame size in samples
        hop_length: Hop size in samples

    Returns:
        Array of RMS values (length n_frames)
    """
    if LIBROSA_AVAILABLE:
        rms = librosa.feature.rms(
            y=audio,
            frame_length=frame_length,
            hop_length=hop_length
        )[0]
    else:
        # Manual RMS calculation
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
    frame_length: int = config.FRAME_LENGTH,
    hop_length: int = config.HOP_LENGTH
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Short-Time Fourier Transform.

    Parameters:
        audio: Audio array
        frame_length: Frame size in samples
        hop_length: Hop size in samples

    Returns:
        Tuple of (magnitude_spectrogram, phase_spectrogram)
        magnitude: (n_bins, n_frames) array
        phase: (n_bins, n_frames) array
    """
    if LIBROSA_AVAILABLE:
        stft_complex = librosa.stft(
            audio,
            n_fft=frame_length,
            hop_length=hop_length
        )
    else:
        # Use scipy for STFT
        f, t, stft_complex = scipy_signal.stft(
            audio,
            nperseg=frame_length,
            noverlap=frame_length - hop_length,
            nfft=frame_length
        )

    magnitude = np.abs(stft_complex)
    phase = np.angle(stft_complex)

    return magnitude, phase


def compute_spectral_features(
    audio: np.ndarray,
    sr: int,
    frame_length: int = config.FRAME_LENGTH,
    hop_length: int = config.HOP_LENGTH
) -> Dict[str, np.ndarray]:
    """
    Compute spectral features: centroid, bandwidth, rolloff.

    Parameters:
        audio: Audio array
        sr: Sample rate (Hz)
        frame_length: Frame size in samples
        hop_length: Hop size in samples

    Returns:
        Dictionary with keys:
            - 'spectral_centroid': array of length n_frames
            - 'spectral_bandwidth': array of length n_frames
            - 'spectral_rolloff': array of length n_frames
    """
    if LIBROSA_AVAILABLE:
        # Use librosa for convenience
        centroid = librosa.feature.spectral_centroid(
            y=audio,
            sr=sr,
            n_fft=frame_length,
            hop_length=hop_length
        )[0]

        bandwidth = librosa.feature.spectral_bandwidth(
            y=audio,
            sr=sr,
            n_fft=frame_length,
            hop_length=hop_length
        )[0]

        rolloff = librosa.feature.spectral_rolloff(
            y=audio,
            sr=sr,
            n_fft=frame_length,
            hop_length=hop_length,
            roll_percent=config.SPECTRAL_ROLLOFF_PERCENT
        )[0]

    else:
        # Manual computation using STFT
        magnitude, _ = compute_stft(audio, frame_length, hop_length)

        # Frequency bins
        freqs = np.fft.rfftfreq(frame_length, 1.0 / sr)

        # Spectral centroid: weighted mean frequency
        centroid = np.sum(freqs[:, np.newaxis] * magnitude, axis=0) / (
            np.sum(magnitude, axis=0) + 1e-8
        )

        # Spectral bandwidth: weighted std of frequencies around centroid
        freq_diff = (freqs[:, np.newaxis] - centroid[np.newaxis, :]) ** 2
        bandwidth = np.sqrt(
            np.sum(freq_diff * magnitude, axis=0) / (np.sum(magnitude, axis=0) + 1e-8)
        )

        # Spectral rolloff: frequency below which X% of energy is contained
        cumsum_mag = np.cumsum(magnitude, axis=0)
        total_energy = cumsum_mag[-1, :]
        threshold = config.SPECTRAL_ROLLOFF_PERCENT * total_energy

        rolloff = np.zeros(magnitude.shape[1])
        for i in range(magnitude.shape[1]):
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
    frame_length: int = config.FRAME_LENGTH,
    hop_length: int = config.HOP_LENGTH
) -> np.ndarray:
    """
    Compute spectral flux: frame-to-frame change in magnitude spectrum.

    Parameters:
        audio: Audio array
        frame_length: Frame size in samples
        hop_length: Hop size in samples

    Returns:
        Array of spectral flux values (length n_frames)
        First frame has flux = 0
    """
    magnitude, _ = compute_stft(audio, frame_length, hop_length)

    # Compute frame-to-frame difference
    flux = np.zeros(magnitude.shape[1], dtype=np.float32)
    for i in range(1, magnitude.shape[1]):
        diff = magnitude[:, i] - magnitude[:, i-1]
        # Only consider increases (half-wave rectification)
        diff = np.maximum(diff, 0)
        flux[i] = np.sum(diff)

    return flux


def compute_zcr(
    audio: np.ndarray,
    frame_length: int = config.FRAME_LENGTH,
    hop_length: int = config.HOP_LENGTH
) -> np.ndarray:
    """
    Compute zero crossing rate per frame.

    Parameters:
        audio: Audio array
        frame_length: Frame size in samples
        hop_length: Hop size in samples

    Returns:
        Array of ZCR values (length n_frames)
    """
    if LIBROSA_AVAILABLE:
        zcr = librosa.feature.zero_crossing_rate(
            audio,
            frame_length=frame_length,
            hop_length=hop_length
        )[0]
    else:
        # Manual ZCR calculation
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
    sr: int,
    frame_length: int = config.FRAME_LENGTH,
    hop_length: int = config.HOP_LENGTH
) -> np.ndarray:
    """
    Compute onset strength envelope (transient density proxy).

    Parameters:
        audio: Audio array
        sr: Sample rate (Hz)
        frame_length: Frame size in samples
        hop_length: Hop size in samples

    Returns:
        Array of onset strength values (length n_frames)
    """
    if LIBROSA_AVAILABLE:
        onset_env = librosa.onset.onset_strength(
            y=audio,
            sr=sr,
            n_fft=frame_length,
            hop_length=hop_length
        )
    else:
        # Use spectral flux as onset strength proxy
        onset_env = compute_spectral_flux(audio, frame_length, hop_length)

    return onset_env


def compute_mfcc_stats(
    audio: np.ndarray,
    sr: int,
    frame_length: int = config.FRAME_LENGTH,
    hop_length: int = config.HOP_LENGTH,
    n_mfcc: int = config.N_MFCC
) -> np.ndarray:
    """
    Compute MFCC coefficients per frame.

    Parameters:
        audio: Audio array
        sr: Sample rate (Hz)
        frame_length: Frame size in samples
        hop_length: Hop size in samples
        n_mfcc: Number of MFCC coefficients

    Returns:
        Array of shape (n_mfcc, n_frames)
    """
    if LIBROSA_AVAILABLE:
        mfccs = librosa.feature.mfcc(
            y=audio,
            sr=sr,
            n_mfcc=n_mfcc,
            n_fft=frame_length,
            hop_length=hop_length
        )
    else:
        # Simplified MFCC computation using mel spectrogram
        warnings.warn(
            "librosa not available, MFCC computation may be less accurate"
        )

        # Compute magnitude spectrogram
        magnitude, _ = compute_stft(audio, frame_length, hop_length)

        # Create mel filterbank (simplified)
        n_mels = 40
        mel_filters = _create_mel_filterbank(
            frame_length // 2 + 1, n_mels, sr
        )

        # Apply mel filters
        mel_spec = mel_filters @ magnitude

        # Log compression
        log_mel = np.log(mel_spec + 1e-8)

        # DCT to get MFCCs
        mfccs = scipy_signal.dct(log_mel, axis=0, type=2, norm='ortho')[:n_mfcc]

    return mfccs.astype(np.float32)


def _create_mel_filterbank(
    n_bins: int,
    n_mels: int,
    sr: int,
    fmin: float = 0.0,
    fmax: Optional[float] = None
) -> np.ndarray:
    """
    Create mel filterbank matrix (simplified version).

    Parameters:
        n_bins: Number of FFT bins
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
    def hz_to_mel(hz):
        return 2595 * np.log10(1 + hz / 700.0)

    def mel_to_hz(mel):
        return 700 * (10 ** (mel / 2595.0) - 1)

    # Create mel-spaced frequencies
    mel_min = hz_to_mel(fmin)
    mel_max = hz_to_mel(fmax)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = mel_to_hz(mel_points)

    # Convert to FFT bin indices
    bin_points = np.floor((n_bins - 1) * hz_points / fmax).astype(int)

    # Create filterbank
    filterbank = np.zeros((n_mels, n_bins))
    for i in range(n_mels):
        left = bin_points[i]
        center = bin_points[i + 1]
        right = bin_points[i + 2]

        # Rising slope
        if center > left:
            filterbank[i, left:center] = np.linspace(0, 1, center - left)

        # Falling slope
        if right > center:
            filterbank[i, center:right] = np.linspace(1, 0, right - center)

    return filterbank


def extract_all_features(
    audio: np.ndarray,
    sr: int,
    frame_length: int = config.FRAME_LENGTH,
    hop_length: int = config.HOP_LENGTH
) -> Dict:
    """
    Extract all frame-level features from audio.

    This is the main entry point for feature extraction.
    All features are aligned to the same frame grid.

    Parameters:
        audio: Audio array
        sr: Sample rate (Hz)
        frame_length: Frame size in samples
        hop_length: Hop size in samples

    Returns:
        Dictionary containing:
            - 'rms': RMS energy array (n_frames,)
            - 'spectral_centroid': array (n_frames,)
            - 'spectral_bandwidth': array (n_frames,)
            - 'spectral_rolloff': array (n_frames,)
            - 'spectral_flux': array (n_frames,)
            - 'zcr': zero crossing rate array (n_frames,)
            - 'onset_strength': array (n_frames,)
            - 'mfcc': array (n_mfcc, n_frames)
            - 'metadata': dict with frame_length, hop_length, n_frames, sr
    """
    # Compute all features
    rms = compute_rms_energy(audio, frame_length, hop_length)
    spectral = compute_spectral_features(audio, sr, frame_length, hop_length)
    spectral_flux = compute_spectral_flux(audio, frame_length, hop_length)
    zcr = compute_zcr(audio, frame_length, hop_length)
    onset_strength = compute_onset_strength(audio, sr, frame_length, hop_length)
    mfcc = compute_mfcc_stats(audio, sr, frame_length, hop_length)

    # Verify all features have same length
    n_frames = len(rms)
    features_to_check = [
        spectral['spectral_centroid'],
        spectral['spectral_bandwidth'],
        spectral['spectral_rolloff'],
        spectral_flux,
        zcr,
        onset_strength
    ]

    for feat in features_to_check:
        if len(feat) != n_frames:
            raise ValueError(
                f"Feature length mismatch: expected {n_frames}, got {len(feat)}"
            )

    if mfcc.shape[1] != n_frames:
        raise ValueError(
            f"MFCC frame count mismatch: expected {n_frames}, got {mfcc.shape[1]}"
        )

    # Package everything
    return {
        'rms': rms,
        'spectral_centroid': spectral['spectral_centroid'],
        'spectral_bandwidth': spectral['spectral_bandwidth'],
        'spectral_rolloff': spectral['spectral_rolloff'],
        'spectral_flux': spectral_flux,
        'zcr': zcr,
        'onset_strength': onset_strength,
        'mfcc': mfcc,
        'metadata': {
            'frame_length': frame_length,
            'hop_length': hop_length,
            'n_frames': n_frames,
            'sample_rate': sr,
            'duration_seconds': n_frames * hop_length / sr
        }
    }

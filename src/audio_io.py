"""
Audio I/O Module

Handles audio loading, preprocessing, and normalization.
All operations are deterministic and reproducible.
"""

import numpy as np
from typing import Tuple, Dict, Optional
import warnings

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    warnings.warn(
        "librosa not available, falling back to scipy for audio loading. "
        "Only WAV files will be supported."
    )

if not LIBROSA_AVAILABLE:
    from scipy.io import wavfile
    from scipy import signal as scipy_signal
    from scipy.interpolate import interp1d

import config


def load_audio(file_path: str, target_sr: Optional[int] = None) -> Tuple[np.ndarray, int]:
    """
    Load audio file and return as numpy array.

    Uses librosa if available (supports mp3, flac, wav), falls back to
    scipy.io.wavfile (WAV only).

    Parameters:
        file_path: Path to audio file
        target_sr: Target sample rate (None = use native rate)

    Returns:
        Tuple of (audio_array, sample_rate)
        audio_array: mono float32 array in range [-1.0, 1.0]
        sample_rate: sample rate in Hz

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file format is unsupported
    """
    if LIBROSA_AVAILABLE:
        # Load at native sample rate first, then use our linear resampler for C++ parity
        audio, sr = librosa.load(file_path, sr=None, mono=True)
        audio = audio.astype(np.float32)

        # Resample using our linear interpolation (not librosa's default)
        if target_sr is not None and sr != target_sr:
            audio = resample_audio(audio, sr, target_sr)
            sr = target_sr

        return audio, sr
    else:
        # Fallback to scipy (WAV only)
        sr, audio = wavfile.read(file_path)

        # Convert to float32 and normalize based on dtype
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0
        elif audio.dtype == np.int32:
            audio = audio.astype(np.float32) / 2147483648.0
        elif audio.dtype == np.float32:
            audio = audio.astype(np.float32)
        else:
            raise ValueError(f"Unsupported audio dtype: {audio.dtype}")

        # Convert to mono if stereo
        if len(audio.shape) > 1:
            audio = convert_to_mono(audio, method='average')

        # Resample if needed
        if target_sr is not None and sr != target_sr:
            audio = resample_audio(audio, sr, target_sr)
            sr = target_sr

        return audio, sr


def convert_to_mono(audio: np.ndarray, method: str = 'average') -> np.ndarray:
    """
    Convert stereo/multi-channel audio to mono.

    Parameters:
        audio: Audio array (can be 1D mono or 2D multi-channel)
        method: Conversion method - 'average', 'left', 'right'

    Returns:
        Mono audio array (1D)

    Raises:
        ValueError: If method is invalid or audio shape is unexpected
    """
    # Guard: validate method early
    if method not in ['average', 'left', 'right']:
        raise ValueError(f"Unknown mono conversion method: {method}")

    # Guard: already mono
    if len(audio.shape) == 1:
        return audio

    # Guard: unexpected shape
    if len(audio.shape) != 2:
        raise ValueError(f"Unexpected audio shape: {audio.shape}")

    # Multi-channel: determine format
    # If shape[0] >= shape[1], format is (samples, channels)
    # If shape[0] < shape[1], format is (channels, samples)
    is_samples_first = audio.shape[0] >= audio.shape[1]

    if method == 'average':
        return np.mean(audio, axis=1 if is_samples_first else 0)
    elif method == 'left':
        return audio[:, 0] if is_samples_first else audio[0, :]
    else:  # method == 'right'
        return audio[:, -1] if is_samples_first else audio[-1, :]


def resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """
    Resample audio to target sample rate using linear interpolation.

    Uses linear interpolation for C++ parity with juce::LinearInterpolator.
    Output length uses ceil to match C++ implementation.

    Parameters:
        audio: Audio array
        orig_sr: Original sample rate (Hz)
        target_sr: Target sample rate (Hz)

    Returns:
        Resampled audio array
    """
    if orig_sr == target_sr:
        return audio

    # Use ceil for output length to match C++ JUCE implementation
    num_samples = int(np.ceil(len(audio) * target_sr / orig_sr))

    if LIBROSA_AVAILABLE:
        # Use linear interpolation for C++ parity
        return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr, res_type='linear')
    else:
        # Linear interpolation using scipy.interpolate.interp1d
        old_indices = np.arange(len(audio))
        new_indices = np.linspace(0, len(audio) - 1, num_samples)
        interpolator = interp1d(old_indices, audio, kind='linear', fill_value='extrapolate')
        return interpolator(new_indices).astype(np.float32)


def normalize_audio(
    audio: np.ndarray,
    method: str = 'peak'
) -> Tuple[np.ndarray, float]:
    """
    Normalize audio amplitude.

    Parameters:
        audio: Audio array
        method: Normalization method
            - 'peak': Scale so max absolute value is 1.0
            - 'loudness': Scale to target RMS level (from config)

    Returns:
        Tuple of (normalized_audio, normalization_factor)
        normalization_factor: value used for scaling (for metadata)

    Raises:
        ValueError: If method is invalid
    """
    if method == 'peak':
        # Peak normalization
        peak = np.abs(audio).max()
        if peak == 0:
            # Silent audio
            return audio, 1.0
        factor = 1.0 / peak
        return audio * factor, factor

    elif method == 'loudness':
        # RMS-based loudness normalization
        rms = np.sqrt(np.mean(audio ** 2))
        if rms == 0:
            # Silent audio
            return audio, 1.0

        # Convert target dB to linear scale
        target_linear = 10 ** (config.RMS_TARGET_DB / 20.0)
        factor = target_linear / rms
        return audio * factor, factor

    else:
        raise ValueError(f"Unknown normalization method: {method}")


def trim_silence(
    audio: np.ndarray,
    sr: int,
    threshold_db: float = -60.0,
    frame_length: int = 2048,
    hop_length: int = 512
) -> Tuple[np.ndarray, Tuple[int, int]]:
    """
    Trim leading and trailing silence from audio.

    Parameters:
        audio: Audio array
        sr: Sample rate (Hz)
        threshold_db: Silence threshold in dB
        frame_length: Frame size for energy calculation (samples)
        hop_length: Hop size between frames (samples)

    Returns:
        Tuple of (trimmed_audio, (start_sample, end_sample))
        start_sample: index of first non-silent sample
        end_sample: index of last non-silent sample

    Note:
        Uses frame-wise RMS energy to detect silence
    """
    if len(audio) == 0:
        return audio, (0, 0)

    # Calculate frame-wise RMS energy
    num_frames = 1 + (len(audio) - frame_length) // hop_length
    energy = np.zeros(num_frames)

    for i in range(num_frames):
        start = i * hop_length
        end = start + frame_length
        frame = audio[start:end]
        energy[i] = np.sqrt(np.mean(frame ** 2))

    # Convert threshold to linear scale
    threshold_linear = 10 ** (threshold_db / 20.0)

    # Find first and last non-silent frames
    non_silent = energy > threshold_linear
    if not np.any(non_silent):
        # All silence
        return audio, (0, len(audio))

    first_frame = np.argmax(non_silent)
    last_frame = len(non_silent) - np.argmax(non_silent[::-1]) - 1

    # Convert frame indices to sample indices
    start_sample = first_frame * hop_length
    end_sample = min(last_frame * hop_length + frame_length, len(audio))

    return audio[start_sample:end_sample], (start_sample, end_sample)


def preprocess_audio(
    file_path: str,
    target_sr: Optional[int] = None,
    normalize_method: Optional[str] = None,
    trim: Optional[bool] = None
) -> Dict:
    """
    Load and preprocess audio track with full pipeline.

    Pipeline: load → mono → resample → normalize → trim (optional)

    Parameters:
        file_path: Path to audio file
        target_sr: Target sample rate (None = use config default)
        normalize_method: Normalization method (None = use config default)
        trim: Whether to trim silence (None = use config default)

    Returns:
        Dictionary containing:
            - 'audio': preprocessed audio array (float32)
            - 'sample_rate': sample rate (Hz)
            - 'duration': duration in seconds
            - 'preprocessing': dict of preprocessing steps applied
                - 'original_sr': original sample rate
                - 'resampled': whether audio was resampled
                - 'normalization_method': normalization method used
                - 'normalization_factor': scaling factor applied
                - 'trimmed': whether silence was trimmed
                - 'trim_samples': (start, end) if trimmed
    """
    # Use config defaults if not specified
    if target_sr is None:
        target_sr = config.TARGET_SAMPLE_RATE
    if normalize_method is None:
        normalize_method = config.NORMALIZATION_METHOD
    if trim is None:
        trim = config.TRIM_SILENCE

    # Initialize preprocessing metadata
    preprocessing = {}

    # Load audio
    audio, orig_sr = load_audio(file_path, target_sr=None)
    preprocessing['original_sr'] = orig_sr

    # Resample if needed
    if orig_sr != target_sr:
        audio = resample_audio(audio, orig_sr, target_sr)
        preprocessing['resampled'] = True
    else:
        preprocessing['resampled'] = False

    # Normalize
    audio, norm_factor = normalize_audio(audio, method=normalize_method)
    preprocessing['normalization_method'] = normalize_method
    preprocessing['normalization_factor'] = float(norm_factor)

    # Trim silence (optional)
    if trim:
        audio, trim_samples = trim_silence(
            audio,
            target_sr,
            threshold_db=config.SILENCE_THRESHOLD_DB,
            frame_length=config.SILENCE_FRAME_LENGTH,
            hop_length=config.SILENCE_HOP_LENGTH
        )
        preprocessing['trimmed'] = True
        preprocessing['trim_samples'] = trim_samples
    else:
        preprocessing['trimmed'] = False
        preprocessing['trim_samples'] = None

    # Calculate duration
    duration = len(audio) / target_sr

    return {
        'audio': audio,
        'sample_rate': target_sr,
        'duration': duration,
        'preprocessing': preprocessing
    }


def validate_audio(audio: np.ndarray, sr: int, max_duration: Optional[float] = None) -> None:
    """
    Validate audio array for processing.

    Parameters:
        audio: Audio array to validate
        sr: Sample rate (Hz)
        max_duration: Maximum allowed duration in seconds (None = use config)

    Raises:
        ValueError: If audio is invalid
    """
    if max_duration is None:
        max_duration = config.MAX_TRACK_DURATION_SEC

    if len(audio) == 0:
        raise ValueError("Audio array is empty")

    if not np.isfinite(audio).all():
        raise ValueError("Audio contains NaN or infinite values")

    duration = len(audio) / sr
    if duration > max_duration:
        raise ValueError(
            f"Audio duration ({duration:.1f}s) exceeds maximum "
            f"({max_duration:.1f}s)"
        )

    if duration < 1.0:
        raise ValueError(f"Audio too short: {duration:.2f}s (minimum 1.0s)")

"""
Kernel Parameters Module - All Tunable Constants

These parameters control kernel behavior and must be synchronized
between Python and C++ implementations.

For C++ port: Each dataclass becomes a struct with the same fields.

USAGE:
    from src.kernel_params import KernelConfig, DEFAULT_CONFIG

    # Use default config
    config = DEFAULT_CONFIG

    # Create custom config
    custom = KernelConfig(
        frame=FrameParams(frame_length=4096),
        tension=TensionParams(smooth_alpha=0.5)
    )
"""

from dataclasses import dataclass, field
from typing import Dict


@dataclass(frozen=True)
class FrameParams:
    """
    Frame-level analysis parameters.

    For C++ port: struct FrameParams { int frame_length; int hop_length; int sample_rate; };

    Attributes:
        frame_length: Samples per STFT frame (default 2048 = ~93ms at 22050 Hz)
        hop_length: Hop between frames (default 512 = ~23ms, 4x overlap)
        sample_rate: Target sample rate in Hz (default 22050)
    """
    frame_length: int = 2048
    hop_length: int = 512
    sample_rate: int = 22050


@dataclass(frozen=True)
class BlockParams:
    """
    Block aggregation parameters.

    For C++ port: struct BlockParams { float block_duration_sec; };

    Attributes:
        block_duration_sec: Duration of each block in seconds (default 0.5s = 2 blocks/beat at 120 BPM)
    """
    block_duration_sec: float = 0.5


@dataclass(frozen=True)
class TensionParams:
    """
    Tension curve computation parameters.

    For C++ port: struct TensionParams {
        float rms_weight, onset_weight, centroid_weight, bandwidth_weight;
        float smooth_alpha, percentile_lower, percentile_upper;
    };

    Attributes:
        rms_weight: Weight for RMS energy component (default 0.4)
        onset_weight: Weight for onset density component (default 0.3)
        centroid_weight: Weight for spectral centroid component (default 0.2)
        bandwidth_weight: Weight for spectral bandwidth component (default 0.1)
        smooth_alpha: EWMA smoothing factor (default 0.3, lower = more smoothing)
        percentile_lower: Lower percentile for robust normalization (default 5.0)
        percentile_upper: Upper percentile for robust normalization (default 95.0)
        normalization_mode: 'robust_track' or 'anchored' (default 'robust_track')
    """
    rms_weight: float = 0.4
    onset_weight: float = 0.3
    centroid_weight: float = 0.2
    bandwidth_weight: float = 0.1
    smooth_alpha: float = 0.3
    percentile_lower: float = 5.0
    percentile_upper: float = 95.0
    normalization_mode: str = 'robust_track'

    def get_weights(self) -> Dict[str, float]:
        """Get weights as dictionary for kernel functions."""
        return {
            'rms': self.rms_weight,
            'onset_density': self.onset_weight,
            'spectral_centroid': self.centroid_weight,
            'spectral_bandwidth': self.bandwidth_weight
        }


@dataclass(frozen=True)
class NoveltyParams:
    """
    Novelty curve computation parameters.

    For C++ port: struct NoveltyParams { int lookback_blocks; int smooth_window; };

    Attributes:
        lookback_blocks: Number of past blocks for context comparison (default 16 = 8 seconds)
        smooth_window: Moving average window for smoothing (default 3 blocks)
    """
    lookback_blocks: int = 16
    smooth_window: int = 3


@dataclass(frozen=True)
class FatigueParams:
    """
    Fatigue curve computation parameters.

    For C++ port: struct FatigueParams {
        int window_blocks, smooth_window;
        float gain_up, gain_down, novelty_spike_threshold;
        float similarity_weight, novelty_weight, variance_weight;
        bool use_leaky_integrator;
    };

    Attributes:
        window_blocks: Rolling window size (default 32 = 16 seconds)
        smooth_window: Moving average window for smoothing (default 5 blocks)
        gain_up: Fatigue accumulation rate per block (default 0.02)
        gain_down: Fatigue recovery rate per block (default 0.08 = 4x faster)
        novelty_spike_threshold: Threshold for triggering recovery (default 0.5)
        similarity_weight: Weight for self-similarity in boring score (default 0.5)
        novelty_weight: Weight for inverse novelty in boring score (default 0.3)
        variance_weight: Weight for inverse variance in boring score (default 0.2)
        use_leaky_integrator: Use leaky integrator model vs weighted average (default True)
    """
    window_blocks: int = 32
    smooth_window: int = 5
    gain_up: float = 0.02
    gain_down: float = 0.08
    novelty_spike_threshold: float = 0.5
    similarity_weight: float = 0.5
    novelty_weight: float = 0.3
    variance_weight: float = 0.2
    use_leaky_integrator: bool = True

    def get_boring_weights(self) -> Dict[str, float]:
        """Get boring signal weights as dictionary for kernel functions."""
        return {
            'self_similarity': self.similarity_weight,
            'inverse_novelty': self.novelty_weight,
            'inverse_variance': self.variance_weight
        }


@dataclass(frozen=True)
class AnchoredNormParams:
    """
    Anchored normalization reference values for cross-track comparability.

    For C++ port: struct AnchoredNormParams {
        float rms_max_dbfs, onset_max, centroid_max_hz, bandwidth_max_hz;
    };

    Attributes:
        rms_max_dbfs: Reference maximum RMS in dBFS (default -6.0)
        onset_max: Reference maximum onset strength (default 50.0)
        centroid_max_hz: Reference maximum spectral centroid in Hz (default 8000.0)
        bandwidth_max_hz: Reference maximum spectral bandwidth in Hz (default 6000.0)
    """
    rms_max_dbfs: float = -6.0
    onset_max: float = 50.0
    centroid_max_hz: float = 8000.0
    bandwidth_max_hz: float = 6000.0


@dataclass(frozen=True)
class MFCCParams:
    """
    MFCC computation parameters.

    For C++ port: struct MFCCParams { int n_mfcc; int n_mels; };

    Attributes:
        n_mfcc: Number of MFCC coefficients (default 13)
        n_mels: Number of mel filterbank bands (default 40)
    """
    n_mfcc: int = 13
    n_mels: int = 40


@dataclass(frozen=True)
class SpectralParams:
    """
    Spectral analysis parameters.

    For C++ port: struct SpectralParams { float rolloff_percent; };

    Attributes:
        rolloff_percent: Spectral rolloff percentage (default 0.85 = 85%)
    """
    rolloff_percent: float = 0.85


@dataclass(frozen=True)
class NormalizationParams:
    """
    General normalization parameters.

    For C++ port: struct NormalizationParams { float percentile_lower; float percentile_upper; };

    Attributes:
        percentile_lower: Lower percentile for clipping (default 1.0)
        percentile_upper: Upper percentile for clipping (default 99.0)
    """
    percentile_lower: float = 1.0
    percentile_upper: float = 99.0


@dataclass
class KernelConfig:
    """
    Complete kernel configuration aggregating all parameter groups.

    For C++ port: This becomes a struct containing all sub-structs,
    or individual sub-structs can be passed to functions.

    Example usage:
        config = KernelConfig()  # All defaults
        config = KernelConfig(tension=TensionParams(smooth_alpha=0.5))  # Override specific params
    """
    frame: FrameParams = field(default_factory=FrameParams)
    block: BlockParams = field(default_factory=BlockParams)
    tension: TensionParams = field(default_factory=TensionParams)
    novelty: NoveltyParams = field(default_factory=NoveltyParams)
    fatigue: FatigueParams = field(default_factory=FatigueParams)
    anchored: AnchoredNormParams = field(default_factory=AnchoredNormParams)
    mfcc: MFCCParams = field(default_factory=MFCCParams)
    spectral: SpectralParams = field(default_factory=SpectralParams)
    normalization: NormalizationParams = field(default_factory=NormalizationParams)

    def to_dict(self) -> Dict:
        """
        Export all parameters as a flat dictionary for JSON serialization.

        Returns:
            Dictionary with all parameter values
        """
        return {
            # Frame params
            'frame_length': self.frame.frame_length,
            'hop_length': self.frame.hop_length,
            'sample_rate': self.frame.sample_rate,

            # Block params
            'block_duration_sec': self.block.block_duration_sec,

            # Tension params
            'tension_weights': self.tension.get_weights(),
            'tension_smooth_alpha': self.tension.smooth_alpha,
            'tension_percentile_lower': self.tension.percentile_lower,
            'tension_percentile_upper': self.tension.percentile_upper,
            'tension_normalization_mode': self.tension.normalization_mode,

            # Novelty params
            'novelty_lookback_blocks': self.novelty.lookback_blocks,
            'novelty_smooth_window': self.novelty.smooth_window,

            # Fatigue params
            'fatigue_window_blocks': self.fatigue.window_blocks,
            'fatigue_smooth_window': self.fatigue.smooth_window,
            'fatigue_gain_up': self.fatigue.gain_up,
            'fatigue_gain_down': self.fatigue.gain_down,
            'fatigue_novelty_spike_threshold': self.fatigue.novelty_spike_threshold,
            'fatigue_boring_weights': self.fatigue.get_boring_weights(),
            'fatigue_use_leaky_integrator': self.fatigue.use_leaky_integrator,

            # Anchored normalization params
            'anchored_rms_max_dbfs': self.anchored.rms_max_dbfs,
            'anchored_onset_max': self.anchored.onset_max,
            'anchored_centroid_max_hz': self.anchored.centroid_max_hz,
            'anchored_bandwidth_max_hz': self.anchored.bandwidth_max_hz,

            # MFCC params
            'n_mfcc': self.mfcc.n_mfcc,
            'n_mels': self.mfcc.n_mels,

            # Spectral params
            'spectral_rolloff_percent': self.spectral.rolloff_percent,

            # Normalization params
            'normalization_percentile_lower': self.normalization.percentile_lower,
            'normalization_percentile_upper': self.normalization.percentile_upper,
        }


# Default configuration instance
DEFAULT_CONFIG = KernelConfig()


def validate_config(config: KernelConfig) -> bool:
    """
    Validate configuration parameters for consistency.

    Parameters:
        config: KernelConfig instance to validate

    Returns:
        True if config is valid

    Raises:
        ValueError: If configuration is invalid
    """
    # Check weights sum to ~1.0
    tension_weights = config.tension.get_weights()
    tension_sum = sum(tension_weights.values())
    if not (0.99 <= tension_sum <= 1.01):
        raise ValueError(f"Tension weights must sum to 1.0, got {tension_sum}")

    fatigue_weights = config.fatigue.get_boring_weights()
    fatigue_sum = sum(fatigue_weights.values())
    if not (0.99 <= fatigue_sum <= 1.01):
        raise ValueError(f"Fatigue boring weights must sum to 1.0, got {fatigue_sum}")

    # Check positive values
    if config.frame.frame_length <= 0:
        raise ValueError("frame_length must be positive")
    if config.frame.hop_length <= 0:
        raise ValueError("hop_length must be positive")
    if config.frame.sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if config.block.block_duration_sec <= 0:
        raise ValueError("block_duration_sec must be positive")

    # Check ranges
    if not (0.0 < config.tension.smooth_alpha <= 1.0):
        raise ValueError("tension_smooth_alpha must be in (0, 1]")
    if not (0.0 <= config.fatigue.gain_up <= 1.0):
        raise ValueError("fatigue_gain_up must be in [0, 1]")
    if not (0.0 <= config.fatigue.gain_down <= 1.0):
        raise ValueError("fatigue_gain_down must be in [0, 1]")
    if not (0.0 <= config.fatigue.novelty_spike_threshold <= 1.0):
        raise ValueError("fatigue_novelty_spike_threshold must be in [0, 1]")

    return True


# Validate default config on import
validate_config(DEFAULT_CONFIG)

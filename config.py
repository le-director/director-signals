"""
director-signals - Configuration

All tunable parameters, weights, and constants with documentation.
Every default value includes rationale.
"""

from typing import Dict, List

# =============================================================================
# FRAME-LEVEL PARAMETERS
# =============================================================================

# Frame length for STFT analysis (samples)
# Why: 2048 samples at 22050 Hz ≈ 93ms window, good balance for musical content
FRAME_LENGTH: int = 2048

# Hop length between frames (samples)
# Why: 512 samples = 4x overlap, standard for good time-frequency resolution
HOP_LENGTH: int = 512

# Target sample rate for processing (Hz)
# Why: 22050 Hz captures musical content up to 11kHz (covers most energy),
#      reduces computation vs 44100 Hz while maintaining quality
TARGET_SAMPLE_RATE: int = 22050

# =============================================================================
# BLOCK AGGREGATION PARAMETERS
# =============================================================================

# Block duration in seconds
# Why: 0.5 seconds ≈ 2 blocks per beat at 120 BPM, provides musical-time resolution
#      while aggregating enough frames for stable statistics
BLOCK_DURATION_SEC: float = 0.5

# Statistics to compute per block from frame features
# Why: Mean captures central tendency, median is robust to outliers,
#      std captures variability, percentiles capture distribution shape
BLOCK_AGGREGATION_STATS: List[str] = ['mean', 'median', 'std', 'p25', 'p75']

# =============================================================================
# LONG-HORIZON WINDOW PARAMETERS
# =============================================================================

# Lookback window for novelty computation (blocks)
# Why: 16 blocks = 8 seconds at 0.5s/block, captures recent musical context
#      for change detection without being too sensitive to local variation
NOVELTY_LOOKBACK_BLOCKS: int = 16

# Window size for fatigue/repetition analysis (blocks)
# Why: 32 blocks = 16 seconds, long enough to detect sustained repetition
#      (e.g., 2 bars repeated 4 times at 120 BPM)
FATIGUE_WINDOW_BLOCKS: int = 32

# Pre-drop window for build detection (seconds)
# Why: 4 seconds is typical build duration in electronic music,
#      long enough to detect rising tension pattern
DROP_PRE_WINDOW_SEC: float = 4.0

# Post-drop window for impact measurement (seconds)
# Why: 4 seconds captures initial drop impact and stabilization,
#      covers typical drop phrase length
DROP_POST_WINDOW_SEC: float = 4.0

# =============================================================================
# CURVE COMPUTATION WEIGHTS
# =============================================================================

# Weights for tension/energy curve components
# Why: RMS (0.4) is primary driver of perceived intensity,
#      onset density (0.3) captures rhythmic drive and impact,
#      spectral centroid (0.2) reflects brightness/aggression,
#      spectral bandwidth (0.1) adds fullness/richness perception
TENSION_WEIGHTS: Dict[str, float] = {
    'rms': 0.4,
    'onset_density': 0.3,
    'spectral_centroid': 0.2,
    'spectral_bandwidth': 0.1
}

# Weights for fatigue curve components
# Why: Self-similarity (0.5) is primary indicator of repetition,
#      inverse novelty (0.3) confirms lack of change,
#      inverse variance (0.2) captures sustained predictability
FATIGUE_WEIGHTS: Dict[str, float] = {
    'self_similarity': 0.5,
    'inverse_novelty': 0.3,
    'inverse_variance': 0.2
}

# =============================================================================
# FATIGUE LEAKY INTEGRATOR PARAMETERS
# =============================================================================

# Enable leaky integrator mode for fatigue (vs. original weighted average)
# Why: Leaky integrator provides more realistic fatigue accumulation and recovery
FATIGUE_USE_LEAKY_INTEGRATOR: bool = True

# Rate of fatigue accumulation per block when content is "boring"
# Why: 0.02 = slow accumulation, takes ~50 blocks (25s) to reach 1.0 from 0
#      This matches subjective experience of gradual fatigue buildup
FATIGUE_GAIN_UP: float = 0.02

# Rate of fatigue recovery per block when content is "interesting"
# Why: 0.08 = 4x faster than accumulation, recovery should be quicker than buildup
#      Novel content provides relief faster than repetition causes fatigue
FATIGUE_GAIN_DOWN: float = 0.08

# Novelty spike threshold for accelerated recovery
# Why: 0.5 = top 50% of novelty range triggers "interesting" mode
#      Below this, fatigue accumulates; above this, fatigue recovers
FATIGUE_NOVELTY_SPIKE_THRESHOLD: float = 0.5

# Weights for "boring" signal in leaky integrator
# These determine what contributes to fatigue accumulation
FATIGUE_BORING_WEIGHTS: Dict[str, float] = {
    'self_similarity': 0.5,
    'inverse_novelty': 0.3,
    'inverse_variance': 0.2
}

# =============================================================================
# SMOOTHING PARAMETERS
# =============================================================================

# EWMA alpha for tension curve smoothing
# Why: 0.3 provides moderate smoothing, balances responsiveness to changes
#      while reducing frame-to-frame noise. Lower = more smoothing.
TENSION_SMOOTH_ALPHA: float = 0.3

# Moving average window for novelty curve (blocks)
# Why: 3-block window (1.5 seconds) reduces noise while preserving
#      transition sharpness for boundary detection
NOVELTY_SMOOTH_WINDOW: int = 3

# Moving average window for fatigue curve (blocks)
# Why: 5-block window (2.5 seconds) provides stability for slow-changing signal,
#      fatigue builds gradually so more smoothing is appropriate
FATIGUE_SMOOTH_WINDOW: int = 5

# =============================================================================
# DETECTION THRESHOLDS
# =============================================================================

# Minimum prominence for tension peaks to be considered drop candidates
# Why: 0.3 in normalized units means peak must rise 0.3 above surrounding baseline,
#      filters out minor fluctuations while catching significant events
DROP_PROMINENCE_THRESHOLD: float = 0.3

# Minimum duration of rising tension before drop (seconds)
# Why: 2 seconds filters out brief transients, requires sustained build
#      typical of intentional musical tension creation
DROP_MIN_BUILD_DURATION_SEC: float = 2.0

# Minimum slope for tension rise to qualify as "build" (normalized units per second)
# Why: 0.05 per second is gentle rise over 10 seconds, catches gradual builds
#      while filtering flat sections before peaks
DROP_MIN_BUILD_SLOPE: float = 0.05

# Threshold below which novelty indicates stagnation
# Why: 0.4 in normalized [0,1] range means very little change,
#      indicates repetitive content
STAGNANT_NOVELTY_THRESHOLD: float = 0.4

# Threshold above which fatigue indicates over-looping
# Why: 0.6 in normalized [0,1] range means high self-similarity and low variance,
#      strong indicator of problematic repetition
STAGNANT_FATIGUE_THRESHOLD: float = 0.6

# Minimum duration to label a section as stagnant (seconds)
# Why: 4 seconds filters brief repeated phrases, identifies sustained problems
#      (roughly 1-2 bars that feel "stuck")
STAGNANT_MIN_DURATION_SEC: float = 4.0

# Confidence threshold for section boundary detection
# Why: 0.5 means only report boundaries with >50% confidence,
#      avoids over-segmentation from minor changes
BOUNDARY_CONFIDENCE_THRESHOLD: float = 0.5

# Minimum distance between detected peaks (blocks)
# Why: 20 blocks = 10 seconds, prevents detecting multiple peaks
#      for same musical event (drops typically 16-32 bars apart)
PEAK_MIN_DISTANCE_BLOCKS: int = 20

# =============================================================================
# DROP CLASSIFICATION PARAMETERS
# =============================================================================

# Minimum number of components with meaningful contrast for "drop" classification
# Why: A true drop should show contrast in multiple dimensions (RMS + onset at minimum),
#      not just a single feature spike
DROP_MIN_CONTRAST_COMPONENTS: int = 2

# Minimum contrast threshold per component for "meaningful" contrast
# Why: 0.2 = 20% relative change is noticeable to listeners
DROP_COMPONENT_CONTRAST_THRESHOLD: float = 0.2

# Minimum post-drop persistence duration (seconds)
# Why: True drops maintain energy for at least 4 seconds (1-2 bars at 120 BPM),
#      brief spikes that decay immediately are not drops
DROP_POST_PERSISTENCE_SEC: float = 4.0

# Minimum fraction of peak tension to maintain during persistence window
# Why: 0.7 = post-drop must stay above 70% of peak value to count as sustained
DROP_PERSISTENCE_THRESHOLD: float = 0.7

# =============================================================================
# PLATEAU DETECTION PARAMETERS
# =============================================================================

# Minimum duration to consider as plateau (seconds)
# Why: 8 seconds (~4 bars at 120 BPM) is minimum to establish a sustained regime
PLATEAU_MIN_DURATION_SEC: float = 8.0

# Tension threshold to be considered "high" for plateau detection
# Why: 0.7 = top 30% of tension range indicates sustained high-energy section
PLATEAU_TENSION_THRESHOLD: float = 0.7

# Within plateau, minimum negative-to-positive contrast for a peak to be a drop
# Why: 0.3 = need significant dip before peak to differentiate from plateau ripple
PLATEAU_DROP_MIN_CONTRAST: float = 0.3

# =============================================================================
# AUDIO PREPROCESSING PARAMETERS
# =============================================================================

# Normalization method: 'peak' or 'loudness'
# Why: 'peak' is simple, deterministic, and sufficient for relative analysis.
#      Scales audio to [-1.0, 1.0] range based on maximum absolute value.
NORMALIZATION_METHOD: str = 'peak'

# Target RMS level for loudness normalization (if using 'loudness' method)
# Why: -20 dBFS is moderate level, leaves headroom, typical for analysis
RMS_TARGET_DB: float = -20.0

# Silence threshold for trimming (dB)
# Why: -60 dB is well below musical content, catches true silence/noise floor
SILENCE_THRESHOLD_DB: float = -60.0

# Whether to trim leading/trailing silence by default
# Why: False preserves original timing, important for drop detection alignment.
#      User can enable if needed.
TRIM_SILENCE: bool = False

# Frame length for silence detection (samples)
# Why: Same as analysis frame length for consistency
SILENCE_FRAME_LENGTH: int = 2048

# Hop length for silence detection (samples)
# Why: Same as analysis hop for consistency
SILENCE_HOP_LENGTH: int = 512

# =============================================================================
# OUTPUT PARAMETERS
# =============================================================================

# JSON schema version
# Why: Versioning allows future format changes while maintaining compatibility
# v1.1.0: Added tension normalization info, event_type classification, fatigue leaky integrator
SCHEMA_VERSION: str = "1.1.0"

# Kernel version (algorithm version, bump when DSP logic changes)
# Why: Allows tracking which algorithm version produced specific outputs
# v1.1.1: Added versioned golden outputs structure with real audio support
KERNEL_VERSION: str = "1.1.1"

# Timebase version (bump when block/time calculation logic changes)
# Why: Timebase is foundational - changes affect all downstream outputs
TIMEBASE_VERSION: str = "1"

# Number of top drop candidates to return
# Why: 10 is manageable for review, covers most real drops in typical tracks
#      (most tracks have 3-8 drops)
TOP_N_DROPS: int = 10

# Plot resolution (dots per inch)
# Why: 150 DPI is good balance of quality and file size for screen viewing
#      and documentation, higher than default 100 DPI
PLOT_DPI: int = 150

# Plot figure size (width, height in inches)
# Why: 14x10 inches at 150 DPI = 2100x1500 pixels, provides detail
#      while fitting on typical screens. Wider than tall for timeline view.
PLOT_FIGSIZE: tuple = (14, 10)

# Number of MFCC coefficients to compute
# Why: 13 is standard (captures spectral envelope), first coefficient
#      relates to energy, rest capture timbre
N_MFCC: int = 13

# Spectral rolloff percentage
# Why: 85% is standard, represents frequency below which 85% of
#      spectral energy is contained, indicates brightness
SPECTRAL_ROLLOFF_PERCENT: float = 0.85

# =============================================================================
# NORMALIZATION PARAMETERS
# =============================================================================

# Method for normalizing block features: 'robust', 'percentile', or 'zscore'
# Why: 'robust' uses median and IQR, resistant to outliers,
#      appropriate for music with dynamic range variation
BLOCK_NORMALIZE_METHOD: str = 'robust'

# Percentiles for percentile normalization
# Why: 1st and 99th percentiles clip extreme outliers while preserving
#      most of the distribution, more robust than min/max
PERCENTILE_LOWER: float = 1.0
PERCENTILE_UPPER: float = 99.0

# =============================================================================
# TENSION NORMALIZATION PARAMETERS
# =============================================================================

# Tension normalization mode: 'robust_track' or 'anchored'
# Why: 'robust_track' preserves relative contrast within a track using percentiles.
#      'anchored' uses physically meaningful reference values for cross-track comparability.
TENSION_NORMALIZATION_MODE: str = 'robust_track'

# Percentiles for tension robust scaling (wider than general percentiles to preserve contrast)
# Why: 5th-95th percentiles are more robust to outliers than 1st-99th while preserving
#      internal contrast in high-energy tracks (avoids "flat near 1.0" problem)
TENSION_PERCENTILE_LOWER: float = 5.0
TENSION_PERCENTILE_UPPER: float = 95.0

# Anchored mode reference values (used when mode='anchored')
# These represent "typical maximum" values for well-mastered commercial tracks

# Maximum RMS in dBFS (0 dBFS = digital maximum)
# Why: -6 dBFS is typical loudness for modern commercial productions
ANCHORED_RMS_MAX_DBFS: float = -6.0

# Maximum onset strength (arbitrary units from librosa onset_strength)
# Why: 50 is typical maximum for dense rhythmic sections
ANCHORED_ONSET_MAX: float = 50.0

# Maximum meaningful spectral centroid in Hz
# Why: 8000 Hz covers most musical brightness, beyond this is noise/artifacts
ANCHORED_CENTROID_MAX_HZ: float = 8000.0

# Maximum meaningful spectral bandwidth in Hz
# Why: 6000 Hz represents very broad spectrum typical of mastered music
ANCHORED_BANDWIDTH_MAX_HZ: float = 6000.0

# =============================================================================
# PERFORMANCE PARAMETERS
# =============================================================================

# Maximum track duration to process (seconds)
# Why: 600 seconds (10 minutes) covers most tracks, prevents memory issues
#      with extremely long files. Can be increased if needed.
MAX_TRACK_DURATION_SEC: float = 600.0

# Number of parallel workers for batch processing (0 = auto)
# Why: 0 uses CPU count, enables parallel track processing for directories
N_WORKERS: int = 0

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_frames_per_block(sample_rate: int = TARGET_SAMPLE_RATE) -> int:
    """
    Calculate number of frames per block.

    Parameters:
        sample_rate: Audio sample rate (Hz)

    Returns:
        Number of frames that fit in one block
    """
    samples_per_block = int(BLOCK_DURATION_SEC * sample_rate)
    frames_per_block = samples_per_block // HOP_LENGTH
    return frames_per_block


def get_blocks_per_second() -> float:
    """
    Calculate number of blocks per second.

    Returns:
        Blocks per second (inverse of BLOCK_DURATION_SEC)
    """
    return 1.0 / BLOCK_DURATION_SEC


def seconds_to_blocks(seconds: float) -> int:
    """
    Convert seconds to number of blocks.

    Parameters:
        seconds: Duration in seconds

    Returns:
        Number of blocks (rounded)
    """
    return int(seconds / BLOCK_DURATION_SEC)


def blocks_to_seconds(blocks: int) -> float:
    """
    Convert blocks to seconds.

    Parameters:
        blocks: Number of blocks

    Returns:
        Duration in seconds
    """
    return blocks * BLOCK_DURATION_SEC


def validate_config() -> bool:
    """
    Validate configuration parameters for consistency.

    Returns:
        True if config is valid

    Raises:
        ValueError: If configuration is invalid
    """
    # Check weights sum to ~1.0
    tension_sum = sum(TENSION_WEIGHTS.values())
    if not (0.99 <= tension_sum <= 1.01):
        raise ValueError(f"TENSION_WEIGHTS must sum to 1.0, got {tension_sum}")

    fatigue_sum = sum(FATIGUE_WEIGHTS.values())
    if not (0.99 <= fatigue_sum <= 1.01):
        raise ValueError(f"FATIGUE_WEIGHTS must sum to 1.0, got {fatigue_sum}")

    # Check positive values
    if FRAME_LENGTH <= 0 or HOP_LENGTH <= 0:
        raise ValueError("FRAME_LENGTH and HOP_LENGTH must be positive")

    if BLOCK_DURATION_SEC <= 0:
        raise ValueError("BLOCK_DURATION_SEC must be positive")

    if TARGET_SAMPLE_RATE <= 0:
        raise ValueError("TARGET_SAMPLE_RATE must be positive")

    # Check threshold ranges
    if not (0.0 <= DROP_PROMINENCE_THRESHOLD <= 1.0):
        raise ValueError("DROP_PROMINENCE_THRESHOLD must be in [0, 1]")

    if not (0.0 <= STAGNANT_NOVELTY_THRESHOLD <= 1.0):
        raise ValueError("STAGNANT_NOVELTY_THRESHOLD must be in [0, 1]")

    if not (0.0 <= STAGNANT_FATIGUE_THRESHOLD <= 1.0):
        raise ValueError("STAGNANT_FATIGUE_THRESHOLD must be in [0, 1]")

    # Check EWMA alpha
    if not (0.0 < TENSION_SMOOTH_ALPHA <= 1.0):
        raise ValueError("TENSION_SMOOTH_ALPHA must be in (0, 1]")

    return True


# Validate on import
validate_config()

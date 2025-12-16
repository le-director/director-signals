# Parameter Documentation

Complete guide to all configuration parameters with rationale and tuning guidelines.

## Table of Contents

1. [Frame-Level Parameters](#frame-level-parameters)
2. [Block Aggregation Parameters](#block-aggregation-parameters)
3. [Long-Horizon Window Parameters](#long-horizon-window-parameters)
4. [Curve Computation Weights](#curve-computation-weights)
5. [Smoothing Parameters](#smoothing-parameters)
6. [Detection Thresholds](#detection-thresholds)
7. [Audio Preprocessing](#audio-preprocessing)
8. [Output Parameters](#output-parameters)
9. [Tuning Guidelines](#tuning-guidelines)

---

## Frame-Level Parameters

### FRAME_LENGTH
- **Value**: `2048` samples
- **Units**: samples
- **Why**: At 22050 Hz, this is ~93ms—good balance for musical content. Longer frames improve frequency resolution but reduce time resolution.
- **Trade-offs**:
  - Larger → better frequency resolution, worse time resolution
  - Smaller → better time resolution, worse frequency resolution
- **Tuning**: Rarely needs adjustment. Use 4096 for low-frequency focus, 1024 for transient precision.

### HOP_LENGTH
- **Value**: `512` samples
- **Units**: samples
- **Why**: 4x overlap (FRAME_LENGTH/HOP_LENGTH=4) is standard for good time-frequency resolution.
- **Trade-offs**:
  - Smaller → more frames, better time resolution, higher computation
  - Larger → fewer frames, faster processing, coarser time resolution
- **Tuning**: Keep at FRAME_LENGTH/4 for standard overlap. Only change if FRAME_LENGTH changes.

### TARGET_SAMPLE_RATE
- **Value**: `22050` Hz
- **Units**: Hz
- **Why**: Captures musical content up to 11 kHz (Nyquist = sr/2), where most musical energy resides. Reduces computation vs 44100 Hz while maintaining quality.
- **Trade-offs**:
  - Higher → more detail, slower processing
  - Lower → less detail, faster processing
- **Tuning**:
  - Use 44100 for full-spectrum analysis
  - Use 16000 for speech/vocal focus
  - Use 11025 for maximum speed (acceptable for rough analysis)

---

## Block Aggregation Parameters

### BLOCK_DURATION_SEC
- **Value**: `0.5` seconds
- **Units**: seconds
- **Why**: ~2 blocks per beat at 120 BPM. Provides musical-time resolution while aggregating enough frames for stable statistics.
- **Trade-offs**:
  - Larger → smoother curves, less temporal detail, faster processing
  - Smaller → more detailed curves, noisier, slower processing
- **Tuning**:
  - Use 1.0s for slower music or smoother curves
  - Use 0.25s for very detailed analysis (e.g., drum & bass)
  - If BPM known, use bar duration (e.g., 2.0s for 120 BPM 4/4)

### BLOCK_AGGREGATION_STATS
- **Value**: `['mean', 'median', 'std', 'p25', 'p75']`
- **Why**:
  - `mean`: Central tendency
  - `median`: Robust to outliers
  - `std`: Captures variability
  - `p25/p75`: Distribution shape
- **Tuning**:
  - Remove percentiles to reduce dimensionality
  - Add 'min'/'max' for extreme value tracking
  - Use only 'mean' for fastest processing

---

## Long-Horizon Window Parameters

### NOVELTY_LOOKBACK_BLOCKS
- **Value**: `16` blocks = 8 seconds
- **Units**: blocks
- **Why**: 8 seconds captures recent musical context for change detection without being too sensitive to local variation.
- **Trade-offs**:
  - Larger → detects broader structural changes, misses subtle transitions
  - Smaller → more sensitive, may flag minor variations
- **Tuning**:
  - Use 32 (16s) for long-form analysis (DJ sets)
  - Use 8 (4s) for rapid genre shifts (electronic music)

### FATIGUE_WINDOW_BLOCKS
- **Value**: `32` blocks = 16 seconds
- **Units**: blocks
- **Why**: 16 seconds is long enough to detect sustained repetition (e.g., 2 bars repeated 4 times at 120 BPM).
- **Trade-offs**:
  - Larger → detects longer-term repetition
  - Smaller → detects shorter loops
- **Tuning**:
  - Use 64 (32s) for progressive house (long builds)
  - Use 16 (8s) for hip-hop (shorter loops)

### DROP_PRE_WINDOW_SEC
- **Value**: `4.0` seconds
- **Units**: seconds
- **Why**: 4 seconds is typical build duration in electronic music before drops.
- **Tuning**:
  - Use 8.0s for progressive/trance (longer builds)
  - Use 2.0s for trap/dubstep (short builds)

### DROP_POST_WINDOW_SEC
- **Value**: `4.0` seconds
- **Units**: seconds
- **Why**: 4 seconds captures initial drop impact and stabilization, covers typical drop phrase.
- **Tuning**:
  - Use 8.0s for measuring sustained drop energy
  - Use 2.0s for just the immediate impact

---

## Curve Computation Weights

### TENSION_WEIGHTS
- **Value**: `{'rms': 0.4, 'onset_density': 0.3, 'spectral_centroid': 0.2, 'spectral_bandwidth': 0.1}`
- **Units**: weights summing to 1.0
- **Rationale**:
  - **RMS (0.4)**: Loudness is primary driver of perceived intensity
  - **Onset density (0.3)**: Rhythmic drive and impact are key to energy
  - **Spectral centroid (0.2)**: Brightness correlates with aggression
  - **Spectral bandwidth (0.1)**: Fullness adds to perceived richness
- **Tuning by Genre**:
  - **EDM/Dance**: Increase onset_density to 0.4, reduce RMS to 0.3 (rhythmic focus)
  - **Ambient/Classical**: Increase spectral_centroid/bandwidth, reduce onset_density (timbral focus)
  - **Rock/Metal**: Increase RMS to 0.5 (loudness-driven)

### FATIGUE_WEIGHTS
- **Value**: `{'self_similarity': 0.5, 'inverse_novelty': 0.3, 'inverse_variance': 0.2}`
- **Units**: weights summing to 1.0
- **Rationale**:
  - **Self-similarity (0.5)**: Direct repetition is strongest indicator
  - **Inverse novelty (0.3)**: Lack of change confirms stagnation
  - **Inverse variance (0.2)**: Low variance shows predictability
- **Tuning**:
  - Increase self_similarity for loop detection
  - Increase inverse_variance for texture stagnation

---

## Smoothing Parameters

### TENSION_SMOOTH_ALPHA
- **Value**: `0.3`
- **Units**: EWMA alpha (0 to 1)
- **Why**: 0.3 provides moderate smoothing, balances responsiveness to changes while reducing frame-to-frame noise.
- **Behavior**:
  - Alpha = 1.0: no smoothing
  - Alpha = 0.1: heavy smoothing
  - Alpha = 0.3: moderate smoothing (default)
- **Tuning**:
  - Use 0.5 for more responsive curves
  - Use 0.1 for very smooth curves

### NOVELTY_SMOOTH_WINDOW
- **Value**: `3` blocks
- **Units**: blocks
- **Why**: 3-block window (1.5s) reduces noise while preserving transition sharpness.
- **Tuning**:
  - Use 5 for smoother novelty
  - Use 1 for no smoothing (raw distances)

### FATIGUE_SMOOTH_WINDOW
- **Value**: `5` blocks
- **Units**: blocks
- **Why**: 5-block window (2.5s) provides stability for slow-changing signal. Fatigue builds gradually so more smoothing is appropriate.
- **Tuning**:
  - Use 7-9 for very smooth fatigue
  - Use 3 for more reactive fatigue

---

## Detection Thresholds

### DROP_PROMINENCE_THRESHOLD
- **Value**: `0.5`
- **Units**: normalized tension units
- **Why**: Peak must rise 0.5 above surrounding baseline in [0,1] scale. Filters minor fluctuations while catching significant events.
- **Tuning**:
  - Use 0.7 for fewer, more confident drops
  - Use 0.3 for more drop candidates (higher recall)

### DROP_MIN_BUILD_DURATION_SEC
- **Value**: `2.0` seconds
- **Units**: seconds
- **Why**: 2 seconds filters brief transients, requires sustained build typical of intentional tension creation.
- **Tuning**:
  - Use 4.0s for progressive genres (longer builds)
  - Use 1.0s for aggressive genres (short builds)

### DROP_MIN_BUILD_SLOPE
- **Value**: `0.1` per second
- **Units**: normalized units per second
- **Why**: 0.1 per second = gentle rise over 5 seconds. Catches gradual builds while filtering flat sections.
- **Tuning**:
  - Use 0.2 for steeper build requirement
  - Use 0.05 for accepting gentler builds

### STAGNANT_NOVELTY_THRESHOLD
- **Value**: `0.3`
- **Units**: normalized [0,1]
- **Why**: 0.3 means very little change, indicates repetitive content.
- **Tuning**:
  - Use 0.4 for detecting more subtle stagnation
  - Use 0.2 for only very repetitive sections

### STAGNANT_FATIGUE_THRESHOLD
- **Value**: `0.7`
- **Units**: normalized [0,1]
- **Why**: 0.7 means high self-similarity and low variance, strong indicator of problematic repetition.
- **Tuning**:
  - Use 0.8 for very strict stagnation
  - Use 0.6 for earlier detection

### STAGNANT_MIN_DURATION_SEC
- **Value**: `8.0` seconds
- **Units**: seconds
- **Why**: 8 seconds filters brief repeated phrases, identifies sustained problems (~2-4 bars that feel "stuck").
- **Tuning**:
  - Use 16.0s for only very long stagnant sections
  - Use 4.0s for flagging shorter repetition

### BOUNDARY_CONFIDENCE_THRESHOLD
- **Value**: `0.5`
- **Units**: confidence [0,1]
- **Why**: Only report boundaries with >50% confidence, avoids over-segmentation.
- **Tuning**:
  - Use 0.7 for fewer, clearer boundaries
  - Use 0.3 for more boundary suggestions

### PEAK_MIN_DISTANCE_BLOCKS
- **Value**: `20` blocks = 10 seconds
- **Units**: blocks
- **Why**: Prevents detecting multiple peaks for same event. Drops typically 16-32 bars apart (~32-64 seconds at 120 BPM).
- **Tuning**:
  - Use 40 (20s) for sparse drop detection
  - Use 10 (5s) for dense electronic music

---

## Audio Preprocessing

### NORMALIZATION_METHOD
- **Value**: `'peak'`
- **Options**: 'peak', 'loudness'
- **Why**: Peak is simple, deterministic, sufficient for relative analysis. Scales to [-1, 1] based on max abs value.
- **Tuning**:
  - Use 'loudness' for RMS-based normalization (more perceptually uniform)

### RMS_TARGET_DB
- **Value**: `-20.0` dBFS
- **Units**: dB full scale
- **Why**: -20 dBFS is moderate level, leaves headroom, typical for analysis.
- **Tuning**: Only relevant if NORMALIZATION_METHOD = 'loudness'

### SILENCE_THRESHOLD_DB
- **Value**: `-60.0` dB
- **Units**: dB
- **Why**: -60 dB is well below musical content, catches true silence/noise floor.
- **Tuning**:
  - Use -40 dB for very quiet passages
  - Use -80 dB for only absolute silence

### TRIM_SILENCE
- **Value**: `False`
- **Why**: Preserves original timing, important for drop detection alignment. User can enable if needed.
- **Tuning**: Set True if leading/trailing silence is problematic

---

## Output Parameters

### SCHEMA_VERSION
- **Value**: `"1.0.0"`
- **Why**: Versioning allows future format changes while maintaining compatibility.

### TOP_N_DROPS
- **Value**: `10`
- **Why**: 10 is manageable for review, covers most real drops in typical tracks (most have 3-8).
- **Tuning**:
  - Use 5 for only top drops
  - Use 20 for comprehensive analysis

### PLOT_DPI
- **Value**: `150`
- **Units**: dots per inch
- **Why**: 150 DPI is good balance of quality and file size for screen viewing.
- **Tuning**:
  - Use 300 for print quality
  - Use 100 for smaller files

### PLOT_FIGSIZE
- **Value**: `(14, 10)` inches
- **Units**: inches (width, height)
- **Why**: 14x10 at 150 DPI = 2100x1500 pixels, provides detail while fitting typical screens.

### N_MFCC
- **Value**: `13`
- **Why**: 13 is standard (captures spectral envelope). First coefficient relates to energy, rest capture timbre.
- **Tuning**:
  - Use 20 for more timbre detail
  - Use 8 for faster processing

### SPECTRAL_ROLLOFF_PERCENT
- **Value**: `0.85` (85%)
- **Why**: Standard, represents frequency below which 85% of spectral energy is contained.

---

## Tuning Guidelines

### For Different Genres

**Electronic Dance Music (EDM)**:
```python
BLOCK_DURATION_SEC = 0.5
DROP_MIN_BUILD_DURATION_SEC = 2.0
TENSION_WEIGHTS = {'rms': 0.3, 'onset_density': 0.4, ...}  # Rhythmic focus
```

**Hip-Hop/Rap**:
```python
BLOCK_DURATION_SEC = 0.5
FATIGUE_WINDOW_BLOCKS = 16  # Shorter loops
STAGNANT_MIN_DURATION_SEC = 4.0  # Flag shorter repetition
```

**Progressive/Trance**:
```python
BLOCK_DURATION_SEC = 1.0  # Smoother curves
DROP_MIN_BUILD_DURATION_SEC = 8.0  # Longer builds
NOVELTY_LOOKBACK_BLOCKS = 32  # Broader context
```

**Classical/Ambient**:
```python
TENSION_WEIGHTS = {'rms': 0.2, 'spectral_centroid': 0.4, ...}  # Timbral focus
DROP_PROMINENCE_THRESHOLD = 0.7  # Fewer false positives
```

### For Different Use Cases

**High Precision (few false positives)**:
```python
DROP_PROMINENCE_THRESHOLD = 0.7
DROP_MIN_BUILD_SLOPE = 0.2
STAGNANT_FATIGUE_THRESHOLD = 0.8
```

**High Recall (catch all candidates)**:
```python
DROP_PROMINENCE_THRESHOLD = 0.3
DROP_MIN_BUILD_DURATION_SEC = 1.0
STAGNANT_NOVELTY_THRESHOLD = 0.4
```

**Fast Processing**:
```python
TARGET_SAMPLE_RATE = 16000
BLOCK_DURATION_SEC = 1.0
BLOCK_AGGREGATION_STATS = ['mean']  # Only mean
```

**Maximum Detail**:
```python
TARGET_SAMPLE_RATE = 44100
BLOCK_DURATION_SEC = 0.25
TENSION_SMOOTH_ALPHA = 0.5  # Less smoothing
```

---

## Parameter Interaction Effects

### Frame ↔ Block Duration
- Larger BLOCK_DURATION_SEC aggregates more frames → smoother but less detail
- Must ensure: `BLOCK_DURATION_SEC * TARGET_SAMPLE_RATE / HOP_LENGTH >= 1`

### Smoothing ↔ Detection
- More smoothing (lower TENSION_SMOOTH_ALPHA) → fewer, broader peaks
- Less smoothing → more peaks, may need higher DROP_PROMINENCE_THRESHOLD

### Window Size ↔ Sensitivity
- Larger NOVELTY_LOOKBACK_BLOCKS → less sensitive to local changes
- Larger FATIGUE_WINDOW_BLOCKS → detects longer-term patterns only

---

## Validation

All parameters are validated on config import:
- Weights must sum to 1.0
- Thresholds must be in [0, 1]
- Durations must be positive
- EWMA alpha must be in (0, 1]

To override parameters programmatically:
```python
import config
config.BLOCK_DURATION_SEC = 1.0  # Override for session
```

To override via CLI:
```bash
python cli.py track.wav --output results/ --block-duration 1.0
```

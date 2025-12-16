# DSP Kernel Specification v1.0

**This file is the authoritative reference for Phase 2 C++ implementation.**

## Overview

The DSP kernel computes three perceptual curves from audio:
- **Tension**: Perceived intensity/energy over time
- **Novelty**: Degree of change relative to recent context
- **Fatigue**: Repetition/stagnation indicator

All algorithms are deterministic, explainable, and suitable for real-time C++ implementation.

---

## Processing Pipeline

```
Audio Input (mono, float32, [-1.0, 1.0])
    │
    ▼
┌─────────────────────────────────────┐
│  STAGE 1: Frame-Level Features      │
│  - RMS energy                       │
│  - Spectral features (STFT-based)   │
│  - Onset strength (spectral flux)   │
│  - MFCCs (optional)                 │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│  STAGE 2: Block Aggregation         │
│  - Aggregate frames → blocks        │
│  - Compute statistics per block     │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│  STAGE 3: Feature Normalization     │
│  - Robust (median/IQR)              │
│  - Percentile scaling               │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│  STAGE 4: Curve Computation         │
│  - Tension: weighted combination    │
│  - Novelty: cosine distance         │
│  - Fatigue: leaky integrator        │
└─────────────────────────────────────┘
    │
    ▼
Output: Three curves, each [0, 1] range
```

---

## Function Reference

### 1. Frame-Level Feature Extraction

#### 1.1 `compute_rms_energy`

**Purpose**: Compute RMS (root-mean-square) energy per frame.

**Inputs**:
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| audio | float32[] | - | Audio samples, normalized [-1, 1] |
| frame_length | int | 2048 | Samples per frame |
| hop_length | int | 512 | Hop between frames |

**Output**: float32[n_frames] where `n_frames = 1 + (len(audio) - frame_length) // hop_length`

**Equation**:
```
rms[i] = sqrt(mean(audio[i*hop : i*hop + frame_length]^2))
```

**C++ Notes**: Simple loop, SIMD-vectorizable.

---

#### 1.2 `compute_stft`

**Purpose**: Compute Short-Time Fourier Transform.

**Inputs**:
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| audio | float32[] | - | Audio samples |
| frame_length | int | 2048 | FFT size |
| hop_length | int | 512 | Hop between frames |

**Output**: (magnitude[n_bins, n_frames], phase[n_bins, n_frames]) where `n_bins = frame_length/2 + 1`

**Notes**:
- Uses Hann window (implicit in scipy.signal.stft)
- Magnitude = |STFT|, Phase = angle(STFT)

**C++ Notes**: Use JUCE FFT or FFTW library.

---

#### 1.3 `compute_spectral_features`

**Purpose**: Compute spectral centroid, bandwidth, and rolloff.

**Inputs**:
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| audio | float32[] | - | Audio samples |
| sr | int | 22050 | Sample rate (Hz) |
| frame_length | int | 2048 | FFT size |
| hop_length | int | 512 | Hop size |
| rolloff_percent | float | 0.85 | Rolloff percentage |

**Outputs**:
- `spectral_centroid`: float32[n_frames] in Hz
- `spectral_bandwidth`: float32[n_frames] in Hz
- `spectral_rolloff`: float32[n_frames] in Hz

**Equations**:
```
freqs = [0, sr/(2*n_bins), 2*sr/(2*n_bins), ..., sr/2]  // n_bins frequency values
mag = STFT magnitude

centroid = sum(freqs * mag) / sum(mag)
bandwidth = sqrt(sum((freqs - centroid)^2 * mag) / sum(mag))
rolloff = freq where cumsum(mag) >= rolloff_percent * sum(mag)
```

---

#### 1.4 `compute_spectral_flux`

**Purpose**: Compute frame-to-frame magnitude change (onset indicator).

**Inputs**: Same as compute_stft

**Output**: float32[n_frames], values >= 0

**Equation**:
```
flux[0] = 0
flux[i] = sum(max(0, mag[:,i] - mag[:,i-1]))  // half-wave rectification
```

---

#### 1.5 `compute_zcr`

**Purpose**: Compute zero-crossing rate per frame.

**Output**: float32[n_frames], values in [0, 1]

**Equation**:
```
signs = sign(frame)  // -1, 0, +1
signs[signs == 0] = 1  // treat zero as positive
zcr = count(sign_changes) / (2 * frame_length)
```

---

#### 1.6 `compute_onset_strength`

**Purpose**: Compute onset/transient strength envelope.

**Implementation**: Uses spectral_flux as proxy.

**Output**: float32[n_frames], values >= 0

---

#### 1.7 `compute_mfcc_stats`

**Purpose**: Compute Mel-Frequency Cepstral Coefficients.

**Additional Parameters**:
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| n_mfcc | int | 13 | Number of coefficients |
| n_mels | int | 40 | Number of mel bands |

**Output**: float32[n_mfcc, n_frames]

**Algorithm**:
1. Compute magnitude spectrogram
2. Apply mel filterbank (triangular filters in mel scale)
3. Log compression: log(mel_spec + 1e-8)
4. DCT type-II to get cepstral coefficients

---

### 2. Block Aggregation

#### 2.1 `frames_to_blocks`

**Purpose**: Aggregate frame features into musical-time blocks.

**Inputs**:
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| feature_array | float32[] | - | Frame-level feature |
| sr | int | 22050 | Sample rate |
| frame_hop | int | 512 | Frame hop size |
| block_duration_sec | float | 0.5 | Block duration |
| agg_stat | string | 'mean' | Aggregation statistic |

**Output**: float32[n_blocks] where `n_blocks = n_frames // frames_per_block`

**Supported Statistics**: 'mean', 'median', 'std', 'p25', 'p75', 'min', 'max'

**Calculation**:
```
frames_per_block = (block_duration_sec * sr) // frame_hop
n_blocks = n_frames // frames_per_block

for i in 0..n_blocks:
    block_frames = feature[i*fpb : (i+1)*fpb]
    blocks[i] = agg_stat(block_frames)
```

---

### 3. Normalization

#### 3.1 `normalize_block_features`

**Purpose**: Normalize features to common scale.

**Methods**:

| Method | Equation | Output Range |
|--------|----------|--------------|
| robust | (x - median) / IQR | ~[-2, 2] |
| percentile | (x - p1) / (p99 - p1), clipped | [0, 1] |
| zscore | (x - mean) / std | ~[-3, 3] |

---

#### 3.2 `smooth_curve`

**Purpose**: Apply temporal smoothing.

**Methods**:

**EWMA (Exponential Weighted Moving Average)**:
```
alpha = 0.3  // lower = more smoothing
smoothed[0] = curve[0]
smoothed[i] = alpha * curve[i] + (1 - alpha) * smoothed[i-1]
```

**Moving Average**:
```
window = 3  // blocks
kernel = ones(window) / window
smoothed = convolve(curve, kernel)  // edge-padded
```

---

#### 3.3 `normalize_tension_robust`

**Purpose**: Normalize tension curve preserving internal contrast.

**Equation**:
```
p_low = percentile(curve, 5)
p_high = percentile(curve, 95)
normalized = clip((curve - p_low) / (p_high - p_low), 0, 1)
```

---

### 4. Curve Computation

#### 4.1 `compute_tension_curve`

**Purpose**: Compute perceived intensity/energy over time.

**Components** (default weights):
| Component | Weight | Description |
|-----------|--------|-------------|
| RMS | 0.4 | Loudness proxy |
| Onset density | 0.3 | Rhythmic impact |
| Spectral centroid | 0.2 | Brightness |
| Spectral bandwidth | 0.1 | Fullness |

**Algorithm (robust_track mode)**:
1. Normalize each component using percentile scaling (5th-95th)
2. Weighted combination: `tension = 0.4*rms + 0.3*onset + 0.2*centroid + 0.1*bandwidth`
3. Final normalization using percentile scaling
4. EWMA smoothing (alpha=0.3)

**Outputs**:
- tension_raw: [0, 1] unsmoothed
- tension_smooth: [0, 1] EWMA smoothed

---

#### 4.2 `compute_novelty_curve`

**Purpose**: Compute degree of change relative to recent context.

**Parameters**:
| Parameter | Default | Description |
|-----------|---------|-------------|
| lookback_blocks | 16 | Context window (8 seconds) |
| smooth_window | 3 | Moving average window |

**Algorithm**:
1. Z-score normalize block features (per column)
2. For each block i:
   - context = mean(features[i-lookback:i])
   - distance[i] = cosine_distance(features[i], context)
3. Normalize distances to [0, 1]
4. Apply moving average smoothing

**Cosine Distance**:
```
cosine_distance(a, b) = 1 - (a · b) / (|a| * |b|)
```

**Output**: float32[n_blocks], values in [0, 1]

---

#### 4.3 `compute_self_similarity_matrix`

**Purpose**: Compute how similar each block is to recent past (repetition indicator).

**Algorithm**:
1. Z-score features
2. For each block i:
   - Compute cosine similarity to each block in lookback window
   - similarity[i] = max(similarities)  // most similar = most repeated

**Output**: float32[n_blocks], values in [0, 1]. Higher = more repetitive.

---

#### 4.4 `compute_feature_variance`

**Purpose**: Compute rolling variance as stagnation indicator.

**Algorithm**:
1. For each block i:
   - window = features[i-window_size:i+1]
   - variance[i] = mean(var(window, axis=0))
2. Normalize to [0, 1]

**Output**: float32[n_blocks], values in [0, 1]. Lower = more stagnant.

---

#### 4.5 `compute_fatigue_leaky_integrator`

**Purpose**: Compute fatigue using perceptually-motivated model.

**Parameters**:
| Parameter | Default | Description |
|-----------|---------|-------------|
| gain_up | 0.02 | Accumulation rate |
| gain_down | 0.08 | Recovery rate (4x faster) |
| novelty_spike_threshold | 0.5 | Threshold for "interesting" |

**Boring Score Weights**:
| Component | Weight |
|-----------|--------|
| self_similarity | 0.5 |
| inverse_novelty | 0.3 |
| inverse_variance | 0.2 |

**Algorithm**:
```
fatigue[0] = 0
for t in 1..n_blocks:
    boring = 0.5*similarity[t] + 0.3*(1-novelty[t]) + 0.2*(1-variance[t])

    if novelty[t] > 0.5 or is_boundary[t]:
        # Interesting: recover quickly
        delta = -gain_down * (1 + novelty[t])
    else:
        # Boring: accumulate slowly
        delta = gain_up * boring

    fatigue[t] = clamp(fatigue[t-1] + delta, 0, 1)
```

**Output**: float32[n_blocks], values in [0, 1]

---

#### 4.6 `compute_fatigue_curve`

**Purpose**: Master fatigue computation function.

**Algorithm**:
1. Compute self_similarity using `compute_self_similarity_matrix`
2. Compute variance using `compute_feature_variance`
3. Compute fatigue using `compute_fatigue_leaky_integrator`
4. Apply moving average smoothing (window=5)

---

## State Management

### FatigueState Class

Only the fatigue leaky integrator requires persistent state between blocks.

```cpp
struct FatigueState {
    float fatigue_value = 0.0f;

    void reset() {
        fatigue_value = 0.0f;
    }
};
```

**For real-time processing**: Use `fatigue_leaky_integrator_step()` which takes FatigueState and updates it per block.

---

## Default Parameters

### Frame Parameters
| Parameter | Value | Rationale |
|-----------|-------|-----------|
| frame_length | 2048 | ~93ms at 22050 Hz |
| hop_length | 512 | ~23ms, 4x overlap |
| sample_rate | 22050 | Captures musical content up to 11kHz |

### Block Parameters
| Parameter | Value | Rationale |
|-----------|-------|-----------|
| block_duration_sec | 0.5 | ~2 blocks/beat at 120 BPM |

### Tension Parameters
| Parameter | Value |
|-----------|-------|
| rms_weight | 0.4 |
| onset_weight | 0.3 |
| centroid_weight | 0.2 |
| bandwidth_weight | 0.1 |
| smooth_alpha | 0.3 |
| percentile_lower | 5.0 |
| percentile_upper | 95.0 |

### Novelty Parameters
| Parameter | Value |
|-----------|-------|
| lookback_blocks | 16 (8 seconds) |
| smooth_window | 3 |

### Fatigue Parameters
| Parameter | Value |
|-----------|-------|
| window_blocks | 32 (16 seconds) |
| smooth_window | 5 |
| gain_up | 0.02 |
| gain_down | 0.08 |
| novelty_spike_threshold | 0.5 |

---

## What is NOT Part of the Kernel

The following are explicitly **excluded** from the kernel module:

1. **Audio I/O**
   - File loading/saving
   - Format conversion
   - Resampling

2. **Event Detection**
   - Drop detection
   - Stagnant segment detection
   - Section boundary detection
   - Peak finding with prominence

3. **Visualization**
   - Plot generation
   - Curve rendering

4. **Export**
   - JSON serialization
   - File writing

5. **CLI/Orchestration**
   - Argument parsing
   - Pipeline coordination

These belong in higher-level modules that consume kernel outputs.

---

## C++ Porting Guidelines

### Memory Management
- Pre-allocate all arrays based on known sizes at initialization
- Use `std::vector<float>` or `juce::AudioBuffer<float>`
- Avoid dynamic allocation during processing

### SIMD Opportunities
- RMS computation (dot product)
- STFT magnitude (vectorized abs)
- Spectral feature loops
- Normalization operations

### Thread Safety
- All functions are pure/stateless except `fatigue_leaky_integrator_step`
- FatigueState should be per-track or protected if shared

### Real-Time Adaptations
1. **Block-by-block processing**: Process one block at a time instead of full arrays
2. **Circular buffers**: For lookback windows (novelty, self-similarity)
3. **Pre-computed filterbanks**: Mel filterbank computed once at startup
4. **Incremental STFT**: Use overlap-add for streaming

### Numerical Precision
- Use float32 throughout (matches Python implementation)
- Add 1e-8 epsilon to denominators to avoid division by zero
- Clip outputs to valid ranges after computation

---

## Validation

Use golden reference outputs in `golden_outputs/` directory to validate C++ implementation:

1. Generate references: `python golden_reference.py --output golden_outputs/`
2. Load reference JSON
3. Run C++ kernel with same audio
4. Compare output checksums

Tolerance: Outputs should match within float32 precision (~1e-6 relative error).

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2025-12-12 | Initial extraction from director-signals |

# Architecture Documentation

## System Overview

director-signals is a modular, deterministic system for offline musical structure analysis. The architecture prioritizes:

1. **Modularity**: Each module has a single responsibility and minimal coupling
2. **Determinism**: Same input always produces identical output
3. **Explainability**: All algorithms use interpretable rules, no black boxes
4. **Testability**: Every component can be tested independently with synthetic audio

## Module Dependency Graph

```
cli.py
  ├─> audio_io (audio loading/preprocessing)
  ├─> features (frame-level extraction)
  ├─> aggregation (block-level aggregation)
  ├─> metrics (curve computation)
  ├─> events (event detection)
  └─> export (JSON/plot generation)

config.py (imported by all modules)
```

Modules are designed to be pipeline stages with minimal dependencies.

## Data Flow

```
Audio File
  ↓
[audio_io] → Preprocessed Audio (mono, resampled, normalized)
  ↓
[features] → Frame-Level Features (RMS, spectral, MFCCs, onset strength)
  ↓
[aggregation] → Block Features (statistics over musical-time windows)
  ↓
[aggregation] → Normalized Block Features
  ↓
[metrics] → Long-Horizon Curves (tension, novelty, fatigue)
  ↓
[events] → Detected Events (drops, stagnant segments, boundaries)
  ↓
[export] → JSON + Plots
```

## Module Descriptions

### config.py

**Purpose**: Central configuration with all tunable parameters

**Responsibilities**:
- Define frame/block/window sizes
- Set detection thresholds
- Specify curve computation weights
- Document rationale for each default

**Key Functions**:
- `validate_config()`: Ensures parameter consistency
- Helper functions for unit conversions

**Design Principles**:
- Every parameter has a documented "why"
- Type hints for all constants
- Validation on import to catch config errors early

---

### src/audio_io.py

**Purpose**: Audio loading and preprocessing

**Responsibilities**:
- Load audio from files (supports WAV, MP3, FLAC via librosa)
- Convert to mono
- Resample to target sample rate
- Normalize amplitude (peak or loudness-based)
- Optionally trim silence
- Record all preprocessing steps

**Key Functions**:
- `load_audio()`: Load from file with format detection
- `preprocess_audio()`: Full preprocessing pipeline
- `normalize_audio()`: Peak or RMS-based normalization
- `validate_audio()`: Sanity checks

**Design Principles**:
- Graceful fallback to scipy if librosa unavailable
- Deterministic: no random operations
- Complete metadata recording for reproducibility

**Edge Cases Handled**:
- Silent audio (zero RMS)
- DC offset in audio
- Stereo vs mono detection
- Sample rate mismatches

---

### src/features.py

**Purpose**: Frame-level feature extraction

**Responsibilities**:
- Compute Short-Time Fourier Transform (STFT)
- Extract spectral features (centroid, bandwidth, rolloff)
- Compute temporal features (RMS, ZCR, onset strength)
- Extract MFCCs for timbre representation
- Ensure all features aligned to same frame grid

**Key Functions**:
- `extract_all_features()`: Main entry point, returns all features
- `compute_spectral_features()`: Centroid, bandwidth, rolloff
- `compute_onset_strength()`: Transient density proxy
- `compute_mfcc_stats()`: Mel-frequency cepstral coefficients

**Design Principles**:
- Vectorized operations (no Python loops over frames)
- Consistent frame alignment across all features
- Fallback implementations when librosa unavailable

**Feature Descriptions**:
- **RMS Energy**: Loudness proxy, correlates with perceived volume
- **Spectral Centroid**: Brightness, weighted mean frequency
- **Spectral Bandwidth**: Fullness, spread around centroid
- **Spectral Rolloff**: 85% energy point, indicates high-frequency content
- **Spectral Flux**: Frame-to-frame spectral change, detects transients
- **Zero Crossing Rate**: Noisiness/percussiveness indicator
- **Onset Strength**: Transient density, rhythmic activity proxy
- **MFCCs**: Timbre/texture representation

---

### src/aggregation.py

**Purpose**: Convert frame-level features to block-level and smooth curves

**Responsibilities**:
- Aggregate frames into musical-time blocks
- Compute robust statistics per block (mean, median, std, percentiles)
- Normalize features to common scale
- Smooth curves (EWMA, moving average, Savitzky-Golay)

**Key Functions**:
- `frames_to_blocks()`: Time-based aggregation with statistics
- `aggregate_frame_features()`: Apply to all features, build feature matrix
- `normalize_block_features()`: Robust/percentile/z-score normalization
- `smooth_curve()`: Multiple smoothing methods

**Design Principles**:
- Time-aligned blocks (fixed duration or bar-based if BPM known)
- Robust statistics resistant to outliers
- Preserve curve length after smoothing

**Normalization Methods**:
- **Robust**: `(x - median) / IQR` - resistant to outliers
- **Percentile**: Scale to [0,1] using 1st/99th percentiles
- **Z-score**: `(x - mean) / std` - standard normalization

---

### src/metrics.py

**Purpose**: Compute long-horizon perceptual curves

**Responsibilities**:
- Compute tension/energy curve from weighted features
- Compute novelty as distance to recent context
- Compute self-similarity for fatigue detection
- Compute fatigue from similarity + low novelty + low variance
- Compute drop impact scores

**Key Functions**:
- `compute_tension_curve()`: Weighted combination of intensity features
- `compute_novelty_curve()`: Cosine distance to recent mean
- `compute_fatigue_curve()`: Combined repetition indicator
- `compute_drop_impact_scores()`: Pre/post contrast measurement
- `compute_all_curves()`: Main entry point

**Algorithm Details**:

**Tension Curve**:
1. Extract normalized components: RMS, onset density, spectral centroid/bandwidth
2. Weighted sum: 40% RMS + 30% onset + 20% centroid + 10% bandwidth
3. Normalize to [0, 1]
4. Smooth with EWMA (α=0.3)

**Novelty Curve**:
1. Z-score all block features
2. For each block, compute mean of recent N blocks (lookback window)
3. Compute cosine distance between current and recent mean
4. Normalize and smooth with moving average

**Fatigue Curve**:
1. Compute self-similarity: max cosine similarity to recent window
2. Invert novelty (high fatigue = low novelty)
3. Compute feature variance in window, invert (high fatigue = low variance)
4. Weighted combination: 50% similarity + 30% inverse novelty + 20% inverse variance
5. Normalize and smooth

**Drop Impact**:
1. Extract pre-drop window (4s before)
2. Extract post-drop window (4s after)
3. Compute feature deltas: RMS, onset strength, spectral centroid/bandwidth
4. Weighted sum of positive deltas as impact score

**Design Principles**:
- All curves in [0, 1] range for interpretability
- Intermediate signals exposed for debugging
- Lookback windows tuned for musical timescales

---

### src/events.py

**Purpose**: Detect musical events using heuristic rules

**Responsibilities**:
- Detect candidate drops from tension peaks
- Detect stagnant/over-looped segments
- Detect section boundaries
- Rank drops by combined detection + impact

**Key Functions**:
- `detect_candidate_drops()`: Tension peak + build + drop pattern
- `detect_stagnant_segments()`: Low novelty + high fatigue regions
- `detect_section_boundaries()`: Novelty peaks + tension changes
- `rank_drop_candidates()`: Combine confidence + impact scores

**Drop Detection Algorithm**:
1. Find tension peaks with prominence > threshold
2. For each peak, check:
   - **Build pattern**: Rising tension for ≥2s before peak
   - **Drop pattern**: Sustained energy for 4s after peak
3. Calculate confidence score:
   - 40% from prominence
   - 30% from build presence (+ bonus for slope)
   - 20% from drop pattern
   - Bonus from impact score if audio provided
4. Return ranked list

**Stagnant Segment Detection**:
1. Find contiguous blocks where novelty < 0.3 AND fatigue > 0.7
2. Filter segments < 8 seconds
3. Label by severity (novelty/fatigue levels)

**Design Principles**:
- All detections include rule breakdown for transparency
- Confidence scores explain which rules were satisfied
- Tunable thresholds for different sensitivity needs

---

### src/export.py

**Purpose**: Generate JSON outputs and visualizations

**Responsibilities**:
- Create versioned JSON schema
- Export metrics, summary, and segments
- Generate time-series plots with event markers
- Handle numpy → JSON conversion

**Key Functions**:
- `create_metrics_json()`: Complete analysis data
- `create_summary_json()`: Top-level statistics
- `export_all_outputs()`: Orchestrate all exports
- `plot_curves_and_events()`: Main visualization
- `plot_tension_components()`: Component breakdown

**JSON Schema**:
```json
{
  "schema_version": "1.0.0",
  "track_metadata": {...},
  "params": {...},
  "curves": {
    "tension_raw": {
      "values": [...],
      "sampling_interval_sec": 0.5,
      "start_time_sec": 0.25,
      "length": 120,
      "description": "...",
      "range": [0.0, 1.0]
    },
    ...
  },
  "events": {
    "candidate_drops": [...],
    "ranked_drops": [...],
    "stagnant_segments": [...],
    "boundaries": [...]
  }
}
```

**Plot Design**:
- 3 vertically stacked subplots (tension, novelty, fatigue)
- Shared time axis
- Drop candidates marked as vertical lines
- Stagnant segments shaded
- Section boundaries as dotted lines

**Design Principles**:
- Versioned schema for future compatibility
- All numeric arrays include metadata (sampling rate, start time, normalization)
- Non-interactive matplotlib backend for server use

---

### cli.py

**Purpose**: Command-line interface and orchestration

**Responsibilities**:
- Parse arguments
- Dispatch to single file / directory / demo mode
- Orchestrate full pipeline
- Print concise summaries
- Handle errors gracefully

**Key Functions**:
- `process_single_track()`: Full pipeline for one track
- `process_directory()`: Batch processing
- `run_demo_mode()`: Synthetic test tracks
- `main()`: CLI entry point

**Usage Modes**:
1. **Single file**: `python cli.py track.wav --output results/`
2. **Directory**: `python cli.py tracks/ --output results/`
3. **Demo**: `python cli.py --demo --output demo/`

**Design Principles**:
- Verbose flag for debugging
- Parameter overrides via CLI
- Fail gracefully with error messages
- Return non-zero exit code on failure

---

## Testing Strategy

### Synthetic Audio Generators

**generate_build_then_drop()**:
- First half: rising amplitude + increasing transient rate
- Second half: high amplitude + steady kick pattern
- Returns audio + expected drop time

**generate_repetitive_loop()**:
- 2-second melodic pattern repeated for duration
- Tests fatigue detection

**generate_section_contrast()**:
- Quiet simple melody → loud multi-harmonic
- Returns audio + transition time

### Test Coverage

- **Unit tests**: Each module independently
- **Integration tests**: Full pipeline end-to-end
- **Determinism tests**: Verify identical outputs
- **Performance tests**: 3-minute track < 30 seconds
- **Schema tests**: Validate JSON structure

### Test Principles

- No external audio files required
- Known ground truth for assertions
- Tolerance windows for time-based detections
- Deterministic (no randomness in tests or code)

---

## Design Patterns

### Pipeline Pattern

Each module is a stage that:
1. Takes standardized input (dict/array)
2. Performs transformation
3. Returns standardized output

Allows independent testing and swapping implementations.

### Configuration Object Pattern

`config.py` as single source of truth:
- All modules import config
- CLI can override via params dict
- Validation happens once on import

### Metadata Preservation

Every transformation records its parameters:
- Audio preprocessing steps
- Feature extraction settings
- Normalization parameters
- Detection thresholds

Enables reproducibility and debugging.

### Graceful Degradation

Optional features degrade gracefully:
- Librosa unavailable → use scipy (WAV only)
- Audio too short → skip or return empty
- Feature missing → use zeros

---

## Performance Considerations

### Memory

- Store features, not audio copies
- Use float32 instead of float64
- Delete intermediate arrays when done

### Computation

- Vectorized operations (numpy/scipy)
- No Python loops over frames
- STFT computed once, reused
- Block aggregation reduces data size early

### Scalability

- 3-minute track: ~15 seconds (target)
- 10-minute track: ~50 seconds
- Bottleneck: STFT computation and MFCC

---

## Future Extensions (Phase 2+)

### Real-time Processing

- Streaming frame aggregation
- Incremental curve updates
- JUCE integration

### Machine Learning

- Learned tension weights
- Drop classifier
- Fatigue regression model

### Advanced Features

- BPM detection for bar-aligned blocks
- Harmonic/percussive separation
- Key detection
- Genre-specific models

---

## Error Handling

### Input Validation

- File format checks
- Duration limits (config.MAX_TRACK_DURATION_SEC)
- Sample rate sanity checks
- NaN/infinity detection

### Graceful Failures

- Invalid audio → clear error message
- Missing features → use defaults
- Zero variance → avoid division by zero

### Debugging

- Verbose flag for progress tracking
- Intermediate signals in outputs
- Rule breakdowns for detections

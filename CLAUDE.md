# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

director-signals is an offline audio analysis pipeline that models long-horizon musical structure (tension, fatigue, impact) using deterministic, explainable algorithms. No machine learning, no real-time constraints, no plugins.

**Core Purpose**: Ingest audio tracks and output interpretable perceptual curves (tension/energy, novelty, repetition/fatigue, drop impact) plus detected events (candidate drops, stagnant sections, transitions).

## Development Commands

### Environment Setup
```bash
# Activate virtual environment
source myenv/bin/activate  # macOS/Linux
# or
myenv\Scripts\activate  # Windows

# Install dependencies (once implemented)
pip install -r requirements.txt
```

### Running Analysis
```bash
# Run on audio file
python cli.py <audio_file> --output <output_dir>

# Run on directory of files
python cli.py <input_dir> --output <output_dir>

# Run demo/synthetic test mode (no external files needed)
python cli.py --demo --output demo_output
```

### Testing
```bash
# Run all tests
pytest tests/

# Run synthetic audio tests specifically
pytest tests/test_synthetic.py

# Run with verbose output
pytest -v tests/
```

## Versioning

The system uses three version identifiers to track changes:

- **KERNEL_VERSION** (`config.py`): Algorithm version - bump when DSP logic changes
- **TIMEBASE_VERSION** (`config.py`, `src/timebase.py`): Block/time calculation version
- **SCHEMA_VERSION** (`config.py`): Output JSON format version

All versions are recorded in golden outputs and analysis results for reproducibility.

## Architecture

The codebase follows a modular pipeline structure with minimal coupling:

```
config.py              → All constants, weights, documented parameters, versions
src/
  audio_io.py         → Loading, resampling, mono conversion, normalization
  features.py         → Frame-level feature extraction (RMS, spectral features, MFCCs, onset strength)
  aggregation.py      → Frame → Block → Long-horizon curve conversion
  kernel.py           → Core DSP functions (portable to C++)
  kernel_params.py    → Kernel configuration dataclasses
  timebase.py         → Canonical time axis utilities
  metrics.py          → Deterministic curve algorithms (novelty, fatigue, drop impact)
  events.py           → Detection heuristics (drops, stagnant segments, boundaries)
  export.py           → JSON schema + plot generation
cli.py                 → Argument parsing and orchestration
golden_reference.py    → Golden output generation and validation
tests/
  test_synthetic.py   → Synthetic audio generation + assertion suite
  test_kernel.py      → Kernel isolation and determinism tests
  test_timebase.py    → Timebase calculation tests
```

## Key Concepts & Terminology

- **Frame**: Short analysis window (e.g., 2048 samples) with hop (e.g., 512 samples)
- **Block**: Aggregated window in musical time (e.g., 1 beat, 1 bar, or N seconds)
- **Long-horizon**: Minutes-scale context (3–5 minutes) computed from block sequences
- **Tension/Energy Curve**: Scalar per block capturing perceived intensity/drive
- **Novelty Curve**: Degree of change relative to recent context
- **Repetition/Fatigue Curve**: Sustained self-similarity / low novelty indicator
- **Drop Impact**: Contrast score between pre-drop and post-drop windows

## Output Structure

Per-track analysis generates an output directory containing:
- `metrics.json`: All curves, parameters, detected events
- `summary.json`: Top-level summary with key stats
- `plots.png`: Overlaid curves vs time with event markers
- `segments.json`: Detected low-novelty segments and candidate drops with timestamps

## Critical Design Constraints

1. **Determinism is mandatory**: Same input → same output across runs
   - Fix all random seeds (prefer no randomness)
   - Feature extraction must be reproducible
   - Record all parameters in outputs

2. **No ML/AI**: Pure signal processing and deterministic heuristics only
   - No PyTorch, TensorFlow, scikit-learn training
   - No embeddings or learned models
   - Explainable rules only

3. **No external files for tests**: Tests use synthetic audio generation
   - Must run anywhere without downloading audio
   - Synthetic tracks with known ground truth (builds, drops, loops)
   - Assert detectors behave correctly within tolerance windows

4. **Modular independence**: Each module should operate standalone
   - Features extracted without knowing about metrics
   - Metrics computed without knowing about event detection
   - Clear data contracts between modules

## Feature Extraction Pipeline

Frame-level features (computed per analysis window):
- RMS energy
- Spectral centroid, bandwidth, rolloff (85%)
- Spectral flux (frame-to-frame magnitude change)
- Zero crossing rate
- Onset strength envelope (transient density proxy)
- MFCC summary (13 coefficients, reducible to statistics)

Block aggregation: frames → robust statistics (mean, median, std, percentiles)

Long-horizon curves:
1. **Tension/Energy**: Weighted combination of normalized features (loudness, onset density, spectral)
2. **Novelty**: Distance between current block and recent context (cosine or L2 on z-scored vectors)
3. **Fatigue**: Self-similarity + low variance over sustained windows
4. **Drop Impact**: Contrast deltas (RMS, onset strength, spectral changes) across drop boundary

## Event Detection Logic

All detectors use explainable heuristic rules:

**Candidate Drops**:
- Tension curve peaks with prominence threshold
- Require pre-drop build (positive slope over minimum duration)
- Post-drop stabilization pattern
- Return confidence score based on rule satisfaction

**Stagnant/Over-looped Sections**:
- Novelty below threshold AND fatigue above threshold
- Sustained for ≥ K seconds/bars
- Output with average novelty/fatigue scores

**Section Boundaries** (optional):
- Novelty peaks + tension changes
- Low confidence unless clearly detected

## Configuration Philosophy

All tunables live in `config.py` with documentation:
- Frame/hop sizes
- Block duration (seconds or bar-based with BPM)
- Smoothing parameters (EWMA, moving average)
- Detection thresholds
- Weight combinations for tension curve
- Lookback windows for novelty/fatigue

Document why each default was chosen.

## Testing Strategy

Synthetic audio generators create controlled test signals:
1. **Build-then-drop**: Rising amplitude + transients → sudden bass/kick
2. **Repetitive loop**: Constant pattern sustained for long duration
3. **Section contrast**: Quiet verse → loud chorus transition

Assertions verify:
- Tension peaks near constructed drops
- Fatigue increases in repetitive regions
- Novelty spikes at designed transitions
- Drop detector returns events within tolerance windows
- JSON schema keys and array lengths are correct
- Performance: 3-minute track processed within time budget

## Golden Outputs

Golden outputs are versioned reference outputs used to validate the C++ Phase 2 port.

### Directory Structure
```
golden_outputs/
├── kernel_v{KERNEL_VERSION}/
│   ├── README.md
│   └── timebase_v{TIMEBASE_VERSION}/
│       ├── synthetic/
│       │   ├── build_drop/
│       │   │   ├── metrics.json
│       │   │   ├── segments.json
│       │   │   └── summary.json
│       │   ├── repetitive_loop/
│       │   └── contrast/
│       │
│       └── real_local/          # not committed (commercial audio)
│           ├── haunted/
│           └── kendrick/
```

### Generation Commands
```bash
# Generate synthetic golden outputs
python golden_reference.py --output golden_outputs/

# Generate from manifest (real audio files)
python golden_reference.py --manifest reference_manifest.json --output golden_outputs/

# Validate existing golden outputs
python golden_reference.py --validate --output golden_outputs/
```

### Reference Manifest

For testing with real (commercial) audio, create `reference_manifest.json` (not committed):
```json
{
  "tracks": [
    {
      "name": "track_name",
      "path": "/absolute/path/to/audio.wav",
      "start_sec": null,
      "end_sec": null,
      "bpm": null,
      "tags": ["genre"],
      "expected": {}
    }
  ]
}
```

The `real_local/` directory and `reference_manifest.json` are gitignored to avoid committing commercial audio references.

## JSON Schema Structure

Versioned output with:
```json
{
  "schema_version": "...",
  "track_metadata": {
    "duration": "...",
    "sample_rate": "...",
    "preprocessing": "..."
  },
  "params": { /* all tunables used */ },
  "curves": {
    "tension_raw": [...],
    "tension_smooth": [...],
    "novelty": [...],
    "fatigue": [...]
  },
  "events": {
    "candidate_drops": [
      {
        "time": "...",
        "score": "...",
        "rule_breakdown": { /* which conditions met */ },
        "component_deltas": { /* RMS, onset, spectral */ }
      }
    ],
    "stagnant_segments": [...],
    "boundaries": [...]
  }
}
```

## Dependencies

Standard Python scientific stack:
- numpy, scipy: Core numerical operations
- matplotlib: Plot generation
- librosa (optional): Audio loading convenience (document fallback)

Avoid heavy ML frameworks. Keep memory usage reasonable by storing features, not audio copies.

## Implementation Principles

1. **Read existing patterns first**: Before adding features, examine how similar modules work
2. **Test with synthetic audio**: All behavior should be verifiable without external files
3. **Document parameter choices**: Every default value needs a "why"
4. **Expose intermediate signals**: Don't hide computations; include them in JSON for debugging
5. **Normalize robustly**: Use median/IQR or percentile scaling, not just min-max
6. **Smooth appropriately**: Document smoothing rationale (balance responsiveness vs noise)

## Common Pitfalls to Avoid

- Don't use dynamic imports
- Don't monkey patch external libraries
- Don't create abstractions until you have 3+ similar use cases
- Don't add ML dependencies or trained models
- Don't make the system depend on external audio files for tests
- Don't sacrifice determinism for convenience
- Don't hide intermediate computations—expose them in outputs

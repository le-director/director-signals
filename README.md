# director-signals

A deterministic, offline audio analysis system for modelling long-horizon musical structure
(tension, novelty, fatigue, and impact) using explainable signal processing — no machine learning.

This repository contains a **Python reference implementation** used to prototype and inspect
the core signal-processing ideas behind *The Director*. It exists to make the signals
reproducible, inspectable, and open to critique.

## Overview

The system ingests full audio tracks and produces:

- **Perceptual curves**: tension / energy, novelty, repetition / fatigue, drop impact
- **Detected events**: candidate drops, stagnant sections, transition boundaries
- **Evaluation outputs**: interpretable JSON + visualisations

All algorithms are deterministic, explainable, and reproducible.
The same input will always produce the same output.

This code is **not** a real-time engine or DAW plugin. It is a research and prototyping artefact.

## Design Principles

- **No ML / AI** — pure signal processing and heuristic rules
- **Deterministic** — no randomness, no training data
- **Explainable** — every curve and event is traceable to inputs
- **Long-horizon** — minutes-scale structure, not frame-level effects
- **Testable** — complete synthetic test suite, no external audio required

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Basic Usage

```bash
# Run demo mode (no audio files needed)
python cli.py --demo --output demo_output/

# Analyse a single track
python cli.py track.wav --output results/

# Analyse all files in a directory
python cli.py tracks/ --output results/

# Verbose output
python cli.py track.wav --output results/ --verbose
```

### Running Tests

```bash
pytest tests/ -v
```

## Output Structure

For each analysed track:

```
output_dir/track_name/
├── track_name_metrics.json
├── track_name_summary.json
├── track_name_segments.json
├── track_name_plots.png
└── track_name_tension_components.png
```

## Core Concepts

### Time Scales

- **Frame**: short STFT window (2048 samples ≈ 93 ms @ 22050 Hz)
- **Block**: musically meaningful aggregation (0.5 s default)
- **Long-horizon**: minutes-scale analysis over block sequences

### Curves

1. **Tension / Energy**
   Composite measure of perceived intensity (RMS, onset density, spectral features).

2. **Novelty**
   Degree of change relative to recent context; highlights transitions and violations.

3. **Fatigue**
   Sustained predictability and low variance; highlights stagnation and over-looping.

4. **Drop Impact**
   Contrast between pre-drop build and post-drop energy persistence.

## Architecture

```
config.py          → Parameters and constants
src/
  audio_io.py      → Audio loading and preprocessing
  features.py      → Frame-level feature extraction
  aggregation.py   → Block aggregation and smoothing
  metrics.py       → Long-horizon curve computation
  events.py        → Event detection heuristics
  export.py        → JSON and plot generation
cli.py             → Command-line interface
tests/
  test_synthetic.py → Synthetic ground-truth tests
```

## Theoretical Context

After this system was built, its behaviour was found to closely align with
David Huron’s **ITPRA** framework (Imagination–Tension–Prediction–Reaction–Appraisal)
described in *Sweet Anticipation*.

This project does **not** claim to validate or prove the theory.
It represents an independent operationalisation that appears consistent with it.

**v1.0 is dedicated to the memory of David Huron.**

## Limitations

- Offline analysis only
- No real-time processing or DAW integration
- No machine learning
- Fixed block duration (no BPM inference)

## License

MIT

## Contact

Issues and discussion via GitHub.


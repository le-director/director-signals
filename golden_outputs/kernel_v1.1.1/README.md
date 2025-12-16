# Golden Reference Outputs

## Kernel Version: 1.1.1
## Timebase Version: 1
## Schema Version: 1.1.0

Generated: 2025-12-13T20:58:59.419981Z

## Audio Source

Synthetic tracks are loaded from committed fixtures in `fixtures/synthetic_audio/`.
The `audio_checksum` in each metrics.json is the SHA256 of the WAV file bytes.

## Directory Structure

```
kernel_v1.1.1/
├── timebase_v1/
│   ├── synthetic/
│   │   ├── build_drop/      # Rising build then high-energy drop (60s)
│   │   ├── repetitive_loop/ # Constant repeating pattern (60s)
│   │   └── contrast/        # Quiet verse to loud chorus (60s)
│   │
│   └── real_local/          # Real audio (not committed)
```

## Synthetic Test Tracks

All synthetic tracks are 60 seconds, loaded from `fixtures/synthetic_audio/`:

1. **build_drop**: Rising amplitude/transients (0-30s) then sudden drop (30-60s)
2. **repetitive_loop**: Constant 2-second A-major triad loop
3. **contrast**: Quiet 220Hz verse (0-30s) then loud multi-harmonic chorus (30-60s)

## Per-Track Outputs

- `metrics.json`: All curves, parameters, and intermediate values
- `segments.json`: Detected events (drops, stagnant segments, boundaries)
- `summary.json`: High-level summary statistics

## Validation

Run `python golden_reference.py --validate` to verify:
- audio_checksum matches fixture WAV SHA256
- time axis max <= duration
- curve lengths == canonical block count

# Changelog

## [4.0.0] — AOC v4

Renames the project to **AOC v4**, consolidates the voice-control, synthesis,
and DAW-integration surfaces into one product, and ships full technical and
integration documentation.

### Added
- `TECHNICAL.md` — full reference manual.
- `TECHNOLOGY.md` — engineering overview.
- `DAW_INTEGRATION.md` — end-to-end DAW guide with REAPER / Ableton / FL snippets.
- `daw --listen`, `daw --exec` subcommands.
- `daw` extra group in `pyproject.toml`.

### Changed
- Project name: `aoc-magic-v3` → `aoc-v4`.
- Version: `3.1.0` → `4.0.0`.
- Console banner and `--help` now read `AOC v4`.

### Unchanged
- Engine inventory from 3.1.0 (kokoro, piper, pyttsx3, xtts, f5 TTS; mkl, knn, none VC).
- Config schema, defense, rate limit, injection guard, wake-lock.

## [3.1.0] — Fix known limitations

XTTS, F5, and real MKL-VC integrated. See prior changelog for details.

## [3.0.0] — AOC Magic v3 unified release

Unified voice control + offline synthesis into one program.

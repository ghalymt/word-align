# WordAlign 2.0 — TODO / Backlog

The original feature modules are implemented. This pass focuses on integration
coverage and portable deployment; heavy model/backend installation is intentionally
deferred until a bundle is being prepared for distribution.

## High Priority

### Stage Caching ✅
**Files:** `wordalign/core/cache.py`, `wordalign/core/fingerprint.py`
Disk-based cache keyed by audio fingerprint + plugin/model/version/language.
Changing subtitle formatting (CPL) reuses ASR outputs — only segmentation re-runs.

### SQLite Project Store ✅
**File:** `wordalign/core/database.py`
Projects, jobs, job stages, QA issues, settings, and glossary persisted in
SQLite (WAL mode). Enables crash recovery and resumable jobs.

### GUI Backend Connection ✅
**File:** `wordalign/gui/server.py`
Local HTTP API (no external dependencies) serving hardware info, engines,
models, projects, jobs, glossary, cache stats, and pipeline execution.
Web GUI (`app.html`) connects automatically; `python -m wordalign --gui`.

### Thinned CLI (use PipelineRunner) ✅
**File:** `wordalign/cli_v2.py`
CLI delegates to `PipelineRunner` with `PrintSink`. `python -m wordalign`
runs cli_v2 only (2.1 removed the silent fallback to the v1 cli.py).

## Medium Priority

### Configurable Segmenter ✅
`segment.py` exposes `set_config(SegmentationConfig)` and `get_config()`.
`set_layout()` remains as a thin compatibility wrapper, and `PipelineRunner`
restores the prior layout after each job. All v1 tests pass.

### LLM QA Provider ✅
**Files:** `wordalign/qa/providers/__init__.py`
`LocalModelProvider` (transformers, lazy-loaded from local HF cache),
`OpenAICompatibleProvider` (any OpenAI-compatible endpoint), `NoOpProvider`.
Chunking + dedup already existed in `qa/engine.py`.

### Local Re-alignment (Edit → Surgical Repair) ✅
**File:** `wordalign/qa/realignment.py`
Edit detection via diff, anchor-based timestamp interpolation, surgical
replacement of only the affected words.

### Waveform Viewer ✅
**File:** `wordalign/gui/waveform.py`
Min/max peak generation (WAV direct, other formats via ffmpeg decode),
JSON output for canvas rendering with seek + boundary overlay.

## Low Priority

### Portable .exe Build ✅
**Files:** `build.spec`, `build.py`, `scripts/build_windows.ps1`, `scripts/Launch WordAlign.vbs`
PyInstaller one-folder build with bundled FFmpeg, a double-click launcher, and
optional model staging. Run `python build.py` or the PowerShell wrapper.

### PySide6 Native GUI ✅
**Files:** `wordalign/gui/native/` (__init__.py, main_window.py, workers.py)
Real Qt app: native file dialogs, QThread pipeline worker, progress bar,
event log, QA issue tree, model path config. `python -m wordalign.gui.native`.

### Diarization Plugin ✅
**File:** `wordalign/engines/adapters/diarization_adapter.py`
EnginePlugin with `Capability.DIARIZE`. Uses pyannote.audio when installed;
falls back to silence-gap pseudo-speaker assignment. The pipeline's
`--diarize` (2.1, `wordalign/diarize.py`) uses pyannote only, with no
heuristic fallback, and labels cues in SRT, WebVTT, ASS and transcripts.

### Glossary / Custom Vocabulary ✅
Schema + CRUD in `wordalign/core/database.py` (project-level and global).
Used by QA to identify likely misspellings.

### Benchmark / Calibration Mode ✅
**File:** `wordalign/benchmark.py`
WER (Levenshtein), runtime, peak memory (tracemalloc), timestamp coverage,
confidence distribution. `create_profile_from_benchmark()` writes a profile.

### MFA Multilingual Support ✅
`wordalign/engines/mfa_engine.py` — language→model map for en/ar/zh/fr/de/es/pt;
`MFAWrapper(language=...)` threads it through G2P + acoustic selection.

### Job Manifest (Output Provenance) ✅
**File:** `wordalign/core/manifest.py`
Version, input fingerprint, profile, engines/models, runtime versions,
hardware, stage durations, warnings, word stats.

### GPU Resource Scheduler ✅
**File:** `wordalign/core/gpu_scheduler.py`
Semaphore-limited GPU slots, OOM detection, recovery suggestions
(smaller chunks / CPU fallback / smaller model).

### Crash Recovery ✅
**File:** `wordalign/core/recovery.py`
Detects stale pending/running jobs in SQLite, reports last completed stage,
builds a resume plan (stage + cache key).

### Code-Switching / Per-Word Language Detection ✅
**File:** `wordalign/qa/codeswitch.py`
Script detection (Arabic/CJK/Cyrillic/Greek/Hebrew) + langid when available
+ English stopword heuristic; sliding-window smoothing; per-word language
populates `WordResult.language`.

---

## Status Summary

| Area | Status |
| --- | --- |
| v1 alignment and segmentation tests | ✅ passing |
| v2 core/profile/plugin tests | ✅ passing |
| TODO module tests | ✅ passing |
| Pipeline/GUI integration tests | ✅ added |
| Full frozen/backend smoke test | ⏳ requires heavy local backends |


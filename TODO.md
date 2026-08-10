# WordAlign 2.0 — TODO / Backlog

Features designed but not yet implemented, ordered by priority.

## High Priority

### Stage Caching
**Status:** Architecture designed, not implemented
**What:** Cache outputs for every expensive stage (Vosk, WhisperX, Qwen, consensus, alignment). Changing `max_cpl` from 42→32 should reuse all ASR outputs and only re-run segmentation.
**Cache key components:** audio fingerprint, plugin ID, plugin version, model ID, model revision, language, engine settings hash.
**Files to create:** `wordalign/core/cache.py`, `wordalign/core/fingerprint.py`
**Estimated effort:** 2–3 days

### SQLite Project Store
**Status:** Schema designed in `docs/proposal.md` (Section 9), not implemented
**What:** Persist projects, jobs, job stages, QA issues, glossary, and settings in SQLite. Enables crash recovery and resumable jobs.
**Files to create:** `wordalign/core/database.py`, migration scripts
**Estimated effort:** 2 days

### GUI Backend Connection
**Status:** HTML GUI exists as static prototype; no Python backend
**What:** Connect the web GUI to the Python pipeline via a local HTTP API (Flask/FastAPI). The GUI should be able to: select files, run PipelineRunner, display progress events, show QA issues, play audio.
**Files to create:** `wordalign/gui/server.py`, API endpoints
**Estimated effort:** 3–5 days

### Thinned CLI (use PipelineRunner)
**Status:** PipelineRunner implemented; CLI still contains original orchestration
**What:** Refactor `cli.py` to call `PipelineRunner.run()` instead of duplicating the pipeline flow. The CLI should translate `PipelineEvent`s to terminal output via `PrintSink`.
**Risk:** Must preserve identical output files. Add golden-output test before refactoring.
**Estimated effort:** 1 day

## Medium Priority

### Segmenter Global Removal
**Status:** `SegmentationConfig` class exists; `segment.py` still uses module-level globals (`_max_cpl`, `_max_lines`, etc.)
**What:** Refactor `segment.py` functions to accept `SegmentationConfig` as a parameter instead of reading globals. Makes multi-job operation safe.
**Risk:** High — touches many call sites. All 23 v1 tests must pass unchanged.
**Estimated effort:** 1–2 days

### LLM QA Provider
**Status:** Interface defined (`qa/engine.py`), chunking logic implemented, no real LLM wired
**What:** Implement `LocalModelProvider` (llama.cpp) and `OpenAICompatibleProvider`. Wire up the chunking, deduplication, and caching.
**Files to create:** `wordalign/qa/providers/local.py`, `wordalign/qa/providers/openai_compat.py`
**Estimated effort:** 2–3 days

### Local Re-alignment (Edit → Surgical Repair)
**Status:** Design complete in `docs/proposal.md` (Phase 10), not implemented
**What:** When the user edits transcript text in the review editor, identify the edited span, extend the audio region slightly, keep reliable neighboring timestamps as anchors, rerun local alignment, and replace only affected timestamps.
**Estimated effort:** 3–4 days

### Waveform Viewer
**Status:** GUI placeholder in place; no peak generation
**What:** Generate cached min/max peak data from audio. Render in the GUI with clickable seeking, subtitle boundary overlays, and QA issue markers.
**Files to create:** `wordalign/gui/waveform.py` (peak generation), JS canvas rendering
**Estimated effort:** 2 days

## Low Priority

### Portable .exe Build
**Status:** Not started
**What:** PyInstaller one-folder deployment for Windows. Bundle FFmpeg. No Python installation required by end users.
**Files to create:** `build.spec`, `build.py`, `scripts/build_windows.ps1`
**Estimated effort:** 1–2 days

### PySide6 Native GUI
**Status:** HTML GUI delivered as interim; PySide6 stubs created but empty
**What:** Native Qt application with proper threading, system tray, native file dialogs. Replace HTML GUI for production desktop use.
**Estimated effort:** 5–7 days

### Diarization Plugin
**Status:** `WordResult.speaker` field exists; no diarization engine
**What:** Implement a diarization plugin (e.g., pyannote.audio) that assigns speaker IDs. Display in transcript view as `SPEAKER 1:` / `SPEAKER 2:`.
**Estimated effort:** 3–4 days

### Glossary / Custom Vocabulary
**Status:** SQLite schema designed; not implemented
**What:** Project-level and global glossary. Used by QA to identify likely misspellings and avoid falsely flagging valid names.
**Estimated effort:** 1–2 days

### Benchmark / Calibration Mode
**Status:** Designed in proposal; not implemented
**What:** Run selected engines on test audio with a reference transcript. Report WER, runtime, peak memory, timestamp coverage, confidence distribution. Allow "Create profile from benchmark".
**Estimated effort:** 2–3 days

### MFA Multilingual Support
**Status:** `english_us_arpa` hardcoded
**What:** Thread a language parameter through `MFAWrapper` to support non-English forced alignment. Requires language-specific G2P and acoustic model selection.
**Estimated effort:** 1 day

### Job Manifest (Output Provenance)
**Status:** `PipelineResult` has `job_manifest_path` field; not populated
**What:** Write `job_manifest.json` with WordAlign version, input fingerprint, pipeline profile, engines, models, runtime versions, stage durations, warnings, and timestamp source statistics.
**Estimated effort:** 0.5 days

### GPU Resource Scheduler
**Status:** Not started
**What:** Prevent launching multiple large GPU models simultaneously. Detect OOM and offer retry with smaller chunks, CPU fallback, or smaller model.
**Estimated effort:** 2 days

### Crash Recovery
**Status:** Depends on SQLite project store
**What:** Save job/project state frequently. On restart, detect incomplete jobs and offer to continue from last completed stage.
**Estimated effort:** 2 days (after SQLite)

### Code-Switching / Per-Word Language Detection
**Status:** `WordResult.language` field exists; no detection logic
**What:** Allow language detection per span rather than assuming one language for the entire recording.
**Estimated effort:** 2–3 days

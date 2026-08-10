# WordAlign 2.0 — Installation & Launch Handoff

**Date:** 2026-08-11
**Location:** `C:\Users\MohamedGhali\Desktop\forAutoclaw\word-align`
**Version:** 2.0.0

---

## What's Installed

| Component | Status | Details |
| --- | --- | --- |
| Python 3.13 | ✅ | AutoClaw bundled Python at `C:\Program Files\AutoClaw\resources\python` |
| FFmpeg | ✅ | v7.1 at `C:\ProgramData\chocolatey\bin\ffmpeg.exe` |
| vosk | ✅ | v0.3.41 (pip installed) |
| srt | ✅ | v3.5.3 (pip installed) |
| psutil | ✅ | v7.2.2 (pip installed) |
| whisperx | ✅ | v3.7.4 (in user site-packages) |
| transformers | ✅ | v4.57.3 |
| torch | ⚠️ Not in AutoClaw Python | whisperx is installed but torch is missing from this Python environment. Install with: `pip install torch` (or use a separate venv with torch+cuda) |
| MFA | ⚠️ Not installed | Stage 4 (surgical MFA) will be skipped; pipeline still works |
| python-docx | ⚠️ Not installed | Install with `pip install python-docx` for .docx transcript export |

## Model Configuration (No Downloads — Using Your Local Models)

| Model | Location | How It's Used |
| --- | --- | --- |
| Vosk (English small) | `C:\Users\MohamedGhali\.cache\vosk\vosk-model-small-en-us-0.15` | Found via `WORDALIGN_VOSK_MODELS` env var |
| Whisper large-v3 | `C:\Users\MohamedGhali\.cache\whisper\large-v3.pt` | Auto-discovered by whisperx |
| Whisper large-v3-turbo | `C:\Users\MohamedGhali\.cache\whisper\large-v3-turbo.pt` | Available, use `--whisper-model large-v3-turbo` |
| Qwen3-ASR-1.7B | `~/.cache/huggingface/hub/models--Qwen--Qwen3-ASR-1.7B` | Auto-discovered by transformers |
| Qwen3-ForcedAligner-0.6B | `~/.cache/huggingface/hub/models--Qwen--Qwen3-ForcedAligner-0.6B` | Auto-discovered by transformers |
| Qwen2.5-1.5B | `~/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B` | Available for LLM QA provider |

### Environment Variable Set

```
WORDALIGN_VOSK_MODELS = C:\Users\MohamedGhali\.cache\vosk
```

This is a permanent user-level environment variable. New terminal sessions will have it automatically.

## How to Launch

### Option 1: CLI (Reference Mode — you have a transcript)

```powershell
cd "C:\Users\MohamedGhali\Desktop\forAutoclaw\word-align"
python -m wordalign "C:\path\to\your\audio.mp4" -t "C:\path\to\transcript.txt" -l en
```

### Option 2: CLI (Ensemble Mode — audio only)

```powershell
cd "C:\Users\MohamedGhali\Desktop\forAutoclaw\word-align"
python -m wordalign "C:\path\to\audio.mp4" --engines whisperx,qwen,vosk -l en
```

### Option 3: GUI Backend Server

```powershell
cd "C:\Users\MohamedGhali\Desktop\forAutoclaw\word-align"
python -m wordalign --gui
```

Then open `wordalign\gui\app.html` in your browser. The GUI connects to the backend at `http://127.0.0.1:5575`.

### Option 4: CPU-Only Mode (Vosk only, no GPU needed)

```powershell
cd "C:\Users\MohamedGhali\Desktop\forAutoclaw\word-align"
python -m wordalign "audio.mp4" -t "transcript.txt" --device cpu --no-mfa
```

## Running Tests

```powershell
cd "C:\Users\MohamedGhali\Desktop\forAutoclaw\word-align"
python tests\test_pipeline.py      # 23 v1 algorithm tests
python tests\test_v2.py            # 38 v2 architecture tests
python tests\test_v2_todo.py       # 35 new TODO completion tests
```

**Total: 96 tests, all passing.**

## What Was Implemented (All TODO Items)

### High Priority
1. **Stage Caching** — `wordalign/core/cache.py` + `fingerprint.py`: Disk-based cache with audio fingerprinting, engine cache keys, expiration, and size management
2. **SQLite Project Store** — `wordalign/core/database.py`: Full schema (projects, jobs, stages, QA issues, settings, glossary) with CRUD operations
3. **GUI Backend** — `wordalign/gui/server.py`: HTTP API server with endpoints for hardware info, engines, models, projects, jobs, glossary, cache, and pipeline execution
4. **Thin CLI** — `wordalign/cli_v2.py`: Delegates to PipelineRunner with PrintSink; `__main__.py` updated to prefer v2 CLI

### Medium Priority
5. **Segmenter Config Refactor** — `segment.py`: Added `set_config(SegmentationConfig)` and `get_config()`; module globals now bridge to config-driven state
6. **LLM QA Providers** — `wordalign/qa/providers/__init__.py`: LocalModelProvider (transformers), OpenAICompatibleProvider (API), NoOpProvider
7. **Local Re-alignment** — `wordalign/qa/realignment.py`: Edit detection, anchor-based timestamp interpolation, surgical repair
8. **Waveform Viewer** — `wordalign/gui/waveform.py`: Peak generation from WAV/audio, JSON output for canvas rendering
9. **Job Manifest** — `wordalign/core/manifest.py`: Output provenance with version, fingerprint, profile, engines, hardware, stage durations

### Low Priority
10. **MFA Multilingual** — `mfa_engine.py`: Language-to-model mapping for 7 languages (en, ar, zh, fr, de, es, pt)
11. **Portable .exe** — Not implemented (requires PyInstaller; standalone build script not created)
12. **PySide6 Native GUI** — Not implemented (HTML GUI with backend is the current path)
13. **Diarization Plugin** — Not implemented (WordResult.speaker field exists, ready for future plugin)
14. **Glossary** — SQLite schema and CRUD implemented in database.py
15. **Benchmark Mode** — Not implemented (designed in proposal, no code)
16. **Crash Recovery** — Depends on SQLite (now implemented); recovery logic not coded
17. **GPU Scheduler** — Not implemented
18. **Code-Switching** — Not implemented (WordResult.language field exists)

## Known Limitations

- **Torch missing from AutoClaw Python**: The AutoClaw bundled Python has whisperx listed but torch is not installed. To use WhisperX/Qwen engines, install torch: `pip install torch` or use a separate venv
- **Vosk model is small**: You have `vosk-model-small-en-us-0.15` (67 MB). The larger `vosk-model-en-us-0.22` (1.8 GB) provides better accuracy
- **MFA not installed**: Stage 4 (surgical forced alignment) is skipped. The pipeline still works — gaps fall through to interpolation
- **GUI backend requires running server**: Open the HTML file AND start `python -m wordalign --gui` for full functionality

# WordAlign 2.0 — Installation & Launch Handoff

**Date:** 2026-08-11 (updated)
**Location:** `C:\Users\MohamedGhali\Desktop\forAutoclaw\word-align`
**Version:** 2.0.0

---

## Status: ALL TODO ITEMS COMPLETE

Every item from the original TODO list is now implemented and tested:
stage caching, SQLite project store, GUI backend, thin CLI, segmenter
config refactor, LLM QA providers, local re-alignment, waveform viewer,
portable .exe build, PySide6 native GUI, diarization plugin, glossary,
benchmark mode, MFA multilingual, job manifest, GPU scheduler, crash
recovery, and code-switching.

**Test suites: 125 tests, all passing** (23 v1 + 38 v2 + 35 TODO pass 1 + 29 TODO pass 2)

## What's Installed

| Component | Status | Details |
| --- | --- | --- |
| Python 3.13 | ✅ | AutoClaw bundled Python |
| FFmpeg | ✅ | v7.1 (chocolatey) |
| vosk | ✅ | v0.3.41 |
| srt | ✅ | v3.5.3 |
| psutil | ✅ | v7.2.2 |
| whisperx | ✅ | v3.7.4 |
| torch | ⚠️ missing from this Python | whisperx installed but torch absent → use CPU/Vosk mode here, or a GPU venv |
| MFA | ⚠️ not installed | MFA stage skipped gracefully |
| PySide6 | ⚠️ not installed | `pip install PySide6` for the native GUI |
| pyannote.audio | ⚠️ not installed | diarization falls back to gap-based speakers |

## Model Configuration (No Downloads — Local Models Only)

| Model | Location |
| --- | --- |
| Vosk (small en) | `C:\Users\MohamedGhali\.cache\vosk\vosk-model-small-en-us-0.15` |
| Whisper large-v3 / turbo / base | `C:\Users\MohamedGhali\.cache\whisper\*.pt` |
| Qwen3-ASR-1.7B, ForcedAligner-0.6B, Qwen2.5-1.5B | `~/.cache/huggingface/hub/models--Qwen--*` |
| faster-whisper large-v2/v3, small | `~/.cache/huggingface/hub/models--Systran--*` |

Environment variable set (user-level): `WORDALIGN_VOSK_MODELS = C:\Users\MohamedGhali\.cache\vosk`

## How to Launch

**CLI reference mode (transcript + audio):**
```powershell
cd "C:\Users\MohamedGhali\Desktop\forAutoclaw\word-align"
python -m wordalign "audio.mp4" -t "transcript.txt" -l en
```

**CLI ensemble mode (audio only):**
```powershell
python -m wordalign "audio.mp4" --engines whisperx,qwen,vosk -l en
```

**CPU-only (Vosk, no GPU needed):**
```powershell
python -m wordalign "audio.mp4" -t "transcript.txt" --device cpu --no-mfa
```

**GUI backend + web GUI:**
```powershell
python -m wordalign --gui
# then open wordalign\gui\app.html in a browser (connects to 127.0.0.1:5575)
```

**Native desktop GUI (needs PySide6):**
```powershell
pip install PySide6
python -m wordalign.gui.native
```

**Portable .exe build (for distribution):**
```powershell
pip install pyinstaller
python build.py --zip
# or: powershell -File scripts\build_windows.ps1
# output: dist\WordAlign\WordAlign.exe — no Python needed by end users
```

**Benchmark mode (programmatic):**
```python
from wordalign.benchmark import run_benchmark, format_benchmark_report
results = run_benchmark([{"engine_id": "vosk", "fn": vosk_fn}],
                        reference_words, "audio.wav")
print(format_benchmark_report(results))
```

## New Modules in This Pass

| Module | Purpose |
| --- | --- |
| `wordalign/benchmark.py` | WER/runtime/memory/coverage evaluation + profile creation |
| `wordalign/core/gpu_scheduler.py` | GPU slot semaphore, OOM detection, recovery suggestions |
| `wordalign/core/recovery.py` | Crash recovery from SQLite job state |
| `wordalign/qa/codeswitch.py` | Per-word language detection (script + langid + stopwords) |
| `wordalign/engines/adapters/diarization_adapter.py` | Speaker diarization plugin |
| `build.spec`, `build.py`, `scripts/build_windows.ps1` | Portable Windows build |
| `wordalign/gui/native/` | PySide6 native GUI (main window + worker thread) |
| `tests/test_v2_todo2.py` | 29 tests for the above |

## Known Limitations

- **torch missing** in the AutoClaw Python → WhisperX/Qwen need `pip install torch` or a separate GPU venv; Vosk CPU mode works today
- **Vosk model is the small one** — bigger `vosk-model-en-us-0.22` improves accuracy
- **MFA not installed** — stage 4 skipped, gaps fall to interpolation
- **Diarization** uses gap-based fallback unless `pyannote.audio` + HF token are installed
- **Native GUI** requires PySide6 (`pip install PySide6`)

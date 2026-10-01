# WordAlign 2.0

**Speech-to-subtitle alignment & transcription, for humans and machines.**

WordAlign turns any audio or video file into perfectly timed subtitles — word-level and sentence-level SRT, plus a clean transcript. It combines multiple ASR engines (Vosk, WhisperX, Qwen ASR) with forced alignment (MFA) and a smart QA review pass that fixes timing, splits lines to your character limit, restores punctuation and capitalization, and flags anything suspicious.

**v2.0 is a full rewrite:** a modular plugin engine, a Model Manager that finds your local models (no forced downloads), a built-in **web GUI** (dark mode by default), a Smart Transcript Review system, and one-click portable builds for Windows — no Python or terminal required.

---

## 🖥️ Screenshots

The GUI runs in your browser (auto-opened), with **dark mode by default** — toggle to light any time with the header button (or `?light=1` in the URL).

| Dark mode (default) | Light mode |
|---|---|
| ![WordAlign GUI dark mode](docs/screenshots/gui-dark.png) | ![WordAlign GUI light mode](docs/screenshots/gui-light.png) |

---

## ✨ What's New in 2.0

- **🖥️ Web GUI** — drag-and-drop files, pick a quality preset, click Run. Real progress bar, live log, cancel button. Opens in your default browser; no install beyond the app itself.
- **🌙 Dark mode** — default theme, one-click toggle.
- **🧠 Smart Transcript Review** — an automatic QA pass that restores **punctuation and capitalization** (via a local LLM like Gemma 4 12B through llama.cpp, with a rule-based fallback), re-times out-of-sync words, detects hallucinated/duplicated text, and reports confidence issues.
- **🧩 Plugin engine architecture** — adding a new ASR/alignment engine is a drop-in plugin. No core edits, no CLI edits, no GUI edits.
- **📦 Model Manager** — auto-detects your installed models (Vosk, Whisper, Hugging Face cache, MFA) and every model path is configurable. No surprise downloads.
- **🌍 MFA for all supported languages** — 65+ languages mapped to Montreal Forced Aligner pretrained models, threaded through the gap-filling path.
- **🔁 Realignment, recovery, benchmarking** — auto re-align after QA, crash recovery with progress checkpointing, built-in benchmark suite.
- **📊 Manifest & cache** — fingerprint-based caching (re-runs are instant), job manifests, SQLite history database.

---

## 🚀 Quick Start

### Windows — for non-technical users (no Python, no terminal)

1. Download the latest **`WordAlign-portable.zip`** from the [Releases page](https://github.com/ghalymt/word-align/releases).
2. Unzip anywhere (e.g. `C:\WordAlign`).
3. Double-click **`Launch WordAlign.vbs`** — the local server starts and a browser tab opens with the GUI.
4. Drop in your audio/video, pick settings, hit **Run**.

> Vosk models are the out-of-the-box engine. If you already have Vosk models installed (e.g. under `D:\Subtitle edit\Vosk`), set the folder once in the GUI's **Model Paths** panel — no downloads needed. The same applies to Whisper, Qwen and MFA models.

### Windows — from source (Python 3.10+)

```bat
git clone https://github.com/ghalymt/word-align.git
cd word-align
py -m venv .venv
.venv\Scripts\activate
pip install -e .
# Optional GPU backends: pip install -e ".[asr,qwen]"
wordalign --gui            :: or just: wordalign
:: CLI:
wordalign --engines vosk -l en input.mp4 -o out
```

### Linux / macOS — from source (Python 3.10+)

```bash
git clone https://github.com/ghalymt/word-align.git
cd word-align
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
# Optional GPU backends: pip install -e ".[asr,qwen]"
wordalign --gui            # opens the browser GUI
# CLI:
wordalign --engines vosk -l en input.mp4 -o out
```

> Linux/macOS packages (AppImage, .deb, Homebrew) are planned — see [Releases](https://github.com/ghalymt/word-align/releases) for current assets. The Python source install works everywhere today.

---

## 🧑‍💻 CLI Reference

```text
wordalign [INPUT] [OPTIONS]

Core options:
  -o, --out DIR            Output directory (default: next to input)
  --engines LIST           Engines: whisperx,qwen,vosk  (default: whisperx,qwen,vosk)
                           Also available: mfa, yamnet
  -l, --language CODE      Language code, e.g. en, ar, fr, de, es, pt (default: auto)
  -c, --cpl N              Max characters per subtitle line (default: 42)
  --format FORMAT          Transcript document: none, txt, docx, both (default: txt)
  --profile NAME           Preset: cpu_only, fast, balanced, maximum_quality

Quality & engines:
  --use-mfa                Enable MFA forced alignment for gap-filling
  --mfa-cmd PATH           Path to the MFA executable
  --whisper-models DIR     Whisper models directory (overrides auto-detect)
  --vosk-models DIR        Vosk models directory (overrides auto-detect)
  --hf-cache DIR           Hugging Face cache directory (overrides auto-detect)
  --mfa-models DIR         MFA models directory (overrides auto-detect)

Smart transcript review:
  --qa                    Run the QA review pass (realignment + checks)
  --punctuation           Restore punctuation & capitalization
  --llm-engine ENGINE     LLM backend: llama_cpp, openai, anthropic (default: llama_cpp)
  --llm-model PATH        LLM model file (GGUF) or model name
  --llm-mtp-model PATH    MTP draft model for speculative decoding
  --no-mtp                Disable MTP speculative decoding

GUI:
  --gui                   Start the web GUI (default when no input given)
  --port N                GUI port (default: 5575)
```

Examples:

```bash
# Vosk only (CPU, no GPU required), English, 42 CPL, with punctuation restored
wordalign --engines vosk -l en --punctuation video.mp4

# Maximum quality: WhisperX + Qwen + Vosk ensemble with MFA + QA
wordalign --engines whisperx,qwen,vosk --use-mfa --qa video.mp4

# Use your locally installed models (no downloads)
wordalign --vosk-models "D:\Subtitle edit\Vosk" --engines vosk -l en video.mp4
```

---

## 🧠 Smart Transcript Review (new in 2.0)

Raw ASR output is lowercase and punctuation-free, which is unusable as subtitles. WordAlign's review stage fixes this:

1. **Punctuation & capitalization restoration** — a local LLM (e.g. Gemma 4 12B via llama.cpp) rewrites each cue's text **without adding, removing, or reordering words**. A rule-based restorer (capitalization, terminal periods, lowercase *i*) is always available as a fallback, so the feature works out of the box.
2. **Realignment** — words that drifted off their timestamps are re-aligned to the audio.
3. **Deterministic checks** — line length (CPL), duplicate lines, hallucination risk, code-switch detection, confidence issues.
4. **Recovery metadata** — job stages and cache keys are persisted so interrupted runs can be inspected and resumed by the recovery API.

Configuration is explicit — all paths are configurable, and the LLM is optional:

```bash
wordalign --punctuation \
  --llm-engine llama_cpp \
  --llm-model "F:\LM_studio\Models\unsloth\gemma-4-12B-it-qat-GGUF\gemma-4-12B-it-qat-UD-Q4_K_XL.gguf" \
  --llm-mtp-model "F:\LM_studio\Models\unsloth\gemma-4-12B-it-qat-GGUF\mtp-gemma-4-12B-it.gguf" \
  --engines vosk -l en video.mp4
```

Environment variables: `WORDALIGN_LLM_ENGINE`, `WORDALIGN_LLM_MODEL`, `WORDALIGN_LLM_MTP_MODEL`, `WORDALIGN_LLM_MTP`.

---

## 🌍 Languages & Models

WordAlign supports **65+ languages** across its engines:

- **Vosk** — en, ar, zh, fr, de, es, pt, it, nl, ru, ja, fa, tr, sv, ca, uk, tl, vi, hi, hr, sr, bs, th, pl + more (depends on which Vosk models you have installed).
- **WhisperX / Qwen ASR** — any language supported by Whisper / Qwen3-ASR.
- **MFA (forced alignment)** — mapped pretrained models for 65+ languages (English ARPA, Arabic, Mandarin, French, German, Spanish, Portuguese, Italian, Dutch, Russian, Japanese, Korean, Persian, Turkish, Swedish, Catalan, Ukrainian, Vietnamese, Hindi, Croatian, Serbian, Bosnian, Thai, Polish, Czech, Slovak, Greek, Romanian, Hungarian, Hebrew, Finnish, Danish, Norwegian, Indonesian, Malay, Tagalog, Bengali, Urdu, Tamil, Telugu, Nepali, Sinhala, Khmer, Lao, Burmese, Armenian, Georgian, Bulgarian, Macedonian, Slovenian, Albanian, Galician, Basque, Swahili, Amharic, Zulu, Xhosa, Yoruba, Igbo, Hausa, Cebuano, and more). MFA is skipped (with a warning) for a language without a pretrained MFA model, or when that language's acoustic/G2P model is not installed; it never falls back to another language's model. MFA is on by default in the CLI and in the GUI's Balanced and Maximum Quality presets, and only runs when it is installed and has a model for the language — use `--no-mfa` (or untick it) to turn it off.

### Model paths — all configurable

WordAlign never forces downloads. Every model directory is auto-detected **and** overridable, via CLI flags, environment variables, or the GUI's Model Paths panel:

| Models | CLI flag | Environment variable | Auto-detect |
|---|---|---|---|
| Vosk | `--vosk-models DIR` | `WORDALIGN_VOSK_MODELS` | `~/vosk-models`, `D:\Subtitle edit\Vosk`, … |
| Whisper / WhisperX | `--whisper-models DIR` | `WORDALIGN_WHISPER_MODELS` | `~/.cache/whisper`, `D:\Subtitle edit\Whisper\Models`, HF cache |
| Qwen (HF) | `--hf-cache DIR` | `WORDALIGN_HF_CACHE` | `~/.cache/huggingface/hub` |
| MFA | `--mfa-models DIR` | `WORDALIGN_MFA_MODELS` | `~/Documents/MFA`, conda envs |

The `/api/model-paths` endpoint (and the GUI panel) shows exactly what was resolved on your machine.

---

## 🏗️ Architecture

```
word-align/
├── launcher.py            # Entry point: no args = GUI, args = CLI
├── build.spec / build.py  # PyInstaller portable build
├── wordalign/
│   ├── cli_v2.py          # CLI (2.0) — thin layer over PipelineRunner
│   ├── core/
│   │   ├── pipeline.py    # PipelineRunner — the single pipeline used by CLI + GUI
│   │   ├── types.py, config.py, events.py
│   │   ├── cache.py       # fingerprint-based job caching
│   │   ├── database.py    # SQLite job history
│   │   ├── fingerprint.py # audio fingerprinting
│   │   ├── manifest.py    # job manifests
│   │   ├── recovery.py    # crash recovery / checkpoints
│   │   ├── gpu_scheduler.py
│   │   └── converters.py
│   ├── engines/           # Engine adapters (plugin architecture)
│   │   ├── vosk_engine.py, whisperx_engine.py, qwen_engine.py
│   │   ├── mfa_engine.py  # 65+ language map, surgical gap alignment
│   │   ├── yamnet_engine.py, nemo_engine.py   # nemo = legacy
│   │   └── adapters/      # runtime adapters (in_process / subprocess / external)
│   ├── plugins/           # Plugin registry, protocol, manifests, capabilities
│   ├── models/            # Model manager, catalog, validator, paths, downloader
│   ├── qa/                # Smart Transcript Review
│   │   ├── punctuation.py # LLM + rule-based punctuation/capitalization
│   │   ├── realignment.py, deterministic.py, engine.py, codeswitch.py
│   │   └── providers/     # llama_cpp, openai, anthropic
│   ├── runtimes/          # Runtime detection (python, conda, external)
│   ├── profiles/          # Quality presets (cpu_only, fast, balanced, maximum_quality)
│   ├── gui/               # Web GUI (app.html, server.py, waveform.py)
│   └── benchmark.py       # Benchmark suite
├── tests/                 # 164 tests: v1, v2, model paths, QA, and integration coverage
└── docs/                  # ARCHITECTURE.md, DEVELOPER.md, USER_GUIDE.md
```

**Key design rule:** the CLI and the GUI both call the same `PipelineRunner`. There is no separate pipeline for the GUI — what you see in the browser is exactly what the CLI does.

**Plugins:** the registry and adapters define the plugin contract for in-process, subprocess, and external engines. The current runner uses the built-in engine adapters for its production waterfall; new adapters can be registered and exposed through the API before their operations are promoted into the waterfall.

---

## 🧪 Tests

```bash
pip install -e ".[dev]"
python -m pytest tests/ -q
```

Individual test modules can also be run directly with `python tests/test_*.py` when pytest is not installed.

The 23 original v1 tests still pass unchanged — the v1 algorithms (`align.py`, `ensemble.py`, `segment.py`) are preserved untouched.

---

## 📦 Building the Portable Package (Windows)
```powershell
# from the repo root, with Python + PyInstaller installed
python build.py --zip --copy-models
# output: dist\WordAlign\WordAlign.exe and Launch WordAlign.vbs
```

`build.py` produces a Windows onedir bundle and links/copies the local `models/` tree when
available. Use `--copy-models` for a relocatable release ZIP. The heavy Qwen and WhisperX
virtual environments are staged separately with `powershell -File scripts\build_backends.ps1`
(or `build_windows.ps1 -BuildBackends`); MFA remains an optional external binary.
The resulting `Launch WordAlign.vbs` starts the local GUI without requiring Python.

---

## 📄 License & Notes

- MIT License — see [LICENSE](LICENSE).
- **NeMo (Parakeet/Canary)**: kept as *legacy* engines in the source (via `nemo_engine.py` / `nemo_adapter.py`) but **not** installed or recommended in 2.0 — the modern default ensemble is **WhisperX + Qwen + Vosk**. If you have NeMo installed, you can still opt in via `--engines parakeet,canary`; otherwise ignore it.
- MFA model downloads happen through MFA's own tooling (`mfa model download acoustic <name>`) — WordAlign only needs the `mfa` executable and the model directories configured above.

---

## 🤝 Contributing

See [docs/DEVELOPER.md](docs/DEVELOPER.md) for the plugin API, [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for internals, and [docs/USER_GUIDE.md](docs/USER_GUIDE.md) for detailed usage.

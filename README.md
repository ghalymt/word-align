# word-align

**Word-accurate subtitle timing from any transcript + audio — or from audio alone.**

`word-align` produces word-level and broadcast-quality sentence-level SRT files by
cascading multiple speech engines, each one only touching what the previous stage
couldn't solve. The alignment and segmentation logic comes out of thousands of
hours of real professional captioning work across 7+ languages.

Version 2.0 adds a **modular plugin architecture**, a **model manager**, a **GUI**,
and a **Smart Transcript Review** system — while preserving every algorithm and
all test coverage from v1.0.

---

## Quick Start (5 Minutes)

### The 30-second version

```bash
git clone https://github.com/ghalymt/word-align
cd word-align
pip install -r requirements.txt
pip install vosk
python -m wordalign audio.mp4 -t transcript.txt --vosk-models /path/to/vosk-models
```

You get two SRT files: `audio_word_level.srt` and `audio_sentence_level.srt`.

### What you need before you start

| Requirement | Why | How to get it |
| --- | --- | --- |
| **Python 3.10+** | Runs word-align | [python.org](https://www.python.org/downloads/) — check "Add to PATH" on Windows |
| **FFmpeg** | Decodes audio/video | See below |
| **NVIDIA GPU (optional)** | Makes WhisperX/Qwen 10–50× faster | CPU mode works, just slower |

### Installing FFmpeg

**Windows (easiest):**
1. Download from [gyan.dev](https://www.gyan.dev/ffmpeg/builds/) — get the "essentials" zip
2. Extract it somewhere, e.g. `C:\ffmpeg`
3. Add `C:\ffmpeg\bin` to your PATH:
   - Press `Win + S`, type "environment variables"
   - Click "Edit the system environment variables"
   - Click "Environment Variables"
   - Under "User variables", find "Path", click Edit
   - Click New, paste `C:\ffmpeg\bin`
   - Click OK on all three windows
4. Open a new terminal and verify: `ffmpeg -version`

**macOS:**
```bash
brew install ffmpeg
```

**Linux:**
```bash
sudo apt install ffmpeg      # Debian/Ubuntu
sudo dnf install ffmpeg      # Fedora
```

### Step-by-step install

```bash
# 1. Download the code
git clone https://github.com/ghalymt/word-align
cd word-align

# 2. Create a virtual environment (keeps things clean)
python -m venv venv

# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# 3. Install the core package
pip install -r requirements.txt

# 4. Install the engines you want (pick one or more)
pip install vosk                         # Vosk — fast, CPU-only, great timestamps
pip install python-docx                  # For .docx transcript export
pip install textgrid                     # For MFA forced alignment

# 5. Check that everything is ready
python preflight.py --lang en
```

The `preflight.py` script tells you exactly what's installed and what's missing.
It will never fail on optional components — only on things that truly block a run.

### Two ways to use word-align

**Way 1 — You have a transcript (Reference Mode):**

If you already have a verbatim text transcript of the audio, word-align will
time-align every single word to the audio:

```bash
python -m wordalign interview.mp4 -t interview.txt --vosk-models ~/vosk-models
```

**Way 2 — You only have audio (Ensemble Mode):**

No transcript? Word-align builds one by making multiple AI engines vote on
what was said:

```bash
python -m wordalign interview.mp4 --engines whisperx,qwen,vosk -l en
```

### Getting Vosk models (for the fastest engine)

Vosk is the fastest engine and provides the best timestamps. You need to
download a model for your language:

1. Go to [alphacephei.com/vosk/models](https://alphacephei.com/vosk/models)
2. Download the model for your language (e.g. `vosk-model-en-us-0.22` for English, ~1.8 GB)
3. Extract it to a folder, e.g. `C:\vosk-models\` on Windows or `~/vosk-models/` on Mac/Linux
4. Point word-align at it:

```bash
# Option A: pass the path each time
python -m wordalign audio.mp4 -t transcript.txt --vosk-models C:\vosk-models

# Option B: set it once and forget about it
# Windows (Command Prompt):
set WORDALIGN_VOSK_MODELS=C:\vosk-models
# Windows (PowerShell):
$env:WORDALIGN_VOSK_MODELS = "C:\vosk-models"
# macOS/Linux:
export WORDALIGN_VOSK_MODELS=/home/me/vosk-models
```

### Getting the GPU engines (optional, for ensemble mode)

WhisperX and Qwen3-ASR are GPU-based engines that provide higher accuracy.
They install automatically with `pip install -r requirements.txt` (which
installs WhisperX + PyTorch). Qwen needs its own environment:

```bash
# Create a separate venv for Qwen (it needs a different PyTorch version)
python -m venv qwen-venv

# Activate it:
# Windows:
qwen-venv\Scripts\activate
# macOS/Linux:
source qwen-venv/bin/activate

# Install Qwen
pip install qwen-asr

# Tell word-align where to find it:
# Windows:
set WORDALIGN_QWEN_PYTHON=C:\path\to\qwen-venv\Scripts\python.exe
# macOS/Linux:
export WORDALIGN_QWEN_PYTHON=/path/to/qwen-venv/bin/python
```

### Opening the GUI

The GUI is a web page — just open it in your browser:

```
word-align/wordalign/gui/app.html
```

Double-click it, or:
```bash
# Windows:
start wordalign\gui\app.html
# macOS:
open wordalign/gui/app.html
# Linux:
xdg-open wordalign/gui/app.html
```

The GUI shows you the Home page with three options: create subtitles from
audio, align an existing transcript, or review a completed project.

> **Note:** The GUI is currently a visual prototype. It demonstrates the full
> interface but doesn't yet connect to the Python backend. Use the CLI for
> real processing. The backend connection is on the [TODO list](TODO.md).

---

## What's new in 2.0

| Feature | Status | What it does |
| --- | --- | --- |
| PipelineRunner | ✅ Done | One API for both CLI and GUI — no more CLI-only pipeline |
| Plugin architecture | ✅ Done | Add engines without touching existing code |
| Engine adapters | ✅ Done | All 6 engines wrapped behind a clean interface |
| Subprocess protocol | ✅ Done | Run engines with conflicting dependencies in separate processes |
| Model manager | ✅ Done | Download, validate, and manage models from the GUI |
| Hardware detector | ✅ Done | Detects GPU, VRAM, CUDA, FFmpeg automatically |
| Smart QA | ✅ Done | Flags suspicious transcript regions (3 signals) |
| Pipeline profiles | ✅ Done | 4 presets: Fast, Balanced, Best Quality, CPU Only |
| Web GUI | ✅ Done | 7-page app: Home, Project, Review, Models, Engines, Pipelines, Diagnostics |
| Typed data structures | ✅ Done | WordResult, SegmentResult replace untyped dicts |
| Developer docs | ✅ Done | Plugin API, extension guide, architecture reference |
| Stage caching | 📋 Planned | Reuse ASR results when only subtitle formatting changes |
| SQLite project store | 📋 Planned | Save/resume jobs, crash recovery |
| GUI backend | 📋 Planned | Connect web GUI to Python pipeline |
| Portable .exe | 📋 Planned | No Python installation needed |
| LLM review | 📋 Planned | AI reads transcript and flags semantic anomalies |
| Local re-alignment | 📋 Planned | Edit text → surgically fix timestamps |
| Waveform viewer | 📋 Planned | Audio visualization in the review editor |

Full backlog with effort estimates: [TODO.md](TODO.md)

---

## How it works

```
                        ┌─────────────────────────────────────────────┐
  audio ──┬──────────►  │  ALIGNMENT WATERFALL (cheapest first)       │
          │             │                                             │
transcript┤             │  1. Vosk (parallel CPU)   ~90% of words     │
 (or none:│             │  2. rough SRT (optional)  cheap extra hits  │
 ensemble │             │  3. WhisperX + wav2vec2   GPU precision     │
 builds   │             │  4. MFA "surgical mode"   gaps only         │
 one)     │             │  5. interpolation         the last few      │
          │             └──────────────┬──────────────────────────────┘
          │                            ▼
          │             ┌─────────────────────────────────────────────┐
          └───────────► │  3-PHASE ITERATIVE SEGMENTER                │
                        │  sentence merging → CPL/duration limits →   │
                        │  clause-aware line balancing (7 languages)  │
                        └──────────────┬──────────────────────────────┘
                                       ▼
                        ┌─────────────────────────────────────────────┐
                        │  SMART QA (v2.0)                            │
                        │  agreement + repetition + timing + (LLM)    │
                        └──────────────┬──────────────────────────────┘
                                       ▼
                  name_word_level.srt   name_sentence_level.srt
                        transcript.txt/.docx   audio_tags.srt
```

**Why a waterfall?** No single engine solves word timing well. Vosk is fast
but misses words; WhisperX is precise but expensive; MFA is the gold standard
but impractical on a full recording. The waterfall runs each engine **only on
what remains unsolved**, so you get MFA-grade precision at a fraction of the
cost. In production, the first CPU pass typically covers ~90% of words before
the GPU is ever touched.

---

## CLI examples

```bash
# Reference mode: you have a verbatim transcript
python -m wordalign interview.mp4 -t interview.txt --vosk-models ~/vosk-models

# Ensemble mode: audio only, engines vote on the transcript
python -m wordalign interview.mp4 --engines whisperx,qwen,vosk -l en

# With a Word transcript deliverable (highlights uncertain words)
python -m wordalign interview.mp4 -l en --doc docx

# Vertical/social video subtitles (32 chars per line)
python -m wordalign reel.mp4 -t reel.txt --max-cpl 32

# Force CPU only (no GPU)
python -m wordalign interview.mp4 -t interview.txt --device cpu

# Check what's installed
python preflight.py --vosk-models /path/to/vosk-models --lang en
```

Run `python -m wordalign --help` to see all options.

---

## Output files

| File | What's in it |
| --- | --- |
| `*_word_level.srt` | One subtitle per word — maximum precision |
| `*_sentence_level.srt` | Merged, balanced, broadcast-quality cues |
| `*_transcript.txt` | Ensemble transcript with `[HH:MM:SS]` timestamps |
| `*_transcript.docx` | Same, but low-confidence words are highlighted yellow |
| `*_audio_tags.srt` | `[laughs]`, `[music]` style event track (`--tags`) |
| `*_combined.srt` | Dialogue + events merged (`--tags`) |

---

## Smart Transcript Review (v2.0)

After alignment, the QA engine flags suspicious regions — it does NOT silently
rewrite them. The human reviewer is always the final authority.

| Signal | What it finds | Needs |
| --- | --- | --- |
| Ensemble disagreement | Words where engines disagree | Ensemble mode |
| Repetition | Repeated phrases (possible hallucination) | Any mode |
| Timing anomaly | Impossible speaking rate, overlaps, interpolation runs | Any mode |
| LLM semantic review | Phrases that don't make sense in context | Optional LLM |

Each flagged issue has a severity (low/medium/high), a category, a
human-readable explanation, and a risk score with component breakdown.

```python
from wordalign.qa.engine import QAEngine
from wordalign.core.converters import dicts_to_words

words = dicts_to_words(aligned_words)
engine = QAEngine(enable_llm=False)
issues = engine.review(words)

for issue in issues:
    print(f"[{issue.severity}] {issue.category} at {issue.start:.1f}s: {issue.explanation}")
```

---

## Adding a new engine (v2.0 plugin system)

Adding an engine takes ~20 lines of code and requires **zero changes** to
existing files:

```python
from wordalign.plugins.base import EngineDescriptor, EnginePlugin, HealthStatus
from wordalign.plugins.capabilities import Capability
from wordalign.plugins.registry import PluginRegistry

class MyEngineAdapter(EnginePlugin):
    def descriptor(self):
        return EngineDescriptor(
            engine_id="my_engine",
            display_name="My Engine",
            version="1.0",
            capabilities=Capability.TRANSCRIBE | Capability.VOTE,
            runtime_type="in_process",
            supported_languages=["en"],
            models=["my-model-v1"],
            hardware={"cpu": True, "cuda": True},
        )

    def health_check(self):
        return HealthStatus(ready=True, runtime_status="ready")

    def transcribe(self, request):
        words = my_engine.transcribe(request.audio_path)
        return {"words": words, "events": [], "engine_id": "my_engine"}

# Register — done. It appears in Engines, Models, and Pipeline pages automatically.
registry = PluginRegistry()
registry.register(MyEngineAdapter())
```

Full guide: [docs/developer.md](docs/developer.md)

---

## Tests

```bash
python tests/test_pipeline.py     # 23 v1 tests (algorithm regression)
python tests/test_v2.py           # 38 v2 tests (new architecture)
# Or:
python -m pytest tests/ -q
```

No GPU, audio, or model weights required. **61 tests, all passing.**

---

## Project layout

```
word-align/
├── wordalign/
│   ├── cli.py, align.py, ensemble.py, segment.py   # v1 core (unchanged)
│   ├── config.py, document.py, tags.py, utils.py   # v1 support (unchanged)
│   ├── core/            # v2: PipelineRunner, types, events, config
│   ├── plugins/         # v2: EnginePlugin, Capability, Registry, Protocol
│   ├── engines/adapters/  # v2: Plugin wrappers for each engine
│   ├── models/          # v2: Model catalog, manager, downloader, validator
│   ├── runtimes/        # v2: Hardware + runtime detection
│   ├── qa/              # v2: Smart Transcript Review
│   ├── profiles/        # v2: Pipeline presets (JSON)
│   ├── data/            # v2: Model catalog
│   └── gui/             # v2: Web GUI
├── tests/               # 61 tests (23 v1 + 38 v2)
├── docs/                # Architecture, proposal, developer & user guides
├── preflight.py         # Installation checker
└── requirements.txt
```

---

## Known limitations

- **MFA forced alignment is English-only** (hardcoded `english_us_arpa`)
- **Ensemble mode defaults to English** if `-l` is not given
- **Vosk models are 1–2 GB each** — budget memory or lower `MAX_WORKERS`
- **The web GUI is a prototype** — use the CLI for real processing
- **Sentence integrity can override the duration cap** — this is deliberate

---

## License

MIT © 2026 [Mohamed Ghali](https://www.linkedin.com/in/mohammed-ghaly-subtitler)
— veterinarian turned speech-pipeline engineer; 50,000+ captioning projects
delivered.

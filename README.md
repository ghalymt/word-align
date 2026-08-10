# word-align

**Word-accurate subtitle timing from any transcript + audio — or from audio alone.**

`word-align` produces word-level and broadcast-quality sentence-level SRT files by
cascading multiple speech engines, each one only touching what the previous stage
couldn't solve. The alignment and segmentation logic comes out of thousands of
hours of real professional captioning work across 7+ languages; this repository
is that logic extracted into a documented, tested package.

Version 2.0 introduces a **modular plugin architecture**, a **model manager**,
a **GUI application**, and a **Smart Transcript Review** system — while
preserving every algorithm and all test coverage from v1.0.

---

## What's new in 2.0

| Feature | Status | Description |
| --- | --- | --- |
| **PipelineRunner** | ✅ Implemented | Single orchestration API callable by CLI and GUI — no more CLI-only pipeline |
| **Plugin architecture** | ✅ Implemented | `EnginePlugin` ABC, `Capability` flags, `PluginRegistry`, manifest validation |
| **Engine adapters** | ✅ Implemented | All 6 engines wrapped behind the plugin interface |
| **Subprocess protocol** | ✅ Implemented | Versioned JSON-line protocol for isolated engine execution |
| **Model manager** | ✅ Implemented | Catalog, download, validate, delete, external path registration |
| **Hardware detector** | ✅ Implemented | OS, CPU, RAM, GPU, VRAM, CUDA, FFmpeg detection |
| **Smart QA** | ✅ Implemented | 3 deterministic signals: agreement, repetition, timing anomalies |
| **Pipeline profiles** | ✅ Implemented | 4 presets: fast, balanced, maximum_quality, cpu_only |
| **Web GUI** | ✅ Implemented | 7-page HTML app: Home, Project, Review, Models, Engines, Pipelines, Diagnostics |
| **Typed data structures** | ✅ Implemented | WordResult, SegmentResult, EngineResult replace anonymous dicts |
| **Developer documentation** | ✅ Implemented | Module contract, plugin registration guide, extension points |
| **Stage caching** | 📋 Planned | Architecture designed, not yet implemented |
| **SQLite project store** | 📋 Planned | Schema designed, not yet implemented |
| **Portable .exe build** | 📋 Planned | PyInstaller spec not yet created |
| **LLM QA provider** | 📋 Planned | Interface defined, chunking implemented, no real LLM wired |
| **Local re-alignment** | 📋 Planned | Edit transcript → surgical timestamp repair (design complete) |
| **Waveform viewer** | 📋 Planned | Cached peaks rendering (GUI placeholder in place) |
| **PySide6 native GUI** | 📋 Planned | HTML GUI delivered as interim solution |

See [TODO.md](TODO.md) for the full backlog with priorities.

---

## Architecture

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

### v2.0 architecture (new)

```
    CLI ─────┐
             │
    GUI ─────┼──> PipelineRunner (core/pipeline.py)
             │         │
             │         ├──> Plugin Registry (plugins/registry.py)
             │         ├──> Engine Adapters (engines/adapters/)
             │         ├──> Model Manager (models/manager.py)
             │         ├──> QA Engine (qa/engine.py)
             │         └──> Existing algorithms (align.py, ensemble.py, segment.py)
             │
    future API┘
```

The CLI and GUI both call `PipelineRunner.run()`. There is no separate "GUI
pipeline" and "CLI pipeline". The runner emits structured events; each frontend
translates them into its output format.

See [docs/architecture.md](docs/architecture.md) for the full v1.0 architecture
reference and [docs/proposal.md](docs/proposal.md) for the v2.0 design document.

---

## Install

### Prerequisites

- **Python 3.10+**
- **FFmpeg/FFprobe** on your system PATH (required for audio decoding)
- An **NVIDIA GPU** with CUDA is strongly recommended (CPU mode works but is
  10–50× slower for WhisperX and Qwen)

### Step-by-step

```bash
# 1. Clone the repository
git clone https://github.com/ghalymt/word-align
cd word-align

# 2. Create a virtual environment (recommended)
python -m venv venv

# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# 3. Install core dependencies
pip install -r requirements.txt

# 4. Install optional engines (pick what you need)
pip install vosk                         # Vosk: fast CPU engine / ensemble voter
pip install "nemo_toolkit[asr]"          # NeMo: legacy Parakeet/Canary voters
pip install python-docx                  # .docx transcript export
pip install textgrid                     # MFA TextGrid parsing
pip install tensorflow tensorflow-hub    # YAMNet audio-event tagging (experimental)

# 5. Verify your installation
python preflight.py --lang en
```

The preflight script tells you exactly what is installed, what is missing, and
whether each component is required or optional. It checks CUDA/VRAM, verifies
`ffprobe` actually executes, and exits non-zero only when something genuinely
blocks a run.

### Qwen voter (ensemble mode)

Qwen3-ASR ships its own torch, which usually differs from the WhisperX stack's,
so word-align calls it out-of-process:

```bash
# Create a separate venv for Qwen
python -m venv qwen-venv

# Windows:
qwen-venv\Scripts\activate
# macOS/Linux:
source qwen-venv/bin/activate

pip install qwen-asr

# Point word-align at it:
# Windows:
set WORDALIGN_QWEN_PYTHON=C:\path\to\qwen-venv\Scripts\python.exe
# macOS/Linux:
export WORDALIGN_QWEN_PYTHON=/path/to/qwen-venv/bin/python
```

### Vosk models

Download from [alphacephei.com/vosk/models](https://alphacephei.com/vosk/models)
and point word-align at the directory:

```bash
python -m wordalign audio.mp4 -t transcript.txt --vosk-models /path/to/vosk-models
```

Or set the environment variable:
```bash
# Windows:
set WORDALIGN_VOSK_MODELS=C:\path\to\vosk-models
# macOS/Linux:
export WORDALIGN_VOSK_MODELS=/path/to/vosk-models
```

### Montreal Forced Aligner (optional, stage 4)

Install from [montreal-forced-aligner.readthedocs.io](https://montreal-forced-aligner.readthedocs.io/)
and either put `mfa` on PATH or:

```bash
# Windows:
set WORDALIGN_MFA=C:\path\to\mfa\or\conda\env
# macOS/Linux:
export WORDALIGN_MFA=/path/to/mfa/or/conda/env
```

---

## Use

### Command line (CLI)

```bash
# Reference mode: you have a verbatim transcript
python -m wordalign interview.mp4 -t interview.txt --vosk-models ~/vosk-models

# Ensemble mode: audio only, engines vote on the transcript
python -m wordalign interview.mp4 --engines whisperx,qwen,vosk -l en

# Ensemble mode with a Word transcript deliverable
python -m wordalign interview.mp4 -l en --doc docx

# Extra timing source + experimental audio-event tags
python -m wordalign film.mkv -t film.txt --srt rough_cut.srt --tags

# Force CPU (e.g. no CUDA available, or debugging a GPU issue)
python -m wordalign interview.mp4 -t interview.txt --device cpu

# Vertical/social video subtitles (32 chars per line instead of 42)
python -m wordalign reel.mp4 -t reel.txt --max-cpl 32

# Check what's installed before running
python preflight.py --vosk-models /path/to/vosk-models --lang en
```

Run `python -m wordalign --help` for the full flag list.

### GUI (web interface)

Open `wordalign/gui/app.html` in any web browser. The GUI provides:

- **Home** — quick start wizard
- **New Project** — file picker, quality preset, language selection
- **Review** — transcript view with issue highlighting
- **Models** — download/validate/manage models
- **Engines** — enable/disable engines, view health status
- **Pipelines** — configure engine weights, timing order, QA settings
- **Diagnostics** — hardware info, system report

The GUI is a self-contained HTML file — no server needed. In a future release,
it will connect to the Python backend via a local API.

### Using pipeline profiles (v2.0)

```python
from wordalign.core.config import PipelineProfile

# Load a preset
profile = PipelineProfile.preset("balanced")

# Load from a JSON file
profile = PipelineProfile.from_json("wordalign/profiles/maximum_quality.json")

# Customize
profile.segmentation.max_cpl = 32  # vertical/social video

# Use with PipelineRunner
from wordalign.core.pipeline import PipelineRunner
from wordalign.config import PipelineConfig
from wordalign.core.events import PrintSink

config = PipelineConfig(
    audio_path="interview.mp4",
    transcript_path="interview.txt",
    vosk_models_dir="/path/to/vosk-models",
)
runner = PipelineRunner(config, profile, PrintSink())
result = runner.run()
```

### Adding a new engine plugin (v2.0)

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
            supported_languages=["en", "fr"],
            models=["my-model-v1"],
            hardware={"cpu": True, "cuda": True},
        )

    def health_check(self):
        return HealthStatus(ready=True, runtime_status="ready")

    def transcribe(self, request):
        # Your engine code here
        return {"words": [...], "events": [], "engine_id": "my_engine"}

# Register it — no changes needed to pipeline.py, cli.py, or ensemble.py
registry = PluginRegistry()
registry.register(MyEngineAdapter())
```

See [docs/developer.md](docs/developer.md) for the full developer guide.

---

## Output files

| File | Contents |
| --- | --- |
| `*_word_level.srt` | One cue per word — the precision product |
| `*_sentence_level.srt` | Merged, balanced, CPL/duration-validated cues |
| `*_transcript.txt` | Ensemble mode: consensus transcript with `[HH:MM:SS]` timestamps |
| `*_transcript.docx` | Ensemble mode: `.docx` with low-agreement words highlighted yellow |
| `*_audio_tags.srt` (`--tags`) | Experimental: `[laughs]`-style event track |
| `*_combined.srt` (`--tags`) | Dialogue + events, collision-adjusted |

---

## Segmentation rules

| Flag | Default | Meaning |
| --- | --- | --- |
| `--max-cpl` | `42` | Characters per line — 42 for regular video, 32 for vertical/social |
| `--max-lines` | `2` | Lines per cue |
| `--max-duration-ms` | `7000` | Maximum on-screen time per cue |
| `--min-cue-ms` | `700` | Minimum on-screen time per cue |

Three principles keep the output review-ready:

1. **No forced bad breaks.** The balancer tries a gold-standard split, then a
   relaxed one; if neither is clean, it leaves the line long and visible for
   a human.
2. **Abbreviations and tags stay whole.** `Mr.`, `Mrs.`, `Dr.`, `St.`, `Prof.`
   are not read as sentence ends. Pure `[music]`-style tags are never merged
   into dialogue.
3. **Clean timing.** Overlaps between cues are resolved to zero (trimming the
   less-reliable side, never moving a Vosk-anchored timestamp). Sub-frame cues
   are extended or merged.

---

## Smart Transcript Review (v2.0)

After alignment, the QA engine analyzes the transcript and flags suspicious
regions — without silently rewriting them. The human reviewer remains the
final authority.

### QA signals

| Signal | What it detects | Requires |
| --- | --- | --- |
| **Ensemble disagreement** | Words where engines disagree or confidence is low | Ensemble mode |
| **Repetition** | Repeated n-grams indicating possible hallucination | Any mode |
| **Timing anomaly** | Impossible speaking rates, overlapping timestamps, long interpolation runs | Any mode |
| **LLM semantic review** | Phrases that are semantically incoherent in context | Optional (local or remote LLM) |

Each issue carries a `RiskScore` with component-level breakdown:

```
HIGH RISK

• Ensemble disagreement (80%)
• Low ASR confidence (60%)
```

### Using the QA engine programmatically

```python
from wordalign.qa.engine import QAEngine
from wordalign.core.converters import dicts_to_words

# After running the pipeline, convert results to typed words
words = dicts_to_words(aligned_words)

# Run QA
engine = QAEngine(enable_llm=False)
issues = engine.review(words)

for issue in issues:
    print(f"[{issue.severity}] {issue.category} at {issue.start:.1f}s")
    print(f"  {issue.explanation}")
    print(f"  Text: {issue.original_text}")
```

---

## Tests

```bash
# Run all tests (no GPU, audio, or model weights required)
python tests/test_pipeline.py     # 23 v1 tests: algorithm regression
python tests/test_v2.py           # 38 v2 tests: new architecture

# Or with pytest:
python -m pytest tests/ -q
```

**61 tests total, all passing.** The v1 tests pin the behavior of the
alignment waterfall, ensemble voting, and segmenter. The v2 tests cover
PipelineRunner events, plugin registry, model manager, QA signals,
subprocess protocol, and engine adapters.

---

## Project structure

```
word-align/
├── wordalign/
│   ├── cli.py                     # CLI entry point
│   ├── align.py                   # Timing waterfall (v1, unchanged)
│   ├── ensemble.py                # Consensus voting (v1, unchanged)
│   ├── segment.py                 # 3-phase segmenter (v1, unchanged)
│   ├── config.py                  # v1 PipelineConfig + constants
│   ├── document.py                # Transcript export (.txt/.docx)
│   ├── tags.py                    # YAMNet audio-event tagging
│   ├── utils.py                   # Shared helpers
│   ├── core/                      # v2.0: Pipeline orchestration
│   ├── plugins/                   # v2.0: Plugin system
│   ├── engines/adapters/          # v2.0: Engine plugin wrappers
│   ├── models/                    # v2.0: Model management
│   ├── runtimes/                  # v2.0: Hardware/runtime detection
│   ├── qa/                        # v2.0: Smart Transcript Review
│   ├── profiles/                  # v2.0: Pipeline presets (JSON)
│   ├── data/model_catalog.json    # v2.0: Curated model catalog
│   └── gui/app.html               # v2.0: Web GUI application
├── tests/
│   ├── test_pipeline.py           # 23 v1 algorithm tests
│   └── test_v2.py                 # 38 v2 architecture tests
├── docs/
│   ├── architecture.md            # v1.0 architecture reference
│   ├── proposal.md                # v2.0 engineering proposal
│   ├── developer.md               # Developer guide
│   └── user_guide.md              # User guide
├── preflight.py                   # Installation checker
├── requirements.txt
├── LICENSE                        # MIT
└── README.md
```

---

## Status

**Production-proven:** the alignment waterfall and the 3-phase segmenter.

**GPU-verified (reference mode):** Vosk → WhisperX/wav2vec2 → MFA → interpolation
→ segmenter, run end-to-end on real audio (RTX 4070 Ti).

**GPU-verified (ensemble mode):** WhisperX + Qwen3-ASR + Vosk voting with
the Qwen subprocess bridge, run end-to-end on real audio.

**Experimental:** audio-event tagging (YAMNet), behind `--tags`.

**v2.0 (new):** PipelineRunner, plugin architecture, model manager, QA engine,
pipeline profiles, web GUI. Architecture is complete and tested (61 tests);
integration with real audio requires the v1 engine dependencies to be installed.

---

## Known limitations

- **MFA gap-filling is English-only.** `english_us_arpa` is hardcoded.
- **Ensemble mode assumes English when `-l` is omitted.** Pass `-l` explicitly.
- **Untimed engines can reinforce but never correct the backbone.**
- **Vosk models are large** (1–2 GB each). Budget memory or lower `MAX_WORKERS`.
- **Sentence integrity outranks the duration cap.** This is deliberate.
- **The v2.0 web GUI is a static prototype.** It demonstrates the full UI but
  does not yet connect to a Python backend. Use the CLI for real processing.

---

## License

MIT © 2026 [Mohamed Ghali](https://www.linkedin.com/in/mohammed-ghaly-subtitler)
— veterinarian turned speech-pipeline engineer; 50,000+ captioning projects
delivered.

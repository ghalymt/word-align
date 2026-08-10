# WordAlign 2.0 Developer Guide

## Architecture Overview

WordAlign 2.0 separates the pipeline into four independent layers:

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

### Key Principle

The CLI and GUI both call `PipelineRunner.run()`. There is no separate
"GUI pipeline" and "CLI pipeline". The runner emits structured events;
each frontend translates them into its output format.

---

## Module Contract

### Adding a New Engine Plugin

1. **Create an adapter** in `wordalign/engines/adapters/`:

```python
# wordalign/engines/adapters/my_engine_adapter.py
from ...plugins.base import EngineDescriptor, EnginePlugin, HealthStatus
from ...plugins.capabilities import Capability

class MyEngineAdapter(EnginePlugin):
    def descriptor(self) -> EngineDescriptor:
        return EngineDescriptor(
            engine_id="my_engine",
            display_name="My Engine",
            version="1.0",
            capabilities=Capability.TRANSCRIBE | Capability.VOTE,
            runtime_type="in_process",  # or "subprocess" or "external"
            supported_languages=["en", "fr"],
            models=["my-model-v1"],
            hardware={"cpu": True, "cuda": True},
        )

    def health_check(self) -> HealthStatus:
        try:
            import my_engine  # noqa
            return HealthStatus(ready=True, runtime_status="ready")
        except ImportError:
            return HealthStatus(ready=False, runtime_status="missing",
                                missing_components=["my_engine"])

    def transcribe(self, request) -> dict:
        # Call your engine, return {"words": [...], "events": [...]}
        words = my_engine.transcribe(request.audio_path, request.language)
        return {"words": words, "events": [], "engine_id": "my_engine"}
```

2. **Register it** (in your application startup or `engines/adapters/__init__.py`):

```python
from wordalign.plugins.registry import PluginRegistry
from wordalign.engines.adapters.my_engine_adapter import MyEngineAdapter

registry = PluginRegistry()
registry.register(MyEngineAdapter())
```

3. **Add model entries** to `wordalign/data/model_catalog.json`:

```json
{
    "id": "my-model-v1",
    "display_name": "My Model v1",
    "plugin_id": "my_engine",
    "source": {"type": "huggingface", "repository": "org/my-model"},
    ...
}
```

**That's it.** No changes to `pipeline.py`, `cli.py`, `ensemble.py`,
or the model manager GUI. The engine automatically appears in the
Engines page, Models page, and Pipeline editor.

### What you DO NOT need to change:
- `cli.py` — no engine imports
- `pipeline.py` — queries by capability, not by name
- `ensemble.py` — voting logic is generic
- GUI code — reads from registry/catalog

---

## Capability Flags

```python
from wordalign.plugins.capabilities import Capability

Capability.TRANSCRIBE    # produces word/text from audio
Capability.WORD_TIMING   # produces word-level timestamps
Capability.FORCED_ALIGN  # aligns given text to audio
Capability.VOTE          # participates in ensemble voting
Capability.DIARIZE       # assigns speaker IDs
Capability.AUDIO_EVENTS  # detects non-speech events
Capability.VAD           # voice activity detection
Capability.TEXT_QA       # reviews transcript text
Capability.AUDIO_QA      # reviews audio directly
Capability.EXPORT        # produces output files
```

A plugin may expose multiple capabilities. The pipeline queries by
capability, not by engine name.

---

## Typed Data Structures

All new code uses typed structures instead of anonymous dicts:

- `WordResult` — one word with timing, confidence, provenance
- `SegmentResult` — one subtitle cue
- `EngineResult` — output from a single engine
- `PipelineResult` — final output of a pipeline run

Conversion helpers bridge to legacy dict format:

```python
from wordalign.core.converters import words_to_dicts, dicts_to_words

# Legacy dict format (for align.py, ensemble.py, segment.py)
dicts = words_to_dicts(typed_words)

# Back to typed
typed_words = dicts_to_words(dicts)
```

---

## Pipeline Profiles

Profiles are JSON files that configure the entire pipeline:

```
wordalign/profiles/
├── fast.json
├── balanced.json
├── maximum_quality.json
└── cpu_only.json
```

Load programmatically:

```python
from wordalign.core.config import PipelineProfile

profile = PipelineProfile.from_json("wordalign/profiles/balanced.json")
# or
profile = PipelineProfile.preset("fast")
```

Create custom profiles by saving a JSON file with the same schema.

---

## Subprocess Protocol

For engines with conflicting dependencies, use the JSON-line subprocess protocol:

1. Worker script reads JSON from stdin, writes JSON to stdout.
2. Messages: `init`, `request`, `progress`, `result`, `error`, `cancel`, `shutdown`.
3. stderr is for human-readable logs.
4. Every request has a unique `request_id`.

See `wordalign/plugins/protocol.py` for the full specification.

---

## QA System

The QA engine runs multiple signals:

1. **Ensemble disagreement** — low confidence words from `build_consensus()`
2. **Repetition detection** — repeated n-grams indicating hallucination
3. **Timing anomalies** — extreme rates, overlaps, interpolation runs
4. **LLM semantic review** — optional, via provider abstraction

```python
from wordalign.qa.engine import QAEngine

engine = QAEngine(enable_llm=False)
issues = engine.review(consensus_words)
# issues: list[TranscriptIssue]
```

Each issue has a `RiskScore` with component-level breakdown.

---

## Testing

Run all tests:

```bash
python tests/test_pipeline.py   # 23 v1 tests (algorithm regression)
python tests/test_v2.py         # 38 v2 tests (new architecture)
```

No GPU, audio, or model weights required.

### Adding a fake plugin for integration tests:

```python
class FakePlugin(EnginePlugin):
    def descriptor(self):
        return EngineDescriptor(
            engine_id="fake", display_name="Fake",
            version="1.0",
            capabilities=Capability.TRANSCRIBE | Capability.VOTE,
            runtime_type="in_process",
            supported_languages=["*"], models=[],
            hardware={"cpu": True, "cuda": False})
    def health_check(self):
        return HealthStatus(ready=True, runtime_status="ready")
```

---

## File Reference

| Path | Purpose |
|---|---|
| `wordalign/core/pipeline.py` | PipelineRunner — single orchestration entry |
| `wordalign/core/types.py` | WordResult, SegmentResult, EngineResult |
| `wordalign/core/events.py` | Event types + EventSink protocol |
| `wordalign/core/config.py` | SegmentationConfig, PipelineProfile, JobContext |
| `wordalign/core/errors.py` | Exception hierarchy |
| `wordalign/core/converters.py` | Typed ↔ legacy dict conversion |
| `wordalign/plugins/base.py` | EnginePlugin ABC, EngineDescriptor |
| `wordalign/plugins/capabilities.py` | Capability flags |
| `wordalign/plugins/registry.py` | PluginRegistry |
| `wordalign/plugins/manifest.py` | PluginManifest dataclass + validation |
| `wordalign/plugins/protocol.py` | Subprocess JSON-line protocol |
| `wordalign/engines/adapters/` | Thin wrappers for existing engines |
| `wordalign/models/catalog.py` | Model catalog parser |
| `wordalign/models/manager.py` | Model install/validate/delete |
| `wordalign/models/validator.py` | Model directory validation |
| `wordalign/models/downloader.py` | Resumable downloads |
| `wordalign/runtimes/detector.py` | Hardware detection |
| `wordalign/runtimes/manager.py` | Runtime environment management |
| `wordalign/qa/engine.py` | QA orchestrator |
| `wordalign/qa/deterministic.py` | Deterministic QA signals |
| `wordalign/qa/issues.py` | TranscriptIssue, RiskScore |
| `wordalign/profiles/*.json` | Pipeline profile presets |
| `wordalign/data/model_catalog.json` | Curated model catalog |
| `wordalign/gui/app.html` | Web-based GUI application |

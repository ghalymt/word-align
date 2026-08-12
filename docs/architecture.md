# word-align Architecture — Current State (v1.0.0)

This document describes the system as it exists today, before the 2.0 refactor.
It is the Phase 0 baseline reference.

---

## 1. Repository Layout

```
word-align/
├── preflight.py              # dependency check script
├── requirements.txt          # core + optional dependencies
├── README.md
├── LICENSE                   # MIT
├── tests/
│   └── test_pipeline.py      # 23 tests, no GPU/audio needed
└── wordalign/
    ├── __init__.py            # version = "1.0.0"
    ├── __main__.py            # guard for Windows multiprocessing spawn
    ├── cli.py                 # ALL orchestration + CLI arg parsing + output writing
    ├── config.py              # PipelineConfig dataclass + constants + model maps
    ├── align.py               # timing waterfall (match, MFA, interpolate, stats)
    ├── ensemble.py            # consensus voting (ROVER descendant)
    ├── segment.py             # 3-phase iterative segmenter (module-level globals)
    ├── document.py            # transcript export (.txt / .docx)
    ├── tags.py                # YAMNet audio-event tagging (experimental)
    ├── utils.py               # text normalization, timecode math, ffmpeg wrappers
    └── engines/
        ├── __init__.py
        ├── vosk_engine.py     # parallel chunked CPU decoding
        ├── whisperx_engine.py # Whisper large-v3 + wav2vec2 alignment
        ├── qwen_engine.py     # Qwen3-ASR adapter (subprocess bridge)
        ├── _qwen_worker.py    # standalone Qwen worker script
        ├── mfa_engine.py      # Montreal Forced Aligner CLI wrapper
        └── nemo_engine.py     # NeMo Parakeet/Canary (legacy voters)
```

---

## 2. Reference-Mode Flow

**Input:** audio/video file + verbatim human transcript (.txt)

```
1.  Parse CLI args → PipelineConfig
2.  Read transcript file → original_text
3.  Auto-detect language (langdetect) if not given
4.  Extract human tags [laughs] etc. → human_tags
5.  Strip tags → text_no_tags
6.  Split into words → human_words
7.  Normalize words → human_words_norm
8.  Create aligned_words: [{"word": w} for w in human_words]
9.  --- WATERFALL ---
10. Vosk: run_vosk_parallel() → match_timestamps() [fill-only]
11. (optional) Rough SRT: parse → linear-interpolate per-word times → match_timestamps()
12. WhisperX: run_whisperx() → match_timestamps()
13. MFA: make_surgical_mfa() → run_mfa_on_gaps() [only on unmatched contiguous runs]
14. Interpolation: interpolate_timestamps() [linear fill between neighbors]
15. Print alignment statistics (per-source counts)
16. --- SEGMENTATION ---
17. set_layout(max_cpl, max_lines, max_duration_ms)  [sets module globals]
18. parse_human_transcript_to_srt_segments()  [maps transcript lines → word spans]
19. run_iterative_merging()  [3-phase sweep with growing char budget 1..84]
20. resolve_overlaps()  [iterative to zero, preserves Vosk timestamps]
21. enforce_min_duration()  [extend or merge sub-frame cues]
22. resolve_overlaps()  [tidy after merge]
23. --- OUTPUT ---
24. Write *_word_level.srt  [one cue per word]
25. Write *_sentence_level.srt  [merged, balanced cues]
26. validate_srt_output()  [overlaps, gaps, duration, CPL]
27. (optional) Document export if ensemble mode
28. (optional) Tag detection if --tags
```

---

## 3. Ensemble-Mode Flow

**Input:** audio/video file only (no transcript)

```
1.  Parse CLI args → PipelineConfig
2.  Language = cfg.language or "en"
3.  --- ENSEMBLE TRANSCRIPTION ---
4.  Run WhisperX → engine_words["whisperx"]
5.  Run Qwen3-ASR → engine_words["qwen"]
6.  Run Vosk → engine_words["vosk"]
7.  (legacy) Run Parakeet/Canary if requested
8.  build_consensus(engine_words) → consensus word list
9.  consensus_to_structured_text() → original_text (newline-broken)
10. --- From here, same as reference mode ---
11. Vosk timestamps seed the waterfall FIRST (best timestamps)
12. Ensemble consensus timestamps fill remaining gaps
13. WhisperX already ran; MFA + interpolation finish
14. Segmentation + output (same as reference mode)
15. Export transcript document (.txt/.docx) with confidence highlighting
```

**Key difference from reference mode:**
- WhisperX runs as the ensemble backbone (not in the waterfall).
- Vosk timestamps are seeded first (before consensus), because Vosk has
  the best timestamps and `match_timestamps` is fill-only.
- The consensus produces both the transcript text AND provisional timestamps.
- A transcript document is exported alongside SRT files.

---

## 4. Transcription Engines

Each engine returns `List[Dict]` with keys: `word`, `start`, `end`, `conf`.

| Engine | File | Timed? | GPU? | Run Mode | Languages |
|---|---|---|---|---|---|
| Vosk | `vosk_engine.py` | Yes (word-level) | No (CPU) | In-process, parallel chunks | 20+ (model-dependent) |
| WhisperX | `whisperx_engine.py` | Yes (word-level via wav2vec2) | Yes (CUDA) | In-process | 99 (Whisper) + alignment model map |
| Qwen3-ASR | `qwen_engine.py` + `_qwen_worker.py` | Yes (word-level via ForcedAligner) | Yes (CUDA) | **Subprocess** (own venv) | 30+ |
| MFA | `mfa_engine.py` | Yes (forced alignment) | No (CPU) | **External CLI** (subprocess) | English only (hardcoded) |
| NeMo Parakeet | `nemo_engine.py` | Yes (word-level) | Yes (CUDA) | In-process | 25 European |
| NeMo Canary-Qwen | `nemo_engine.py` | **No** (text only) | Yes (CUDA) | In-process | English only |

**Engine return contract (all engines):**
```python
List[Dict[str, Any]]  # each dict: {"word": str, "start": float, "end": float, "conf": float}
```
WhisperX additionally returns `(word_segments, audio_events)` — a 2-tuple.

---

## 5. Alignment Sources

The `match_timestamps()` function in `align.py` is **fill-only**: it copies
timestamps from a source onto still-unmatched words using `SequenceMatcher`
on normalized word sequences. It never overwrites a word that already has
a `"matched": True` flag.

Sources are tried in order; each fills only gaps:

| Order | Source | When Used |
|---|---|---|
| 1 | Vosk | Always (if available) — reference mode |
| 1 | Vosk | Ensemble mode: seeded FIRST (before consensus) |
| 2 | Rough SRT | If `--srt` provided |
| 3 | WhisperX | Reference mode (in ensemble, already ran as backbone) |
| 3 | Ensemble consensus | Ensemble mode (after Vosk) |
| 4 | MFA | If available and `--no-mfa` not set |
| 5 | Interpolation | Always (fills whatever remains) |

Each word gets a `"source"` attribute: `"Vosk"`, `"WhisperX"`, `"MFA"`,
`"Interpolated"`, `"Ensemble"`, or `"SRT"`.

---

## 6. Timing Waterfall

Location: `wordalign/align.py`

### match_timestamps(aligned_words, source_words, source_name, human_words_norm)
- Uses `difflib.SequenceMatcher` to align normalized word sequences.
- Only fills words where `matched` is False.
- Updates word dict with `start`, `end`, `matched=True`, `source`.
- Prints match count and percentage.

### make_surgical_mfa(aligned_words, audio_path, mfa_cmd)
- Finds contiguous runs of unmatched words ("gaps").
- Extracts audio slices (with 0.25s padding) around each gap.
- Creates `.wav` + `.lab` pairs for MFA corpus.
- Runs MFA G2P → dictionary generation → alignment.
- Parses TextGrid output, maps times back with chunk offsets.
- Cleans up temp directory.

### interpolate_timestamps(aligned_words)
- Linear interpolation between nearest timed neighbors.
- Fallback: extends from previous word's end (+0.05s start, +0.25s duration).
- Last resort: `i * 0.3` spacing.

### print_alignment_statistics(aligned_words)
- Counts words per source, prints summary.

---

## 7. Ensemble Voting

Location: `wordalign/ensemble.py`

### build_consensus(engine_words, priors)
- Backbone selection: highest-priority timed engine by `BACKBONE_PREFERENCE`
  (WhisperX > Qwen > Vosk).
- For each backbone word:
  - Backbone votes for its own token with `prior * confidence`.
  - Other timed engines vote via time-overlap window (±0.40s) with
    proximity-weighted decay.
  - Untimed engines vote via difflib lexical matching (can only reinforce,
    never correct).
- A challenger token replaces the backbone token only if its total vote
  exceeds the backbone's by `REPLACE_MARGIN` (0.25).
- Each consensus word retains: `word`, `start`, `end`, `conf` (fused agreement).

### DEFAULT_PRIORS (transcription accuracy ranking)
| Engine | Prior |
|---|---|
| WhisperX | 1.00 |
| Qwen | 0.85 |
| Vosk | 0.55 |
| Canary (legacy) | 1.20 |
| Parakeet (legacy) | 1.10 |

### consensus_to_structured_text(consensus)
- Renders consensus as newline-separated lines for the segmenter.
- Breaks at sentence ends, speech pauses (>0.6s gap), or line length >42 chars.

---

## 8. Segmentation

Location: `wordalign/segment.py`

### Module-level configuration globals
```python
_max_cpl = 32              # from config.MAX_CPL
_max_lines = 2
_max_duration_ms = 3000    # from config.MAX_DURATION_MS
_current_max_chars = 32    # varies during iteration
```
These are set by `set_layout()` and `set_max_chars()` at runtime.
**This is a blocking issue for multi-job safety.**

### set_layout(max_cpl, max_lines, max_duration_ms)
Called once by the CLI before segmentation begins.

### parse_human_transcript_to_srt_segments(transcript_text, aligned_words)
- Splits transcript into lines, strips tags.
- Maps each line to its word span via normalized sequence matching.
- Creates segment dicts with start/end timecodes and source attribution.

### run_iterative_merging(segments)
- Sweeps character budget from 1 to `_max_cpl * _max_lines`.
- Each iteration runs 3 phases:
  - **Phase 1**: Merge to sentence boundaries (capitalization-aware,
    abbreviation-aware, tag-isolation-aware).
  - **Phase 2**: Merge short fragments under CPL/duration limits.
  - **Phase 3**: Balance each block into visually even lines
    (gold split → relaxed split → leave long for review).
- `shift_dangling_words()` runs between phases.

### resolve_overlaps(segments, max_passes=6)
- Iterative overlap removal to fixed point.
- Rules: never move Vosk timestamps; prefer trimming earlier END;
  preserve START times.
- Uses `start_source` / `end_source` on each segment.

### enforce_min_duration(segments, min_ms=700)
- Pass 1: extend short cues into available space before next cue.
- Pass 2: merge still-degenerate cues into previous cue.

### validate_srt_output(entries, name)
- Checks: overlaps, large gaps, duration over limit, CPL per line.

---

## 9. Document Output

Location: `wordalign/document.py`

- `group_into_sentences()`: splits at terminal punctuation or max 60 words.
- `group_into_paragraphs()`: splits at >1.75s silence gaps or max 8 sentences.
- `write_txt()`: `[HH:MM:SS]` paragraph timestamps + plain text.
- `write_docx()`: highlighted low-confidence words (<0.5 agreement threshold).
- `export_transcript()`: dispatcher for format selection.

---

## 10. Audio Tagging (Experimental)

Location: `wordalign/tags.py`

- Requires `tensorflow` + `tensorflow-hub` (YAMNet).
- `detect_audio_tags_yamnet()`: runs YAMNet on 16kHz audio, detects events
  from a tag map (e.g., "laughs" → "Laughter" AudioSet class).
- `match_human_tags_to_detections()`: 1-to-1 matching of transcript `[tags]`
  to detected events.
- `optimize_tags()`: merges consecutive identical tags with no dialogue between.
- `adjust_tag_timing()`: nudges tags outside dialogue cues.
- `combine_and_sort_srt()`: merges sentence + tag SRT entries chronologically.

---

## 11. External Processes

### Qwen subprocess bridge
- `qwen_engine.py` launches `_qwen_worker.py` via `subprocess.run()`.
- Worker runs in a separate venv (different torch version).
- Communication: sentinel-tagged JSON on stdout (`__WA_QWEN_JSON__` prefix).
- Worker handles its own chunking, VRAM management, OOM recovery.
- Returns: `{"language": str, "words": [...]}`.

### MFA CLI
- `mfa_engine.py` wraps the `mfa` executable via `subprocess.run()`.
- Resolves MFA path: explicit arg → env var → PATH → conda env search.
- Augments PATH for conda env (DLLs, Libraries).
- Commands: `mfa g2p` (dictionary), `mfa align` (alignment).
- Parses TextGrid output files.

### FFmpeg/FFprobe
- `utils.create_chunk_wav()`: ffmpeg to extract + resample audio chunks.
- `utils.get_audio_duration()`: ffprobe to get duration.

---

## 12. Model Paths

### Vosk models
- Directory: `--vosk-models` or `WORDALIGN_VOSK_MODELS` env var.
- Model name resolved via `VOSK_MODEL_MAP` in `config.py` (e.g., `en` → `vosk-model-en-us-0.22`).
- `PipelineConfig.vosk_model_path(language)` returns full path if it exists.

### WhisperX models
- Model size: `--whisper-model` (default `large-v3`).
- Downloaded automatically by WhisperX (HuggingFace cache).
- Alignment models: `ALIGNMENT_MODEL_MAP` in `whisperx_engine.py`
  maps language codes to wav2vec2 checkpoints.

### Qwen models
- ASR: `Qwen/Qwen3-ASR-1.7B` (default, configurable).
- Aligner: `Qwen/Qwen3-ForcedAligner-0.6B` (default, configurable).
- Cache dir: `--qwen-models` or `WORDALIGN_QWEN_MODELS` env var.
- Python: `--qwen-python` or `WORDALIGN_QWEN_PYTHON` env var.

### MFA models
- G2P model: `english_us_arpa` (hardcoded).
- Acoustic model: `english_us_arpa` (hardcoded).
- Located at `~/Documents/MFA/pretrained_models/`.

### NeMo models (legacy)
- Parakeet: `nvidia/parakeet-tdt-0.6b-v3`.
- Canary-Qwen: `nvidia/canary-qwen-2.5b`.
- Downloaded automatically by NeMo (HuggingFace cache).

---

## 13. Dependencies

### Core (required)
| Package | Purpose |
|---|---|
| `whisperx>=3.1` | Whisper transcription + wav2vec2 alignment |
| `torch>=2.1` | PyTorch backend |
| `srt>=3.5` | SRT parsing/composing |
| `librosa>=0.10` | Audio loading (YAMNet, Qwen) |
| `psutil>=5.9` | Memory checks |
| `langdetect>=1.0.9` | Language auto-detection |

### Optional
| Package | Purpose |
|---|---|
| `vosk>=0.3.45` | Vosk engine / voter |
| `qwen-asr` | Qwen3-ASR voter (usually in separate venv) |
| `nemo_toolkit[asr]>=2.0` | Legacy Parakeet/Canary voters |
| `tensorflow>=2.15` | YAMNet audio-event tagging |
| `tensorflow-hub>=0.16` | YAMNet model loading |
| `textgrid>=1.5` | MFA TextGrid parsing |
| `python-docx>=1.1` | .docx transcript export |

### External binaries
- `ffmpeg` / `ffprobe` — must be on PATH.
- `mfa` — Montreal Forced Aligner (optional, on PATH or via env var).

---

## 14. Configuration

### PipelineConfig (config.py)
Central dataclass with fields for:
- Audio/transcript/SRT paths
- Language, output directory
- Vosk models directory, MFA command
- Feature toggles: `use_vosk`, `use_mfa`, `use_tags`
- Whisper model size, compute device
- Ensemble engines list, Qwen settings
- Document format, timestamps
- Subtitle layout: `max_cpl`, `max_lines`, `max_duration_ms`, `min_cue_ms`

### Module-level constants (config.py)
- `MAX_CPL = 32` (overridden by CLI default 42)
- `MAX_DURATION_MS = 3000` (overridden by CLI default 7000)
- `MIN_LINE_RATIO = 0.33`
- `ITERATION_START = 1`, `ITERATION_END = 32`
- `VOSK_CHUNK_SECONDS = 600`
- `MAX_WORKERS = 4`
- `TERMINAL_PUNCT_PATTERN` — regex for sentence-ending punctuation
- `NON_TERMINAL_ABBREVIATIONS` — set of abbreviations that look like sentence ends
- `PREFER_NEW_LINE_WORDS` — words that start lines better (7 languages)
- `CLAUSE_MARKERS` — conjunctions + relative pronouns (7 languages)
- `TAG_MAP` — human tag → AudioSet class name
- `VOSK_MODEL_MAP` — language code → Vosk model directory name

### Environment variables
| Variable | Purpose |
|---|---|
| `WORDALIGN_VOSK_MODELS` | Vosk models directory |
| `WORDALIGN_MFA` | MFA executable or conda env path |
| `WORDALIGN_QWEN_PYTHON` | Python in Qwen venv |
| `WORDALIGN_QWEN_MODELS` | Qwen model cache directory |

---

## 15. Output Files

All outputs land next to the audio file (or in `-o DIR`):

| File | Contents |
|---|---|
| `*_word_level.srt` | One cue per word — the precision product |
| `*_sentence_level.srt` | Merged, balanced, CPL/duration-validated cues |
| `*_transcript.txt` | Ensemble mode: consensus transcript (paragraphed, timestamped) |
| `*_transcript.docx` | Ensemble mode: .docx with low-agreement words highlighted |
| `*_audio_tags.srt` | `--tags`: `[laughs]`-style event track |
| `*_combined.srt` | `--tags`: dialogue + events, collision-adjusted |

---

## 16. Test Coverage

`tests/test_pipeline.py` — 23 tests, no GPU/audio/weights required.

Covers:
- Ensemble correction (challenger overturns backbone)
- Margin protection (single dissenter can't overturn)
- Backbone selection (WhisperX chosen, not highest prior)
- Fill-only waterfall (later sources don't overwrite)
- Interpolation monotonicity
- CPL enforcement in segmentation
- Tag stripping regex
- Document paragraphing and export
- Per-line CPL validation (not per-block)
- Final-entry validation
- Untimed engine reinforcement characterization
- Terminal punctuation with closing quotes (curly, guillemet)
- Abbreviation detection
- Segmenter doesn't split on abbreviations
- Overlap resolution: trims end, preserves Vosk
- Overlap resolution: never moves Vosk end
- Overlap resolution: reaches zero on a chain
- Consensus structured text breaks into lines
- Configurable CPL limit
- No-panic balancer leaves unsplittable lines
- Isolated tag never merged
- Minimum duration enforcement (extend + merge)
- Qwen worker output parsing

---

## 17. Known Coupling Issues (Pre-refactor Risks)

These are the issues the 2.0 refactor must address:

1. **Orchestration in cli.py**: `main()` contains the entire pipeline flow
   interleaved with `print()` calls and file I/O. No programmatic API exists.

2. **Module-level globals in segment.py**: `_max_cpl`, `_max_lines`,
   `_max_duration_ms`, `_current_max_chars` are mutable module globals.
   Multiple concurrent jobs would corrupt each other's configuration.

3. **No structured events**: All progress/status communication is via `print()`.
   No way for a GUI to observe progress without parsing stdout.

4. **No cancellation**: Long-running operations have no cancellation token.
   Killing a process is the only way to stop.

5. **Anonymous dicts throughout**: Word results are `Dict[str, Any]` with no
   type safety. The `aligned_words` list is mutated in place by multiple
   functions.

6. **Engine discovery is hardcoded**: `cli.py` imports engines directly and
   calls them by name. No registry, no capability-based dispatch.

7. **No caching**: Rerunning the pipeline always re-runs all ASR. Changing
   `max_cpl` from 42 to 32 re-runs WhisperX.

8. **Qwen subprocess protocol is ad-hoc**: Sentinel-tagged JSON is functional
   but not a generalizable protocol. Only Qwen uses it.

9. **MFA is English-only hardcoded**: `english_us_arpa` is baked into both
   G2P and acoustic model selection.

10. **Preflight is a standalone script**: Not integrated into any programmatic
    API; results are print-only.

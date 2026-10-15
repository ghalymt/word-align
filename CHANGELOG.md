# Changelog

## 2.2.0 — 2026-10-15

Bad audio. A new **robust flow** (`--profile robust`, GUI preset *Robust*)
measures the recording, cleans only the parts that need it, lets several
ASR engines vote, times words with forced alignment on the original audio,
and sorts every cue into a review queue. Every new stage is optional and
skipped with a message when its model is not installed. English first;
see [docs/robust_flow.md](docs/robust_flow.md).

### Upgrade notes (behaviour changes)

- **Word timing uses anchors + forced alignment by default**
  (`--timing anchor`): agreeing engines give anchors, Qwen3-ForcedAligner
  (or wav2vec2) aligns the windows between them on the original audio, MFA
  and pause-aware interpolation fill the rest. `--timing fill_only` brings
  back the 2.1 waterfall; `--no-forced-align` keeps the engine times.
- **Every run with anchor timing (the default) writes `<name>_review.json`**
  (cue tiers Accept / Quick check / Redo with reasons) unless `--no-triage`
  is given.
- **The audio analysis stage runs by default** when its model is present
  (Silero VAD; `--no-vad` turns it off), and the hallucination filter drops
  engine words the audio does not support (`--no-hallucination-filter`).
- **Speaker labels use pyannote Community-1** (falls back to 3.1); accept
  its terms on Hugging Face once.
- **Transcript-free mode skips Vosk** when two timed voters already anchor
  the timing and a forced aligner (Qwen3-ForcedAligner or wav2vec2) is
  installed.
- The Qwen3-ForcedAligner language list in the model catalog was wrong
  (it listed Arabic); it now matches the model: zh, en, yue, fr, de, it, ja,
  ko, pt, ru, es.

### Added

- **Audio analysis:** Silero VAD (ONNX, no torch in the app), VAD-planned
  ASR chunks, WADA-SNR, narrowband and clipping detection, and timed sound
  events (PretrainedSED BEATs) per chunk.
- **Gated cleanup** (`--cleanup auto|all|off`): RoFormer vocal separation for
  loud music, RoFormer crowd removal, MossFormerGAN for steady noise, mixed
  into a separate ASR track; never on phone or clipped audio.
- **ROVER voting** over Cohere Transcribe, Qwen3-ASR, parakeet-ultra,
  WhisperX and Vosk (`--voters`), per VAD chunk, with each word's agreement
  and the losing readings kept for review.
- **Overlap map** (`--overlap-map`) and **overlapped-speech ASR** with
  SE-DiCoW (`--overlap-asr`); two-speaker **dash cues** (`--dash-cues`,
  `--dash-style`).
- **SDH tag proposals** from the sound events (`--tags`).
- **Review queue in the GUI:** loop a cue, apply a voter's alternative with
  one click, accept with Ctrl+Enter. Corrections go to
  `<name>_corrected.srt` (plus an edit log); the pipeline's files are never
  changed.
- **Triage calibration:** `wordalign-bench fit-triage --jobs FOLDER` learns
  the risk weights from your corrected jobs.
- **Benchmark:** `wordalign-bench` builds a suite from public data
  (LibriSpeech alignments, MUSAN noise, FMA music) and scores WER, timing,
  invented words and triage per condition; see
  [docs/benchmark.md](docs/benchmark.md).
- **Setup:** `scripts/install_robust.ps1` (venvs and models),
  `scripts/fetch_models.py` (`--list` shows what is installed),
  `wordalign --doctor`, `scripts/smoke_robust.py` (one real call per
  backend).
- **Configurable locations:** `--models-dir` / `WORDALIGN_MODELS_DIR`,
  `WORDALIGN_VENVS_DIR`, `WORDALIGN_<NAME>_PYTHON` for every backend,
  `WORDALIGN_VAD_MODEL`, `WORDALIGN_SED_DIR`.
- The model catalog lists the new models with their licenses.

### Measured (`wordalign-bench`, quick suite, RTX 4070 Ti)

Five minutes of LibriSpeech speech in nine conditions (clean, noise 5 dB,
music 0 dB, six-talker babble 0 dB, reverb 1.0 s, μ-law phone, a second
talker at −5 dB, music only, noise only), every run with an empty cache.
"2.1" is the 2.1 behaviour run by the 2.2 code
(`--timing fill_only --no-vad --no-hallucination-filter --no-triage`).

| | 2.1 | 2.2 defaults | `--profile robust` |
|---|---|---|---|
| Word-start error, known transcript (mean) | 868 ms | **69 ms** | 70 ms |
| Words within 100 ms, known transcript | 83.0% | **90.1%** | 90.1% |
| WER, no transcript | 16.9% | **15.0%** | 15.3% |
| Word-start error, no transcript (mean) | 225 ms | **73 ms** | 106 ms |
| Invented words per 10 min of music/noise only | 300 | **0** | **0** |
| Erroneous cues the review queue flags (no transcript) | – | 88% | 77% |
| Run time for the 5-minute file | 238–289 s | 234–282 s | 195–268 s |

By condition: reverb WER 43% → 2.6%; babble timing with a known transcript
5.8 s → 0.2 s. Worse than 2.1: transcript-free WER under 0 dB music with
sung vocals (0% → 12–13%: the voters also transcribe the lyrics) and under
0 dB babble (73% → 84–87%, unusable either way). On this synthetic suite the
robust profile's extra voters and cleanup did not beat the 2.2 defaults;
whether they help on real recordings with music beds, crowds and crosstalk
is not measured yet, so compare on your own material (see
docs/benchmark.md).

### Fixed

- Qwen chunks that ran out of GPU memory are split and retried; pieces that
  still fail are reported instead of being silently empty.
- WhisperX finds its venv through the same lookup as the other backends.
- `--doctor` starts MFA instead of only finding it, so a pip install of MFA
  without its native `kalpy` module is reported (install MFA from
  conda-forge).
- A forced-aligned word no longer vanishes from the word-level SRT when the
  aligner gives it no length.

## 2.1.0 — 2026-10-01

New output formats, speaker labels, batch processing, a GUI job queue and
a long list of fixes. Pull request numbers refer to
[ghalymt/word-align](https://github.com/ghalymt/word-align/pulls?q=is%3Apr).

### Upgrade notes (behaviour changes)

- **MFA never borrows the English model.** For a language without a
  pretrained MFA model, or whose model is not installed, MFA is skipped with
  a warning; it used to align with the English acoustic model and produce
  wrong timings (#17).
- **MFA default is the same everywhere:** on in the CLI and in the GUI's
  Balanced and Maximum Quality presets, and it only runs when installed and
  available for the language (#17). The GUI's engine checkboxes now override
  the preset (#10), and explicit `--use-mfa` / `--no-*` flags override
  `--engines` (#3).
- **Ensemble mode output changed.** The primary transcript is split into
  cues (it used to come out as one cue for the whole file) and Vosk/Qwen
  refine its timings (#7).
- **GUI server:** requests from other websites are refused (no CORS headers,
  cross-origin POSTs and foreign `Host` headers get 403). Opening `app.html`
  from disk no longer works; use `wordalign --gui`. For LAN access set
  `WORDALIGN_ALLOWED_HOSTS` (#4, #19).
- **GUI jobs queue** and run one at a time by default
  (`WORDALIGN_MAX_CONCURRENT_JOBS`); jobs left running by a previous session
  are marked *interrupted* on start (#20).
- **`python -m wordalign` no longer falls back to the v1 CLI** when the v2
  CLI fails to import; the real error is shown (#15).
- **The release ZIP holds the app only.** Models go in `models\` next to
  `WordAlign.exe`; `MODELS.txt` in the ZIP says where each kind goes.
  `build.py --zip-with-models` still builds a full offline bundle (#23).

### Added

- **WebVTT and ASS export:** `--formats srt,vtt,ass`, GUI toggles, profile
  `export.vtt` / `export.ass` (#18).
- **Speaker labels:** `--diarize [--speakers N]` with pyannote.audio (needs
  `HF_TOKEN`). Cues never mix speakers; names appear as WebVTT voices, the
  ASS Name field, an SRT prefix and transcript paragraphs (#22).
- **Batch mode:** `--batch FOLDER [--recursive] [--skip-existing]` pairs
  `talk.mp4` with `talk.txt`/`talk.srt`, keeps going after a failure and
  writes `wordalign_batch_report.json` (#21).
- **`--max-lines N`** lays cues out on up to N lines; `--max-lines 1` never
  adds a break (#16).
- **GPU scheduler in use:** WhisperX and Qwen take a process-wide GPU slot,
  so two jobs never load GPU models at once (`WORDALIGN_GPU_SLOTS`) (#20).
- **Streaming uploads:** GUI uploads are written to disk as they arrive
  instead of being held in memory (#23).
- **Tests and CI:** CI on Linux and Windows (Python 3.10 and 3.12) for every
  pull request (#6); golden-file tests through the real pipeline and a
  real-audio timing accuracy job with Vosk (#13).

### Fixed

- Failed or cancelled engine runs were cached and replayed for 30 days (#2).
- Reference mode with an `.srt` transcript: `[tags]` shifted every later cue,
  and preserved cues were not laid out (#8).
- `--punctuation`: the second line of two-line cues was dropped, long files
  never used the LLM, and real cues were rejected as commentary (#9).
- QA issues moved between jobs, global glossary terms were duplicated,
  trailing untimed gaps were missed (#11).
- Windows: crash after a successful run when output was redirected, and
  cp1252 crashes in the WhisperX/Qwen workers, llama-cli and MFA output
  (#12, #14).
- MFA temp directories leaked; `--legacy-ensemble` crashed when every engine
  failed; primary-transcript runs bypassed the stage cache (#12, #14).
- Five-line cues fell back to one long line; impossible layouts
  (`--max-lines 0`, `--cpl 0`, negative durations) were accepted (#16).
- A blank line inside a cue ended the WebVTT cue early (#18).
- Batch: equally named files in different subfolders overwrote each other
  (#21). `--speakers N` without `--diarize` was ignored (#22).
- Uploaded file names with `;` lost their extension; a trailing CR/LF of an
  uploaded file was stripped (#10, #23).
- A fresh clone was missing the `wordalign/models` package because of a
  `.gitignore` rule (#1).

### Security

- The local API sent `Access-Control-Allow-Origin: *`, so any website could
  read it (#4); cross-site POSTs could still start runs or clear the cache,
  and DNS rebinding was possible (#19).
- Server text (error messages, output paths) was inserted into the GUI as
  HTML: an audio file named like an HTML tag could run script in the GUI
  (#25).
- Refused requests and JSON bodies are no longer read into memory whatever
  their size (#19, #25).

## 2.0.0 — 2026-08-10

Full rewrite: plugin engine architecture, Model Manager, web GUI, Smart
Transcript Review, stage cache and job history, portable Windows build.

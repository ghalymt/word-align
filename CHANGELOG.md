# Changelog

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

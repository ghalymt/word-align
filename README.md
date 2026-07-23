# word-align

**Word-accurate subtitle timing from any transcript + audio — or from audio alone.**

`word-align` produces word-level and broadcast-quality sentence-level SRT files by
cascading multiple speech engines, each one only touching what the previous
stage couldn't solve. The alignment and segmentation logic comes out of
thousands of hours of real professional captioning work across 7+ languages;
this repository is that logic extracted into a documented, tested package.
See [Status](#status) for what is production-proven and what is not.

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
                  name_word_level.srt   name_sentence_level.srt
```

## Why a waterfall?

No single engine solves word timing well. Vosk is fast but misses words;
WhisperX's wav2vec2 forced alignment is precise but expensive and can drift on
long silences; Montreal Forced Aligner is the gold standard but impractical to
run on a full recording. The waterfall runs each engine **only on what remains
unsolved**, so you get MFA-grade precision at a fraction of MFA's cost. In
production use on long-form dialogue, the first (CPU-only) pass typically
covers ~90% of words before the GPU is ever touched, and MFA runs on isolated
gap slices instead of whole files. Every word in the output is traceable to
its source engine — the per-engine breakdown is printed at the end of every
run.

## Transcript-free ensemble mode (beta)

Given audio with no reference transcript, `word-align` builds one by **making
the engines vote**. The 2026 Hugging Face Open ASR Leaderboard's conclusion
after evaluating 60+ systems is that there is no catch-all model: LLM-decoder
models (NVIDIA Canary-Qwen) lead English accuracy, Whisper large-v3 leads
language coverage (99 languages), and Parakeet TDT leads speed and silence
robustness. So instead of picking one, `wordalign.ensemble` implements a
time-anchored, confidence-weighted voting scheme (a lightweight descendant of
ROVER, Fiscus 1997):

- **WhisperX** is the backbone: best multilingual coverage *and* the word
  timestamps everything downstream needs.
- **Parakeet TDT** and **Canary-Qwen** (optional, via NeMo) act as
  high-accuracy challengers; a backbone word is replaced only when the
  weighted vote against it clears a margin.
- **Vosk** contributes an architecturally independent opinion for cheap.
- Every consensus word keeps a fused agreement score, and low-agreement words
  are flagged for human review — because in professional captioning, knowing
  *where the transcript is uncertain* is as valuable as the transcript itself.

The consensus is then fed through the exact same alignment waterfall as a
human transcript, so both modes share one battle-tested code path.

## Install

```bash
git clone https://github.com/ghalymt/word-align
cd word-align
pip install -r requirements.txt          # core (WhisperX path)
pip install vosk                         # optional: waterfall stage 1 / voter
pip install "nemo_toolkit[asr]"          # optional: Parakeet & Canary voters
pip install tensorflow tensorflow-hub    # optional: experimental tagging
pip install textgrid                     # optional: MFA gap-filling
```

`ffmpeg`/`ffprobe` must be on PATH. For the MFA stage, install
[Montreal Forced Aligner](https://montreal-forced-aligner.readthedocs.io/) and
either put `mfa` on PATH or point `WORDALIGN_MFA` at the executable or its
conda env. Vosk models go in any directory, referenced via `--vosk-models` or
`WORDALIGN_VOSK_MODELS`.

Because most of those pieces are optional and several are heavy, there is a
preflight check that tells you exactly what is present, what is missing, and
which of the two categories it falls into:

```bash
python preflight.py --vosk-models /path/to/vosk-models --lang en
```

It reports CUDA and VRAM, verifies `ffprobe` actually executes rather than
merely existing on PATH, and exits non-zero only when something genuinely
blocks a run — optional gaps are listed but never fail the check.

## Use

```bash
# Reference mode: you have a verbatim transcript
python -m wordalign interview.mp4 -t interview.txt --vosk-models ~/vosk-models

# Ensemble mode: audio only, engines vote on the transcript
python -m wordalign interview.mp4 --engines whisperx,parakeet,vosk -l en

# Ensemble mode with a Word transcript deliverable
python -m wordalign interview.mp4 -l en --doc docx

# Extra timing source + experimental audio-event tags
python -m wordalign film.mkv -t film.txt --srt rough_cut.srt --tags

# Force CPU (e.g. no CUDA available, or debugging a GPU issue)
python -m wordalign interview.mp4 -t interview.txt --device cpu
```

Run `python -m wordalign --help` for the full flag list. The ones worth
knowing: `--engines` picks the ensemble voters, `--device {cuda,cpu}` forces
the compute device, `--no-vosk` / `--no-mfa` skip waterfall stages, `--doc`
selects the transcript document format, and `-o` redirects the output
directory.

Outputs land next to the audio (or in `-o DIR`):

| file                        | contents                                        |
| --------------------------- | ----------------------------------------------- |
| `*_word_level.srt`          | one cue per word — the precision product        |
| `*_sentence_level.srt`      | merged, balanced, CPL/duration-validated cues   |
| `*_transcript.txt` / `.docx`| ensemble mode: consensus transcript document — paragraphed by speech pauses, `[HH:MM:SS]` stamps, and (docx) low-agreement words highlighted for review. Select with `--doc`: `txt`, `docx`, or `both` |
| `*_audio_tags.srt` (`--tags`) | experimental: `[laughs]`-style event track    |
| `*_combined.srt` (`--tags`) | dialogue + events, collision-adjusted           |

## Segmentation rules

The segmenter enforces professional captioning constraints: 32 characters per
line, 3-second maximum block duration, balanced line splits that respect
protected phrases and prefer breaking **before** function words, with
language-aware clause markers for EN/FR/ES/DE/NL/IT/PT. The merge sweep runs
with a character budget growing 1→32, which lets small merges settle before
large ones are considered — greedy single-pass merging produces measurably
worse line breaks.

## Tests

```bash
python tests/test_pipeline.py     # or: python -m pytest tests/ -q
```

12 tests, no GPU / audio / model weights required. They cover ensemble voting
behaviour (challenger correction, the replacement margin, backbone selection,
the reinforce-only role of untimed engines), the fill-only semantics of the
waterfall, interpolation monotonicity, CPL enforcement in the segmenter and
in the output validator, tag stripping, terminal-punctuation detection, and
transcript paragraphing. Several are regression tests pinned to specific bugs
— each names the bug it guards against. Engine adapters need real audio and
are not covered.

## Status

**Production-proven:** the alignment waterfall and the 3-phase segmenter.
This is the logic that shipped tens of thousands of captioning jobs, and the
test suite pins its behaviour.

**GPU-verified (reference mode):** the packaged reference-mode pipeline —
Vosk → WhisperX/wav2vec2 → MFA gap-filling → interpolation → segmenter — has
been run end-to-end on real audio on an RTX 4070 Ti, producing valid word- and
sentence-level SRT (all lines within the CPL limit, timings monotonic). Run
`python preflight.py` first; it reports exactly what's installed.

**Beta — voting logic tested, NeMo voters not yet exercised on real audio:**
transcript-free ensemble mode. The consensus algorithm is unit-tested, but the
NeMo Parakeet/Canary adapters need a heavy optional install and have not been
run against real audio in their packaged form. Treat first ensemble runs as a
shakedown, and please open an issue if a voter misbehaves.

**Experimental:** audio-event tagging (YAMNet), behind `--tags`.

Ensemble voter weights are leaderboard-informed defaults; tune
`wordalign/ensemble.py::DEFAULT_PRIORS` for your domain.

## Known limitations

Worth knowing before you reach for these paths:

- **The MFA gap-filling stage is English-only.** `english_us_arpa` is
  hardcoded for both the G2P and acoustic models. Non-English audio still
  works — MFA is stage 4 of 5, and the waterfall degrades to interpolation —
  but you lose that stage's precision. Threading a language parameter through
  `MFAWrapper` is the obvious next contribution.
- **Ensemble mode assumes English when `-l` is omitted.** The language is
  needed up front to choose voters, before any audio has been decoded, so
  there is no auto-detection in this mode. Pass `-l` explicitly for anything
  that isn't English.
- **Untimed engines can reinforce the backbone but never correct it.**
  Canary-Qwen returns text without word timestamps, so its votes are
  projected onto backbone positions by exact-token matching — which means a
  vote only ever lands on a word the backbone already produced. In practice
  Canary is a strong *confidence* signal and a strong defender of contested
  slots, but it cannot propose a substitution. Fuzzy projection would change
  this and is the second obvious contribution.
- **Vosk models are large.** They are loaded once per worker process and
  reused across chunks; with `MAX_WORKERS = 4` and a 1.8 GB English model,
  budget memory accordingly or lower `MAX_WORKERS` in `config.py`.
- **Sentence integrity outranks the duration cap.** A single sentence longer
  than `MAX_DURATION_MS` is kept whole rather than split mid-clause; the
  output validator reports it as `duration_long` instead. This is deliberate
  — a cue that breaks a sentence badly is worse than a cue that lingers — but
  it means a clean run can still report duration issues. Lower
  `MAX_CPL`/`ITERATION_END` if you would rather force earlier breaks.

## License

MIT © 2026 [Mohamed Ghali](https://www.linkedin.com/in/mohammed-ghaly-subtitler)
— veterinarian turned speech-pipeline engineer; 50,000+ captioning projects
delivered.

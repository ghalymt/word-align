# The robust flow (bad audio)

WordAlign 2.2 adds a set of stages for recordings that defeat a single ASR
engine: background music, crowd noise, steady noise, phone audio, people
talking over each other, and long silences where engines invent text. The
stages are switched on by the **`robust` profile** (CLI `--profile robust`,
GUI preset **Robust**). Every stage is optional: anything that is not
installed is skipped with a message, and the run carries on.

English is the best-supported language. The other languages work where the
models support them (see [Models](#models-and-licenses)).

```text
media ─► 1 analysis ─► 2 cleanup ─► 3 voting ─► 4 speakers ─► 5 timing ─► 6 tags ─► 7–8 review queue
         VAD, SNR,     only where    ROVER over   overlap map   anchors +    SDH tag    Accept / Quick
         sound events  it helps      3–5 engines  (+ SE-DiCoW)  forced       proposals  check / Redo
                                                                alignment
```

## Quick start

```powershell
# 1. Install the optional backends and models (Windows; run once)
.\scripts\install_robust.ps1 -Bundle E:\Wordalign\dist\WordAlign -Models E:\Wordalign\models

# 2. Check what was found
wordalign --doctor

# 3. Run
wordalign talk.mp4 --profile robust -l en                 # no transcript
wordalign talk.mp4 -t talk.txt --profile robust -l en     # align your transcript
```

Each run writes `<name>_review.json` next to the subtitles. Open the job in
the GUI to work through the review queue, or read the file yourself.

## Installing

`scripts/install_robust.ps1` creates the extra Python environments next to
the existing `qwen` and `whisperx` ones and downloads the models into the
models folder. It is idempotent: run it again to add a component; `-Rebuild`
recreates a venv.

| Component | What it is | Where it runs | Size |
|---|---|---|---|
| `vad` | Silero VAD 6.2.1 (ONNX) | main app (onnxruntime) | ~2 MB |
| `sed` | PretrainedSED BEATs sound events | `whisperx` venv | ~0.4 GB |
| `diarize` | pyannote speaker-diarization-community-1 (gated) | `whisperx` venv | ~0.1 GB |
| `separate` | Mel-Band / BS RoFormer (audio-separator) | `separate` venv | ~1 GB |
| `denoise` | MossFormerGAN SE 16 kHz (ClearerVoice) | `denoise` venv | ~0.2 GB |
| `cohere` | Cohere Transcribe 03-2026 (gated) | `cohere` venv | ~4 GB |
| `parakeet` | parakeet-ultra | `parakeet` venv | ~2.5 GB |
| `sedicow` | SE-DiCoW (overlapped speech) | `whisperx` venv | ~4 GB |

```powershell
.\scripts\install_robust.ps1 -Components vad,sed,diarize      # just the analysis + speakers
.\scripts\install_robust.ps1 -Bundle D:\WordAlign -Models D:\models -SkipModels
```

Models can also be fetched on their own, on any OS:

```bash
python scripts/fetch_models.py --models-dir E:/Wordalign/models --list     # what is there
python scripts/fetch_models.py --models-dir E:/Wordalign/models vad sed    # fetch some
```

Gated Hugging Face models (Cohere Transcribe, pyannote Community-1) need you
to accept their terms on huggingface.co once and log in
(`huggingface-cli login`) or set `HF_TOKEN`.

### Where WordAlign looks

| What | Default | Override |
|---|---|---|
| Models root | `<app>/models` | `--models-dir DIR` or `WORDALIGN_MODELS_DIR` |
| Backend venvs | `<app>/venvs/<name>` (portable) or `<project>/venvs/<name>` | `WORDALIGN_VENVS_DIR` |
| One backend's Python | — | `WORDALIGN_<NAME>_PYTHON`, e.g. `WORDALIGN_COHERE_PYTHON` |
| Silero VAD file | `<models>/vad/silero_vad.onnx` | `WORDALIGN_VAD_MODEL` |
| Sound-event code + checkpoint | `<models>/sed/PretrainedSED` | `WORDALIGN_SED_DIR` |
| Hugging Face models | `<models>/huggingface/hub` | `--hf-cache` / `WORDALIGN_HF_CACHE` |
| Triage weights | `~/.wordalign/triage_weights.json` | `WORDALIGN_HOME` |

`<NAME>` is one of `SEPARATE`, `DENOISE`, `COHERE`, `PARAKEET`, `WHISPERX`,
`QWEN`, and `SED`, `DIARIZE`, `SEDICOW` (these three default to the
`whisperx` venv). Nothing is read from the Hugging Face cache in your user
folder unless you point a setting at it.

`wordalign --doctor` lists every stage, what it found and where.

## The stages

### 1. Analysis

Silero VAD gives a speech probability every 32 ms and the speech segments.
Speech is cut into ASR chunks of at most 28 s at the quietest frame, never
across a pause longer than 2 s. For every chunk WordAlign measures:

- **SNR** (WADA-SNR, in dB),
- **narrowband** (phone audio: little energy above 3.4 kHz),
- **clipping** (share of samples at full scale),
- **sound events** from PretrainedSED (music, crowd, laughter, applause,
  noise) at 40 ms resolution.

These give each chunk its conditions: `music`, `crowd`, `low_snr` (< 10 dB),
`very_low_snr` (< 3 dB), `noise` (sound-event model *and* SNR < 25 dB),
`narrowband`, `clipping`, and later `overlap`. The quality map is cached with
the stage cache. `--no-vad` turns the stage off; `--no-sound-events` keeps
the VAD and measures but skips the sound-event model.

### 2. Cleanup (gated)

Cleanup only ever produces a second track, `asr_audio`, for the engines
that read text. Timing is always measured on the original audio. With
`--cleanup auto` (the robust profile) a chunk is cleaned only when it needs
it:

| Chunk | Treatment |
|---|---|
| music under speech and SNR < 8 dB | RoFormer vocal separation |
| crowd under speech and SNR < 3 dB | RoFormer crowd removal |
| other speech with SNR < 10 dB | MossFormerGAN, mixed back at 50 % |
| narrowband or clipped | never (enhancement makes these worse) |

Cleaned chunks are cross-faded into the original. `--cleanup all` cleans
every chunk with music or crowd regardless of SNR; `--cleanup off` (the
default outside the robust profile) skips the stage.

### 3. Voting (transcript-free mode)

Without a transcript, every voter transcribes the same VAD chunks of
`asr_audio`, and ROVER builds a word network and votes per chunk, weighting
each engine by a prior (Cohere 1.0, Qwen 0.95, SE-DiCoW 0.9, parakeet 0.85,
WhisperX 0.8, Vosk 0.4; change them in a profile's `voter_priors`). An
engine that returns nothing for a chunk votes for silence, which is how
invented "Thank you." lines disappear. Each word keeps its agreement
(`text_conf`) and the losing readings (`alternatives`), which the review
queue shows.

Choose the voters with `--voters cohere,qwen,parakeet,whisperx` (the robust
profile's set) or add `vosk`. Cohere has no timestamps; it still votes on
the words.

**Hallucination filter.** Before voting, words that the audio does not
support are removed: long repetition loops, short loops in low-speech
audio, and stock phrases ("thank you for watching") sitting alone in
non-speech. Removed words are listed in the review file.
`--no-hallucination-filter` keeps them.

### 4. Speakers and overlap

With `--overlap-map` (robust profile) or `--diarize`, pyannote Community-1
(falling back to 3.1) runs before ASR. Its turns give the overlap map:
cues inside overlapped speech get a higher review risk. `--diarize` also
labels the speakers in the subtitles; `--dash-cues` joins a quick
two-speaker exchange into one dash cue (`-Are you coming?` / `-Yes.`;
`--dash-style "- "` adds a space).

`--overlap-asr` (transcript-free, English) additionally runs SE-DiCoW, a
speaker-conditioned Whisper, on the overlapped regions so the second
speaker's words are transcribed too. It implies `--diarize`. It is off in
every profile because it is slow and only pays off on real crosstalk.

### 5. Timing: anchors and forced alignment

`--timing anchor` (the default from 2.2) replaces the v2.1 fill-the-gaps
waterfall:

1. Every timed source (Vosk, Qwen3-ASR, parakeet, WhisperX, a rough SRT) is
   matched to the final words. A run of 3+ matching words is an **anchor**;
   it is **strong** when two sources agree within 150 ms.
2. Windows between anchors (or, in transcript-free mode, the voted VAD
   chunks) are aligned with **Qwen3-ForcedAligner** on the *original*
   audio (wav2vec2 when Qwen is not installed or the language is not
   covered). A window result is rejected when it collapses, places too few
   words, or disagrees with the strong anchors; a rejected window is
   retried on the cleaned track, and otherwise keeps the anchor times.
3. MFA, then pause-aware interpolation, fill what is left. Interpolation
   uses the VAD so a guessed word lands in speech, not silence, and every
   guessed word is flagged for review.

In transcript-free mode Vosk is skipped when two timed voters and the forced
aligner are available. `--no-forced-align` keeps the engine/anchor times;
`--timing fill_only` brings back the v2.1 waterfall.

Qwen3-ForcedAligner covers zh, en, yue, fr, de, it, ja, ko, pt, ru, es.

### 6. Tags

With `--tags`, the sound-event detections become SDH tags (`[music]`,
`[laughter]`, `[applause]`, ...). In reference mode they are matched to the
tags in your transcript; in transcript-free mode they are written as
proposals for review.

### 7–8. Review queue

Every cue gets a risk score from its words (voter disagreement, words no
engine heard, interpolated or weakly confirmed timing), its conditions
(overlap, music, low SNR, ...) and hallucination removals, and one of three
tiers:

| Tier | Meaning |
|---|---|
| **Accept** | risk < 0.25: nothing suggests an error |
| **Quick check** | worth a listen; most need no change |
| **Redo** | predicted error share ≥ 30 %, the point where retyping beats editing |

`<name>_review.json` holds the tiers, the reasons in plain words, each
word's confidence and alternatives, and a `queue` of cue numbers riskiest
first. `--no-triage` skips it.

**In the GUI** a finished job shows step 4, *Review*: filter to Redo +
Quick check, play a cue on a loop (`Esc`), fix the text (one click applies a
voter's alternative), and accept (`Ctrl+Enter` moves to the next cue).
Every save writes, next to the job's files:

- `<name>_corrected.srt` — the subtitles with your fixes (the pipeline's own
  files are never changed),
- `<name>_review_edits.json` — what you did to each cue,
- `<name>_review_log.jsonl` — an append-only log of every change.

**Calibrating the tiers.** The weights behind the risk score are a starting
point. After you have corrected a few jobs (in the GUI, or by saving your
own `<name>_corrected.srt` next to `<name>_review.json`), re-fit them:

```bash
wordalign-bench fit-triage --jobs "%USERPROFILE%\WordAlign\outputs"
wordalign-bench fit-triage --jobs D:\reviewed --dry-run    # show, do not save
```

A cue counts as "needed a fix" when its words changed or its start or end
moved by more than 150 ms. The new weights go to
`~/.wordalign/triage_weights.json` and every later run uses them.

## Settings reference

| CLI flag | Profile key (`"robust": {...}`) | robust profile | default |
|---|---|---|---|
| `--no-vad` | `analysis` | on | on |
| `--no-sound-events` | `sound_events` | on | on |
| `--cleanup off\|auto\|all` | `cleanup` | auto | off |
| `--voters LIST` | `voters` | cohere, qwen, parakeet, whisperx | the `--engines` set |
| `--no-hallucination-filter` | `hallucination_filter` | on | on |
| `--overlap-map` | `overlap_map` | on | off |
| `--overlap-asr` | `overlap_asr` | off | off |
| `--diarize`, `--dash-cues`, `--dash-style` | `diarize`, `dash_cues` | off | off |
| `--timing anchor\|fill_only` | — | anchor | anchor |
| `--no-forced-align` | `forced_align` | on | on |
| `--no-triage` | `triage` | on | on |

A flag on the command line always wins over the profile. `maximum_quality`
uses the same stages as `robust`; `fast` and `cpu_only` skip the
sound-event model and forced alignment.

## What it costs

On an RTX 4070 Ti (12 GB), a 45 s English clip ran in about 3½ minutes
with the robust profile, most of it loading four ASR models once each. The
per-engine time grows roughly linearly with the audio after that; cleanup
only runs on the chunks that need it. GPU engines run one at a time
(`WORDALIGN_GPU_SLOTS`), so 12 GB is enough for every model in the list.

## Troubleshooting

- **A stage says "skipped"** — run `wordalign --doctor`; it names the
  missing venv or model and the command that installs it.
- **pyannote: "accept the terms"** — open the model page on huggingface.co
  while logged in, accept, and make sure `HF_TOKEN` or `huggingface-cli
  login` is set for the user that runs WordAlign.
- **Out of GPU memory in Qwen** — chunks that fail are retried in halves;
  pieces that still fail are reported and left to the other voters.
- **The review queue is silent** — the GUI keeps a small Opus copy of the
  audio for playback; it needs ffmpeg with libopus (AAC or WAV are tried
  next).
- **Results look like v2.1** — check that the profile is `robust` and that
  `--timing fill_only` is not set.

## Models and licenses

| Model | License | Languages |
|---|---|---|
| Silero VAD 6.2.1 | MIT | any |
| PretrainedSED BEATs (strong) | MIT | any |
| pyannote speaker-diarization-community-1 | CC-BY-4.0 (gated) | any |
| Mel-Band / BS RoFormer (audio-separator) | per checkpoint | any |
| MossFormerGAN SE 16 kHz | Apache-2.0 | any |
| Cohere Transcribe 03-2026 | Apache-2.0 (gated) | ar, de, el, en, es, fr, it, ja, ko, nl, pl, pt, vi, zh |
| parakeet-ultra | CC-BY-4.0 | 25 European languages |
| SE-DiCoW | Apache-2.0 | en |
| Qwen3-ASR-1.7B / Qwen3-ForcedAligner-0.6B | Apache-2.0 | see above |

See also [the benchmark](benchmark.md) for how the flow is measured.

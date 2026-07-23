"""word-align: word-accurate subtitle timing via multi-engine alignment."""
# 0.10.0: Qwen3-ASR replaces NeMo as the default ensemble voter -- a *timed*
# challenger (word timestamps via Qwen3-ForcedAligner) reached through a
# subprocess bridge to its own venv. Ensemble mode is now verified end-to-end
# on GPU: vote weight follows transcription accuracy (WhisperX > Qwen > Vosk),
# Vosk seeds the timing, and the consensus is line-structured so segmentation
# no longer collapses to a single cue. Builds on 0.9.x (Windows mp guard,
# abbreviation-aware splitting, source-aware overlap resolution).
__version__ = "0.10.0"

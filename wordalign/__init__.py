"""word-align: word-accurate subtitle timing via multi-engine alignment."""
# 1.0.0: both modes verified end-to-end on GPU and the subtitle layout follows
# the .srt standard (configurable 42x2 / 32x2). Reference mode (Vosk -> WhisperX
# -> MFA -> interpolation -> segmenter) is production-derived and GPU-verified;
# ensemble mode fuses WhisperX + Qwen3-ASR + Vosk with a no-panic balancer,
# isolated-tag protection, zero-overlap timing, and a minimum-duration guard.
# 23 tests. See the git history for the 0.9.x -> 1.0.0 path.
__version__ = "2.1.0"

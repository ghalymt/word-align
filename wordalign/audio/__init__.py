"""Audio analysis and preparation for bad-audio robustness.

Stage 1 (``analysis``) measures the audio before any ASR runs: where the
speech is (Silero VAD), what else is going on (sound-event classes), how
loud the noise is (WADA-SNR), and whether the channel is narrowband or
clipped. Stage 2 (``prep``) uses that map to clean only the stretches where
cleaning is known to help, producing a second track for the ASR engines;
word timing always reads the original audio.
"""

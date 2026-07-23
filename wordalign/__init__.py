"""word-align: word-accurate subtitle timing via multi-engine alignment."""
# 0.11.0: configurable subtitle layout -- per-line CPL x line count (42x2 for
# regular video, 32x2 for vertical), max duration, min cue duration -- plus a
# "no-panic" balancer that never forces an ugly break, isolated-tag protection,
# overlap resolution iterated to zero, and a minimum-duration guard that clears
# sub-frame cues. 0.10.0 added the Qwen3-ASR timed ensemble voter; 0.9.x fixed
# the Windows mp guard, abbreviation splitting, and source-aware overlaps.
__version__ = "0.11.0"

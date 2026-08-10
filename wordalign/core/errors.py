"""Exception hierarchy for WordAlign."""
from __future__ import annotations


class WordAlignError(Exception):
    """Base exception for all WordAlign errors."""


class EngineError(WordAlignError):
    """An engine failed during execution."""
    def __init__(self, engine_id: str, message: str, recoverable: bool = False):
        self.engine_id = engine_id
        self.recoverable = recoverable
        super().__init__(f"[{engine_id}] {message}")


class OOMError(EngineError):
    """GPU out of memory."""
    def __init__(self, engine_id: str, message: str = "CUDA out of memory"):
        super().__init__(engine_id, message, recoverable=True)


class RuntimeMissingError(EngineError):
    """Required runtime/dependency is not installed."""
    def __init__(self, engine_id: str, missing: list[str]):
        self.missing = missing
        super().__init__(engine_id,
                         f"Missing: {', '.join(missing)}", recoverable=False)


class ModelNotValidatedError(EngineError):
    """Model directory failed validation."""
    def __init__(self, engine_id: str, reason: str):
        super().__init__(engine_id, f"Model invalid: {reason}", recoverable=False)


class SubprocessCrashError(EngineError):
    """A subprocess worker crashed."""
    def __init__(self, engine_id: str, stderr_tail: str):
        self.stderr_tail = stderr_tail
        super().__init__(engine_id, f"Worker crashed:\n{stderr_tail}",
                         recoverable=False)


class SubprocessTimeoutError(EngineError):
    """A subprocess worker timed out."""
    def __init__(self, engine_id: str, timeout_seconds: float):
        super().__init__(engine_id,
                         f"Timed out after {timeout_seconds:.0f}s",
                         recoverable=True)


class PipelineCancelled(WordAlignError):
    """Pipeline was cancelled by the user."""
    pass


class ProfileValidationError(WordAlignError):
    """A pipeline profile failed validation."""
    pass

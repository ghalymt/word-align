"""GPU resource scheduler.

Prevents launching multiple large GPU models simultaneously, detects
OOM conditions, and offers fallback strategies (smaller chunks, CPU
fallback, or smaller model).

The scheduler is a per-process semaphore with GPU memory awareness:
- Engines acquire a GPU slot before loading models
- Concurrent GPU loads are limited (default: 1)
- OOM errors are detected and mapped to fallback suggestions
"""
from __future__ import annotations

import itertools
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Union

from .errors import OOMError


@dataclass
class GPUSlot:
    """A held GPU reservation."""
    engine_id: str
    model_id: Optional[str]
    acquired_at: float
    estimated_mb: int = 0
    slot_id: int = 0


@dataclass
class OOMRecovery:
    """Fallback strategies after an OOM."""
    suggest_smaller_chunks: bool = False
    suggest_cpu_fallback: bool = False
    suggest_smaller_model: bool = False
    message: str = ""


class GPUScheduler:
    """Serializes GPU access across engines in this process.

    Parameters
    ----------
    max_concurrent : int
        Maximum number of GPU slots that can be held at once (default 1).
    """

    def __init__(self, max_concurrent: int = 1):
        self.max_concurrent = max_concurrent
        self._sem = threading.BoundedSemaphore(max_concurrent)
        self._lock = threading.Lock()
        # Keyed per acquisition, not per engine: two jobs may both hold (or
        # wait for) a "whisperx" slot, and each must release its own.
        self._slots: Dict[int, GPUSlot] = {}
        self._ids = itertools.count(1)
        self._total_acquires = 0
        self._total_releases = 0

    @property
    def active_slots(self) -> List[GPUSlot]:
        with self._lock:
            return list(self._slots.values())

    @property
    def active_count(self) -> int:
        return len(self.active_slots)

    def acquire(self, engine_id: str, model_id: Optional[str] = None,
                estimated_mb: int = 0,
                timeout_seconds: float = 300.0) -> GPUSlot:
        """Acquire a GPU slot, blocking until one is free.

        Raises
        ------
        TimeoutError
            If no slot becomes free within ``timeout_seconds``.
        """
        acquired = self._sem.acquire(timeout=timeout_seconds)
        if not acquired:
            raise TimeoutError(
                f"GPU slot busy for {engine_id} after {timeout_seconds:.0f}s; "
                "another engine is holding the GPU. Try --device cpu or "
                "reduce concurrent engines."
            )
        slot = GPUSlot(
            engine_id=engine_id,
            model_id=model_id,
            acquired_at=time.time(),
            estimated_mb=estimated_mb,
        )
        with self._lock:
            slot.slot_id = next(self._ids)
            self._slots[slot.slot_id] = slot
            self._total_acquires += 1
        return slot

    def release(self, slot: Union[GPUSlot, str]) -> bool:
        """Release a held slot (or the oldest slot held by an engine id).

        Returns False, and releases nothing, if no such slot is held. The
        old version released the semaphore unconditionally, which raised
        ValueError on a double release and could free a slot someone else
        held.
        """
        with self._lock:
            if isinstance(slot, GPUSlot):
                key = slot.slot_id if slot.slot_id in self._slots else None
            else:
                key = next((k for k, s in self._slots.items()
                            if s.engine_id == slot), None)
            if key is None:
                return False
            del self._slots[key]
            self._total_releases += 1
        self._sem.release()
        return True

    def release_all(self) -> int:
        """Release all held slots. Returns how many were released."""
        with self._lock:
            slots = list(self._slots.values())
        return sum(1 for slot in slots if self.release(slot))

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release_all()


def detect_oom(exc: Exception) -> bool:
    """Detect CUDA out-of-memory conditions from an exception."""
    text = str(exc).lower()
    return any(token in text for token in (
        "out of memory", "cuda out of memory", "cuda error: out of memory",
        "oom", "cublas", "memoryerror", "allocate memory",
    ))


def suggest_recovery(engine_id: str, exc: Exception,
                     chunk_seconds: Optional[float] = None,
                     has_cpu_fallback: bool = True,
                     has_smaller_model: bool = True) -> OOMRecovery:
    """Build a recovery suggestion from an OOM exception."""
    if not detect_oom(exc):
        return OOMRecovery(message=str(exc))

    rec = OOMRecovery(
        suggest_smaller_chunks=chunk_seconds is not None and chunk_seconds > 10,
        suggest_cpu_fallback=has_cpu_fallback,
        suggest_smaller_model=has_smaller_model,
    )
    parts = [f"[{engine_id}] GPU out of memory."]
    if rec.suggest_smaller_chunks:
        parts.append(f"Lower --qwen-chunk-seconds below {chunk_seconds:.0f}s.")
    if rec.suggest_cpu_fallback:
        parts.append("Retry with --device cpu.")
    if rec.suggest_smaller_model:
        parts.append("Use a smaller model (e.g. whisper large-v3-turbo).")
    rec.message = " ".join(parts)
    return rec


def run_with_gpu_slot(scheduler: GPUScheduler,
                      engine_id: str,
                      fn: Callable,
                      model_id: Optional[str] = None,
                      estimated_mb: int = 0,
                      timeout_seconds: float = 300.0,
                      *args, **kwargs):
    """Run ``fn`` while holding a GPU slot; release on exit (even on error)."""
    slot = scheduler.acquire(engine_id, model_id=model_id,
                             estimated_mb=estimated_mb,
                             timeout_seconds=timeout_seconds)
    try:
        return fn(*args, **kwargs)
    finally:
        scheduler.release(slot)


_SHARED: Optional[GPUScheduler] = None
_SHARED_LOCK = threading.Lock()


def shared_scheduler() -> GPUScheduler:
    """The process-wide scheduler every pipeline run shares.

    WORDALIGN_GPU_SLOTS sets how many GPU engines may run at once
    (default 1: one model on the GPU at a time).
    """
    global _SHARED
    with _SHARED_LOCK:
        if _SHARED is None:
            try:
                slots = max(1, int(os.environ.get("WORDALIGN_GPU_SLOTS", "1")))
            except ValueError:
                slots = 1
            _SHARED = GPUScheduler(max_concurrent=slots)
        return _SHARED

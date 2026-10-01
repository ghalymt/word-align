"""LLM QA provider interface and stubs.

Provides the interface for LLM-based semantic review of transcripts.
Two providers are defined:

- LocalModelProvider: wraps a local llama.cpp / transformers model
- OpenAICompatibleProvider: talks to an OpenAI-compatible API endpoint

Both implement the same interface expected by QAEngine._run_llm_review().
"""
from __future__ import annotations

import json
import os
from typing import Any, Optional


class LLMQAProvider:
    """Base interface for LLM-based QA review."""

    def review_chunk(self, center: list, before: list, after: list) -> list[dict]:
        """Review a chunk of transcript words.

        Parameters
        ----------
        center : list[WordResult]
            The words to review (target chunk).
        before : list[WordResult]
            Context words before the chunk.
        after : list[WordResult]
            Context words after the chunk.

        Returns
        -------
        list[dict]
            Findings, each with keys:
            - word_ids: list of word IDs that are suspicious
            - confidence: 0.0-1.0
            - suggested_text: optional corrected text
            - reason: human-readable explanation
        """
        raise NotImplementedError


class LocalModelProvider(LLMQAProvider):
    """Local LLM provider using transformers or llama.cpp.

    Loads a model from the local HuggingFace cache. No downloads are triggered
    if the model is already cached.
    """

    def __init__(self, model_name: str = "Qwen/Qwen2.5-1.5B",
                 device: Optional[str] = None,
                 cache_dir: Optional[str] = None):
        self.model_name = model_name
        self.device = device
        if cache_dir:
            self.cache_dir = cache_dir
        elif os.environ.get("HF_HOME"):
            self.cache_dir = os.environ["HF_HOME"]
        else:
            # Default to the project's local <models>/huggingface/hub so
            # the LLM QA provider never reaches for ~/.cache.
            from ...models.paths import _project_models_root
            self.cache_dir = str(_project_models_root() / "huggingface" / "hub")
        self._model = None
        self._tokenizer = None

    def _ensure_loaded(self):
        """Lazy-load the model on first use."""
        if self._model is not None:
            return
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch

            kwargs = {"cache_dir": self.cache_dir} if self.cache_dir else {}
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name, **kwargs)
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                **kwargs,
            )
            if self.device:
                self._model = self._model.to(self.device)
        except Exception as exc:
            raise RuntimeError(f"Failed to load local model {self.model_name}: {exc}")

    def review_chunk(self, center: list, before: list, after: list) -> list[dict]:
        """Review using local model generation."""
        self._ensure_loaded()

        # Build the prompt
        text_before = " ".join(w.text for w in before[-10:])
        text_center = " ".join(w.text for w in center)
        text_after = " ".join(w.text for w in after[:10])

        prompt = (
            "You are a transcript quality reviewer. "
            "Read the following transcript segment and identify any words "
            "that look incorrect, nonsensical, or like speech recognition errors. "
            "Return a JSON array of findings, each with 'word_ids', 'confidence', "
            "'suggested_text', and 'reason'. If no issues, return [].\n\n"
            f"Context before: {text_before}\n"
            f"Review this: {text_center}\n"
            f"Context after: {text_after}\n\n"
            "Findings (JSON array):"
        )

        try:
            inputs = self._tokenizer(prompt, return_tensors="pt")
            if hasattr(self._model, "device"):
                inputs = {k: v.to(self._model.device) for k, v in inputs.items()}
            outputs = self._model.generate(**inputs, max_new_tokens=256, temperature=0.3)
            response = self._tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:],
                                              skip_special_tokens=True)
            findings = json.loads(response)
            if isinstance(findings, list):
                return findings
        except Exception:
            pass

        return []


class OpenAICompatibleProvider(LLMQAProvider):
    """OpenAI-compatible API provider.

    Works with any endpoint that accepts the OpenAI chat completions format.
    """

    def __init__(self,
                 base_url: str = "http://127.0.0.1:1234/v1",
                 api_key: str = "not-needed",
                 model: str = "local-model"):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model

    def review_chunk(self, center: list, before: list, after: list) -> list[dict]:
        """Review using OpenAI-compatible API."""
        import urllib.request

        text_before = " ".join(w.text for w in before[-10:])
        text_center = " ".join(w.text for w in center)
        text_after = " ".join(w.text for w in after[:10])

        prompt = (
            "You are a transcript quality reviewer. "
            "Read the following transcript segment and identify any words "
            "that look incorrect, nonsensical, or like speech recognition errors. "
            "Return a JSON array of findings, each with 'word_ids', 'confidence', "
            "'suggested_text', and 'reason'. If no issues, return [].\n\n"
            f"Context before: {text_before}\n"
            f"Review this: {text_center}\n"
            f"Context after: {text_after}\n\n"
            "Findings (JSON array):"
        )

        payload = json.dumps({
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3,
            "max_tokens": 256,
        }).encode("utf-8")

        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
        )

        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                content = data["choices"][0]["message"]["content"]
                findings = json.loads(content)
                if isinstance(findings, list):
                    return findings
        except Exception:
            pass

        return []


class NoOpProvider(LLMQAProvider):
    """Provider that always returns no findings. Default when no LLM is configured."""

    def review_chunk(self, center: list, before: list, after: list) -> list[dict]:
        return []

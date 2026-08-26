"""
Model-agnostic LLM client.

Every agent in the harness (planner, attacker, verifier, optimizer) and the target
under test talk to a model through this one interface. It speaks the OpenAI
chat-completions protocol, which is the de-facto standard implemented by OpenAI,
Azure OpenAI, vLLM, Ollama (`/v1`), Together, Groq, OpenRouter, and most local
servers — so you point each role at whatever endpoint you're authorized to use by
setting `base_url` and `model`.

Design goals:
  * Reproducibility — temperature, top_p, and seed are explicit and logged.
  * Robustness — retries with exponential backoff on transient errors.
  * No hard vendor lock — the only dependency is the `openai` python package,
    which is itself just an HTTP client for the standard protocol.

The client never bundles credentials. Supply them via environment variables or the
constructor. Nothing here is specific to any single provider.
"""

from __future__ import annotations

import os
import time
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class GenerationConfig:
    """Decoding parameters. Keep temperature low for the graders (verifier) and
    moderate for the generators (planner/attacker/optimizer)."""

    temperature: float = 0.7
    top_p: float = 1.0
    max_tokens: int = 1024
    seed: Optional[int] = None          # passed through when the backend supports it
    stop: Optional[List[str]] = None


@dataclass
class LLMClient:
    """A thin, retrying wrapper over an OpenAI-compatible chat endpoint."""

    model: str
    base_url: Optional[str] = None                 # e.g. http://localhost:11434/v1 for Ollama
    api_key_env: str = "OPENAI_API_KEY"
    api_key: Optional[str] = None                  # overrides the env var if set
    max_retries: int = 4
    timeout: float = 60.0
    default: GenerationConfig = field(default_factory=GenerationConfig)
    _client: object = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "The 'openai' package is required. Install with: pip install openai"
            ) from exc

        key = self.api_key or os.environ.get(self.api_key_env, "not-needed")
        self._client = OpenAI(api_key=key, base_url=self.base_url, timeout=self.timeout)

    def chat(
        self,
        messages: List[Dict[str, str]],
        config: Optional[GenerationConfig] = None,
    ) -> str:
        """Send a chat request and return the assistant's text.

        Retries transient failures with jittered exponential backoff. Raises the
        last exception if every attempt fails, so callers can decide how to degrade.
        """
        cfg = config or self.default
        last_exc: Optional[Exception] = None

        for attempt in range(self.max_retries):
            try:
                kwargs = dict(
                    model=self.model,
                    messages=messages,
                    temperature=cfg.temperature,
                    top_p=cfg.top_p,
                    max_tokens=cfg.max_tokens,
                )
                if cfg.seed is not None:
                    kwargs["seed"] = cfg.seed
                if cfg.stop:
                    kwargs["stop"] = cfg.stop
                resp = self._client.chat.completions.create(**kwargs)
                return (resp.choices[0].message.content or "").strip()
            except Exception as exc:  # noqa: BLE001 — we retry on anything transient
                last_exc = exc
                if attempt == self.max_retries - 1:
                    break
                sleep = (2 ** attempt) + random.uniform(0, 0.5)
                time.sleep(sleep)

        raise RuntimeError(f"LLM call failed after {self.max_retries} attempts: {last_exc}")

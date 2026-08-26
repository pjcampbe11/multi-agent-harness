"""
Pairwise diversity scoring for attack plans.

The X-Teaming paper measures plan diversity as the mean *pairwise* dissimilarity of
plan embeddings, reporting an average score of 0.702 (vs. 0.278 for the strongest
prior method). We reproduce that metric:

    diversity(S) = mean over all i<j of ( 1 - cosine_similarity(e_i, e_j) )

where e_k is an embedding of plan k's flattened text.

Primary backend is sentence-transformers (`all-MiniLM-L6-v2`) — the same compact
embedding model used elsewhere in this operator's tooling. If that package is not
installed, we fall back to a deterministic lexical embedding (character n-gram +
token hashing) so the harness still runs and still ranks plans sensibly, just with
a coarser notion of "different". The fallback is clearly flagged in `backend`.
"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from typing import List, Sequence

import numpy as np

_TOKEN_RE = re.compile(r"[a-z0-9']+")


@dataclass
class DiversityScorer:
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    _model: object = None
    backend: str = "uninitialized"

    def __post_init__(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
            self.backend = "sentence-transformers"
        except Exception:  # noqa: BLE001 — any failure -> lexical fallback
            self._model = None
            self.backend = "lexical-fallback"

    # ---- embeddings -------------------------------------------------------
    def embed(self, texts: Sequence[str]) -> np.ndarray:
        if self._model is not None:
            vecs = self._model.encode(list(texts), normalize_embeddings=True)
            return np.asarray(vecs, dtype=np.float32)
        return np.stack([self._lexical_embed(t) for t in texts])

    @staticmethod
    def _lexical_embed(text: str, dim: int = 512) -> np.ndarray:
        """Hash tokens and character trigrams into a fixed-width unit vector.

        Deterministic and dependency-free. Not semantic, but it reliably separates
        plans that use different personas/strategies/wording.
        """
        vec = np.zeros(dim, dtype=np.float32)
        toks = _TOKEN_RE.findall(text.lower())
        grams = toks + [text[i : i + 3] for i in range(max(0, len(text) - 2))]
        for g in grams:
            h = int(hashlib.md5(g.encode("utf-8")).hexdigest(), 16)
            vec[h % dim] += 1.0
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    # ---- scoring ----------------------------------------------------------
    def pairwise_diversity(self, texts: Sequence[str]) -> float:
        """Mean pairwise (1 - cosine similarity). Range ~[0, 2], practically [0, 1]."""
        if len(texts) < 2:
            return 0.0
        emb = self.embed(texts)
        sims = emb @ emb.T
        n = len(texts)
        total, count = 0.0, 0
        for i in range(n):
            for j in range(i + 1, n):
                total += 1.0 - float(sims[i, j])
                count += 1
        return total / count if count else 0.0

    def most_redundant_index(self, texts: Sequence[str]) -> int:
        """Index of the plan most similar to the rest — the best candidate to drop
        or regenerate when trimming a set toward higher diversity."""
        emb = self.embed(texts)
        sims = emb @ emb.T
        np.fill_diagonal(sims, 0.0)
        return int(np.argmax(sims.sum(axis=1)))

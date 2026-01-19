# embeddings/tender_embedder.py

from __future__ import annotations

import math
import os
from typing import Iterable, List, Sequence, Union

from sentence_transformers import SentenceTransformer

try:
    from FlagEmbedding import BGEM3FlagModel
except Exception:  # pragma: no cover - optional dependency
    BGEM3FlagModel = None

DEFAULT_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-m3")
DEFAULT_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "auto").strip().lower()
DEFAULT_USE_FP16 = os.getenv("EMBEDDING_USE_FP16", "true").strip().lower()


def _is_truthy(value: str) -> bool:
    return value in {"1", "true", "yes", "y", "on"}


def _supports_fp16() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False


def _normalize_vector(vec: List[float]) -> List[float]:
    norm = math.sqrt(sum(float(v) * float(v) for v in vec))
    if norm <= 0:
        return vec
    return [float(v) / norm for v in vec]


def _normalize_vectors(vectors: List[List[float]]) -> List[List[float]]:
    return [_normalize_vector(vec) for vec in vectors]


class TenderEmbedder:
    def __init__(self, model_name: str | None = None, provider: str | None = None) -> None:
        self.model_name = model_name or DEFAULT_MODEL_NAME
        self.provider = (provider or DEFAULT_PROVIDER).strip().lower()
        self._backend = "sentence-transformers"
        self.model = self._init_model()

    def _init_model(self):
        if self.provider not in {"auto", "sentence-transformers", "flagembedding"}:
            raise ValueError(f"Unknown embedding provider: {self.provider}")

        use_flag = self.provider == "flagembedding" or (
            self.provider == "auto" and _is_bge_m3(self.model_name) and BGEM3FlagModel is not None
        )

        if use_flag:
            if BGEM3FlagModel is None:
                raise RuntimeError("FlagEmbedding is required for BGEM3FlagModel but is not installed.")
            use_fp16 = _is_truthy(DEFAULT_USE_FP16) and _supports_fp16()
            self._backend = "flagembedding"
            return BGEM3FlagModel(self.model_name, use_fp16=use_fp16)

        st_kwargs = {}
        if _is_bge_m3(self.model_name):
            st_kwargs["trust_remote_code"] = True
        self._backend = "sentence-transformers"
        return SentenceTransformer(self.model_name, **st_kwargs)

    def embed(self, texts: Union[str, Sequence[str]]) -> List[List[float]]:
        """
        Encode text(s) into embedding vectors.
        """
        normalized = self._normalize_texts(texts)
        if not normalized:
            return []

        if self._backend == "flagembedding":
            output = self.model.encode(
                normalized,
                return_dense=True,
                return_sparse=False,
                return_colbert_vecs=False,
            )
            dense = output.get("dense_vecs") if isinstance(output, dict) else None
            if dense is None:
                return []
            if hasattr(dense, "tolist"):
                dense = dense.tolist()
            return _normalize_vectors(dense)

        vectors = self.model.encode(
            normalized,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return vectors.tolist()

    @staticmethod
    def _normalize_texts(texts: Union[str, Sequence[str]]) -> List[str]:
        if isinstance(texts, str):
            return [texts]

        if isinstance(texts, Iterable):
            return [t for t in texts if isinstance(t, str) and t.strip()]

        return []


def _is_bge_m3(model_name: str) -> bool:
    return model_name.strip().lower().startswith("baai/bge-m3")

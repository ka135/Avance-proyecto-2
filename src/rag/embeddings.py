"""Embeddings LOCALES (ONNX vía fastembed): el texto del corpus no sale de la máquina."""
import hashlib
import os
import re
from typing import List

import numpy as np

_model = None


def _fastembed(name: str):
    global _model
    if _model is None or getattr(_model, "_name", None) != name:
        from fastembed import TextEmbedding
        _model = TextEmbedding(model_name=name)
        _model._name = name
    return _model


def _hash_embed(texts: List[str], dim: int = 384):
    """Backend de PRUEBA (sin red): bolsa de palabras hasheada. No usar en producción."""
    out = np.zeros((len(texts), dim), dtype="float32")
    for i, t in enumerate(texts):
        for w in re.findall(r"\w+", t.lower()):
            out[i, int(hashlib.md5(w.encode()).hexdigest(), 16) % dim] += 1
    return out


def embed_texts(texts: List[str], model_name: str) -> List[List[float]]:
    if os.getenv("EMBEDDING_BACKEND", "fastembed") == "hash":
        arr = _hash_embed(texts)
    else:
        arr = np.array(list(_fastembed(model_name).embed(texts)), dtype="float32")
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return (arr / norms).tolist()

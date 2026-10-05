"""Recuperación híbrida: similitud de coseno (semántica) + coincidencia de términos exactos.

Los embeddings son flojos con códigos y nombres propios (p. ej. "E04", "TX-200"); la parte léxica
corrige eso. Con hybrid=False se usa solo coseno (útil como línea base en la evaluación).
"""
import re
import unicodedata
from dataclasses import dataclass

from .config import RagConfig
from .embeddings import embed_texts
from .vectorstore import get_collection

_STOP = {"que", "como", "cual", "cuales", "cuando", "para", "por", "con", "del", "los", "las", "una",
         "uno", "esta", "este", "hay", "mi", "si", "es", "en", "de", "la", "el", "se", "un", "lo",
         "al", "me", "su", "sus", "no", "ya", "mas", "muy", "son", "ser", "hago", "debo", "puedo"}


@dataclass
class Retrieved:
    text: str
    source: str
    section: str
    page: int
    chunk_id: str
    score: float          # puntaje usado para ordenar (híbrido o coseno)
    cosine: float = 0.0   # similitud de coseno pura


def _tokens(text: str) -> set:
    t = unicodedata.normalize("NFD", text.lower())
    t = "".join(c for c in t if unicodedata.category(c) != "Mn")
    return {w for w in re.findall(r"[a-z0-9]+", t) if len(w) >= 2 and w not in _STOP}


def retrieve(query: str, cfg: RagConfig) -> list:
    col = get_collection(cfg)
    pool = min(col.count(), max(cfg.top_k * 3, 10)) if cfg.hybrid else cfg.top_k
    res = col.query(query_embeddings=embed_texts([query], cfg.embedding_model), n_results=pool,
                    include=["documents", "metadatas", "distances"])
    items = []
    for doc, m, d in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        cos = round(1 - d, 4)
        items.append(Retrieved(doc, m["source"], m["section"], m["page"], m["chunk_id"], cos, cos))

    if cfg.hybrid:
        q = _tokens(query)
        for it in items:
            lex = len(q & _tokens(f"{it.section} {it.text}")) / len(q) if q else 0.0
            it.score = round((1 - cfg.lexical_weight) * it.cosine + cfg.lexical_weight * lex, 4)
        items.sort(key=lambda x: x.score, reverse=True)
    return items[:cfg.top_k]

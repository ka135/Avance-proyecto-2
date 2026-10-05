"""Índice vectorial persistente en Chroma, con metadatos para citar la fuente."""
import sys

try:  # Streamlit Cloud trae un sqlite antiguo; chromadb necesita >= 3.35
    __import__("pysqlite3")
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass

import chromadb

from .chunking import chunk_corpus
from .config import CHROMA_DIR, DOCS_DIR, RagConfig
from .embeddings import embed_texts
from .ingest import load_documents


def _client():
    return chromadb.PersistentClient(path=str(CHROMA_DIR))


def get_collection(cfg: RagConfig):
    return _client().get_or_create_collection(
        name=cfg.collection_name, metadata={"hnsw:space": "cosine"})


def build_index(cfg: RagConfig, docs_dir=DOCS_DIR, rebuild: bool = False):
    """Ingesta -> chunking -> embeddings -> Chroma. Idempotente: si ya existe, la reutiliza."""
    client = _client()
    if rebuild:
        try:
            client.delete_collection(cfg.collection_name)
        except Exception:
            pass
    col = client.get_or_create_collection(
        name=cfg.collection_name, metadata={"hnsw:space": "cosine"})
    if col.count() > 0:
        return col

    chunks = chunk_corpus(load_documents(docs_dir), cfg.chunk_size, cfg.chunk_overlap)
    # El título de la sección se antepone SOLO para embeber (mejora la recuperación);
    # el texto almacenado y mostrado es el fragmento original.
    to_embed = [f"{c.section}\n{c.text}" if c.section else c.text for c in chunks]
    vectors = embed_texts(to_embed, cfg.embedding_model)
    for i in range(0, len(chunks), 256):
        b = slice(i, i + 256)
        col.add(
            ids=[c.chunk_id for c in chunks[b]],
            documents=[c.text for c in chunks[b]],
            embeddings=vectors[b],
            metadatas=[{"source": c.source, "section": c.section,
                        "page": c.page or 0, "chunk_id": c.chunk_id} for c in chunks[b]],
        )
    return col

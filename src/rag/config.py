"""Configuración central del pipeline RAG."""
import hashlib
import os
from dataclasses import dataclass
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

ROOT = Path(__file__).resolve().parents[2]
DOCS_DIR = ROOT / "documentos"
CHROMA_DIR = ROOT / "chroma_db"
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def get_secret(name: str, default=None):
    """Lee una credencial desde variable de entorno o st.secrets (nunca del repo)."""
    value = os.getenv(name)
    if value:
        return value
    try:
        import streamlit as st
        return st.secrets.get(name, default)
    except Exception:
        return default


@dataclass(frozen=True)
class RagConfig:
    chunk_size: int = 800          # caracteres por fragmento
    chunk_overlap: int = 120       # solapamiento entre fragmentos consecutivos
    top_k: int = 4                 # fragmentos enviados al LLM
    hybrid: bool = True            # similitud semántica + coincidencia de palabras exactas
    lexical_weight: float = 0.3    # peso de la parte léxica en el puntaje híbrido
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    max_history_messages: int = 6  # últimos mensajes usados en el chat

    @property
    def collection_name(self) -> str:
        backend = os.getenv("EMBEDDING_BACKEND", "fastembed")
        tag = hashlib.md5(f"{self.embedding_model}{backend}".encode()).hexdigest()[:6]
        return f"manuales_c{self.chunk_size}_o{self.chunk_overlap}_{tag}"


PROVIDERS = {
    "gemini": {"key": "GEMINI_API_KEY", "model": "gemini-3.8-flash", "base_url": None},
    "mistral": {"key": "MISTRAL_API_KEY", "model": "mistral-small-latest", "base_url": "https://api.mistral.ai/v1"},
    "groq": {"key": "GROQ_API_KEY", "model": "openai/gpt-oss-120b", "base_url": "https://api.groq.com/openai/v1"},
}


def provider() -> str:
    p = str(get_secret("LLM_PROVIDER", "gemini")).lower()
    if p not in PROVIDERS:
        raise RuntimeError(f"LLM_PROVIDER debe ser uno de {list(PROVIDERS)}")
    return p


def api_key():
    return get_secret(PROVIDERS[provider()]["key"])


def base_url():
    return PROVIDERS[provider()]["base_url"]


def llm_model() -> str:
    explicit = get_secret("LLM_MODEL") or (get_secret("GEMINI_MODEL") if provider() == "gemini" else None)
    return explicit or PROVIDERS[provider()]["model"]


def gemini_model() -> str:  # compatibilidad
    return llm_model()
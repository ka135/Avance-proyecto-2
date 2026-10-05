"""Construye (o reconstruye) el índice vectorial: python scripts/build_index.py [--rebuild]"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.rag.config import RagConfig  # noqa: E402
from src.rag.vectorstore import build_index  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--chunk-size", type=int, default=RagConfig.chunk_size)
    ap.add_argument("--overlap", type=int, default=RagConfig.chunk_overlap)
    a = ap.parse_args()
    cfg = RagConfig(chunk_size=a.chunk_size, chunk_overlap=a.overlap)
    col = build_index(cfg, rebuild=a.rebuild)
    print(f"Colección '{cfg.collection_name}' lista con {col.count()} fragmentos.")

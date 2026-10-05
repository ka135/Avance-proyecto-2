"""Evaluación del RAG con Ragas (faithfulness, answer_relevancy, context_precision, context_recall).

Uso:
  python eval/run_ragas.py --tag baseline --chunk-size 300 --overlap 0 --top-k 2
  python eval/run_ragas.py --tag mejora   --chunk-size 800  --overlap 120 --top-k 4
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402
from langchain_core.embeddings import Embeddings  # noqa: E402
from ragas import EvaluationDataset, RunConfig, evaluate  # noqa: E402
from ragas.embeddings import LangchainEmbeddingsWrapper  # noqa: E402
from ragas.llms import LangchainLLMWrapper  # noqa: E402
from ragas.metrics import (Faithfulness, LLMContextPrecisionWithReference,  # noqa: E402
                           LLMContextRecall, ResponseRelevancy)

from src.rag.config import RagConfig, api_key, base_url, get_secret, llm_model, provider  # noqa: E402
from src.rag.embeddings import embed_texts  # noqa: E402
from src.rag.pipeline import RagAssistant  # noqa: E402

OUT = Path(__file__).resolve().parent / "resultados"
METRICS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]


class LocalEmbeddings(Embeddings):
    """Mismos embeddings locales del pipeline, para answer_relevancy."""
    def __init__(self, model): self.model = model
    def embed_documents(self, texts): return embed_texts(texts, self.model)
    def embed_query(self, text): return embed_texts([text], self.model)[0]


def answer_text(a: dict) -> str:
    steps = " ".join(f"{i}. {s}" for i, s in enumerate(a.get("pasos", []), 1))
    return f"{a['respuesta']} {steps}".strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--chunk-size", type=int, default=RagConfig.chunk_size)
    ap.add_argument("--overlap", type=int, default=RagConfig.chunk_overlap)
    ap.add_argument("--top-k", type=int, default=RagConfig.top_k)
    ap.add_argument("--questions", default=str(Path(__file__).with_name("preguntas_eval.json")))
    ap.add_argument("--limit", type=int, default=0, help="usar solo las primeras N preguntas (prueba rápida)")
    ap.add_argument("--no-hybrid", action="store_true", help="solo similitud de coseno (sin parte léxica)")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--sleep", type=float, default=4.0, help="pausa entre preguntas (límites del plan gratuito)")
    ap.add_argument("--reuse", action="store_true", help="reutiliza respuestas ya generadas del tag")
    a = ap.parse_args()

    cfg = RagConfig(chunk_size=a.chunk_size, chunk_overlap=a.overlap, top_k=a.top_k, hybrid=not a.no_hybrid)
    questions = json.load(open(a.questions, encoding="utf-8"))
    if a.limit:
        questions = questions[:a.limit]
    cache = OUT / f"{a.tag}_respuestas.json"

    if a.reuse and cache.exists():
        rows = json.load(open(cache, encoding="utf-8"))
    else:
        bot = RagAssistant(cfg, rebuild=True)
        print(f"Índice: {cfg.collection_name} ({bot.collection.count()} fragmentos)")
        rows = []
        for q in questions:
            out = bot.ask(q["pregunta"])
            rows.append({"id": q["id"], "tipo": q["tipo"], "user_input": q["pregunta"],
                         "retrieved_contexts": [c.text for c in out["contexts"]],
                         "retrieved_sources": [c.chunk_id for c in out["contexts"]],
                         "response": answer_text(out["answer"]), "reference": q["ground_truth"]})
            print(f"[{q['id']}] ok"); time.sleep(a.sleep)
        json.dump(rows, open(cache, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    ds = EvaluationDataset.from_list([{k: r[k] for k in
                                       ("user_input", "retrieved_contexts", "response", "reference")} for r in rows])
    judge_model = get_secret("LLM_JUDGE_MODEL") or llm_model()
    if provider() == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        chat = ChatGoogleGenerativeAI(model=judge_model, google_api_key=api_key(), temperature=0)
    else:
        from langchain_openai import ChatOpenAI
        chat = ChatOpenAI(model=judge_model, api_key=api_key(), base_url=base_url(), temperature=0)
    judge = LangchainLLMWrapper(chat)
    emb = LangchainEmbeddingsWrapper(LocalEmbeddings(cfg.embedding_model))
    result = evaluate(
        ds, metrics=[Faithfulness(), ResponseRelevancy(strictness=1),
                     LLMContextPrecisionWithReference(), LLMContextRecall()],
        llm=judge, embeddings=emb,
        run_config=RunConfig(max_workers=a.workers, timeout=240, max_retries=8))

    df = result.to_pandas()
    # Normaliza nombres de columnas de métricas entre versiones de Ragas
    rename = {}
    for c in df.columns:
        for m in METRICS:
            if m in c:
                rename[c] = m
    df = df.rename(columns=rename)
    df.insert(0, "id", [r["id"] for r in rows])
    df.insert(1, "tipo", [r["tipo"] for r in rows])
    df.to_csv(OUT / f"{a.tag}_detalle.csv", index=False, encoding="utf-8-sig")

    summary = {"tag": a.tag, "config": {"chunk_size": cfg.chunk_size, "overlap": cfg.chunk_overlap,
                                        "top_k": cfg.top_k, "hybrid": cfg.hybrid, "embedding": cfg.embedding_model,
                                        "llm": f"{provider()}:{llm_model()}"},
               "global": df[METRICS].mean().round(3).to_dict(),
               "por_tipo": df.groupby("tipo")[METRICS].mean().round(3).to_dict(orient="index")}
    json.dump(summary, open(OUT / f"{a.tag}_resumen.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(pd.DataFrame([summary["global"]]).to_string(index=False))
    print("Guardado en", OUT)


if __name__ == "__main__":
    main()

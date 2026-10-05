"""Orquestador RAG conversacional: reescritura -> recuperación -> prompt -> generación."""
from .config import RagConfig
from .generator import call_llm, generate_answer
from .prompts import CONDENSE_PROMPT, build_prompt, format_history
from .retriever import retrieve
from .vectorstore import build_index


class RagAssistant:
    def __init__(self, cfg: RagConfig = None, rebuild: bool = False):
        self.cfg = cfg or RagConfig()
        self.collection = build_index(self.cfg, rebuild=rebuild)

    def standalone_question(self, question: str, history) -> str:
        """Convierte una pregunta de seguimiento en una pregunta autónoma para buscar bien."""
        if not history:
            return question
        recent = history[-self.cfg.max_history_messages:]
        try:
            rewritten = call_llm(
                CONDENSE_PROMPT.format(historial=format_history(recent), pregunta=question),
                json_mode=False).strip().strip('"')
            return rewritten or question
        except Exception:  # noqa: BLE001  (si falla, se busca con la pregunta original)
            return question

    def ask(self, question: str, history=None) -> dict:
        history = (history or [])[-self.cfg.max_history_messages:]
        query = self.standalone_question(question, history)
        contexts = retrieve(query, self.cfg)
        answer = generate_answer(build_prompt(question, contexts, history))
        # Si el modelo omitió fuente/confianza, se completan de forma coherente
        if not answer.get("fuente"):
            answer["fuente"] = contexts[0].source if (answer["en_corpus"] and contexts) else "ninguna"
        if not answer.get("confianza"):
            answer["confianza"] = "media" if answer["en_corpus"] else "baja"
        return {"answer": answer, "contexts": contexts, "standalone_question": query}

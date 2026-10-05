"""Chat conversacional RAG (Streamlit). Credenciales: variable de entorno o st.secrets."""
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.rag.config import DOCS_DIR, RagConfig, api_key, llm_model, provider  # noqa: E402
from src.rag.pipeline import RagAssistant  # noqa: E402

st.set_page_config(page_title="Asistente de Soporte Técnico (RAG)", page_icon="🛠️")
st.title("🛠️ Asistente de Soporte Técnico")
st.caption("Responde únicamente con base en los manuales indexados y cita sus fuentes.")

ICON = {"alta": "🟢", "media": "🟡", "baja": "🔴"}

with st.sidebar:
    st.header("Configuración")
    top_k = st.slider("Fragmentos recuperados (top_k)", 1, 8, RagConfig.top_k)
    st.markdown("**Manuales indexados**")
    for p in sorted(DOCS_DIR.iterdir()):
        if p.is_file():
            st.markdown(f"- `{p.name}`")
    st.caption(f"LLM: {provider()} · {llm_model()}")
    if st.button("🗑️ Nueva conversación"):
        st.session_state.messages = []
        st.rerun()


@st.cache_resource(show_spinner="Construyendo índice vectorial…")
def load_assistant(top_k: int):
    return RagAssistant(RagConfig(top_k=top_k))


if not api_key():
    st.error("Falta la clave de API del proveedor configurado. Defínela como variable de entorno o en los Secrets.")
    st.stop()

assistant = load_assistant(top_k)
st.session_state.setdefault("messages", [])


def render_answer(msg):
    a = msg["answer"]
    st.markdown(a["respuesta"])
    if a.get("pasos"):
        st.markdown("**Pasos:**")
        for i, s in enumerate(a["pasos"], 1):
            st.markdown(f"{i}. {s}")
    if a.get("en_corpus", True):
        st.caption(f"{ICON.get(a['confianza'], '⚪')} Confianza: {a['confianza']} · 📄 Fuente: {a['fuente']}")
    else:
        st.info("Esta información no está en los manuales indexados, por eso no se muestra una solución.")
    with st.expander(f"Fragmentos consultados ({len(msg['contexts'])})"):
        if msg.get("standalone_question") and msg["standalone_question"] != msg["question"]:
            st.caption(f"Pregunta reformulada para la búsqueda: _{msg['standalone_question']}_")
        for c in msg["contexts"]:
            st.markdown(f"**{c.source}** · {c.section or 'N/A'} · `{c.chunk_id}` · similitud {c.score:.2f}")
            st.code(c.text, language=None)


for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        if m["role"] == "user":
            st.markdown(m["content"])
        else:
            render_answer(m)

if prompt := st.chat_input("Escribe tu consulta técnica…"):
    # El historial para el LLM son solo los turnos anteriores (texto)
    history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Buscando en los manuales…"):
            try:
                out = assistant.ask(prompt, history)
            except Exception as e:  # noqa: BLE001
                st.error(f"No se pudo generar la respuesta: {e}")
                st.session_state.messages.pop()
                st.stop()
        msg = {"role": "assistant", "content": out["answer"]["respuesta"], "answer": out["answer"],
               "contexts": out["contexts"], "standalone_question": out["standalone_question"],
               "question": prompt}
        render_answer(msg)
    st.session_state.messages.append(msg)

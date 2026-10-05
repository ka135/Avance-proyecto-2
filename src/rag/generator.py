"""Integración con Gemini: reintentos ante 503/429 y parseo robusto del JSON."""
import json
import re
import time

from .config import api_key, base_url, llm_model, provider
from .prompts import SYSTEM_PROMPT

_RETRIABLE = ("503", "429", "UNAVAILABLE", "RESOURCE_EXHAUSTED", "overloaded")


def _gemini(prompt, system, json_mode):
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=api_key())
    config = types.GenerateContentConfig(
        system_instruction=system, temperature=0.1,
        response_mime_type="application/json" if json_mode else None)
    return client.models.generate_content(model=llm_model(), contents=prompt, config=config).text or ""


def _openai_compat(prompt, system, json_mode):
    """Mistral y Groq usan la API compatible con OpenAI."""
    from openai import OpenAI
    client = OpenAI(api_key=api_key(), base_url=base_url())
    messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
    kwargs = {"model": llm_model(), "messages": messages, "temperature": 0.1}
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    return client.chat.completions.create(**kwargs).choices[0].message.content or ""


def call_llm(prompt: str, system: str = None, json_mode: bool = True, retries: int = 5) -> str:
    if not api_key():
        raise RuntimeError(f"Falta la clave de API del proveedor '{provider()}' (variable de entorno o secrets).")
    fn = _gemini if provider() == "gemini" else _openai_compat
    for attempt in range(retries):
        try:
            return fn(prompt, system, json_mode)
        except Exception as e:  # noqa: BLE001
            if attempt < retries - 1 and any(t in str(e) for t in _RETRIABLE):
                time.sleep(2 ** attempt * 2)
                continue
            raise


def parse_json(raw: str) -> dict:
    clean = re.sub(r"^```(?:json)?|```$", "", (raw or "").strip(), flags=re.M).strip()
    try:
        data = json.loads(clean)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", clean, flags=re.S)
        try:
            data = json.loads(m.group(0)) if m else {}
        except json.JSONDecodeError:
            data = {}
    if not isinstance(data, dict) or "respuesta" not in data:
        data = {"respuesta": raw or "No se pudo interpretar la respuesta del modelo.",
                "pasos": [], "fuente": "ninguna", "confianza": "baja", "en_corpus": False}
    data.setdefault("en_corpus", True)
    data.setdefault("fuente", None)      # None = el modelo no lo devolvió; el pipeline lo completa
    data.setdefault("confianza", None)
    # Limpieza: modelos pequeños a veces meten fuente/confianza/en_corpus dentro de "pasos"
    pasos = data.get("pasos") or []
    if isinstance(pasos, str):
        pasos = [pasos]
    clean = []
    for p in pasos:
        if isinstance(p, dict):
            continue
        t = str(p).strip()
        if t and not re.match(r"^(fuente|confianza|en_corpus)\b\s*[:=]?", t, flags=re.I):
            clean.append(t)
    data["pasos"] = clean
    return data


def generate_answer(prompt: str) -> dict:
    return parse_json(call_llm(prompt, system=SYSTEM_PROMPT, json_mode=True))

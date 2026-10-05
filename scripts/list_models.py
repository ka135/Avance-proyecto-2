"""Lista los modelos Gemini disponibles para tu clave: python scripts/list_models.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from google import genai  # noqa: E402
from src.rag.config import get_secret  # noqa: E402

client = genai.Client(api_key=get_secret("GEMINI_API_KEY"))
for m in client.models.list():
    acts = getattr(m, "supported_actions", None) or []
    if "generateContent" in acts:
        print(m.name.replace("models/", ""))

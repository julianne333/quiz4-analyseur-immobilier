from __future__ import annotations

import json
import os
from typing import Any

from dotenv import load_dotenv
import streamlit as st

try:
    from google import genai
except ImportError:  # pragma: no cover - optional dependency at runtime
    genai = None


DEFAULT_MODEL = "gemini-2.5-flash-lite"


def _read_streamlit_secret(key: str) -> str | None:
    try:
        return st.secrets.get(key)
    except Exception:  # pragma: no cover - depends on Streamlit runtime/config
        return None


def _get_api_key() -> str | None:
    load_dotenv()
    return (
        _read_streamlit_secret("GEMINI_API_KEY")
        or _read_streamlit_secret("GOOGLE_API_KEY")
        or os.getenv("GEMINI_API_KEY")
        or os.getenv("GOOGLE_API_KEY")
    )


def _get_model_name() -> str:
    load_dotenv()
    return (
        _read_streamlit_secret("GEMINI_MODEL")
        or os.getenv("GEMINI_MODEL")
        or DEFAULT_MODEL
    )


def ai_is_configured() -> bool:
    return bool(_get_api_key() and genai is not None)


def _coerce_for_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _coerce_for_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_coerce_for_json(item) for item in value]
    if hasattr(value, "item"):
        try:
            return _coerce_for_json(value.item())
        except ValueError:
            return str(value)
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except TypeError:
            return str(value)
    if isinstance(value, float) and value != value:
        return None
    return value


@st.cache_data(show_spinner=False, ttl=900)
def _generate_gemini_summary(prompt: str, model: str) -> dict[str, str]:
    api_key = _get_api_key()

    if not api_key or genai is None:
        return {"source": "fallback", "text": "", "error": "AI not configured"}

    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=model,
            contents=prompt,
        )
        text = (response.text or "").strip()
        if not text:
            return {"source": "fallback", "text": "", "error": "Empty AI response"}
        return {"source": "gemini", "text": text}
    except Exception as exc:  # pragma: no cover - network/runtime dependent
        return {"source": "fallback", "text": "", "error": str(exc)}


def generate_investment_narrative(
    *,
    context_type: str,
    payload: dict[str, Any],
    fallback_text: str,
) -> dict[str, str]:
    prompt = f"""
Tu es analyste junior dans un fonds d'investissement immobilier.
Redige une note executive en francais sur un ton professionnel, concise et actionnable.
Contraintes:
- 110 a 160 mots
- pas de liste a puces
- aucun markdown, aucun titre, aucun caractere ** ou ##
- base-toi uniquement sur les donnees fournies
- cite clairement l'opportunite principale, le risque principal et la prochaine action recommandee
- ne mentionne pas que tu es une IA

Type de contexte: {context_type}

Donnees:
{json.dumps(_coerce_for_json(payload), ensure_ascii=True, indent=2)}
""".strip()

    model = _get_model_name()
    result = _generate_gemini_summary(prompt, model)
    if result["source"] == "gemini":
        return result

    return {
        "source": "fallback",
        "text": fallback_text,
        "error": result.get("error", "Fallback used"),
    }


def generate_summary_from_prompt(*, prompt: str, fallback_text: str) -> dict[str, str]:
    model = _get_model_name()
    result = _generate_gemini_summary(prompt, model)
    if result["source"] == "gemini":
        return result

    return {
        "source": "fallback",
        "text": fallback_text,
        "error": result.get("error", "Fallback used"),
    }

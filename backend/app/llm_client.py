from typing import List, Dict, Optional
import requests

from .config import settings


class LLMError(RuntimeError):
    pass


def llm_chat(messages: List[Dict[str, str]]) -> str:
    """
    Calls local Ollama chat endpoint.
    messages: [{"role": "system"|"user"|"assistant", "content": "..."}]
    """
    url = f"{settings.OLLAMA_URL}/api/chat"
    payload = {
        "model": settings.OLLAMA_MODEL,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": 0.7,
            "num_predict": 300
        }
    }

    try:
        r = requests.post(url, json=payload, timeout=120)
        if r.status_code != 200:
            raise LLMError(f"Ollama HTTP {r.status_code}: {r.text[:500]}")
        data = r.json()
        content = (data.get("message") or {}).get("content")
        if not content:
            raise LLMError(f"Ollama returned empty content: {data}")
        return content.strip()
    except requests.RequestException as e:
        raise LLMError(f"Ollama request failed: {e}")


def simple_fallback_reply(pred_label: Optional[str], baseline_ready: bool) -> str:
    """
    Fallback if Ollama is down. Keeps the pipeline testable.
    """
    if not baseline_ready:
        return (
            "I’m here. Before we go deep—quick check: are you feeling stressed, tired, or both? "
            "Also, what’s the *one* task you need to tackle next?"
        )

    if pred_label == "High":
        return (
            "Okay—keep it tiny. Breathe in 4 seconds, out 6 seconds (x3). "
            "Now tell me the next *single* step you can do in 5 minutes."
        )
    if pred_label == "Medium":
        return (
            "Let’s simplify. What’s the deadline, and what’s the hardest part right now? "
            "We’ll make a short plan."
        )
    return (
        "I’m with you. Tell me what’s going on, and what you’ve already tried so far."
    )
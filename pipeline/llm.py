"""Thin Gemini client: JSON-schema-constrained output, rate limiting, retries,
and automatic fallback across model names (free-tier models change over time)."""
from __future__ import annotations

import json
import os
import time

from . import config as C

_client = None
_model_idx = 0
_last_call = 0.0


def available() -> bool:
    return bool(os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))


def _get_client():
    global _client
    if _client is None:
        from google import genai
        _client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))
    return _client


def model_name() -> str:
    return C.GEMINI_MODELS[min(_model_idx, len(C.GEMINI_MODELS) - 1)]


def generate_json(prompt: str, schema=None, system: str | None = None, temperature: float = 0.1,
                  max_retries: int = 6) -> dict:
    """Call Gemini and return parsed JSON. `schema` may be a Pydantic model class."""
    global _model_idx, _last_call
    from google.genai import types

    client = _get_client()
    cfg = types.GenerateContentConfig(
        temperature=temperature,
        response_mime_type="application/json",
        response_schema=schema,
        system_instruction=system,
    )
    delay = 10.0
    for attempt in range(max_retries):
        wait = C.LLM_MIN_INTERVAL_S - (time.time() - _last_call)
        if wait > 0:
            time.sleep(wait)
        _last_call = time.time()
        try:
            resp = client.models.generate_content(model=model_name(), contents=prompt, config=cfg)
            if getattr(resp, "parsed", None) is not None:
                p = resp.parsed
                return p.model_dump() if hasattr(p, "model_dump") else p
            return json.loads(resp.text)
        except Exception as e:  # noqa: BLE001
            msg = str(e)
            if ("404" in msg or "not found" in msg.lower()) and _model_idx < len(C.GEMINI_MODELS) - 1:
                print(f"[llm] model {model_name()} unavailable; falling back")
                _model_idx += 1
                continue
            if "429" in msg or "RESOURCE_EXHAUSTED" in msg or "503" in msg or "500" in msg:
                if "per day" in msg.lower() or "PerDay" in msg:
                    if _model_idx < len(C.GEMINI_MODELS) - 1:
                        print(f"[llm] daily quota hit on {model_name()}; switching model")
                        _model_idx += 1
                        continue
                print(f"[llm] rate limited/unavailable (attempt {attempt + 1}); sleeping {delay:.0f}s")
                time.sleep(delay)
                delay = min(delay * 2, 120)
                continue
            print(f"[llm] error (attempt {attempt + 1}): {msg[:300]}")
            time.sleep(3)
    raise RuntimeError("LLM call failed after retries")


def generate_text(prompt: str, system: str | None = None, temperature: float = 0.2) -> str:
    from google.genai import types

    client = _get_client()
    for i in range(len(C.GEMINI_MODELS)):
        try:
            resp = client.models.generate_content(
                model=C.GEMINI_MODELS[i], contents=prompt,
                config=types.GenerateContentConfig(temperature=temperature, system_instruction=system))
            return resp.text
        except Exception as e:  # noqa: BLE001
            last = e
            continue
    raise RuntimeError(f"LLM text call failed: {last}")

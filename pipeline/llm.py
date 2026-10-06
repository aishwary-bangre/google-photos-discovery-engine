"""LLM client with provider + model rotation.

Free tiers have small *per-model, per-day* quotas, so this client:
  * discovers which Gemini 'flash' models the key can use,
  * rotates to the next model after repeated 429/503 errors,
  * falls back to Groq (if GROQ_API_KEY is set) when every Gemini model is exhausted.
Outputs are always re-validated by the caller with Pydantic.
"""
from __future__ import annotations

import json
import os
import re
import time

from . import config as C

_gem_client = None
_models: list[tuple[str, str]] | None = None  # (provider, model)
_idx = 0
_last_call = 0.0

GROQ_MODELS = ["llama-3.3-70b-versatile", "openai/gpt-oss-120b", "llama-3.1-8b-instant"]


def available() -> bool:
    return bool(os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or os.environ.get("GROQ_API_KEY"))


def _gemini():
    global _gem_client
    if _gem_client is None:
        from google import genai
        _gem_client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))
    return _gem_client


def _discover() -> list[tuple[str, str]]:
    global _models
    if _models is not None:
        return _models
    models: list[tuple[str, str]] = []
    if os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"):
        found = []
        try:
            for m in _gemini().models.list():
                name = m.name.split("/")[-1]
                actions = getattr(m, "supported_actions", None) or []
                if "generateContent" not in actions:
                    continue
                if "flash" not in name or re.search(r"image|tts|audio|live|embed|exp|preview-\d\d-\d\d", name):
                    continue
                found.append(name)
        except Exception as e:  # noqa: BLE001
            print(f"[llm] model discovery failed ({e}); using defaults")
        preferred = [m for m in C.GEMINI_MODELS if m in found] or C.GEMINI_MODELS
        rest = sorted([m for m in found if m not in preferred], reverse=True)
        models += [("gemini", m) for m in preferred + rest]
    if os.environ.get("GROQ_API_KEY"):
        models += [("groq", m) for m in GROQ_MODELS]
    print(f"[llm] model rotation: {[m for _, m in models]}")
    _models = models
    return models


def current_provider() -> str:
    ms = _discover()
    return ms[min(_idx, len(ms) - 1)][0] if ms else "none"


def model_name() -> str:
    ms = _discover()
    return ms[min(_idx, len(ms) - 1)][1] if ms else "none"


def _call(provider: str, model: str, prompt: str, schema, system: str | None, temperature: float) -> dict:
    if provider == "gemini":
        from google.genai import types
        cfg = types.GenerateContentConfig(temperature=temperature, response_mime_type="application/json",
                                          response_schema=schema, system_instruction=system)
        resp = _gemini().models.generate_content(model=model, contents=prompt, config=cfg)
        if getattr(resp, "parsed", None) is not None:
            p = resp.parsed
            return p.model_dump() if hasattr(p, "model_dump") else p
        return json.loads(resp.text)
    from groq import Groq
    sys_msg = (system or "") + "\nRespond with a single valid JSON object only."
    if schema is not None and hasattr(schema, "model_json_schema"):
        sys_msg += "\nJSON schema to follow exactly:\n" + json.dumps(schema.model_json_schema())
    resp = Groq(api_key=os.environ["GROQ_API_KEY"]).chat.completions.create(
        model=model, temperature=temperature, response_format={"type": "json_object"},
        messages=[{"role": "system", "content": sys_msg}, {"role": "user", "content": prompt}])
    return json.loads(resp.choices[0].message.content)


def generate_json(prompt: str, schema=None, system: str | None = None, temperature: float = 0.1) -> dict:
    global _idx, _last_call
    models = _discover()
    if not models:
        raise RuntimeError("no LLM provider configured")
    fails = 0
    while _idx < len(models):
        provider, model = models[_idx]
        wait = C.LLM_MIN_INTERVAL_S - (time.time() - _last_call)
        if wait > 0:
            time.sleep(wait)
        _last_call = time.time()
        try:
            return _call(provider, model, prompt, schema, system, temperature)
        except Exception as e:  # noqa: BLE001
            msg = str(e)
            quota = any(k in msg for k in ("429", "RESOURCE_EXHAUSTED", "rate_limit", "Rate limit"))
            overloaded = any(k in msg for k in ("503", "500", "UNAVAILABLE", "overloaded"))
            not_found = "404" in msg or "NOT_FOUND" in msg or "not found" in msg.lower()
            daily = "PerDay" in msg or "per day" in msg.lower() or "RPD" in msg or "TPD" in msg
            fails += 1
            print(f"[llm] {provider}/{model} error #{fails}: {msg[:220]}")
            if not_found or daily or (quota and fails >= 3) or (overloaded and fails >= 4) or fails >= 5:
                _idx += 1
                fails = 0
                if _idx < len(models):
                    print(f"[llm] rotating to {models[_idx][0]}/{models[_idx][1]}")
                continue
            time.sleep(15 * fails)
    raise RuntimeError("all LLM models exhausted for today")


def generate_text(prompt: str, system: str | None = None, temperature: float = 0.2) -> str:
    """Plain-text generation for the dashboard (tries each model once)."""
    last = None
    for provider, model in _discover():
        try:
            if provider == "gemini":
                from google.genai import types
                r = _gemini().models.generate_content(
                    model=model, contents=prompt,
                    config=types.GenerateContentConfig(temperature=temperature, system_instruction=system))
                return r.text
            from groq import Groq
            r = Groq(api_key=os.environ["GROQ_API_KEY"]).chat.completions.create(
                model=model, temperature=temperature,
                messages=[{"role": "system", "content": system or ""}, {"role": "user", "content": prompt}])
            return r.choices[0].message.content
        except Exception as e:  # noqa: BLE001
            last = e
    raise RuntimeError(f"LLM text call failed: {last}")

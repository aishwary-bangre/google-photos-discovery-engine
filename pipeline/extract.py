"""Stage 3 — LLM structured coding of every gated record.

- Batched (N records per call) with a JSON schema enforced by the API.
- Every output is re-validated with Pydantic; ids not in the batch are dropped.
- evidence_quote is checked to be a real substring of the source text
  (anti-hallucination); unverifiable quotes are flagged, never shown as evidence.
- Results are cached per record id, so the stage resumes after quota errors.
"""
from __future__ import annotations

import json
import os
import re

import pandas as pd
from pydantic import ValidationError

from . import config as C
from . import llm
from .schema import CODEBOOK, BatchResult, RecordTag

CACHE = C.CACHE_DIR / "tags.jsonl"
SYSTEM = ("You are a UX researcher coding user feedback about finding old photos in Google Photos. "
          "Be literal: code only what the text supports. Never invent details.")


def _norm(s: str) -> str:
    return re.sub(r"\W+", " ", (s or "").lower()).strip()


def _load_cache() -> dict[str, dict]:
    if not CACHE.exists():
        return {}
    out = {}
    for line in CACHE.read_text(encoding="utf-8").splitlines():
        if line.strip():
            d = json.loads(line)
            out[d["id"]] = d
    return out


def _append_cache(rows: list[dict]):
    with CACHE.open("a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _mock_tag(rec: dict) -> dict:
    """Offline stand-in used ONLY for local tests (LLM_MOCK=1). Never used in CI."""
    t = rec["full_text"].lower()
    stage = ("interpret" if "search" in t else "browse" if "scroll" in t else "unclear")
    return RecordTag(id=rec["id"], relevant=("find" in t or "search" in t), failure_stage=stage,
                     failure_mode="mock", evidence_quote=rec["full_text"][:60]).model_dump()


def _build_prompt(batch: pd.DataFrame) -> str:
    items = [{"id": r.id, "source": r.source, "text": r.full_text[:1200]} for r in batch.itertuples()]
    return (CODEBOOK + "\n\nCode each item below. Return {\"results\": [...]} with one object per id.\n\n"
            + json.dumps(items, ensure_ascii=False))


def run() -> pd.DataFrame:
    gated = pd.read_parquet(C.PROC_DIR / "gated.parquet")
    cache = _load_cache()
    todo = gated[~gated["id"].isin(cache.keys())]
    mock = os.environ.get("LLM_MOCK") == "1"
    if not mock and not llm.available() and len(todo):
        raise SystemExit("[extract] GEMINI_API_KEY not set")
    print(f"[extract] {len(gated)} gated, {len(cache)} cached, {len(todo)} to code "
          f"({'MOCK' if mock else llm.model_name()})")

    texts = dict(zip(gated["id"], gated["full_text"]))
    for start in range(0, len(todo), C.LLM_BATCH_SIZE):
        batch = todo.iloc[start:start + C.LLM_BATCH_SIZE]
        ids = set(batch["id"])
        if mock:
            rows = [_mock_tag(r) for r in batch.to_dict("records")]
        else:
            try:
                raw = llm.generate_json(_build_prompt(batch), schema=BatchResult, system=SYSTEM)
            except RuntimeError as e:
                print(f"[extract] stopping early: {e}. Re-run to resume from cache.")
                break
            rows = []
            for item in (raw.get("results") or []):
                try:
                    tag = RecordTag.model_validate(item)
                except ValidationError as ve:
                    print(f"[extract] invalid item dropped: {str(ve)[:120]}")
                    continue
                if tag.id in ids:
                    rows.append(tag.model_dump())
        for r in rows:
            q = _norm(r.get("evidence_quote"))
            r["quote_verified"] = bool(q) and q in _norm(texts.get(r["id"], ""))
            r["model"] = "mock" if mock else llm.model_name()
        _append_cache(rows)
        cache.update({r["id"]: r for r in rows})
        done = min(start + C.LLM_BATCH_SIZE, len(todo))
        print(f"[extract] {done}/{len(todo)} coded")

    tags = pd.DataFrame([cache[i] for i in gated["id"] if i in cache])
    if tags.empty:
        raise SystemExit("[extract] nothing coded — check the gate stage and API key")
    merged = gated.merge(tags, on="id", how="inner")
    merged.to_parquet(C.PROC_DIR / "tagged.parquet", index=False)
    rel = merged[merged["relevant"]]
    print(f"[extract] tagged {len(merged)}; relevant {len(rel)}; "
          f"verified quotes {rel['quote_verified'].mean():.0%}" if len(rel) else "[extract] no relevant rows")
    return merged

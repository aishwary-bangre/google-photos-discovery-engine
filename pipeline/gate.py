"""Stage 2 — relevance gate.

Raw feedback about Google Photos is mostly about backup, storage, pricing and
crashes. This stage keeps only records that plausibly describe *trying to find
an existing photo*, in two passes:
  1. keyword pre-filter (cheap, high recall)
  2. semantic similarity to problem 'anchor' statements (precision ranking)
The LLM later makes the final relevant / not-relevant call per record.
"""
from __future__ import annotations

import hashlib
import json
import re

import numpy as np
import pandas as pd

from . import config as C
from . import embed


def load_raw() -> pd.DataFrame:
    rows = []
    for p in sorted(C.RAW_DIR.glob("*.jsonl")):
        with p.open(encoding="utf-8") as f:
            rows += [json.loads(l) for l in f if l.strip()]
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["full_text"] = (df["title"].fillna("") + ". " + df["text"].fillna("")).str.strip(". ").str.strip()
    # de-duplicate identical texts across sources (PullPush overlaps Reddit)
    df["text_hash"] = df["full_text"].str.lower().str.replace(r"\W+", " ", regex=True).map(
        lambda s: hashlib.md5(s.encode()).hexdigest())
    df = df.drop_duplicates("text_hash").drop_duplicates("id").reset_index(drop=True)
    return df


def run() -> pd.DataFrame:
    df = load_raw()
    if df.empty:
        raise SystemExit("[gate] no raw data found — run the collect stage first")
    n_raw = len(df)
    by_source_raw = df["source"].value_counts().to_dict()

    df = df[df["full_text"].str.len() >= 25]
    kw = re.compile(C.KEYWORD_PATTERN, re.I)
    df = df[df["full_text"].str.contains(kw)].copy()
    n_kw = len(df)

    texts = df["full_text"].str.slice(0, 1500).tolist()
    corpus = texts + C.ANCHORS
    D = embed.encode(texts, fit_corpus=corpus)
    A = embed.encode(C.ANCHORS, fit_corpus=corpus)
    sims = D @ A.T
    df["anchor_sim"] = sims.max(axis=1)
    df["nearest_anchor"] = [C.ANCHORS[i] for i in sims.argmax(axis=1)]

    thr = C.MIN_ANCHOR_SIM if embed.backend().startswith("sentence") else 0.10
    df = df[df["anchor_sim"] >= thr].sort_values("anchor_sim", ascending=False)
    df = df.head(C.MAX_LLM_RECORDS).reset_index(drop=True)

    out = C.PROC_DIR / "gated.parquet"
    df.drop(columns=["meta"], errors="ignore").assign(
        meta_json=df.get("meta", pd.Series([{}] * len(df))).map(lambda m: json.dumps(m, default=str))
    ).to_parquet(out, index=False)

    stats = {
        "raw_records": n_raw, "raw_by_source": by_source_raw,
        "after_keyword_filter": n_kw, "after_semantic_gate": len(df),
        "gated_by_source": df["source"].value_counts().to_dict(),
        "embedding_backend": embed.backend(), "min_anchor_sim": thr,
    }
    (C.PROC_DIR / "gate_stats.json").write_text(json.dumps(stats, indent=2))
    print(f"[gate] {n_raw} raw -> {n_kw} keyword -> {len(df)} semantic  ({embed.backend()})")
    return df

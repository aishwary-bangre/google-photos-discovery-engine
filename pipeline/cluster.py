"""Stage 4 — discover problem themes bottom-up and score them as opportunities.

Relevant records are embedded on their coded failure_mode + evidence, reduced
with UMAP, and clustered with HDBSCAN (noise allowed, so outliers aren't forced
into a theme). Each cluster is named by the LLM from its member summaries, then
scored:

    severity      = mean outcome weight (gave_up 1.0, found_with_effort 0.6, unknown 0.5, found 0.2)
    urgency_norm  = mean urgency / 3
    opportunity   = share_of_relevant x severity x urgency_norm   (rescaled so the top theme = 100)
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from . import config as C
from . import embed, llm

OUTCOME_W = {"gave_up": 1.0, "found_with_effort": 0.6, "unknown": 0.5, "found": 0.2}


def _label_clusters(df: pd.DataFrame) -> dict[int, dict]:
    labels = {}
    mock = os.environ.get("LLM_MOCK") == "1" or not llm.available()
    for cid, g in df[df["cluster"] >= 0].groupby("cluster"):
        sample = g.sort_values("anchor_sim", ascending=False)["failure_mode"].head(15).tolist()
        if mock:
            words = pd.Series(" ".join(sample).lower().split()).value_counts().index[:4]
            labels[cid] = {"name": " / ".join(words), "problem": "; ".join(sample[:2]), "mismatch": ""}
            continue
        prompt = (
            "These are coded failure descriptions from users trying to find old photos in Google Photos, "
            "all in one cluster:\n- " + "\n- ".join(sample) +
            "\n\nReturn JSON {\"name\": <=6 word theme name, \"problem\": one sentence problem statement, "
            "\"mismatch\": one sentence on the gap between what users remember and what the product can use}."
        )
        try:
            labels[cid] = llm.generate_json(prompt)
        except Exception as e:  # noqa: BLE001
            labels[cid] = {"name": f"Cluster {cid}", "problem": sample[0] if sample else "", "mismatch": ""}
            print(f"[cluster] label failed for {cid}: {e}")
    return labels


def run() -> dict:
    df = pd.read_parquet(C.PROC_DIR / "tagged.parquet")
    rel = df[df["relevant"]].copy().reset_index(drop=True)
    if len(rel) < 10:
        raise SystemExit(f"[cluster] only {len(rel)} relevant records — not enough to cluster")

    docs = (rel["failure_mode"].fillna("") + " | " + rel["evidence_quote"].fillna("")).tolist()
    X = embed.encode(docs)

    from sklearn.cluster import HDBSCAN
    min_size = max(6, len(rel) // 40)
    if len(rel) >= 40:
        import umap
        Z = umap.UMAP(n_components=5, n_neighbors=15, min_dist=0.0, metric="cosine",
                      random_state=42).fit_transform(X)
        xy = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, metric="cosine",
                       random_state=42).fit_transform(X)
    else:
        Z, xy = X, X[:, :2]
    rel["cluster"] = HDBSCAN(min_cluster_size=min_size, min_samples=3).fit_predict(Z)
    rel["x"], rel["y"] = xy[:, 0], xy[:, 1]

    labels = _label_clusters(rel)
    rel["theme"] = rel["cluster"].map(lambda c: labels.get(c, {}).get("name", "Unclustered (noise)"))

    # --- opportunity scoring ---
    rel["sev"] = rel["outcome"].map(OUTCOME_W).fillna(0.5)
    rows = []
    for cid, g in rel[rel["cluster"] >= 0].groupby("cluster"):
        share = len(g) / len(rel)
        sev = g["sev"].mean()
        urg = g["urgency"].mean() / 3
        lab = labels.get(cid, {})
        rows.append({
            "cluster": int(cid), "theme": lab.get("name", f"Cluster {cid}"),
            "problem": lab.get("problem", ""), "mismatch": lab.get("mismatch", ""),
            "n": len(g), "share": share, "severity": sev, "urgency_norm": urg,
            "gave_up_rate": (g["outcome"] == "gave_up").mean(),
            "top_stage": g["failure_stage"].value_counts().idxmax(),
            "top_photo_type": g["photo_type"].value_counts().idxmax(),
            "raw_score": share * sev * urg,
        })
    themes = pd.DataFrame(rows)
    if len(themes):
        themes["opportunity"] = (100 * themes["raw_score"] / themes["raw_score"].max()).round(1)
        themes = themes.sort_values("opportunity", ascending=False)
    themes.to_csv(C.PROC_DIR / "themes.csv", index=False)
    rel.drop(columns=["sev"]).to_parquet(C.PROC_DIR / "relevant.parquet", index=False)

    # --- human spot-check sample ---
    cols = ["id", "source", "full_text", "relevant", "photo_type", "cues_remembered", "cues_missing",
            "failure_stage", "outcome", "urgency", "evidence_quote", "quote_verified", "url"]
    spot = pd.concat([df[df["relevant"]].sample(min(40, int(df["relevant"].sum())), random_state=7),
                      df[~df["relevant"]].sample(min(20, int((~df["relevant"]).sum())), random_state=7)])
    spot = spot[[c for c in cols if c in spot.columns]].assign(human_agrees="")
    spot.to_csv(C.PROC_DIR / "spot_check.csv", index=False)

    # --- summary for the dashboard / deck ---
    def vc(col):
        return rel[col].value_counts().to_dict()

    def vc_list(col):
        return rel[col].explode().dropna().value_counts().to_dict()

    summary = {
        "n_tagged": int(len(df)), "n_relevant": int(len(rel)),
        "relevance_rate": float(len(rel) / max(len(df), 1)),
        "by_source": vc("source"), "photo_type": vc("photo_type"), "failure_stage": vc("failure_stage"),
        "outcome": vc("outcome"), "workaround": vc("workaround"), "query_style": vc("query_style"),
        "cues_remembered": vc_list("cues_remembered"), "cues_missing": vc_list("cues_missing"),
        "n_clusters": int((themes["cluster"] >= 0).sum()) if len(themes) else 0,
        "noise_share": float((rel["cluster"] < 0).mean()),
        "quote_verified_rate": float(rel["quote_verified"].mean()),
        "model": str(rel["model"].iloc[0]) if "model" in rel else "",
        "embedding_backend": embed.backend(),
    }
    (C.PROC_DIR / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"[cluster] {summary['n_clusters']} themes from {len(rel)} relevant records "
          f"(noise {summary['noise_share']:.0%})")
    return summary

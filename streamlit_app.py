"""Google Photos 'Memory Mismatch' Discovery Engine — dashboard.

Reads the outputs of the pipeline in data/processed/ and lets anyone:
  * see the research-question answers (RQ1–RQ6) with evidence,
  * compare opportunity themes discovered by clustering,
  * drill into every coded record with a link to its public source,
  * ask the corpus a question (RAG, answers cite record ids),
  * paste any new piece of feedback and watch the coder tag it live.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Photo Retrieval Discovery Engine", page_icon="🔎", layout="wide")

try:  # Streamlit Cloud secrets -> env for the pipeline's LLM client
    if "GEMINI_API_KEY" in st.secrets:
        os.environ["GEMINI_API_KEY"] = st.secrets["GEMINI_API_KEY"]
except Exception:  # noqa: BLE001
    pass

PROC = Path(os.environ.get("DATA_DIR", Path(__file__).parent / "data")) / "processed"

# ---- palette (validated reference palette; light mode) ----
BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#c9c8c2"
ORDINAL = ["#86b6ef", "#5598e7", "#256abf", "#104281"]  # found -> gave up
SEQ = [[0, "#f3f7fd"], [0.15, "#cde2fb"], [0.4, "#86b6ef"], [0.7, "#2a78d6"], [1, "#0d366b"]]
FONT = dict(family="Inter, system-ui, sans-serif", size=14, color="#0b0b0b")

STAGE_ORDER = ["express", "interpret", "evaluate", "refine", "browse", "access", "unclear"]
STAGE_LABEL = {
    "express": "Express — can't phrase the memory",
    "interpret": "Interpret — search misreads the clue",
    "evaluate": "Evaluate — too many look-alikes",
    "refine": "Refine — no way to narrow / pivot",
    "browse": "Browse — endless scrolling, no search",
    "access": "Access — photo hidden / elsewhere",
    "unclear": "Unclear",
}
CUE_LABEL = {
    "time_approx": "Rough time", "life_chapter": "Life chapter", "place": "Place",
    "people": "People", "event_context": "What was happening", "purpose": "Why it was taken",
    "visual_content": "What's in it (visual)", "text_in_image": "Text in the image",
    "source_app": "Where it came from (app)", "nothing_specific": "Only that it exists",
}
CONTEXT_CUES = {"time_approx", "life_chapter", "people", "event_context", "purpose", "source_app"}
OUTCOME_ORDER = ["found", "found_with_effort", "unknown", "gave_up"]


def style(fig, h=380):
    fig.update_layout(height=h, font=FONT, plot_bgcolor="#fcfcfb", paper_bgcolor="#fcfcfb",
                      margin=dict(l=10, r=40, t=80, b=10), hoverlabel=dict(font_size=14),
                      title=dict(y=0.97, yanchor="top"),
                      legend=dict(orientation="h", y=1.0, yanchor="bottom", x=0, title=None, traceorder="normal"))
    fig.update_xaxes(gridcolor="#ecebe6", zeroline=False)
    fig.update_yaxes(gridcolor="#ecebe6", zeroline=False)
    return fig


def hbar(series: pd.Series, title: str, color=BLUE, pct_of: int | None = None, h=360, sort=True):
    s = series.sort_values() if sort else series
    txt = [f"{v} ({v / pct_of:.0%})" if pct_of else str(v) for v in s.values]
    fig = go.Figure(go.Bar(x=s.values, y=s.index, orientation="h", marker_color=color,
                           text=txt, textposition="outside", cliponaxis=False,
                           hovertemplate="%{y}: %{x}<extra></extra>"))
    fig.update_layout(title=title, bargap=0.35)
    style(fig, h)
    fig.update_layout(margin=dict(r=90))
    return fig


@st.cache_data
def load():
    if not (PROC / "relevant.parquet").exists():
        return None
    rel = pd.read_parquet(PROC / "relevant.parquet")
    for c in ("cues_remembered", "cues_missing"):
        rel[c] = rel[c].map(lambda v: list(v) if v is not None else [])
    themes = pd.read_csv(PROC / "themes.csv")
    summary = json.loads((PROC / "summary.json").read_text())
    gate = json.loads((PROC / "gate_stats.json").read_text())
    runlog = json.loads((PROC / "run_log.json").read_text()) if (PROC / "run_log.json").exists() else {}
    return rel, themes, summary, gate, runlog


data = load()
st.title("🔎 Memory-Mismatch Discovery Engine")
st.caption("Why do people fail to find photos they *partly* remember in Google Photos? "
           "An AI pipeline over public Play Store, App Store and Reddit feedback.")

if data is None:
    st.warning("No processed data yet. Run the GitHub Actions workflow **Discovery pipeline** first.")
    st.stop()

rel, themes, summary, gate, runlog = data
N = len(rel)

tabs = st.tabs(["How it works", "Research questions", "Opportunity themes", "Evidence explorer",
                "Ask the engine", "Try the coder live"])

# ======================================================================= 1. HOW IT WORKS
with tabs[0]:
    c = st.columns(4)
    c[0].metric("Public posts & reviews collected", f"{gate['raw_records']:,}")
    c[1].metric("Passed keyword + semantic gate", f"{gate['after_semantic_gate']:,}")
    c[2].metric("Coded as real retrieval attempts", f"{N:,}")
    c[3].metric("Evidence quotes verified verbatim", f"{summary['quote_verified_rate']:.0%}")

    left, right = st.columns([1.1, 1])
    with left:
        st.subheader("Pipeline")
        st.markdown(f"""
1. **Collect** — Play Store (IN/US/GB), App Store RSS (5 countries), Reddit search + comments
   (r/googlephotos, r/GooglePixel, r/Android …), PullPush archive. Runs in GitHub Actions.
2. **Gate** — regex pre-filter for retrieval language → embeddings (`{gate['embedding_backend']}`)
   ranked by similarity to 7 problem 'anchor' statements (threshold {gate['min_anchor_sim']}).
3. **Code** — `{summary.get('model', '')}` tags each record against a fixed **memory-mismatch codebook**
   (Pydantic schema, JSON-constrained): photo type · cues remembered · cues forgotten · verbatim query ·
   failure stage · workaround · outcome · urgency · evidence quote.
   Quotes are checked to exist verbatim in the source; unverified quotes are never shown as evidence.
4. **Discover** — UMAP + HDBSCAN cluster the coded failure modes bottom-up (noise allowed), the LLM
   names each theme, and themes are scored:
   `opportunity = share × severity × urgency` (rescaled, top = 100).
""")
    with right:
        st.subheader("Funnel of evidence")
        steps = ["Collected", "Keyword filter", "Semantic gate", "LLM: relevant"]
        vals = [gate["raw_records"], gate["after_keyword_filter"], gate["after_semantic_gate"], N]
        fig = go.Figure(go.Funnel(y=steps, x=vals, marker_color=[ORDINAL[0], ORDINAL[1], ORDINAL[2], ORDINAL[3]],
                                  textinfo="value+percent initial"))
        st.plotly_chart(style(fig, 320), use_container_width=True)
        src = pd.Series(gate["raw_by_source"]).rename("collected").to_frame()
        src["relevant"] = rel["source"].value_counts()
        st.dataframe(src.fillna(0).astype(int), use_container_width=True)
    if runlog:
        st.caption(f"Last pipeline run: {runlog.get('finished', '')[:16].replace('T', ' ')} UTC")

# ======================================================================= 2. RESEARCH QUESTIONS
with tabs[1]:
    st.markdown("Each chart answers one research question. Hover for counts; every number links "
                "to records in **Evidence explorer**.")

    # RQ1
    st.subheader("RQ1 · Which photos do people struggle to retrieve — and which ones do they give up on?")
    pt = rel.groupby(["photo_type", "outcome"]).size().unstack(fill_value=0)
    pt = pt.reindex(columns=[o for o in OUTCOME_ORDER if o in pt.columns])
    pt = pt.loc[pt.sum(axis=1).sort_values().index]
    fig = go.Figure()
    for i, o in enumerate(pt.columns):
        fig.add_bar(y=pt.index, x=pt[o], name=o.replace("_", " "), orientation="h",
                    marker=dict(color=ORDINAL[min(i, 3)], line=dict(color="#fcfcfb", width=2)),
                    hovertemplate="%{y} · " + o.replace("_", " ") + ": %{x}<extra></extra>")
    fig.update_layout(barmode="stack", title="Records by photo type, split by outcome")
    st.plotly_chart(style(fig, 420), use_container_width=True)

    # RQ2 + RQ3
    st.subheader("RQ2 / RQ3 · What do people remember — and what have they forgotten?")
    remembered = rel["cues_remembered"].explode().dropna().value_counts()
    missing = rel["cues_missing"].explode().dropna().value_counts()
    cues = pd.DataFrame({"Remembered": remembered, "Forgotten": missing}).fillna(0).astype(int)
    cues = cues.sort_values("Remembered")
    cues.index = [CUE_LABEL.get(i, i) for i in cues.index]
    fig = go.Figure()
    fig.add_bar(y=cues.index, x=cues["Remembered"], name="Remembered", orientation="h", marker_color=BLUE,
                text=cues["Remembered"], textposition="outside", cliponaxis=False)
    fig.add_bar(y=cues.index, x=cues["Forgotten"], name="Forgotten", orientation="h", marker_color=ORANGE,
                text=cues["Forgotten"], textposition="outside", cliponaxis=False)
    fig.update_layout(barmode="group", title="Memory cues mentioned (a record can mention several)", bargap=0.25)
    st.plotly_chart(style(fig, 460), use_container_width=True)
    ctx = rel["cues_remembered"].map(lambda l: bool(set(l) & CONTEXT_CUES)).mean()
    content = rel["cues_remembered"].map(lambda l: bool(set(l) & {"visual_content", "text_in_image", "place"})).mean()
    a, b = st.columns(2)
    a.metric("Records remembering *context* (who / when / why / what was happening / source app)", f"{ctx:.0%}")
    b.metric("Records remembering *content* (what's visible / text / place)", f"{content:.0%}")

    # RQ4
    st.subheader("RQ4 · How do people phrase searches when memory is incomplete?")
    a, b = st.columns([1, 1.2])
    with a:
        qs = rel.loc[rel["query_style"] != "none", "query_style"].value_counts()
        if len(qs):
            st.plotly_chart(hbar(qs, "Style of the search they describe", pct_of=int(qs.sum()), h=300),
                            use_container_width=True)
    with b:
        q = rel.loc[rel["query_text"].notna() & (rel["query_text"].str.len() > 1),
                    ["query_text", "query_style", "failure_stage", "url"]]
        st.markdown(f"**{len(q)} verbatim search phrases** users said they typed")
        st.dataframe(q.head(60), use_container_width=True, height=300,
                     column_config={"url": st.column_config.LinkColumn("source")})

    # RQ5
    st.subheader("RQ5 · Where in the retrieval journey does it break?")
    a, b = st.columns([1, 1.2])
    with a:
        stg = rel["failure_stage"].value_counts().reindex(STAGE_ORDER).dropna().astype(int)
        stg.index = [STAGE_LABEL[i] for i in stg.index]
        st.plotly_chart(hbar(stg[::-1], "Failure stage (journey order)", pct_of=N, h=380, sort=False), use_container_width=True)
    with b:
        m = rel.explode("cues_remembered").dropna(subset=["cues_remembered"]).reset_index(drop=True)
        hm = pd.crosstab(m["cues_remembered"], m["failure_stage"])
        hm = hm.reindex(columns=[s for s in STAGE_ORDER if s in hm.columns])
        hm.index = [CUE_LABEL.get(i, i) for i in hm.index]
        fig = px.imshow(hm, color_continuous_scale=SEQ, text_auto=True, aspect="auto",
                        labels=dict(x="Failure stage", y="Cue remembered", color="Records"))
        fig.update_layout(title="Cue remembered × where it broke")
        st.plotly_chart(style(fig, 380), use_container_width=True)

    # RQ6
    st.subheader("RQ6 · What do people do instead?")
    w = rel.loc[rel["workaround"] != "none_mentioned", "workaround"].value_counts()
    if len(w):
        w.index = [i.replace("_", " ") for i in w.index]
        st.plotly_chart(hbar(w, "Workarounds mentioned", pct_of=int(w.sum()), h=340), use_container_width=True)

# ======================================================================= 3. THEMES
with tabs[2]:
    st.markdown("Themes are discovered **bottom-up** (UMAP + HDBSCAN on coded failure modes), named by the LLM, "
                "then scored: `opportunity = share × severity × urgency`, rescaled so the top theme = 100.")
    show = themes[["theme", "opportunity", "n", "share", "gave_up_rate", "severity", "urgency_norm",
                   "top_stage", "top_photo_type", "problem", "mismatch"]]
    st.dataframe(show, use_container_width=True, hide_index=True, column_config={
        "opportunity": st.column_config.ProgressColumn("Opportunity", min_value=0, max_value=100, format="%.0f"),
        "share": st.column_config.NumberColumn("Share", format="%.2f"),
        "gave_up_rate": st.column_config.NumberColumn("Gave up", format="%.2f"),
        "severity": st.column_config.NumberColumn(format="%.2f"),
        "urgency_norm": st.column_config.NumberColumn("Urgency", format="%.2f"),
    })
    if themes.empty:
        st.info("Not enough relevant records to form themes yet.")
        st.stop()
    pick = st.selectbox("Inspect a theme", themes["theme"].tolist())
    sel = rel["theme"] == pick
    fig = go.Figure()
    fig.add_scatter(x=rel.loc[~sel, "x"], y=rel.loc[~sel, "y"], mode="markers", name="Other records",
                    marker=dict(size=8, color=GRAY, line=dict(color="#fcfcfb", width=1)),
                    text=rel.loc[~sel, "theme"], hovertemplate="%{text}<extra></extra>")
    fig.add_scatter(x=rel.loc[sel, "x"], y=rel.loc[sel, "y"], mode="markers", name=pick,
                    marker=dict(size=10, color=BLUE, line=dict(color="#fcfcfb", width=2)),
                    text=rel.loc[sel, "failure_mode"], hovertemplate="%{text}<extra></extra>")
    fig.update_layout(title="Semantic map of failure modes (UMAP 2-D)")
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    st.plotly_chart(style(fig, 440), use_container_width=True)
    ev = rel.loc[sel & rel["quote_verified"]].sort_values("urgency", ascending=False)
    st.markdown(f"**Verified evidence — {pick}** ({sel.sum()} records)")
    for r in ev.head(8).itertuples():
        st.markdown(f"> “{r.evidence_quote}”  \n<span style='color:#52514e'>{r.source} · {r.photo_type} · "
                    f"stage: {r.failure_stage} · outcome: {r.outcome} · [source]({r.url})</span>",
                    unsafe_allow_html=True)

# ======================================================================= 4. EVIDENCE
with tabs[3]:
    f = st.columns(5)
    srcs = f[0].multiselect("Source", sorted(rel["source"].unique()))
    pts = f[1].multiselect("Photo type", sorted(rel["photo_type"].unique()))
    stgs = f[2].multiselect("Failure stage", [s for s in STAGE_ORDER if s in rel["failure_stage"].unique()])
    cue = f[3].multiselect("Cue remembered", sorted(CUE_LABEL), format_func=lambda c: CUE_LABEL[c])
    outs = f[4].multiselect("Outcome", OUTCOME_ORDER)
    v = rel
    if srcs: v = v[v["source"].isin(srcs)]
    if pts: v = v[v["photo_type"].isin(pts)]
    if stgs: v = v[v["failure_stage"].isin(stgs)]
    if outs: v = v[v["outcome"].isin(outs)]
    if cue: v = v[v["cues_remembered"].map(lambda l: bool(set(l) & set(cue)))]
    st.caption(f"{len(v)} of {N} records")
    st.dataframe(v[["evidence_quote", "photo_type", "failure_stage", "outcome", "urgency", "cues_remembered",
                    "cues_missing", "query_text", "workaround", "theme", "source", "url"]],
                 use_container_width=True, height=560, hide_index=True,
                 column_config={"url": st.column_config.LinkColumn("source link")})
    st.download_button("Download these records (CSV)", v.drop(columns=["x", "y"], errors="ignore").to_csv(index=False),
                       "retrieval_evidence.csv", "text/csv")

# ======================================================================= 5. ASK
def _has_key() -> bool:
    return bool(os.environ.get("GEMINI_API_KEY"))


@st.cache_resource
def _index(texts: tuple[str, ...]):
    from sklearn.feature_extraction.text import TfidfVectorizer
    vec = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", sublinear_tf=True)
    return vec, vec.fit_transform(texts)


with tabs[4]:
    st.markdown("Ask a question about the corpus. The engine retrieves the most relevant coded records "
                "and answers **only** from them, citing record ids (citations are validated).")
    if not _has_key():
        st.info("Add `GEMINI_API_KEY` in the app's Streamlit secrets to enable this tab.")
    examples = ["What do people remember about receipts or documents they can't find?",
                "Why does searching by a place or event fail?",
                "What workarounds do people use when search fails?"]
    qtext = st.text_input("Question", placeholder=examples[0])
    st.caption("Try: " + " · ".join(f"*{e}*" for e in examples))
    if qtext and _has_key():
        corpus = (rel["full_text"].str.slice(0, 600) + " || " + rel["failure_mode"].fillna("")).tolist()
        vec, M = _index(tuple(corpus))
        sims = (M @ vec.transform([qtext]).T).toarray().ravel()
        top = rel.iloc[sims.argsort()[::-1][:25]]
        ctx = "\n".join(f"[{r.id}] ({r.source}; type={r.photo_type}; stage={r.failure_stage}; outcome={r.outcome}) "
                        f"{r.full_text[:500]}" for r in top.itertuples())
        from pipeline import llm
        with st.spinner("Thinking…"):
            ans = llm.generate_text(
                f"Records:\n{ctx}\n\nQuestion: {qtext}\n\nAnswer in 4-8 sentences for a product manager. "
                "Use only the records. Cite record ids in square brackets after each claim. "
                "If the records don't answer it, say so.",
                system="You are a careful UX research analyst. Never invent facts or ids.")
        cited = set(re.findall(r"\[([^\]\s]+:[0-9a-f]+)\]", ans))
        valid = cited & set(top["id"])
        st.markdown(ans)
        st.caption(f"{len(valid)}/{len(cited)} citations validated against retrieved records.")
        with st.expander("Cited records"):
            st.dataframe(top[top["id"].isin(valid)][["id", "evidence_quote", "source", "url"]],
                         hide_index=True, use_container_width=True,
                         column_config={"url": st.column_config.LinkColumn("source")})

# ======================================================================= 6. LIVE CODER
with tabs[5]:
    st.markdown("Paste any review, Reddit post or interview note. The same codebook used on the whole corpus "
                "tags it live — this is the engine's core step, testable by anyone.")
    sample = ("I KNOW I took a photo of my landlord's rent receipt sometime last year, it was a WhatsApp image. "
              "Searched 'receipt', 'rent', 'bill' — nothing. Ended up scrolling for 20 minutes and gave up.")
    txt = st.text_area("Feedback text", value=sample, height=130)
    if st.button("Code it", type="primary"):
        if not _has_key():
            st.info("Add `GEMINI_API_KEY` in the app's Streamlit secrets to enable live coding.")
        else:
            from pipeline import llm
            from pipeline.extract import SYSTEM, _norm
            from pipeline.schema import CODEBOOK, RecordTag
            with st.spinner("Coding…"):
                out = llm.generate_json(CODEBOOK + "\n\nCode this item. id='live'.\n\n" + txt,
                                        schema=RecordTag, system=SYSTEM)
            tag = RecordTag.model_validate(out)
            ok = bool(tag.evidence_quote) and _norm(tag.evidence_quote) in _norm(txt)
            a, b = st.columns(2)
            a.metric("Failure stage", STAGE_LABEL.get(tag.failure_stage, tag.failure_stage))
            b.metric("Outcome · urgency", f"{tag.outcome} · {tag.urgency}/3")
            st.markdown(f"**Remembered:** {', '.join(CUE_LABEL.get(c, c) for c in tag.cues_remembered) or '—'}  \n"
                        f"**Forgotten:** {', '.join(CUE_LABEL.get(c, c) for c in tag.cues_missing) or '—'}  \n"
                        f"**Query typed:** {tag.query_text or '—'} ({tag.query_style})  \n"
                        f"**Why it failed:** {tag.failure_mode}  \n"
                        f"**Evidence:** “{tag.evidence_quote}” — {'✅ verified verbatim' if ok else '⚠️ not verbatim'}")
            with st.expander("Raw JSON"):
                st.json(tag.model_dump())

st.divider()
st.caption("Built for a product research case study. Sources are public reviews and posts; links open the original. "
           "Code: github.com/aishwary-bangre/google-photos-discovery-engine")

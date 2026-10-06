# Memory-Mismatch Discovery Engine — Google Photos

**Question:** people remember *something* about an old photo, yet still fail to find it.
What do they remember, what have they forgotten, and where exactly does retrieval break?

This engine answers that from public user feedback at scale. Instead of sentiment analysis, it
codes each post on the **memory-mismatch codebook**: what the person remembered, what they had
forgotten, how they phrased the search, where in the journey it broke, what they did instead and
whether they gave up. It then finds problem themes bottom-up and ranks them as opportunities.

## Pipeline

```
 Play Store (IN/US/GB) ─┐
 App Store RSS (5 cc)  ─┤                 ┌────────────┐   ┌──────────────────────┐   ┌────────────────────┐
 Reddit search+comments ┼─► 1. COLLECT ──►│ 2. GATE    │──►│ 3. CODE (Gemini)     │──►│ 4. DISCOVER        │──► Streamlit
 PullPush archive      ─┤   (Actions)     │ regex +    │   │ Pydantic schema,     │   │ UMAP + HDBSCAN,    │    dashboard
 Apify/CSV imports     ─┘                 │ embeddings │   │ JSON-constrained,    │   │ LLM-named themes,  │
                                          │ vs anchors │   │ verbatim-quote check │   │ opportunity score  │
                                          └────────────┘   └──────────────────────┘   └────────────────────┘
```

| Stage | What it does | Why |
|---|---|---|
| Collect | Scrapes tens of thousands of public reviews/posts; each source fails soft | Breadth across Android, iOS and long-form Reddit stories |
| Gate | Retrieval-language regex → `all-MiniLM-L6-v2` similarity to 7 problem anchors | Most feedback is about backup/storage; keep only "trying to find a photo" |
| Code | Batched Gemini calls with an enforced JSON schema; every quote verified verbatim against its source; per-record cache so runs resume after quota limits | Structured, auditable evidence — not summaries |
| Discover | Embeds coded failure modes → UMAP (5-D) → HDBSCAN (noise allowed) → LLM names each cluster → `opportunity = share × severity × urgency` | Lets the problems emerge from data rather than from my hypotheses |

### Codebook (per record)
`photo_type` · `cues_remembered[]` · `cues_missing[]` · `query_text` (verbatim) · `query_style` ·
`failure_stage` (express / interpret / evaluate / refine / browse / access) · `failure_mode` ·
`workaround` · `outcome` · `urgency` · `evidence_quote` (+ `quote_verified`)

Failure stages map to the metric decomposition:
**Recall → Express → Interpret → Evaluate → Refine → Found**.

## Dashboard tabs
1. **How it works** — evidence funnel and source mix
2. **Research questions** — RQ1 photo types × outcome · RQ2/3 cues remembered vs forgotten ·
   RQ4 how searches are phrased · RQ5 failure stage + cue × stage heatmap · RQ6 workarounds
3. **Opportunity themes** — ranked themes, semantic map, verified quotes with source links
4. **Evidence explorer** — filter every coded record, download CSV
5. **Ask the engine** — RAG over the coded corpus; answers cite record ids, citations validated
6. **Try the coder live** — paste any feedback and see it coded with the same schema

## Run it

**GitHub Actions (recommended):** add repo secret `GEMINI_API_KEY`, then
*Actions → Discovery pipeline → Run workflow*. Results are committed to `data/processed/`.
Re-running only `extract cluster` resumes from the cache if a quota limit stopped a run.

**Locally:**
```bash
pip install -r requirements-pipeline.txt
export GEMINI_API_KEY=...
python -m pipeline.run                      # all stages
python -m pipeline.run --stages extract cluster
streamlit run streamlit_app.py
```

**Extra data:** drop Apify Reddit-scraper JSON or any CSV with a `text` column into
`data/raw/external/` and run with `--sources external`.

## Limitations
- Public feedback over-represents frustrated, vocal users; it shows *what* breaks, not *how often*
  across all users. Primary research (live retrieval tasks) is used to validate.
- English only. App Store RSS exposes only the latest ~500 reviews per country.
- LLM coding is checked by schema validation and verbatim-quote verification. A random sample
  (40 relevant + 20 rejected records) is exported to `data/processed/spot_check.csv` for human review.

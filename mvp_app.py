"""Memory Detective — MVP.

Finds a photo from what you remember *around* it (when-ish, which part of your life,
who was there, why you took it, where it came from) instead of the exact words that
describe what's *in* it. Includes a Google-Photos-style baseline and a task harness
so the two can be compared on the same library.
"""
from __future__ import annotations

import json
import math
import os
import random
import re
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Memory Detective", page_icon="🕵️", layout="wide")
try:
    if "GEMINI_API_KEY" in st.secrets:
        os.environ["GEMINI_API_KEY"] = st.secrets["GEMINI_API_KEY"]
    if "GROQ_API_KEY" in st.secrets:
        os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
except Exception:  # noqa: BLE001
    pass

LIB = Path(__file__).parent / "mvp" / "library"
TODAY = datetime(2026, 10, 6)


# ============================================================ data
@st.cache_data
def load_library():
    d = json.loads((LIB / "library.json").read_text())
    items = pd.DataFrame(d["items"])
    items["dt"] = pd.to_datetime(items["date"])
    items["month"] = items["dt"].dt.strftime("%B %Y")
    items["path"] = items["file"].map(lambda f: str(LIB / f))
    # what a Google-Photos-like index knows: visual labels, OCR text, place, date, named faces
    items["gp_doc"] = (items["labels"] + " " + items["caption"] + " " + items["ocr"] + " " + items["place"] + " "
                       + items["month"] + " " + items["people"].map(" ".join) + " "
                       + items["kind"].replace({"screenshot": "screenshot screenshots"}))
    moments = d["moments"]
    items["moment_name"] = items["moment"].map(lambda m: moments[m]["name"])
    # what Memory Detective additionally understands: life moments, source app, purpose
    items["md_doc"] = (items["gp_doc"] + " " + items["moment_name"] + " " + items["purpose"] + " "
                       + items["source"].replace({"whatsapp": "whatsapp forwarded sent", "download": "downloaded email",
                                                  "camera": "camera photo i took", "screenshot": "screenshot"}))
    return items, moments


items, MOMENTS = load_library()


@st.cache_resource
def tfidf(col: str):
    from sklearn.feature_extraction.text import TfidfVectorizer
    vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words="english")
    return vec, vec.fit_transform(items[col].str.lower())


def text_scores(query: str, col: str) -> np.ndarray:
    vec, M = tfidf(col)
    if not query.strip():
        return np.zeros(len(items))
    return (M @ vec.transform([query.lower()]).T).toarray().ravel()


# ============================================================ cue parsing
PARSE_PROMPT = """You turn a person's fuzzy memory of a photo into search cues for their own photo library.
Today is {today}. Their life moments (from their library):
{moments}

Memory: "{memory}"

Return JSON with keys:
  moments: list of moment ids that plausibly match (may be empty),
  date_from, date_to: ISO dates bounding when it was taken, or null,
  kind: one of photo | screenshot | document | any   (document = photo of paper/ticket/receipt/card),
  source: one of camera | screenshot | whatsapp | download | any,
  people: list of names mentioned,
  terms: 5-12 search words: visible content AND likely printed words (e.g. medicine -> tablet, strip, pharmacy),
  summary: one short line restating what they're looking for.
"""


def llm_parse(memory: str) -> dict | None:
    if not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GROQ_API_KEY")):
        return None
    try:
        from pipeline import llm
        mlist = "\n".join(f"- {k}: {v['name']} ({v['start']} to {v['end']}, {v['place']}, with {', '.join(v['people']) or 'no one'})"
                          for k, v in MOMENTS.items())
        out = llm.generate_json(PARSE_PROMPT.format(today=TODAY.date(), moments=mlist, memory=memory),
                                system="Be literal. Only use moment ids from the list.")
        out["moments"] = [m for m in out.get("moments", []) if m in MOMENTS]
        return out
    except Exception as e:  # noqa: BLE001
        st.session_state["llm_error"] = str(e)[:200]
        return None


KEYWORDS = {  # rule-based fallback when no LLM is available
    "sick_feb26": ["sick", "fever", "ill", "doctor", "medicine", "unwell", "flu"],
    "goa_trip": ["goa", "beach", "trip with friends"], "manali_trip": ["manali", "snow", "mountain", "himachal"],
    "internship_blr": ["internship", "bengaluru", "bangalore", "pg", "office", "rent"],
    "diwali_home": ["diwali", "diya", "rangoli"], "cousin_wedding": ["wedding", "priya", "mehendi", "mehndi"],
    "hostel_life": ["hostel", "college", "class", "lecture", "wifi", "fest", "campus"], "home": ["home", "bruno", "dog", "jaipur"],
}


def rule_parse(memory: str) -> dict:
    t = memory.lower()
    moms = [m for m, kws in KEYWORDS.items() if any(re.search(rf"\b{re.escape(k)}\b", t) for k in kws)]
    kind = "screenshot" if "screenshot" in t or "sent me" in t or "chat" in t else (
        "document" if any(k in t for k in ("receipt", "bill", "ticket", "prescription", "document", "card")) else "any")
    return {"moments": moms, "date_from": None, "date_to": None, "kind": kind, "source": "any", "people": [],
            "terms": re.findall(r"[a-z]{3,}", t), "summary": memory[:80]}


def score(cues: dict, boosts: dict) -> np.ndarray:
    s = 1.0 * text_scores(" ".join(cues.get("terms", [])), "md_doc")
    if cues.get("moments"):
        s += 0.9 * items["moment"].isin(cues["moments"]).to_numpy()
    if cues.get("kind") not in (None, "any"):
        s += 0.5 * (items["kind"] == cues["kind"]).to_numpy()
    if cues.get("source") not in (None, "any"):
        s += 0.3 * (items["source"] == cues["source"]).to_numpy()
    for p in cues.get("people", []):
        s += 0.4 * items["people"].map(lambda l: p.lower() in [x.lower() for x in l]).to_numpy()
    try:
        if cues.get("date_from") and cues.get("date_to"):
            a, b = pd.to_datetime(cues["date_from"]), pd.to_datetime(cues["date_to"]) + pd.Timedelta(days=1)
            s += 0.6 * ((items["dt"] >= a) & (items["dt"] <= b)).to_numpy()
    except Exception:  # noqa: BLE001
        pass
    # answers to clarifying questions & warmer/colder are soft multipliers (memory can be wrong)
    for col, val in boosts.get("match", []):
        s = np.where(items[col] == val, s * 1.8 + 0.3, s * 0.45)
    for iid in boosts.get("warmer", []):
        ref = items.loc[items["id"] == iid].iloc[0]
        days = (items["dt"] - ref["dt"]).abs().dt.days.to_numpy()
        s = s + 0.6 * (items["moment"] == ref["moment"]).to_numpy() + 0.5 * (items["kind"] == ref["kind"]).to_numpy() \
            + 0.4 * np.exp(-days / 10)
    for iid in boosts.get("not_it", []):
        s = np.where(items["id"] == iid, -9, s)
    return s


QUESTIONS = {
    "kind": ("What kind of image was it?", {"photo": "📷 A photo of a place / people / thing",
                                             "document": "🧾 A photo of paper — receipt, ticket, card, prescription",
                                             "screenshot": "📱 A screenshot (chat, payment, app)"}),
    "moment": ("Which part of your life was it from?", None),
    "source": ("How did it get into your phone?", {"camera": "I took it", "screenshot": "I screenshotted it",
                                                   "whatsapp": "Someone sent it (WhatsApp)", "download": "I downloaded it"}),
    "season": ("Roughly when?", None),
}


def best_question(s: np.ndarray, asked: set[str]):
    """Pick the attribute whose answer best splits the current top candidates (max entropy)."""
    top = items.assign(_s=s).sort_values("_s", ascending=False).head(15)
    top = top.assign(season=top["dt"].dt.year.astype(str) + " " + ((top["dt"].dt.month - 1) // 3).map(
        {0: "Jan–Mar", 1: "Apr–Jun", 2: "Jul–Sep", 3: "Oct–Dec"}))
    best, best_h = None, 0.6
    for attr in ("kind", "moment", "source", "season"):
        if attr in asked:
            continue
        p = np.array(list(Counter(top[attr]).values()), dtype=float)
        p /= p.sum()
        h = -(p * np.log2(p)).sum()
        if h > best_h:
            best, best_h = attr, h
    if best is None:
        return None
    vals = [v for v, _ in Counter(top[best]).most_common(4)]
    q, labels = QUESTIONS[best]
    if best == "moment":
        opts = {v: MOMENTS[v]["name"] for v in vals}
    elif best == "season":
        opts = {v: v for v in vals}
    else:
        opts = {v: labels.get(v, v) for v in vals}
    return best, q, opts


# ============================================================ tasks
TASKS = [
    {"id": "T1", "target": "strip_feb26", "brief": "You were down with fever earlier this year and the doctor gave you "
     "a course of tablets. You took a photo of the tablet strip so you could re-order it. **Find that photo.**"},
    {"id": "T2", "target": "cafe_bill_goa", "brief": "On the Goa trip, a friend sent you the location of a small café "
     "and you all went there for dinner. You photographed the bill to split it. **Find the café bill.**"},
    {"id": "T3", "target": "rent_jul26", "brief": "During your internship you stayed in a PG. The office now wants proof "
     "of the rent you paid for **July**. You had photographed each month's receipt. **Find July's receipt.**"},
    {"id": "T4", "target": "wifi_chat", "brief": "Back in the hostel, someone in your wing shared the password of the "
     "new WiFi router in the group chat and you screenshotted it. **Find that screenshot.**"},
]
PLANS = {"A": {"T1": "baseline", "T2": "detective", "T3": "baseline", "T4": "detective"},
         "B": {"T1": "detective", "T2": "baseline", "T3": "detective", "T4": "baseline"}}

ss = st.session_state
ss.setdefault("log", [])
ss.setdefault("active", None)


def start_task(task, mode):
    ss.active = {"task": task, "mode": mode, "t0": time.time(), "turns": 0, "cues": None, "boosts": {"match": [], "warmer": [], "not_it": []},
                 "asked": set(), "history": [], "query": ""}


def finish(found_id: str | None, gave_up=False):
    a = ss.active
    ok = (found_id == a["task"]["target"])
    ss.log.append({"tester": ss.get("tester", ""), "task": a["task"]["id"], "mode": a["mode"], "success": ok and not gave_up,
                   "picked": found_id or "", "seconds": round(time.time() - a["t0"], 1), "turns": a["turns"],
                   "time": datetime.now().strftime("%H:%M")})
    ss.last_result = (ok and not gave_up, a)
    ss.active = None


def grid(df: pd.DataFrame, key: str, warmer=False, n=12):
    cols = st.columns(4)
    for i, r in enumerate(df.head(n).itertuples()):
        with cols[i % 4]:
            st.image(r.path, use_container_width=True)
            st.caption(f"{r.dt:%d %b %Y} · {r.place}")
            b1, b2 = st.columns(2)
            if b1.button("✅ This one", key=f"{key}_pick_{r.id}"):
                finish(r.id)
                st.rerun()
            if warmer and b2.button("🔥 Closer", key=f"{key}_warm_{r.id}", help="Not it, but close — show me things like this"):
                ss.active["boosts"]["warmer"].append(r.id)
                ss.active["boosts"]["not_it"].append(r.id)
                ss.active["turns"] += 1
                st.rerun()


# ============================================================ UI
st.title("🕵️ Memory Detective")
st.caption("Find a photo from what you remember *around* it — not the exact words for what's *in* it.")

with st.sidebar:
    st.subheader("Test session")
    ss.tester = st.text_input("Tester alias", value=ss.get("tester", ""), placeholder="e.g. P3")
    plan = st.radio("Plan", ["A", "B"], horizontal=True, help="Counterbalances which tasks use which mode")
    st.markdown("**You are Riya**, a final-year student. This is your phone library "
                f"({len(items)} photos). You'd remember your own life:")
    for k in ("cousin_wedding", "manali_trip", "diwali_home", "goa_trip", "sick_feb26", "internship_blr"):
        m = MOMENTS[k]
        st.markdown(f"- {m['name']} — {pd.to_datetime(m['start']):%b %Y}, {m['place']}")
    st.markdown("- College hostel in Pune · dog Bruno at home (Jaipur)")
    if ss.log:
        st.divider()
        res = pd.DataFrame(ss.log)
        st.dataframe(res[["task", "mode", "success", "seconds", "turns"]], hide_index=True)
        st.download_button("Download results CSV", res.to_csv(index=False), f"results_{ss.tester or 'tester'}.csv")

tab_test, tab_free, tab_how = st.tabs(["🧪 Retrieval tasks", "🔎 Free explore", "How it works"])

# ------------------------------------------------------------ task runner
with tab_test:
    if ss.get("last_result"):
        ok, a = ss.last_result
        (st.success if ok else st.error)(f"{a['task']['id']} ({a['mode']}): {'found it ✅' if ok else 'not the right photo ❌'} "
                                         f"in {ss.log[-1]['seconds']}s")
        ss.last_result = None
    if ss.active is None:
        done = {r["task"] for r in ss.log}
        st.markdown("Each task describes a memory. Find the photo as you naturally would. "
                    "The mode (classic search vs. Memory Detective) is set by the test plan.")
        for t in TASKS:
            mode = PLANS[plan][t["id"]]
            c1, c2 = st.columns([5, 1])
            c1.markdown(f"**{t['id']}** · *{'Classic search' if mode == 'baseline' else 'Memory Detective'}*  \n{t['brief']}")
            if c2.button("Start" if t["id"] not in done else "Redo", key=f"start_{t['id']}"):
                start_task(t, mode)
                st.rerun()
    else:
        a = ss.active
        st.info(f"**{a['task']['id']}** — {a['task']['brief']}")
        top_bar = st.columns([4, 1])
        if top_bar[1].button("Give up", type="secondary"):
            finish(None, gave_up=True)
            st.rerun()

        if a["mode"] == "baseline":
            st.markdown("##### Search your photos")
            q = st.text_input("Search", key=f"bq_{a['task']['id']}_{a['t0']}",
                              placeholder="Search your photos (things, places, people, text, months)")
            if q and q != a["query"]:
                a["query"], a["turns"] = q, a["turns"] + 1
            if a["query"]:
                s = text_scores(a["query"], "gp_doc")
                res = items.assign(_s=s)
                res = res[res["_s"] > 0.02].sort_values("_s", ascending=False)
                st.caption(f"{len(res)} results for “{a['query']}”")
                if res.empty:
                    st.warning("No results. Try other words — or scroll the timeline below.")
                grid(res, "b", n=16)
            st.markdown("##### Timeline")
            show = st.slider("Photos shown", 12, len(items), 24, step=12, key=f"tl_{a['t0']}")
            grid(items, "tl", n=show)
        else:
            if a["cues"] is None:
                mem = st.text_area("Describe whatever you remember — when-ish, what was going on, who was there, "
                                   "why you took it, where it came from. Fuzzy is fine.", height=100,
                                   key=f"mem_{a['t0']}")
                if st.button("Start looking", type="primary") and mem.strip():
                    with st.spinner("Reading your memory…"):
                        cues = llm_parse(mem) or rule_parse(mem)
                    a["cues"], a["turns"] = cues, 1
                    if cues.get("moments"):
                        a["asked"].add("moment")
                    if cues.get("date_from"):
                        a["asked"].add("season")
                    if cues.get("kind") not in (None, "any"):
                        a["asked"].add("kind")
                    a["history"].append(("you", mem))
                    st.rerun()
            else:
                cues = a["cues"]
                s = score(cues, a["boosts"])
                st.markdown(f"🕵️ **Looking for:** {cues.get('summary', '')}")
                chips = []
                if cues.get("moments"):
                    chips.append("moment: " + " / ".join(MOMENTS[m]["name"] for m in cues["moments"]))
                if cues.get("kind") not in (None, "any"):
                    chips.append(f"kind: {cues['kind']}")
                if cues.get("date_from"):
                    chips.append(f"between {cues['date_from']} and {cues.get('date_to')}")
                for col, val in a["boosts"]["match"]:
                    chips.append(f"{col}: {MOMENTS[val]['name'] if col == 'moment' else val}")
                if chips:
                    st.caption(" · ".join(chips))

                q = best_question(s, a["asked"])
                if q:
                    attr, text, opts = q
                    st.markdown(f"**{text}**")
                    bcols = st.columns(len(opts) + 1)
                    for i, (val, label) in enumerate(opts.items()):
                        if bcols[i].button(label, key=f"q_{attr}_{val}_{a['turns']}"):
                            col = "season" if attr == "season" else attr
                            if attr == "season":
                                yr, qtr = val.split(" ", 1)
                                months = {"Jan–Mar": (1, 3), "Apr–Jun": (4, 6), "Jul–Sep": (7, 9), "Oct–Dec": (10, 12)}[qtr]
                                a["cues"]["date_from"] = f"{yr}-{months[0]:02d}-01"
                                a["cues"]["date_to"] = f"{yr}-{months[1]:02d}-28"
                            else:
                                a["boosts"]["match"].append((col, val))
                            a["asked"].add(attr)
                            a["turns"] += 1
                            st.rerun()
                    if bcols[-1].button("Not sure", key=f"q_skip_{attr}_{a['turns']}"):
                        a["asked"].add(attr)
                        st.rerun()

                more = st.text_input("Remember anything else? (optional)", key=f"more_{a['t0']}_{a['turns']}")
                if more:
                    extra = llm_parse(more) or rule_parse(more)
                    a["cues"]["terms"] = list(cues.get("terms", [])) + list(extra.get("terms", []))
                    a["cues"]["moments"] = list(dict.fromkeys(list(cues.get("moments", [])) + list(extra.get("moments", []))))
                    a["turns"] += 1
                    st.rerun()

                res = items.assign(_s=s).sort_values("_s", ascending=False)
                st.markdown("##### Best matches — tap ✅ if you see it, 🔥 if one is *close*")
                grid(res, "d", warmer=True, n=8)
        if ss.get("llm_error"):
            st.caption(f"(AI parser unavailable, using rules: {ss.llm_error[:80]})")

# ------------------------------------------------------------ free explore
with tab_free:
    st.markdown("Try both on any memory of Riya's library.")
    mem = st.text_input("Your memory", placeholder="the bill from that café in Goa a friend sent me the location of")
    if mem:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Classic keyword search**")
            s = text_scores(mem, "gp_doc")
            r = items.assign(_s=s)
            r = r[r["_s"] > 0.02].sort_values("_s", ascending=False).head(6)
            for x in r.itertuples():
                st.image(x.path, width=160, caption=f"{x.dt:%d %b %Y} · {x.caption}")
            if r.empty:
                st.warning("No results")
        with c2:
            st.markdown("**Memory Detective**")
            cues = llm_parse(mem) or rule_parse(mem)
            st.json({k: cues.get(k) for k in ("moments", "kind", "date_from", "date_to", "terms")}, expanded=False)
            r = items.assign(_s=score(cues, {"match": [], "warmer": [], "not_it": []})).sort_values("_s", ascending=False).head(6)
            for x in r.itertuples():
                st.image(x.path, width=160, caption=f"{x.dt:%d %b %Y} · {x.moment_name}")

# ------------------------------------------------------------ how it works
with tab_how:
    st.markdown("""
**Why it exists.** In public feedback, people who can't find a photo usually remember *context* — roughly when,
which part of their life, who was around, why they took it, where it came from — but search only accepts words for
what is *visible* in the photo. The fallback, scrolling the timeline, breaks when there are many look-alikes
(receipts, payment screenshots, tablet strips).

**What it does differently**
1. **Understands life context.** An LLM maps a fuzzy memory onto the library's *life moments* (trips, illness,
   internship — detected from time + place clusters, like Google Photos' trips/events) and a time window, and expands
   it into likely visible/printed words.
2. **Asks one question at a time** — the one that best splits the current top candidates (entropy over kind, life
   moment, source app, season). Recognition is easier than recall, so every question is multiple-choice.
3. **Warmer/colder.** Tapping 🔥 on a near-miss pulls in photos from the same moment, of the same kind, taken near the
   same date — exactly how people say they navigate ("it was right after…").
4. **Soft, not hard, filters.** Memory is often wrong; answers re-rank instead of excluding.

**Classic search** (baseline) uses what a Google-Photos-style index knows: visual labels, text in the image (OCR),
place, month, named faces and screenshot type.

Library: synthetic receipts/screenshots (all names fictional) + CC-licensed photos from Wikimedia Commons.
""")
    with st.expander("Image credits"):
        st.dataframe(items[["id", "credit"]], hide_index=True, use_container_width=True)

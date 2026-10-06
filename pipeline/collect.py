"""Collectors for public user feedback about photo retrieval.

Every collector returns records in one common shape:
    {id, source, text, title, rating, date, url, meta}
and writes them to data/raw/<source>.jsonl. Each collector fails soft: if a
source blocks us, the run continues with the others and the failure is logged.
"""
from __future__ import annotations

import hashlib
import html
import json
import re
import time
from datetime import datetime, timezone

import requests

from . import config as C

UA = {"User-Agent": "Mozilla/5.0 (research; google-photos-discovery-engine; +https://github.com)"}


def _rid(source: str, key: str) -> str:
    return f"{source}:{hashlib.sha1(key.encode()).hexdigest()[:12]}"


def _write(source: str, records: list[dict]) -> int:
    seen, out = set(), []
    for r in records:
        if r["id"] in seen or not (r.get("text") or "").strip():
            continue
        seen.add(r["id"])
        out.append(r)
    path = C.RAW_DIR / f"{source}.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
    print(f"[collect] {source}: {len(out)} records -> {path.name}")
    return len(out)


# ---------------------------------------------------------------- Play Store
def collect_playstore() -> int:
    from google_play_scraper import Sort, reviews

    records = []
    for country in C.PLAY_COUNTRIES:
        for sort in (Sort.NEWEST, Sort.MOST_RELEVANT):
            token, got = None, 0
            target = C.PLAY_REVIEWS_PER_COUNTRY if sort == Sort.NEWEST else 3000
            while got < target:
                try:
                    batch, token = reviews(C.PLAY_APP_ID, lang="en", country=country,
                                           sort=sort, count=200, continuation_token=token)
                except Exception as e:  # noqa: BLE001
                    print(f"[collect] play {country} {sort} error: {e}")
                    break
                for r in batch:
                    records.append({
                        "id": _rid("play", r["reviewId"]),
                        "source": "play_store",
                        "text": r.get("content") or "",
                        "title": "",
                        "rating": r.get("score"),
                        "date": r.get("at"),
                        "url": f"https://play.google.com/store/apps/details?id={C.PLAY_APP_ID}&reviewId={r['reviewId']}",
                        "meta": {"country": country, "thumbs_up": r.get("thumbsUpCount", 0),
                                 "app_version": r.get("reviewCreatedVersion")},
                    })
                got += len(batch)
                if not token or not batch:
                    break
                time.sleep(0.3)
            print(f"[collect] play {country} {sort.name}: {got}")
    return _write("play_store", records)


# ---------------------------------------------------------------- App Store
def collect_appstore() -> int:
    records = []
    for cc in C.APPSTORE_COUNTRIES:
        for page in range(1, 11):
            url = (f"https://itunes.apple.com/{cc}/rss/customerreviews/page={page}/"
                   f"id={C.APPSTORE_APP_ID}/sortby=mostrecent/json")
            try:
                resp = requests.get(url, headers=UA, timeout=20)
                entries = resp.json().get("feed", {}).get("entry", [])
            except Exception as e:  # noqa: BLE001
                print(f"[collect] appstore {cc} p{page} error: {e}")
                break
            if isinstance(entries, dict):
                entries = [entries]
            entries = [e for e in entries if "im:rating" in e]
            if not entries:
                break
            for e in entries:
                rid = e.get("id", {}).get("label", "")
                records.append({
                    "id": _rid("appstore", rid),
                    "source": "app_store",
                    "text": e.get("content", {}).get("label", ""),
                    "title": e.get("title", {}).get("label", ""),
                    "rating": int(e.get("im:rating", {}).get("label", 0) or 0),
                    "date": e.get("updated", {}).get("label"),
                    "url": f"https://apps.apple.com/{cc}/app/id{C.APPSTORE_APP_ID}?see-all=reviews",
                    "meta": {"country": cc},
                })
            time.sleep(0.5)
    return _write("app_store", records)


# ---------------------------------------------------------------- Reddit
def _reddit_get(url: str, params: dict) -> dict | None:
    for host in ("https://www.reddit.com", "https://old.reddit.com"):
        try:
            r = requests.get(host + url, params=params, headers=UA, timeout=20)
            if r.status_code == 200:
                return r.json()
            print(f"[collect] reddit {host}{url} -> HTTP {r.status_code}")
        except Exception as e:  # noqa: BLE001
            print(f"[collect] reddit error: {e}")
        time.sleep(2)
    return None


def _post_record(d: dict, src: str) -> dict:
    return {
        "id": _rid("reddit", d.get("id", d.get("permalink", ""))),
        "source": src,
        "text": d.get("selftext") or d.get("body") or "",
        "title": d.get("title", ""),
        "rating": None,
        "date": datetime.fromtimestamp(d.get("created_utc", 0), tz=timezone.utc).isoformat(),
        "url": "https://www.reddit.com" + d.get("permalink", "") if d.get("permalink") else d.get("full_link", ""),
        "meta": {"subreddit": d.get("subreddit"), "score": d.get("score"),
                 "num_comments": d.get("num_comments")},
    }


def collect_reddit() -> int:
    records, posts = [], {}
    jobs = [(f"/r/{s}/search.json", {"q": q, "restrict_sr": 1, "limit": 100, "sort": "relevance", "t": "all"})
            for s in C.REDDIT_SUBS for q in C.REDDIT_QUERIES]
    jobs += [("/search.json", {"q": q, "limit": 100, "sort": "relevance", "t": "all"})
             for q in C.REDDIT_GLOBAL_QUERIES]
    blocked = 0
    for path, params in jobs:
        data = _reddit_get(path, params)
        if data is None:
            blocked += 1
            if blocked >= 4:
                print("[collect] reddit appears blocked from this runner; skipping to PullPush")
                break
            continue
        for child in data.get("data", {}).get("children", []):
            d = child.get("data", {})
            text = (d.get("title", "") + " " + d.get("selftext", "")).lower()
            if "photo" not in text and "picture" not in text and "pic" not in text:
                continue
            posts[d["id"]] = d
        time.sleep(1.2)

    for d in posts.values():
        records.append(_post_record(d, "reddit"))

    # Comments of the most-discussed posts carry most of the stories.
    top = sorted(posts.values(), key=lambda d: d.get("num_comments", 0), reverse=True)[:120]
    for d in top:
        data = _reddit_get(f"/comments/{d['id']}.json", {"limit": 100, "depth": 2})
        if not data or len(data) < 2:
            continue
        for c in data[1].get("data", {}).get("children", []):
            cd = c.get("data", {})
            if c.get("kind") != "t1" or len(cd.get("body", "")) < 40:
                continue
            rec = _post_record(cd, "reddit_comment")
            rec["title"] = d.get("title", "")
            records.append(rec)
        time.sleep(1.2)
    return _write("reddit", records)


def collect_pullpush() -> int:
    """PullPush (Pushshift successor) — works when reddit.com blocks CI runners."""
    records = []
    base = "https://api.pullpush.io/reddit/search"
    for kind in ("submission", "comment"):
        for sub in C.REDDIT_SUBS[:5]:
            for q in C.REDDIT_QUERIES:
                try:
                    r = requests.get(f"{base}/{kind}/", params={"q": q, "subreddit": sub, "size": 100},
                                     headers=UA, timeout=30)
                    items = r.json().get("data", []) if r.status_code == 200 else []
                except Exception as e:  # noqa: BLE001
                    print(f"[collect] pullpush error: {e}")
                    items = []
                for d in items:
                    if kind == "comment":
                        d.setdefault("permalink", d.get("permalink", ""))
                    records.append(_post_record(d, "reddit" if kind == "submission" else "reddit_comment"))
                time.sleep(1.0)
    return _write("pullpush", records)


# ---------------------------------------------------------------- Manual imports
def import_external() -> int:
    """Pick up any exported datasets dropped into data/raw/external/ (Apify, CSVs).

    Supported: Apify Reddit Scraper JSON (fields: id/title/body/url/createdAt/communityName),
    generic CSV/JSON with a `text` column.
    """
    import pandas as pd

    ext = C.RAW_DIR / "external"
    if not ext.exists():
        return 0
    records = []
    for p in ext.iterdir():
        try:
            if p.suffix == ".json":
                rows = json.loads(p.read_text(encoding="utf-8"))
            elif p.suffix == ".jsonl":
                rows = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
            elif p.suffix == ".csv":
                rows = pd.read_csv(p).to_dict("records")
            else:
                continue
        except Exception as e:  # noqa: BLE001
            print(f"[collect] could not read {p.name}: {e}")
            continue
        for row in rows:
            text = row.get("body") or row.get("text") or row.get("content") or row.get("comment") or ""
            title = row.get("title") or ""
            key = str(row.get("id") or row.get("url") or text[:80])
            records.append({
                "id": _rid("ext", key), "source": row.get("source") or f"external:{p.stem}",
                "text": str(text), "title": str(title), "rating": row.get("rating"),
                "date": row.get("createdAt") or row.get("date"), "url": row.get("url") or "",
                "meta": {"community": row.get("communityName") or row.get("subreddit")},
            })
    return _write("external", records)


# ---------------------------------------------------------------- Hacker News
def collect_hn() -> int:
    """Hacker News comments/stories via the public Algolia API (long, specific stories)."""
    records = []
    for q in C.HN_QUERIES:
        for tags in ("comment", "story"):
            try:
                r = requests.get("https://hn.algolia.com/api/v1/search",
                                 params={"query": q, "tags": tags, "hitsPerPage": 200}, headers=UA, timeout=30)
                hits = r.json().get("hits", []) if r.status_code == 200 else []
            except Exception as e:  # noqa: BLE001
                print(f"[collect] hn error: {e}")
                hits = []
            for h in hits:
                text = re.sub(r"<[^>]+>", " ", html.unescape(h.get("comment_text") or h.get("story_text") or ""))
                if "photo" not in (text + (h.get("title") or "")).lower():
                    continue
                records.append({
                    "id": _rid("hn", h.get("objectID", "")), "source": "hacker_news",
                    "text": text, "title": h.get("title") or h.get("story_title") or "",
                    "rating": None, "date": h.get("created_at"),
                    "url": f"https://news.ycombinator.com/item?id={h.get('objectID')}", "meta": {},
                })
            time.sleep(0.5)
    return _write("hacker_news", records)


COLLECTORS = {
    "hacker_news": collect_hn,
    "play_store": collect_playstore,
    "app_store": collect_appstore,
    "reddit": collect_reddit,
    "pullpush": collect_pullpush,
    "external": import_external,
}


def run(sources: list[str] | None = None) -> dict:
    stats = {}
    for name, fn in COLLECTORS.items():
        if sources and name not in sources:
            continue
        try:
            stats[name] = fn()
        except Exception as e:  # noqa: BLE001
            print(f"[collect] {name} failed: {e}")
            stats[name] = 0
    return stats

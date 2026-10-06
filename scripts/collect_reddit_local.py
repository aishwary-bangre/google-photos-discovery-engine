"""Collect Reddit posts + comments about finding old photos — run this on YOUR laptop.

Reddit blocks cloud servers (GitHub Actions), but works fine from a home connection.

    pip install requests
    python collect_reddit_local.py

It writes reddit_local.jsonl next to this script (takes ~10 minutes).
Upload that file to the repo folder data/raw/external/ (GitHub: open the folder ->
Add file -> Upload files), then re-run the pipeline workflow.
"""
import json
import time
from datetime import datetime, timezone

import requests

UA = {"User-Agent": "windows:photo-retrieval-research:v1.0 (academic research script)"}
SUBS = ["googlephotos", "GooglePixel", "Android", "pixel_phones", "DataHoarder", "iphone", "ios",
        "techsupport", "NoStupidQuestions", "TipOfMyTongue"]
QUERIES = ["can't find photo", "cannot find a photo", "find old photo", "looking for a photo",
           "google photos search", "search photos not working", "find a picture I took",
           "find screenshot", "remember taking a photo", "photo from years ago", "lost a photo",
           "scrolling to find photo", "search by date photos", "find photo of document"]
GLOBAL = ['"google photos" "can\'t find"', '"google photos" search find old', 'google photos search useless',
          'find a specific photo in google photos', 'find old screenshot phone']


def get(url, params):
    for _ in range(3):
        try:
            r = requests.get("https://www.reddit.com" + url, params=params, headers=UA, timeout=25)
            if r.status_code == 200:
                return r.json()
            if r.status_code == 429:
                time.sleep(20)
                continue
            print("HTTP", r.status_code, url)
            return None
        except Exception as e:  # noqa: BLE001
            print("error", e)
            time.sleep(5)
    return None


def rec(d, kind, title=""):
    return {
        "id": d.get("name") or d.get("id"), "source": kind,
        "title": d.get("title") or title, "text": d.get("selftext") or d.get("body") or "",
        "date": datetime.fromtimestamp(d.get("created_utc", 0), tz=timezone.utc).isoformat(),
        "url": "https://www.reddit.com" + d.get("permalink", ""), "subreddit": d.get("subreddit"),
    }


def main():
    posts = {}
    jobs = [(f"/r/{s}/search.json", {"q": q, "restrict_sr": 1, "limit": 100, "sort": "relevance", "t": "all"})
            for s in SUBS for q in QUERIES]
    jobs += [("/search.json", {"q": q, "limit": 100, "sort": "relevance", "t": "all"}) for q in GLOBAL]
    for i, (path, params) in enumerate(jobs, 1):
        data = get(path, params)
        for c in (data or {}).get("data", {}).get("children", []):
            d = c["data"]
            t = (d.get("title", "") + " " + d.get("selftext", "")).lower()
            if any(w in t for w in ("photo", "picture", "pic ", "pics", "screenshot", "image")):
                posts[d["id"]] = d
        print(f"search {i}/{len(jobs)} — {len(posts)} posts so far")
        time.sleep(1.1)

    out = [rec(d, "reddit") for d in posts.values()]
    top = sorted(posts.values(), key=lambda d: d.get("num_comments", 0), reverse=True)[:250]
    for i, d in enumerate(top, 1):
        data = get(f"/comments/{d['id']}.json", {"limit": 200, "depth": 3})
        if data and len(data) > 1:
            stack = list(data[1]["data"]["children"])
            while stack:
                c = stack.pop()
                if c.get("kind") != "t1":
                    continue
                cd = c["data"]
                if len(cd.get("body", "")) >= 40:
                    out.append(rec(cd, "reddit_comment", d.get("title", "")))
                rep = cd.get("replies")
                if isinstance(rep, dict):
                    stack += rep["data"]["children"]
        print(f"comments {i}/{len(top)} — {len(out)} records")
        time.sleep(1.1)

    with open("reddit_local.jsonl", "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nDone: {len(out)} records -> reddit_local.jsonl")


if __name__ == "__main__":
    main()

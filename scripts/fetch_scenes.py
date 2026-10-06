"""Fetch scene photos for the MVP demo library from Wikimedia Commons (CC-licensed).

Runs in GitHub Actions (Commons is not reachable from every sandbox). Downloads a
480px thumbnail per image into mvp/library/scenes/ and writes scenes_raw.json with
title, license and attribution so the app can credit every image.
"""
import json
import re
import time
from pathlib import Path

import requests

OUT = Path(__file__).resolve().parent.parent / "mvp" / "library"
IMG = OUT / "scenes"
IMG.mkdir(parents=True, exist_ok=True)
UA = {"User-Agent": "photo-retrieval-research/1.0 (student project; contact via GitHub)"}
API = "https://commons.wikimedia.org/w/api.php"

# (slug, query, how many)
QUERIES = [
    ("goa_cafe", "Fontainhas Panaji", 3), ("goa_cafe", "cafe coffee cup table", 3),
    ("goa_cafe", "coffee shop interior", 2), ("wedding_food", "Indian thali", 2),
    ("manali_cafe", "cafe Himachal", 2), ("campus", "College of Engineering Pune", 2),
    ("campus", "IIT Bombay campus", 2), ("fest", "concert crowd stage lights", 2),
    ("library", "library interior books", 2), ("dosa", "dosa", 2), ("chai", "chai glass", 2),
    ("biryani", "biryani", 2), ("dog", "Labrador Retriever", 5), ("monsoon", "rain street Mumbai", 2),
    ("friends", "friends group selfie", 2), ("birthday", "birthday cake candles", 2),
    ("wedding_decor", "wedding marigold decoration", 2),
]
BAD = re.compile(r"map|logo|diagram|chart|svg|flag|coat of arms|stamp|poster|drawing|painting|plan|seal", re.I)


def search(q, n):
    for attempt in range(4):
        try:
            return _search(q, n)
        except Exception as e:  # noqa: BLE001
            print("retry", q, e)
            time.sleep(10 * (attempt + 1))
    return []


def _search(q, n):
    params = {
        "action": "query", "format": "json", "generator": "search", "gsrsearch": f"filetype:bitmap {q}",
        "gsrnamespace": 6, "gsrlimit": 30, "prop": "imageinfo",
        "iiprop": "url|mime|size|extmetadata", "iiurlwidth": 480,
    }
    r = requests.get(API, params=params, headers=UA, timeout=30)
    r.raise_for_status()
    pages = sorted(r.json().get("query", {}).get("pages", {}).values(), key=lambda p: p.get("index", 0))
    out = []
    for p in pages:
        ii = (p.get("imageinfo") or [{}])[0]
        if ii.get("mime") != "image/jpeg" or BAD.search(p["title"]) or ii.get("width", 0) < 800:
            continue
        md = ii.get("extmetadata", {})
        out.append({
            "title": p["title"], "thumb": ii.get("thumburl"), "page": ii.get("descriptionurl"),
            "license": md.get("LicenseShortName", {}).get("value", ""),
            "artist": re.sub(r"<[^>]+>", "", md.get("Artist", {}).get("value", ""))[:80],
            "description": re.sub(r"<[^>]+>", "", md.get("ImageDescription", {}).get("value", ""))[:300],
        })
        if len(out) >= n + 2:  # a couple of spares for manual curation
            break
    return out


def main():
    results = []
    for slug, q, n in QUERIES:
        try:
            items = search(q, n)
        except Exception as e:  # noqa: BLE001
            print("error", slug, e)
            continue
        for i, it in enumerate(items):
            fn = f"{slug}_{abs(hash(it['title'])) % 10**6}.jpg"
            try:
                img = requests.get(it["thumb"], headers=UA, timeout=30)
                if img.status_code != 200:
                    continue
                (IMG / fn).write_bytes(img.content)
            except Exception as e:  # noqa: BLE001
                print("download error", fn, e)
                continue
            results.append({"file": f"scenes/{fn}", "slug": slug, "query": q, **it})
        print(f"{slug}: {len(items)}")
        time.sleep(3)
    prev = json.loads((OUT / "scenes_raw.json").read_text()) if (OUT / "scenes_raw.json").exists() else []
    seen = {p["title"] for p in prev}
    results = prev + [r for r in results if r["title"] not in seen]
    (OUT / "scenes_raw.json").write_text(json.dumps(results, indent=1, ensure_ascii=False))
    print("total", len(results))


if __name__ == "__main__":
    main()

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
    ("goa_beach", "Baga beach Goa", 4), ("goa_sunset", "sunset beach Goa", 3),
    ("goa_cafe", "cafe interior Goa", 4), ("goa_fort", "Fort Aguada Goa", 2),
    ("goa_food", "Goan fish curry thali", 2), ("goa_scooter", "scooter Goa road", 2),
    ("goa_church", "Basilica of Bom Jesus", 2),
    ("diwali_diya", "Diwali diya lamps", 3), ("diwali_rangoli", "rangoli Diwali", 2),
    ("sweets", "Indian sweets mithai", 2),
    ("mehendi", "mehndi hands bride", 2), ("wedding_decor", "Indian wedding decoration mandap", 3),
    ("wedding_food", "Indian wedding buffet food", 2),
    ("manali_temple", "Hidimba Devi Temple", 2), ("manali_snow", "Solang valley snow", 3),
    ("manali_river", "Beas river Manali", 2), ("manali_cafe", "Old Manali cafe", 2),
    ("bengaluru_metro", "Namma Metro Bangalore train", 2), ("office", "office desk laptop monitor", 3),
    ("campus", "university campus India building", 3), ("fest", "college fest concert stage crowd India", 2),
    ("library", "library reading room students", 2), ("dosa", "masala dosa", 2), ("chai", "masala chai cup", 2),
    ("biryani", "Hyderabadi biryani", 2), ("dog", "Labrador retriever dog", 5),
    ("monsoon", "monsoon rain street Pune", 2), ("sunset_city", "Pune skyline sunset", 2),
]
BAD = re.compile(r"map|logo|diagram|chart|svg|flag|coat of arms|stamp|poster|drawing|painting|plan|seal", re.I)


def search(q, n):
    params = {
        "action": "query", "format": "json", "generator": "search", "gsrsearch": f"filetype:bitmap {q}",
        "gsrnamespace": 6, "gsrlimit": 30, "prop": "imageinfo",
        "iiprop": "url|mime|size|extmetadata", "iiurlwidth": 480,
    }
    r = requests.get(API, params=params, headers=UA, timeout=30)
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
            fn = f"{slug}_{i}.jpg"
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
        time.sleep(1)
    (OUT / "scenes_raw.json").write_text(json.dumps(results, indent=1, ensure_ascii=False))
    print("total", len(results))


if __name__ == "__main__":
    main()

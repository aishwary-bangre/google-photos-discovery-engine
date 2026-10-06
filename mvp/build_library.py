"""Assemble the demo library for 'Riya', a final-year student, from
  * utility.json  (synthetic receipts, prescriptions, screenshots …)
  * scenes_raw.json (CC-licensed Wikimedia Commons photos)
Each scene slug is placed in a life 'moment' with dates, place and people, the way a
real phone library has them (EXIF time + location, face groups, source app).
"""
from __future__ import annotations

import json
import random
from datetime import datetime, timedelta
from pathlib import Path

LIB = Path(__file__).resolve().parent / "library"
random.seed(5)

MOMENTS = {
    "cousin_wedding": dict(name="Cousin Priya's wedding", start="2025-01-24", end="2025-01-26", place="Jaipur",
                           people=["Priya", "Mom", "Dad"]),
    "manali_trip": dict(name="Manali trip", start="2025-05-14", end="2025-05-18", place="Manali, Himachal Pradesh",
                        people=["Ananya", "Kabir"]),
    "hostel_life": dict(name="College & hostel (Pune)", start="2024-08-01", end="2026-05-20", place="Pune",
                        people=["Ananya", "Rohan", "Kabir"]),
    "diwali_home": dict(name="Diwali at home", start="2025-10-19", end="2025-10-22", place="Jaipur", people=["Mom", "Dad"]),
    "goa_trip": dict(name="Goa trip with friends", start="2025-12-18", end="2025-12-22", place="Goa",
                     people=["Ananya", "Rohan", "Kabir"]),
    "sick_feb26": dict(name="Down with fever", start="2026-02-10", end="2026-02-14", place="Pune", people=[]),
    "internship_blr": dict(name="Summer internship (Bengaluru)", start="2026-05-30", end="2026-08-14",
                           place="Bengaluru", people=["Rohan"]),
    "home": dict(name="At home in Jaipur", start="2024-06-01", end="2026-09-30", place="Jaipur", people=["Mom", "Dad"]),
}

# slug -> (moment, content labels a vision model would give, source, people-in-photo?)
SCENES = {
    "goa_beach": ("goa_trip", "beach, sea, sand, people, sky, coast", "camera", True),
    "goa_sunset": ("goa_trip", "sunset, beach, sea, silhouette, evening sky", "camera", False),
    "goa_fort": ("goa_trip", "fort, lighthouse, coast, building, palm trees", "camera", False),
    "goa_food": ("goa_trip", "food, thali, rice, curry, fish, plate", "camera", False),
    "goa_scooter": ("goa_trip", "street, scooters, shops, road", "camera", False),
    "goa_church": ("goa_trip", "church, old building, architecture, tourists", "camera", False),
    "goa_cafe": ("goa_trip", "cafe, coffee, table, interior, colourful houses", "camera", False),
    "diwali_diya": ("diwali_home", "oil lamps, flames, candles, night, festival lights", "camera", False),
    "diwali_rangoli": ("diwali_home", "colourful floor art, pattern, flowers, festival", "camera", False),
    "sweets": ("diwali_home", "sweets, dessert, food, plate, shop", "camera", False),
    "mehendi": ("cousin_wedding", "hands, henna, pattern, bride, bangles", "whatsapp", True),
    "wedding_decor": ("cousin_wedding", "decoration, flowers, stage, wedding, marigold", "camera", False),
    "wedding_food": ("cousin_wedding", "food, thali, buffet, plate", "camera", False),
    "manali_temple": ("manali_trip", "temple, wooden building, pagoda, trees, forest", "camera", False),
    "manali_snow": ("manali_trip", "snow, mountains, huts, valley, people", "camera", True),
    "manali_river": ("manali_trip", "river, rocks, rapids, mountains, water", "camera", True),
    "manali_cafe": ("manali_trip", "cafe, mountains, coffee, table", "camera", False),
    "bengaluru_metro": ("internship_blr", "train, metro, station, platform, signboard", "camera", False),
    "office": ("internship_blr", "office, desk, computer, monitors, workspace", "camera", True),
    "campus": ("hostel_life", "building, campus, trees, college", "camera", False),
    "fest": ("hostel_life", "concert, stage, lights, crowd, night", "camera", True),
    "library": ("hostel_life", "library, books, shelves, reading room", "camera", False),
    "dosa": ("hostel_life", "food, dosa, plate, chutney", "camera", False),
    "chai": ("hostel_life", "tea, cup, glass, drink", "camera", False),
    "biryani": ("hostel_life", "food, rice, biryani, plate", "camera", False),
    "monsoon": ("hostel_life", "rain, street, umbrella, wet road", "camera", True),
    "friends": ("hostel_life", "people, group, selfie, smiling", "camera", True),
    "birthday": ("hostel_life", "cake, candles, birthday, celebration", "camera", True),
    "dog": ("home", "dog, pet, labrador, grass", "camera", False),
    "sunset_city": ("hostel_life", "sunset, city, skyline, sky", "camera", False),
}
# indices (by file name) removed after visual curation: off-topic or duplicate shots
EXCLUDE = set(json.loads((LIB / "exclude.json").read_text())) if (LIB / "exclude.json").exists() else set()

UTILITY_LABELS = {  # extra content labels a vision model would add for utility images
    "close-up of a silver tablet strip": "medicine, tablets, pills, strip, pharmacy",
    "photo of a doctor's prescription paper on a table": "document, paper, text, prescription, medical",
    "photo of a printed shop receipt": "receipt, bill, paper, text, document",
    "photo of a printed document with a table of values": "document, paper, text, report",
    "phone screenshot of a payment app": "screenshot, payment, app, text",
    "phone screenshot of a chat conversation": "screenshot, chat, messages, text",
    "photo of a whiteboard with handwriting": "whiteboard, text, handwriting, classroom",
    "photo of a blackboard with chalk writing": "blackboard, chalk, text, classroom",
    "photo of a printed ticket": "ticket, paper, text, document",
    "image of a boarding pass": "ticket, boarding pass, text, document",
    "phone screenshot of a ticket": "screenshot, ticket, text",
    "photo of an identity card": "card, identity card, document, text, photo",
}


def rand_date(m: dict) -> str:
    a, b = datetime.fromisoformat(m["start"]), datetime.fromisoformat(m["end"])
    d = a + timedelta(seconds=random.randint(0, int((b - a).total_seconds()) + 86399))
    return d.replace(hour=random.randint(8, 22)).isoformat(timespec="minutes")


def main():
    items = []
    for u in json.loads((LIB / "utility.json").read_text()):
        m = MOMENTS[u["moment"]]
        items.append({
            "id": Path(u["file"]).stem, "file": u["file"], "kind": "screenshot" if u["source"] == "screenshot" else
            ("document" if u["source"] in ("camera", "download", "whatsapp") else "photo"),
            "source": u["source"], "date": u["date"], "moment": u["moment"], "place": m["place"],
            "people": [], "labels": UTILITY_LABELS.get(u["caption"], "") , "caption": u["caption"],
            "ocr": u["ocr"], "purpose": u.get("purpose", ""), "credit": "Synthetic image (fictional data)",
        })
    for s in json.loads((LIB / "scenes_raw.json").read_text()):
        if s["slug"] not in SCENES or Path(s["file"]).stem in EXCLUDE:
            continue
        mom, labels, source, has_people = SCENES[s["slug"]]
        m = MOMENTS[mom]
        ppl = random.sample(m["people"], k=min(len(m["people"]), random.randint(1, 2))) if has_people and m["people"] else []
        items.append({
            "id": Path(s["file"]).stem, "file": s["file"], "kind": "photo", "source": source,
            "date": rand_date(m), "moment": mom, "place": m["place"], "people": ppl, "labels": labels,
            "caption": labels.split(",")[0], "ocr": "", "purpose": "",
            "credit": f"{s.get('artist') or 'Wikimedia Commons'} · {s.get('license', '')} · {s.get('page', '')}",
        })
    items.sort(key=lambda x: x["date"], reverse=True)
    (LIB / "library.json").write_text(json.dumps({"moments": MOMENTS, "items": items}, indent=1, ensure_ascii=False))
    from collections import Counter
    print(len(items), "items", Counter(i["moment"] for i in items))


if __name__ == "__main__":
    main()

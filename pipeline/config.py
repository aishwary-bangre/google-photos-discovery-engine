"""Central configuration for the Google Photos retrieval discovery engine."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("DATA_DIR", ROOT / "data"))
RAW_DIR = DATA_DIR / "raw"
PROC_DIR = DATA_DIR / "processed"
CACHE_DIR = DATA_DIR / "cache"
for d in (RAW_DIR, PROC_DIR, CACHE_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ---------- Sources ----------
PLAY_APP_ID = "com.google.android.apps.photos"
PLAY_COUNTRIES = ["us", "in"]
PLAY_REVIEWS_PER_COUNTRY = int(os.environ.get("PLAY_REVIEWS_PER_COUNTRY", 80000))

APPSTORE_APP_ID = "962194608"  # Google Photos on iOS
APPSTORE_COUNTRIES = ["us", "in", "gb", "ca", "au"]

REDDIT_SUBS = ["googlephotos", "GooglePixel", "Android", "pixel_phones", "DataHoarder", "googlehome", "iphone"]
REDDIT_QUERIES = [
    "can't find photo", "cannot find a photo", "find old photo", "looking for a photo",
    "search not working", "search photos", "find a picture", "find screenshot",
    "search by", "photo search", "can't remember when", "lost a photo", "where is my photo",
]
REDDIT_GLOBAL_QUERIES = [
    '"google photos" "can\'t find"', '"google photos" search find old picture',
    '"google photos" search useless', '"google photos" find screenshot',
]

# ---------- Relevance gate ----------
# Keyword pre-filter: any hit lets a record through to the semantic gate.
KEYWORD_PATTERN = (
    r"\b(find|finding|found|search|searching|searched|look(?:ing)? for|locate|"
    r"can'?t remember|cannot remember|forgot|forget|scroll|scrolling|where is|"
    r"dig(?:ging)? through|hunt|retriev|remember|memory|memories|old photo|old picture|"
    r"screenshot|receipt|document|recogni[sz]|face group|ask photos)\b"
)
# Anchor statements describing the problem space; records are ranked by cosine
# similarity to the closest anchor.
ANCHORS = [
    "I know the photo exists but I cannot find it in Google Photos",
    "Search does not find the picture I am looking for",
    "I have to scroll through thousands of photos to find one old picture",
    "I remember roughly what the photo was but not when it was taken",
    "I can't find an old screenshot or document photo when I need it",
    "Searching by a word or place gives the wrong results",
    "Too many similar photos, hard to spot the one I need",
]
MAX_LLM_RECORDS = int(os.environ.get("MAX_LLM_RECORDS", 2500))
MIN_ANCHOR_SIM = float(os.environ.get("MIN_ANCHOR_SIM", 0.30))

# ---------- LLM ----------
GEMINI_MODELS = [m for m in [
    os.environ.get("GEMINI_MODEL"),
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.0-flash",
] if m]
LLM_BATCH_SIZE = int(os.environ.get("LLM_BATCH_SIZE", 35))
LLM_MIN_INTERVAL_S = float(os.environ.get("LLM_MIN_INTERVAL_S", 4.0))  # free-tier friendly

# ---------- Embeddings ----------
EMBED_MODEL = os.environ.get("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

HN_QUERIES = ["google photos search", "google photos find photo", "find old photo phone",
              "photo search app", "search my photos", "can't find photo"]

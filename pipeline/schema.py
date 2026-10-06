"""Stage 3 schema — the 'memory mismatch' codebook.

Each piece of feedback is coded on the axes the brief cares about:
what the user remembered, what they had forgotten, how they phrased the search,
and at which stage of the retrieval journey it broke.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

PhotoType = Literal[
    "document_id", "medical", "receipt_payment", "notes_study", "screenshot_info",
    "travel_place", "people_event", "pet", "food", "video", "other", "unspecified",
]
Cue = Literal[
    "time_approx",      # roughly when (year, month, season)
    "life_chapter",     # "when I was in college", "before we moved"
    "place",            # location or venue
    "people",           # who was in it / who was there
    "event_context",    # what was happening (trip, wedding, before/after)
    "purpose",          # why they took it (to remember a bill, a doctor's note)
    "visual_content",   # objects, scene, colours in the image
    "text_in_image",    # words visible in the photo or screenshot
    "source_app",       # came from WhatsApp, a screenshot, a download, another phone
    "nothing_specific", # only knows it exists
]
Stage = Literal[
    "express",    # can't turn what they remember into a query the product accepts
    "interpret",  # query given, product misunderstands / returns wrong or nothing
    "evaluate",   # results shown but too many / too similar to spot the right one
    "refine",     # first attempt failed and there is no good way to narrow or pivot
    "browse",     # no search attempt; manual scrolling / navigation is the pain
    "access",     # photo hidden: archived, other account, not backed up, locked folder
    "unclear",
]
Workaround = Literal[
    "scroll_timeline", "people_faces", "albums", "map_location", "date_jump",
    "other_app", "ask_someone", "gave_up", "none_mentioned", "other",
]
Outcome = Literal["found", "found_with_effort", "gave_up", "unknown"]
QueryStyle = Literal["content_keyword", "context_description", "date_or_place", "person", "text_ocr", "none"]


class RecordTag(BaseModel):
    id: str
    relevant: bool = Field(description="True only if the author is trying (or tried) to find a specific existing photo/video/screenshot in their library")
    photo_type: PhotoType = "unspecified"
    cues_remembered: list[Cue] = Field(default_factory=list)
    cues_missing: list[Cue] = Field(default_factory=list)
    query_text: str | None = Field(default=None, description="Verbatim search phrase if the author quotes what they typed")
    query_style: QueryStyle = "none"
    failure_stage: Stage = "unclear"
    failure_mode: str = Field(default="", description="One neutral sentence: why retrieval failed")
    workaround: Workaround = "none_mentioned"
    outcome: Outcome = "unknown"
    urgency: int = Field(default=1, ge=1, le=3, description="1 casual/nostalgic, 2 needed it, 3 urgent/high-stakes")
    evidence_quote: str = Field(default="", description="Verbatim span (<=200 chars) copied from the text")

    @field_validator("cues_remembered", "cues_missing", mode="before")
    @classmethod
    def _dedupe(cls, v):
        return list(dict.fromkeys(v or []))


class BatchResult(BaseModel):
    results: list[RecordTag]


CODEBOOK = """
CODEBOOK (use exactly these labels)

relevant: true ONLY if the author describes trying to find / retrieve a specific photo, video or
screenshot that already exists in their library (including complaints that search can't find things).
false for: backup/sync failures, storage/pricing, editing, crashes, sharing, general praise, deleted photos
they want restored (unless they are searching for them).

photo_type: document_id | medical | receipt_payment | notes_study | screenshot_info | travel_place |
people_event | pet | food | video | other | unspecified

cues_remembered / cues_missing (lists; only what the text supports):
  time_approx (roughly when) | life_chapter ("in college", "before we moved") | place | people |
  event_context (what was happening) | purpose (why the photo was taken) | visual_content (objects/scene/colour) |
  text_in_image | source_app (WhatsApp, screenshot, download, old phone) | nothing_specific
  cues_missing = things the author explicitly says they don't know/remember (e.g. "don't remember the date").

query_text: the exact words the author says they typed into search, else null.
query_style: content_keyword | context_description | date_or_place | person | text_ocr | none

failure_stage (where the journey broke):
  express   = they can't phrase what they remember in a way search accepts
  interpret = they searched, but results were wrong/empty/search ignored their words
  evaluate  = results contained many similar items; hard to spot the right one
  refine    = after a failed attempt there was no way to narrow down or try another angle
  browse    = no search; the pain is scrolling/navigating the timeline manually
  access    = photo is hidden: archived, locked folder, other account, not backed up, on another device
  unclear

failure_mode: one neutral sentence in your own words.
workaround: scroll_timeline | people_faces | albums | map_location | date_jump | other_app | ask_someone | gave_up | none_mentioned | other
outcome: found | found_with_effort | gave_up | unknown
urgency: 1 casual/nostalgic, 2 needed it, 3 urgent or high-stakes (ID, medical, money, legal, work deadline)
evidence_quote: copy a short span (<=200 chars) VERBATIM from the text that best supports your coding.
If relevant=false, still return the id and relevant=false; other fields may stay default.
"""

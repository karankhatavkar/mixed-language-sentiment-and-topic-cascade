"""Clean + slice-bucket + stratified-sample YouTube comments.

Reads
  data/youtube/videos.csv            — the 3-video in-scope list
  data/raw/youtube/comments_raw.csv  — all comments the scraper has pulled

Writes
  data/processed/youtube_pool.csv    — the labelling pool.

Stage 2 of the YouTube loader. Deliberately strict: a smaller clean pool
beats a larger noisy one.

Steps:
  1. Scope  — keep only comments whose video_id is in videos.csv.
  2. Clean  — dedupe (normalised text), length bounds, strip whitespace.
  3. Spam   — URLs, discount codes, char-repeats, mention/hashtag-only.
  4. Slice  — per comment:
                arabic  : AR letters >= 3 AND Latin letters <  2
                english : Latin letters >= 3 AND AR letters == 0
                mixed   : AR letters >= 2 AND Latin letters >= 3
                (anything else: dropped)
  5. Sample — target TARGETS[slice]; cap MAX_PER_VIDEO per video for
              diversity; prefer longer comments with >=1 like.

Deterministic: SEED=42.

Run: python -m src.build_youtube_pool
"""
from __future__ import annotations

import csv
import random
import re
import unicodedata
from collections import defaultdict

from src.config import PROCESSED_DIR, ROOT, SEED

VIDEOS_CSV = ROOT / "data" / "youtube" / "videos.csv"
RAW_CSV = ROOT / "data" / "raw" / "youtube" / "comments_raw.csv"
POOL_CSV = PROCESSED_DIR / "youtube_pool.csv"

MIN_LEN = 10
MAX_LEN = 500
MAX_PER_VIDEO = 120   # 3 videos, want up to ~240-260 total

TARGETS = {"english": 100, "arabic": 100, "mixed": 60}

URL_RE = re.compile(r"https?://\S+|www\.\S+")
MENTION_RE = re.compile(r"@[\w.]+")
HASHTAG_RE = re.compile(r"#[\w؀-ۿ]+")
ZEROWIDTH_RE = re.compile(r"[​-‏‪-‮﻿]")
CHAR_REPEAT_RE = re.compile(r"(.)\1{5,}")
# Discount codes: 5-8 char token of ASCII uppercase+digit, bounded by anything
# non-Latin-alphanumeric. \b doesn't help here because Arabic script counts as
# a word character in Python's Unicode-aware regex (so "A96JMأقوى" has no \b).
CODE_TOKEN_RE = re.compile(r"(?<![A-Za-z0-9])[A-Z0-9]{5,8}(?![A-Za-z0-9])")


def _letter_counts(s: str) -> tuple[int, int]:
    ar = sum(1 for c in s if "؀" <= c <= "ۿ")
    lat = sum(1 for c in s if "a" <= c.lower() <= "z")
    return ar, lat


def _slice_for(text: str) -> str | None:
    ar, lat = _letter_counts(text)
    if ar >= 2 and lat >= 3:
        return "mixed"
    if ar >= 3 and lat < 2:
        return "arabic"
    if lat >= 3 and ar == 0:
        return "english"
    return None


def _normalize(text: str) -> str:
    t = unicodedata.normalize("NFKC", text)
    t = ZEROWIDTH_RE.sub("", t)
    return re.sub(r"\s+", " ", t).strip()


def _is_spam(text: str) -> bool:
    if URL_RE.search(text):
        return True
    if CHAR_REPEAT_RE.search(text):
        return True
    # Any uppercase+digit mixed token of 5-8 chars is almost always a coupon
    # code in this corpus. Require >=1 digit to avoid clipping acronyms like
    # "SHEIN", "LOVED", "FASHION".
    for tok in CODE_TOKEN_RE.findall(text):
        if any(c.isdigit() for c in tok) and any(c.isalpha() for c in tok):
            return True
    stripped = HASHTAG_RE.sub("", MENTION_RE.sub("", text)).strip()
    if len(stripped) < MIN_LEN:
        return True
    return False


def _in_scope_ids() -> set[str]:
    with VIDEOS_CSV.open(encoding="utf-8") as f:
        return {row["video_id"] for row in csv.DictReader(f)}


def _load_raw(scope: set[str]) -> list[dict]:
    with RAW_CSV.open(encoding="utf-8") as f:
        return [r for r in csv.DictReader(f) if r["video_id"] in scope]


def _clean(rows: list[dict]) -> list[dict]:
    seen: set[str] = set()
    out: list[dict] = []
    stats = defaultdict(int)
    for r in rows:
        t = _normalize(r["text"])
        if not t:
            stats["empty"] += 1
            continue
        if len(t) < MIN_LEN:
            stats["too_short"] += 1
            continue
        if len(t) > MAX_LEN:
            stats["too_long"] += 1
            continue
        key = t.lower()
        if key in seen:
            stats["dupe"] += 1
            continue
        seen.add(key)
        out.append({**r, "text_norm": t})
    print(f"clean:  kept {len(out):>5}  dropped {dict(stats)}")
    return out


def _despam(rows: list[dict]) -> list[dict]:
    kept = [r for r in rows if not _is_spam(r["text_norm"])]
    print(f"spam:   kept {len(kept):>5}  removed {len(rows) - len(kept)}")
    return kept


def _bucket(rows: list[dict]) -> dict[str, list[dict]]:
    buckets: dict[str, list[dict]] = {"english": [], "arabic": [], "mixed": []}
    other = 0
    for r in rows:
        s = _slice_for(r["text_norm"])
        if s is None:
            other += 1
            continue
        buckets[s].append({**r, "slice": s})
    print(f"bucket: EN={len(buckets['english'])}  AR={len(buckets['arabic'])}  "
          f"MIX={len(buckets['mixed'])}  dropped={other}")
    return buckets


def _sample(buckets: dict[str, list[dict]], rng: random.Random) -> list[dict]:
    picked: list[dict] = []
    for slice_name, target in TARGETS.items():
        pool = list(buckets[slice_name])
        for r in pool:
            r["_likes"] = int(r.get("like_count") or 0)
            r["_len"] = len(r["text_norm"])
        rng.shuffle(pool)
        pool.sort(key=lambda r: (r["_likes"] > 0, r["_len"]), reverse=True)

        per_video: dict[str, int] = defaultdict(int)
        chosen: list[dict] = []
        for r in pool:
            if len(chosen) >= target:
                break
            if per_video[r["video_id"]] >= MAX_PER_VIDEO:
                continue
            chosen.append(r)
            per_video[r["video_id"]] += 1

        print(f"sample [{slice_name:>7}] target={target:>3}  got={len(chosen):>3}  "
              f"videos={dict(per_video)}")
        picked.extend(chosen)
    return picked


OUT_COLS = [
    "id", "video_id", "comment_id", "slice",
    "text", "like_count", "reply_count", "published_at",
]


def _assemble(rows: list[dict]) -> list[dict]:
    rows.sort(key=lambda r: (r["slice"], r["video_id"], r["comment_id"]))
    out = []
    counters: dict[str, int] = defaultdict(int)
    for r in rows:
        counters[r["slice"]] += 1
        n = counters[r["slice"]]
        out.append({
            "id": f"youtube_{r['slice']}_{n:03d}",
            "video_id": r["video_id"],
            "comment_id": r["comment_id"],
            "slice": r["slice"],
            "text": r["text_norm"],
            "like_count": r.get("like_count", 0),
            "reply_count": r.get("reply_count", 0),
            "published_at": r.get("published_at", ""),
        })
    return out


def build() -> None:
    rng = random.Random(SEED)
    scope = _in_scope_ids()
    raw = _load_raw(scope)
    print(f"scope:  {len(scope)} videos  ({sorted(scope)})")
    print(f"raw:    {len(raw)} in-scope rows from {RAW_CSV.name}")

    cleaned = _clean(raw)
    cleaned = _despam(cleaned)
    buckets = _bucket(cleaned)
    picked = _sample(buckets, rng)
    rows = _assemble(picked)

    POOL_CSV.parent.mkdir(parents=True, exist_ok=True)
    with POOL_CSV.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=OUT_COLS)
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {POOL_CSV}  ({len(rows)} rows)")


if __name__ == "__main__":
    build()

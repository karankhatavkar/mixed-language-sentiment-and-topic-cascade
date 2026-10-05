
"""Scrape top-level YouTube comments for the SHEIN video shortlist.

Reads data/youtube/videos.csv, pulls every top-level comment for each
video_id, writes data/raw/youtube/comments_raw.csv.

Rerunnable: appends new comments while skipping comment_ids already in the
output. Safe to interrupt and restart.

Needs YOUTUBE_API_KEY in .env (same file as GEMINI_API_KEY).

Quota: commentThreads.list = 1 unit per page (<=100 comments). The 15-video
shortlist is ~10,500 comments -> ~110 units, well under the 10k/day free tier.

Run: python -m src.scrape_youtube
"""
from __future__ import annotations

import csv
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request

from dotenv import load_dotenv

from src.config import ROOT

VIDEOS_CSV = ROOT / "data" / "youtube" / "videos.csv"
RAW_DIR = ROOT / "data" / "raw" / "youtube"
RAW_CSV = RAW_DIR / "comments_raw.csv"

API_ROOT = "https://www.googleapis.com/youtube/v3"
PAGE_SIZE = 100  # commentThreads.list max

COLUMNS = [
    "video_id",
    "comment_id",
    "text",
    "published_at",
    "like_count",
    "reply_count",
    "author_channel_id",
]


def _api_key() -> str:
    load_dotenv()
    key = os.environ.get("YOUTUBE_API_KEY")
    if not key:
        raise RuntimeError(
            "YOUTUBE_API_KEY missing. Add it to .env next to GEMINI_API_KEY."
        )
    return key


def _get(path: str, key: str, **params) -> dict:
    params["key"] = key
    url = f"{API_ROOT}/{path}?{urllib.parse.urlencode(params)}"
    with urllib.request.urlopen(url, timeout=30) as r:
        return json.load(r)


def _fetch_video(video_id: str, key: str) -> list[dict]:
    """All top-level comments for one video. Pages until exhausted."""
    rows: list[dict] = []
    page_token: str | None = None
    while True:
        params = dict(
            part="snippet",
            videoId=video_id,
            maxResults=PAGE_SIZE,
            textFormat="plainText",
            order="time",
        )
        if page_token:
            params["pageToken"] = page_token
        try:
            data = _get("commentThreads", key, **params)
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", "ignore")[:200]
            print(f"  HTTP {e.code}: {body}")
            return rows

        for item in data.get("items", []):
            s = item["snippet"]["topLevelComment"]["snippet"]
            rows.append({
                "video_id": video_id,
                "comment_id": item["snippet"]["topLevelComment"]["id"],
                "text": s.get("textOriginal", ""),
                "published_at": s.get("publishedAt", ""),
                "like_count": s.get("likeCount", 0),
                "reply_count": item["snippet"].get("totalReplyCount", 0),
                "author_channel_id": s.get("authorChannelId", {}).get("value", ""),
            })
        page_token = data.get("nextPageToken")
        if not page_token:
            return rows


def _load_video_ids() -> list[str]:
    with VIDEOS_CSV.open(encoding="utf-8") as f:
        return [row["video_id"] for row in csv.DictReader(f)]


def _load_existing_ids() -> set[str]:
    if not RAW_CSV.exists():
        return set()
    with RAW_CSV.open(encoding="utf-8") as f:
        return {row["comment_id"] for row in csv.DictReader(f)}


def scrape() -> None:
    key = _api_key()
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    videos = _load_video_ids()
    seen = _load_existing_ids()
    print(f"videos:   {len(videos)}  (from {VIDEOS_CSV.name})")
    print(f"existing: {len(seen)} comments already in {RAW_CSV.name}")

    write_header = not RAW_CSV.exists()
    added_total = 0
    with RAW_CSV.open("a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        if write_header:
            w.writeheader()

        for i, vid in enumerate(videos, 1):
            t0 = time.time()
            rows = _fetch_video(vid, key)
            new = [r for r in rows if r["comment_id"] not in seen]
            for r in new:
                w.writerow(r)
                seen.add(r["comment_id"])
            added_total += len(new)
            print(f"[{i:>2}/{len(videos)}] {vid}  "
                  f"fetched={len(rows):>5}  new={len(new):>5}  "
                  f"({time.time() - t0:.1f}s)")

    print(f"\ntotal new comments appended: {added_total}")
    print(f"total rows in {RAW_CSV.name}: {len(seen)}")


if __name__ == "__main__":
    scrape()

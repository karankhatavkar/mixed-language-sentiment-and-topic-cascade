# YouTube source

One brand (**SHEIN**) across 15 videos. The brand's own channel has comments
disabled or near-zero traffic, so videos are hauls/reviews posted *about* SHEIN
by third-party creators. "One brand" is kept at the content level — all 15
videos are SHEIN-focused, so the fixed 8 fashion topics apply uniformly.

## Files

- `videos.csv` — 15-video shortlist (**the source of truth** for the pull).
- `candidates.csv` — the full 46-video scored pool the shortlist was picked
  from. Kept so swaps are reproducible if a video goes private / disables
  comments / gets taken down.

Neither file holds YouTube comments. The scraper (next step) reads
`videos.csv` and writes `data/raw/youtube/comments_raw.csv` (gitignored).

## Columns in `videos.csv`

| column          | meaning                                                      |
|-----------------|--------------------------------------------------------------|
| `slice_target`  | expected dominant slice of this video's comments (sampled)   |
| `video_id`      | 11-char YouTube ID                                           |
| `channel`       | channel display name at scrape time                          |
| `title`         | video title at scrape time                                   |
| `published`     | publish date (YYYY-MM-DD)                                    |
| `comment_count` | total top-level comments reported by the API at scrape time  |
| `ar_pct`        | fraction of sampled comments that are ≥50% Arabic-script     |
| `en_pct`        | fraction of sampled comments with 0 Arabic-script characters |
| `url`           | watch URL                                                    |
| `notes`         | why this video is on the list                                |

Rounded percentages are from a 50-comment relevance-ordered sample (first page
of `commentThreads.list`), not the full comment set — treat them as a hint,
not ground truth.

## How the shortlist was picked

1. **Six YouTube search queries**, chosen to surface three audience profiles:
   AR-dominant hauls (SA, EG), EN-dominant hauls/reviews (US), and
   Gulf-bilingual hauls (AE/Dubai topic). 56 unique video IDs.
2. **Filter**: keep videos with ≥20 total comments and a reachable comment
   endpoint. 46 survived.
3. **Score per video**: sample up to 50 top-level comments, compute Arabic /
   English / mixed shares by Arabic-script character ratio
   (AR ≥ 0.5 → arabic, 0.05 < AR < 0.5 → mixed, AR == 0 → english).
4. **Shortlist**: 6 Arabic + 8 English + 1 wildcard (an EN video with
   non-zero Arabic share), biased toward:
   - **Higher raw comment volume** (gives us a labeling pool to pick from).
   - **Content diversity**: hauls, honest reviews, critical "is it a scam"
     videos — so the 8 fashion topics (quality, price, delivery, sizing,
     returns, customer service, product_praise, other) all have a chance to
     appear and negative sentiment isn't under-represented.

## Known limitation: the mixed slice

**No video surfaced as naturally code-switched.** Audiences follow the
creator's language — AR creators get AR comments, EN creators get EN
comments. Expect the mixed slice to come from the **tail** of both groups:
Arabic comments with embedded English brand/product terms, EN comments with
embedded Arabic emoji/transliteration, and short bilingual replies on the
two bilingual-titled Arabic videos (`4fEycJyCgZU`, `fLl_mp7nWqk`).

Slice assignment for YouTube rows will therefore happen **per comment** at
preprocessing time, using Arabic-script ratio bands, not per video.

## Reproducibility notes

- Video `comment_count` and the AR/EN percentages are snapshots. Re-running
  the scoring script a month later will shift numbers slightly as new
  comments land.
- If a video drops out of the list (deleted, comments disabled), pick the
  next-ranked candidate of the same `slice_target` from `candidates.csv` and
  update `videos.csv` in the same commit.

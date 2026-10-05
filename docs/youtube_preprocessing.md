# YouTube preprocessing

Builds the labelling pool for the YouTube slice of the test set.

Two scripts, two outputs:

| step | script | input | output |
|---|---|---|---|
| 1. scrape | `src/scrape_youtube.py` | `data/youtube/videos.csv` | `data/raw/youtube/comments_raw.csv` |
| 2. clean + sample | `src/build_youtube_pool.py` | raw CSV + `videos.csv` (scope) | `data/processed/youtube_pool.csv` |

The pool CSV is **not** the final labelled set — it is the candidate list that goes to the Gemini Pro judge for topic pre-labels and then to human review for `gold_sentiment` and `gold_topics`.

## Scope decisions

**Brand:** SHEIN. The only brand with enough third-party review volume in both English and Arabic (brand-owned Gulf channels have comments disabled or sub-10 traffic per video).

**Product category:** clothing hauls. Narrowed from an initial 15-video pool to tighten topic coherence — all comments discuss clothing from the same brand, so the fixed 8 fashion topics apply uniformly.

**Three videos:**

| slice_target | video_id | channel | title | raw comments |
|---|---|---|---|---:|
| arabic | `8560RK9-iWw` | Beauty_basket95 | تنسيقات من شي ان | 4,684 |
| arabic | `z9r4kSdW_Kk` | Sara's Makeup | مشترياتي العشوائية من موقع شي ان | 574 |
| english | `CJzUfj-5jFQ` | Emma Jordyn | SUMMER SHEIN HAUL | 810 |

Full provenance and the 46-video candidate pool these were drawn from live in `data/youtube/`.

## Stage 1 — scrape

**Endpoint:** `commentThreads.list?part=snippet&textFormat=plainText&order=time&maxResults=100` (YouTube Data API v3).

**What is captured per comment:**

- `video_id`
- `comment_id`
- `text` (`textOriginal` — raw, not `textDisplay` which contains YouTube's `<a>` HTML)
- `published_at`, `like_count`, `reply_count`
- `author_channel_id` (for de-botting at label time)

**What is skipped:** replies to comments (`comments.list`). Top-level threads are enough for the current pool size; adding replies is a one-flag change if the mixed slice needs boosting later.

**Rerunnable.** The scraper reads existing `comments_raw.csv`, hashes `comment_id`s already present, and only appends new rows. Safe to interrupt and restart.

**Quota:** 1 unit per page of 100 comments. Full pull of the 15-video candidate list took ~85 units — the 3-video scope is a strict subset of what is already on disk, so **the scraper does not need to be rerun** unless you add videos.

**Auth:** `YOUTUBE_API_KEY` in `.env` alongside `GEMINI_API_KEY`. The API key is restricted to the `YouTube Data API v3` scope only.

**Known quirks observed on this pull:**

- `commentCount` from `videos.list` exceeds the number of returned threads because `commentCount` includes replies. Expected.
- A tiny number of `comment_id`s (3 of 8,438, 0.04%) collided across different videos — likely stitched / reposted comments. Dedupe handled it at write time.

## Stage 2 — clean + sample

Reads `videos.csv` (for scope) and `comments_raw.csv` (for data). Keeps only rows whose `video_id` is listed in `videos.csv`. All attrition is logged.

### 2.1 Cleaning

Applied in order:

1. **Normalise** — `unicodedata.NFKC`, strip zero-width (`​-‏`, `‪-‮`, `﻿`), collapse all whitespace runs to a single space.
2. **Length bounds** — drop if `len < 10` or `len > 500` chars. Short end filters emoji-only / "nice 👍" fluff. Long end filters pasted-prayer / copypasta essays.
3. **Dedupe** — on lowercased normalised text. Catches the same comment copy-pasted across threads.

### 2.2 Spam filters

A comment is dropped as spam if any of these fire:

- Contains a URL (`https?://` or `www.`).
- Contains a 6+ run of the same character (`😍😍😍😍😍😍`, `aaaaaa`, `.......`).
- Contains a 5–8 char ASCII `[A-Za-z0-9]` token with at least one digit and at least one letter, bounded by anything non-Latin-alphanumeric. This catches SHEIN discount codes like `A96JM` even when they are glued to Arabic script (`A96JMأقوى كود شي ان`), which `\b` does not catch because Python's Unicode-aware `\b` treats Arabic script as word characters.
- Hashtag/mention stripping leaves fewer than 10 chars of real content.

### 2.3 Slice bucketing (per comment, not per video)

The initial per-video Arabic-script-ratio shows that **no video is naturally code-switched** — audiences follow the creator's language. Slice is therefore assigned at the comment level, not inherited from the video, using *script-presence* rather than a ratio band (Arabic is denser than Latin, so a one-English-word injection into an Arabic sentence looks like 85% AR by ratio but is clearly mixed).

| slice | rule |
|---|---|
| **arabic** | Arabic letters ≥ 3 AND Latin letters < 2 |
| **english** | Latin letters ≥ 3 AND Arabic letters == 0 |
| **mixed** | Arabic letters ≥ 2 AND Latin letters ≥ 3 |
| *dropped* | everything else (too short, pure emoji, Latin-only but < 3 letters, etc.) |

### 2.4 Stratified sampling

Targets: `english=100`, `arabic=100`, `mixed=60`.

Within each slice:

- Score each candidate by `(has_at_least_one_like, text_length)`, descending.
- Shuffle (seeded) within score tiers so ties are broken deterministically but not by `comment_id` lexical order.
- Cap at `MAX_PER_VIDEO = 120` per video — soft diversity ceiling. With only 3 videos the cap only bites on English, where one video has ~800 clean candidates.

**Determinism:** `random.Random(SEED=42)`. Same raw CSV + same `videos.csv` → bit-identical pool.

## Output schema

`data/processed/youtube_pool.csv`:

| column | meaning |
|---|---|
| `id` | `youtube_{slice}_{NNN}` — sequential per slice |
| `video_id` | source video |
| `comment_id` | YouTube ID, for de-dupe and traceback |
| `slice` | `english` / `arabic` / `mixed` |
| `text` | normalised text (NFKC, whitespace-collapsed) |
| `like_count` | int, from the API at scrape time |
| `reply_count` | int, from the API at scrape time |
| `published_at` | ISO 8601, from the API |

**Not present yet:** `gold_sentiment`, `gold_topics`, `split`, `source`. Those are added by the labelling / judge pipeline, which produces the final `data/processed/youtube_dev.csv` and `youtube_test.csv` matching the project's common schema.

## Result

- Scope: 3 videos, **6,068 in-scope raw comments**
- After clean + spam: **5,349** (12% attrition)
- After bucketing: `EN=795`, `AR=4,516`, `MIX=34`
- After sampling: **234 total** — `english=100`, `arabic=100`, `mixed=34`

The mixed slice is the natural ceiling of these 3 videos; raising it requires adding more Arabic-creator SHEIN clothing videos from `data/youtube/candidates.csv` (the English video contributed zero mixed, confirming the "audiences follow creator language" finding).

## Known imperfections (handle at label time)

- **~5 French-Arabic examples** are in the mixed bucket ("vulgaire", "simple même maquillage", "La simplicité fait la beauté", "nuisette", "plus simple plus beau"). The project's `mixed` slice is defined as EN+AR, so these are technically off-slice. Distinguishing EN from FR by character set alone is unreliable, so the filter does not try — drop them during human review.
- **One lowercase discount code** slips through (`sagx6f6 كود شي ان خصم`). The code-token regex is uppercase-only; two-line fix if needed.
- **Zero mixed comments from the English video** (Emma Jordyn, Western audience). All mixed-slice supply is from the two Arabic videos.

## Explicitly NOT done (per project spec)

- No emoji stripping — passed through as text, same as the public-set loader
- No Arabic script normalisation (no diacritic removal, no alef unification, no tatweel strip)
- No Arabizi (Latin-script Arabic) detection — Khaleeji-style transliterations like `Ana bou7di li la7t` land in the `english` bucket by rule, which is a known limitation
- No lowercasing of text
- No translation; the slice is bilingual by design

## Config

Hard-coded in `src/build_youtube_pool.py` (small file, single purpose — not promoted to `src/config.py`):

| key | value |
|---|---|
| `MIN_LEN` | 10 |
| `MAX_LEN` | 500 |
| `MAX_PER_VIDEO` | 120 |
| `TARGETS` | `{"english": 100, "arabic": 100, "mixed": 60}` |
| seed | `SEED` from `src/config.py` (42) |

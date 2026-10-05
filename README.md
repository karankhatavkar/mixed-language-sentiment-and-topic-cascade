# Mixed-language sentiment and topic cascade

Labels social-media posts in English, Arabic, or both for sentiment and 8 fashion topics. Two free HuggingFace models run on every post. Only the hard ones go to Gemini. The three setups below compare that cascade against the simpler extremes.

## Block diagram

```
                 ┌──────────────────────┐
                 │ raw data             │
                 │ (two tracks)         │
                 └──────────┬───────────┘
                            │
                 ┌──────────▼───────────┐
                 │ preprocessing        │
                 │ (per-track rules)    │
                 └──────────┬───────────┘
                            │
                 ┌──────────▼───────────┐
                 │ data/processed/      │
                 │  dev · test · youtube│
                 └──────────┬───────────┘
            ┌───────────────┼───────────────┐
            │               │               │
       ┌────▼────┐     ┌────▼────┐     ┌────▼────┐
       │ Setup A │     │ Setup B │     │ Setup C │
       │ small   │     │ Gemini  │     │ cascade │
       │ models  │     │ on all  │     │ A, then │
       │ only    │     │ posts   │     │ route   │
       └────┬────┘     └────┬────┘     └────┬────┘
            │               │               │
            └───────────────┼───────────────┘
                            │
                 ┌──────────▼───────────┐
                 │ evaluate.py          │
                 │ results/*.md         │
                 └──────────────────────┘
```

## Data tracks

### Track 1: public datasets (3-way sentiment)

| slice | source | dev | test |
|---|---|---:|---:|
| English | Cardiff `tweet_sentiment_multilingual` (en) | 50 | 100 |
| Arabic  | Cardiff `tweet_sentiment_multilingual` (ar) | 50 | 100 |
| Mixed EN+AR | EESA corpus | 50 | 100 |

Gold labels: `positive`, `negative`, `neutral`.

Preprocessing (`src/load_data.py`):

```
raw CSV → strip whitespace → drop empty text → validate labels
       → stratified sample by sentiment (seed=42)
       → id-tag per slice → write dev.csv / test.csv
```

### Track 2: YouTube (4-way sentiment + topics)

Three videos about one product line (SHEIN clothing hauls), picked after scoring 46 candidates for comment volume and language mix:

- 2 Arabic haul channels (`Beauty_basket95`, `Sara's Makeup`)
- 1 English haul channel (`Emma Jordyn`)

Gold labels: `positive`, `negative`, `neutral`, `mixed` + a list of topics per row.

Preprocessing (`src/scrape_youtube.py`, `src/build_youtube_pool.py`, `src/label_youtube.py`, `src/load_data.py`):

```
YouTube Data API v3 → comments_raw.csv   (8,435 raw comments)
         │
         ▼ scope filter to 3 videos
   6,068 in-scope rows
         │
         ▼ clean: NFKC, zero-width strip, whitespace, length bounds, dedupe
         │
         ▼ spam filter: URLs, discount codes, mention-only, char-repeats
         │
         ▼ per-comment bucketing by Arabic-script presence:
             arabic  → >=3 AR letters,  <2 Latin
             english → >=3 Latin letters, 0 AR
             mixed   → >=2 AR letters,  >=3 Latin
         │
         ▼ stratified sample (seed=42, per-video cap)
   100 EN + 100 AR + 34 mixed = 234 rows
         │
         ▼ Flash-Lite pre-label (sentiment + sarcasm + topics + reason)
         │
         ▼ human review of all 234 rows
         │
         ▼ youtube.csv (gold_sentiment + gold_topics)
```

## Models

| role | model | notes |
|---|---|---|
| Small sentiment | `cardiffnlp/twitter-xlm-roberta-base-sentiment` | 3-way, multilingual, CPU |
| Small topics | `MoritzLaurer/mDeBERTa-v3-base-mnli-xnli` | zero-shot, multi-label (sigmoid), CPU |
| Pipeline LLM | `gemini-3.5-flash-lite` | temperature=0, JSON schema, cached |
| Topic judge | `gemini-2.5-pro` | temperature=0, JSON schema, cached |

Every Gemini call is cached in `cache/llm_cache.json` keyed by `model + prompt_version + post_id`, so reruns are free and bit-identical.

## Setups

### Setup A: small only
Run XLM-R for sentiment. Run mDeBERTa zero-shot over the 8 topics. Keep topics scoring at or above 0.5. No LLM.

### Setup B: LLM only
Flash-Lite on every post. Returns 4-way sentiment, sarcasm, a topic list, and a one-line reason. Small models skipped.

### Setup C: cascade
Setup A first. Then route a row to Flash-Lite if any of these fire:

1. XLM-R top-class confidence < 0.7
2. Positive and negative probabilities both ≥ 0.3 (likely mixed)
3. No topic scored ≥ 0.5

Routed rows get the Flash-Lite answer. Unrouted rows keep A's answer. Route reasons are logged per row.

## Results

### Public test: 300 rows, 3-way gold

| Setup | EN F1 | AR F1 | Mixed F1 | Overall F1 | Neg Recall | Topic F1 | LLM calls | $ / 1k |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A | 0.724 | 0.677 | 0.838 | 0.749 | 0.848 | 0.230 | 0% | $0.000 |
| B | 0.761 | 0.714 | 0.939 | 0.805 | 0.826 | 0.435 | 100% | $0.250 |
| C | 0.726 | 0.694 | 0.939 | 0.787 | 0.859 | 0.276 | 70% | $0.174 |

### YouTube: 234 rows, 4-way gold + topics

| Setup | EN F1 | AR F1 | Mixed F1 | Overall F1 | Neg Recall | Topic F1 | LLM calls | $ / 1k |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A | 0.463 | 0.334 | 0.556 | 0.441 | 0.615 | 0.347 | 0% | $0.000 |
| B | 0.945 | 0.899 | 0.715 | 0.909 | 0.975 | 0.904 | 100% | $0.250 |
| C | 0.813 | 0.800 | 0.715 | 0.807 | 0.951 | 0.681 | 76% | $0.190 |

### How to read the numbers

- **Overall F1**: macro-F1 over the gold labels (3 on public, 4 on YouTube).
- **Neg Recall**: fraction of real complaints caught. Direct brand-monitoring signal.
- **Topic F1**: macro-F1 over the 8 topics; multi-label.
- **LLM calls**: share of rows that triggered a Gemini call (always 0% for A, 100% for B).
- **$ / 1k**: estimated Flash-Lite spend per 1,000 posts at the observed call rate.

Full per-slice tables, confusion matrices, and routing-reason breakdowns live in `results/metrics.md` and `results/metrics_youtube.md`.

## Run

```bash
python -m src.load_data                              # build dev.csv, test.csv, youtube.csv
python -m src.pipeline --split test                  # setups A, B, C on public test
python -m src.pipeline --split youtube               # setups A, B, C on YouTube
python -m src.judge --split test                     # Pro topic labels for public test
python -m src.evaluate --split test                  # writes results/metrics.md
python -m src.evaluate --split youtube               # writes results/metrics_youtube.md
```

Requires `GEMINI_API_KEY` and `YOUTUBE_API_KEY` in `.env`.

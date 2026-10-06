# Part B: Sentiment and Topic Pipeline Config

Oct 5, 2026 · @Karan

## Scope and decisions

Part B labels English, Arabic and mixed posts for sentiment and topic, using two free models and Gemini only for hard posts, and measures accuracy and cost three ways.

- **Languages:** English and Arabic (Arabic script), including posts that mix both. Arabizi is out of scope.
- **Sentiment labels:** positive, negative, neutral, mixed. Sarcasm is a separate yes/no flag.
- **No training:** every model is used as downloaded.
- **Emojis:** not handled separately for now. They are treated as normal text and passed to the models as-is.
- **Build order:** base pipeline on public labelled data first. YouTube comments on one brand are added after, through the same loader.
- **Left out:** emoji rules, fine-tuning, BERTopic discovery, confidence intervals, a separate calibration study.

## Datasets

450 labelled posts: 150 per language, each split 50 dev and 100 test, sampled with seed 42 and stratified by label.

| Slice | Source | Dev (50) from | Test (100) from | Label format |
| --- | --- | --- | --- | --- |
| English | [cardiffnlp/tweet\_sentiment\_multilingual](https://huggingface.co/datasets/cardiffnlp/tweet_sentiment_multilingual), config `english` | `validation` split (323) | `test` split (869) | 0 = negative, 1 = neutral, 2 = positive |
| Arabic | Same dataset, config `arabic` | `validation` split (323) | `test` split (869) | 0 = negative, 1 = neutral, 2 = positive |
| Mixed | [EESA corpus](https://github.com/ASTalaat/EESA-Corpus) | `EESA-Dev.csv` (818) | `EESA-Test.csv` (818) | positive / negative / neutral |
| YouTube | Official YouTube Data API, 3 SHEIN clothing-haul videos (2 AR creators, 1 EN creator) | — | 234 total (EN 100 / AR 100 / mixed 34) | Flash-Lite pre-labelled, human reviewed |

**Why keep dev:** it is the only place the confidence cutoffs are picked. Picking them on test would inflate the test score. Both sources already ship a dev split, so it costs nothing extra.

**Common format** after loading, one CSV per split: `id, text, gold_sentiment, slice, split, source`. YouTube rows use the same columns, plus `gold_topics`.

**Label mismatch:** public sets have 3 labels. A `mixed` prediction on them counts as wrong but gets its own column in the confusion table. Mixed and sarcasm are scored properly on YouTube.

## Pipeline and models

Every post goes through two free models; only posts where either is unsure get one Gemini call, which returns the final answer.

&#91;embedded content: pipeline flow · 2 free models, 1 decision, 1 LLM fallback\]

Both free models run on every post; Gemini is called only when either one is unsure, and its answer replaces theirs.

| Role | Model | Output | Confidence |
| --- | --- | --- | --- |
| Sentiment | [cardiffnlp/twitter-xlm-roberta-base-sentiment](https://huggingface.co/cardiffnlp/twitter-xlm-roberta-base-sentiment) | positive / neutral / negative | top probability (3 add to 1) |
| Topic | [MoritzLaurer/mDeBERTa-v3-base-mnli-xnli](https://huggingface.co/MoritzLaurer/mDeBERTa-v3-base-mnli-xnli), zero-shot, `multi_label=True` | a score 0 to 1 per topic | top topic score |
| Hard posts | Gemini 3.5 Flash-Lite (`gemini-3.5-flash-lite`), temperature 0 | 4-way sentiment, sarcasm, topics, reason | not used for routing |
| Topic judge (eval only) | Gemini 3.1 Pro (preview) (`gemini-3.1-pro-preview`), temperature 0 | topics per test post | n/a |
|  |  |  |  |

**Why these models:**

- **XLM-R:** one model reads English, Arabic and mixed posts, trained on \~198M tweets and fine-tuned for sentiment in 8 languages including Arabic. On Saudi-English code-switched text, XLM-RoBERTa beat Arabic-only and larger LLM models (95.5% macro-F1, [AbjadNLP 2026](https://aclanthology.org/2026.abjadnlp-1.30)). Fine-tuned small models also beat zero-shot GPT-4 on Arabic tweet sentiment ([Taqyim](https://arxiv.org/pdf/2306.16322)).
- **mDeBERTa:** zero-shot, so it needs only topic names and no topic labels. Covers 100 languages, 80.2% on Arabic XNLI, returns a score per topic that doubles as routing confidence. Simpler than embedding match (needs phrase lists) or BERTopic (unnamed clusters), and free unlike LLM-only.
- **Trade-off:** neither is tuned for Gulf dialect or fashion, which is why low-confidence posts go to Gemini.

**Routing rules (send to Gemini if any is true):**

1. Sentiment confidence below the cutoff. Default 0.7; try 0.6, 0.7, 0.8 on dev.
2. Positive and negative both at 0.3 or above (likely mixed).
3. No topic scores 0.5 or above. Default 0.5; try 0.4, 0.5, 0.6 on dev.

**Topic settings:** hypothesis template `This post is about {}.`, topic names in English (the model matches across languages). Every topic at or above the threshold is kept, so a post can have two.

**Three setups run on the same posts:**

- A. Small only: XLM-R + mDeBERTa, no LLM. Low topic score becomes `other`.
- B. LLM only: Gemini Flash-Lite on every post.
- C. Cascade: A first, routed posts replaced by Gemini's answer.

**Caching:** every Gemini answer (pipeline and judge) is saved to `cache/llm_cache.json`, keyed by model + prompt version + post id. Reruns are free and identical.

**Gemini settings:** temperature 0, JSON output (`response_mime_type: application/json`), thinking set to its lowest level so output tokens stay small. Key in `GEMINI_API_KEY`.

Cost check ([pricing](https://ai.google.dev/gemini-api/docs/pricing)): Flash-Lite is $0.30 in / $2.50 out per million tokens, with a free tier. About 400 tokens in and 50 out per post is about $0.00025, so LLM-only on all 450 posts is about $0.11. The judge (3.1 Pro, $2 in / $12 out, no free tier) on 300 posts is about $0.42.

## Topics and LLM prompt

Eight topics, the same list for mDeBERTa, Gemini and the judge.

| Topic | Covers |
| --- | --- |
| delivery | late, missing or damaged orders, shipping |
| sizing | fit, wrong size, size charts |
| quality | fabric, stitching, item not as pictured |
| price | too expensive, discounts, sales, value |
| returns\_refunds | returning items, refunds, exchanges |
| customer\_service | support replies, waiting, staff |
| product\_praise | liking a product, style, look |
| other | anything else |

**Gemini prompt (v1), in short:** You label social media posts about a fashion brand. Posts may be English, Arabic, or both. Return JSON only. Sentiment is one of positive, negative, neutral, mixed (mixed = clearly both positive and negative). Sarcasm is true only if the literal words say the opposite of what is meant. Topics are one or more from the list. Reason is one short line.

**Output schema (per post):**

```json
{
  "sentiment": "negative",
  "sarcasm": false,
  "topics": ["delivery"],
  "reason": "Complains the order has not arrived after a week"
}
```

**Pipeline output row:** `id, slice, split, text, gold_sentiment, xlmr_label, xlmr_conf, topic_scores, routed, route_reason, final_sentiment, sarcasm, final_topics, llm_reason, setup`. These map to the `enrichment` block of the Part A mention record.

## Evaluation

Sentiment is scored against the datasets' human labels; topics are scored against an LLM judge that you spot-check.

**Sentiment (test posts, per slice and overall):**

- Macro-F1 (headline number)
- Negative precision and recall
- Confusion table, with a `mixed` column
- LLM call rate and cost per 1,000 posts

Computed with `sklearn.metrics.classification_report` and `confusion_matrix`.

**Choosing cutoffs (dev only):** run setup C at each cutoff, record macro-F1 and LLM call rate, pick the point where accuracy stops improving. Test is run once with the chosen cutoffs.

**Topics (LLM-as-judge):**

1. Gemini 3.1 Pro labels topics on the 300 test posts with the same list and prompt rules.
2. Score setups A, B and C against the judge: precision, recall and F1 per topic, plus macro-F1.
3. You label 20 posts blind, without seeing the judge, and report agreement with the judge. That number says how far to trust it.
4. Public tweets are not about fashion, so most will be `other`. Topic results become meaningful on the YouTube comments.

**Error analysis:** 10 to 15 test posts the cascade got wrong, each tagged with a reason: sarcasm, dialect word, language switch, emoji, weak opinion read as neutral, or gold label looks wrong.

**Results table (README headline):**

| Setup | EN F1 | AR F1 | Mixed F1 | Neg recall | Topic F1 | LLM calls | Cost / 1k |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A. Small only |  |  |  |  |  | 0% | $0 |
| B. LLM only |  |  |  |  |  | 100% | \~$0.25 |
| C. Cascade |  |  |  |  |  |  |  |

## Project setup

One small Python repo, run as five scripts in order; YouTube is a new loader, nothing else changes.

**Folders:**

```
partB/
  data/raw/            downloaded datasets
  data/processed/      dev.csv, test.csv (common format)
  src/config.py        cutoffs, topic list, model ids, prompt version
  src/load_data.py     sample 50 dev + 100 test per slice; later youtube
  src/models.py        XLM-R sentiment + mDeBERTa topics
  src/llm.py           Gemini call, JSON parse, cache
  src/pipeline.py      setups A, B, C
  src/judge.py         Gemini Pro topic labels for test
  src/evaluate.py      metrics, cutoff sweep, results table
  cache/llm_cache.json
  results/             predictions_*.csv, metrics.md, errors.md
  README.md
  .env                 GEMINI_API_KEY (not committed)
```

**Dependencies:** `transformers`, `torch` (CPU is fine), `datasets`, `pandas`, `scikit-learn`, `google-genai`, `python-dotenv`. Model downloads are about 1.1 GB each for XLM-R and mDeBERTa.

**Run order:**

1. `load_data.py` builds dev.csv and test.csv.
2. `pipeline.py --split dev --sweep` runs the cutoff sweep.
3. Set the chosen cutoffs in `config.py`.
4. `pipeline.py --split test` runs A, B, C once.
5. `judge.py` then `evaluate.py` write `results/metrics.md`.

**YouTube track:** a `youtube` source in `load_data.py` writing the same columns plus `gold_topics`. Its upstream (`scrape_youtube.py` → `build_youtube_pool.py` → `label_youtube.py`) is documented in `docs/youtube_preprocessing.md`. The YouTube set is a single 234-row split (no separate dev/test), evaluated once with the dev-tuned cutoffs — see `README.md` for the result tables and `results/metrics_youtube.md` for the full breakdown.

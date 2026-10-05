# Instructions

## Project

Part B of a sentiment and topic pipeline. Labels English, Arabic and mixed (EN+AR) social media posts for sentiment (positive/negative/neutral/mixed) and 8 fashion topics. Runs two free HuggingFace models on every post and routes only low-confidence posts to Gemini. Compares three setups (small-only, LLM-only, cascade) on macro-F1, negative recall, topic F1, LLM call rate and cost.

Full spec: `Part B Sentiment and Topic Pipeline Config.md`.

## Stack

- **Language:** Python 3.11+
- **Sentiment model:** `cardiffnlp/twitter-xlm-roberta-base-sentiment` (3-way, multilingual)
- **Topic model:** `MoritzLaurer/mDeBERTa-v3-base-mnli-xnli` (zero-shot, `multi_label=True`)
- **LLM fallback + judge:** Google Gemini via `google-genai` SDK
  - Pipeline: `gemini-3.5-flash-lite`, temperature 0, JSON mode, lowest thinking level
  - Topic judge (eval only): `gemini-3.1-pro-preview`, temperature 0
- **ML/data:** `transformers`, `torch` (CPU is fine), `datasets`, `pandas`, `scikit-learn`
- **Config:** `python-dotenv` for `GEMINI_API_KEY`
- **Storage:** CSV on disk (`data/processed/*.csv`, `results/predictions_*.csv`), JSON file cache for LLM calls

Stack is locked unless explicitly changed. Don't propose alternatives without a stated reason. In particular: no fine-tuning, no BERTopic, no embedding-based topic matching, no separate Arabizi handling — all explicitly out of scope.

## Environment

- Python 3.12 in `.venv/`, managed by `uv`. Install deps with `uv pip install <pkg>` — there is no `pip` inside the venv.
- No `pyproject.toml` or lockfile yet. Add one before onboarding anyone else.
- **Pinned:** `datasets<3`. The `cardiffnlp/tweet_sentiment_multilingual` dataset still ships as a loader script, which `datasets>=3` refuses to run. Do not bump without first switching the loader (or vendoring the CSVs).

## Repo layout

```text
mixed-lang-sentiment-cascade/
├── CLAUDE.md                                      # this file
├── README.md
├── Part B Sentiment and Topic Pipeline Config.md  # spec (source of truth)
├── .env                                           # GEMINI_API_KEY (not committed)
├── .venv/                                         # uv-managed virtual env (gitignored)
├── data/
│   ├── raw/                                       # downloaded datasets (gitignored)
│   │   ├── eesa/                                  # EESA corpus (manual curl)
│   │   └── tweet_sentiment_multilingual/          # HF-exported CSVs
│   └── processed/                                 # dev.csv, test.csv (common format)
├── docs/                                          # stage notes (preprocessing.md, ...)
├── src/
│   ├── __init__.py
│   ├── config.py                                  # cutoffs, topic list, model ids, prompt version
│   ├── load_data.py                               # 50 dev + 100 test per slice; later youtube
│   ├── models.py                                  # XLM-R sentiment + mDeBERTa topics
│   ├── llm.py                                     # Gemini call, JSON parse, cache
│   ├── pipeline.py                                # setups A, B, C
│   ├── judge.py                                   # Gemini Pro topic labels for test
│   └── evaluate.py                                # metrics, cutoff sweep, results table
├── cache/llm_cache.json                           # keyed by model + prompt version + post id
└── results/                                       # predictions_*.csv, metrics.md, errors.md
```

Each stage gets a short notes file in `docs/` (what it does, config used, result). Keep them point-form, not narrative.

## Data contract

One CSV per split with columns: `id, text, gold_sentiment, slice, split, source`. YouTube rows add `gold_topics`.

Pipeline output row: `id, slice, split, text, gold_sentiment, xlmr_label, xlmr_conf, topic_scores, routed, route_reason, final_sentiment, sarcasm, final_topics, llm_reason, setup`.

Do not change these column names or add/remove columns without updating `evaluate.py` in the same change.

## Routing rules (cascade)

Send a post to Gemini if **any** of these are true:
1. Sentiment confidence below cutoff (default 0.7; sweep 0.6/0.7/0.8 on dev).
2. Positive and negative probabilities both ≥ 0.3 (likely mixed).
3. No topic score ≥ 0.5 (default; sweep 0.4/0.5/0.6 on dev).

Cutoffs are picked on **dev only**. Test is run once with the chosen cutoffs. Never tune on test.

## Dependency policy

**Default: write it yourself.** Every dependency is a liability.

OK to depend on:
- The declared stack: `transformers`, `torch`, `datasets`, `pandas`, `scikit-learn`, `google-genai`, `python-dotenv`.
- Things genuinely hard to get right (model loading, tokenisation, HTTP, metric implementations).

Not OK:
- Wrappers over 5–20 lines of stdlib (`pathlib`, `csv`, `json`, `argparse` are enough).
- "Nicer API" layers over something already in the stack (don't add `langchain` on top of `google-genai`; don't add `polars` alongside `pandas`).
- Prompt-templating or LLM-orchestration frameworks. The whole LLM layer is one `llm.py` with a cache dict.

Before adding a runtime dep, justify in the commit message:
1. What exactly does it do that we can't write in <30 lines of clear code?
2. How often is it used?
3. Transitive footprint?

## Configuration

`src/config.py` is the single source of truth for: topic list, cutoffs, model ids, prompt version, random seed (42), split sizes (50 dev / 100 test per slice).

- Do not call `os.getenv` outside the one place that reads `GEMINI_API_KEY`.
- Do not call `load_dotenv` more than once.
- Fail fast on startup if `GEMINI_API_KEY` is missing when a Gemini-using script runs. No silent "skip LLM" fallbacks — they'd corrupt comparison across setups.

## LLM cache

Every Gemini call (pipeline and judge) is cached in `cache/llm_cache.json`, keyed by `model_id + prompt_version + post_id`. Reruns must be free and bit-identical. If you change the prompt, bump `prompt_version` in `config.py` — don't silently invalidate the cache or leave stale entries that map to the old prompt.

The judge model and the pipeline fallback are separate cache namespaces.

## Code style

- **Small, obvious functions.** A 15-line function with clear names beats a three-class abstraction.
- **No premature abstraction.** Three similar lines is better than a badly-named base class. Extract on the third caller, not the hypothetical one.
- **No error handling for cases that can't happen.** Validate only at boundaries: dataset loading, Gemini JSON parsing, CSV read/write.
- **Determinism:** always use `seed=42` for sampling; `temperature=0` for Gemini. Reruns on the same inputs must produce the same outputs (modulo the cache being warm).
- **No comments explaining *what*** — only *why* when non-obvious. Remove stale TODOs.
- **Keep files focused.** The seven files listed above are the whole repo. Don't add a `utils.py` grab bag.

## Scope guardrails

Hard-locked out of scope — don't propose them, don't half-implement them:
- Emoji preprocessing/rules (passed through as text).
- Fine-tuning any model.
- BERTopic or any topic *discovery* (topic list is fixed, 8 items).
- Confidence intervals or a separate calibration study.
- Arabizi (Latin-script Arabic) handling.
- Any 4-way sentiment scoring against the public datasets — those are 3-label, so a `mixed` prediction on them counts as wrong and gets its own confusion-matrix column. Mixed/sarcasm are scored properly only on YouTube later.

## Run order

1. `python -m src.load_data` → builds `data/processed/dev.csv` and `test.csv`.
2. `python -m src.pipeline --split dev --sweep` → cutoff sweep on dev.
3. Set chosen cutoffs in `src/config.py`.
4. `python -m src.pipeline --split test` → runs setups A, B, C once.
5. `python -m src.judge` then `python -m src.evaluate` → writes `results/metrics.md` and `results/errors.md`.

Adding YouTube later: new `youtube` source in `load_data.py` with the same columns + `gold_topics`; rerun steps 4 and 5. No other file should need to change.

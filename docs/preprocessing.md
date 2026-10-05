# Preprocessing

Builds `data/processed/dev.csv` and `data/processed/test.csv` from three raw sources. Run: `python -m src.load_data`.

## Common target schema

- Columns: `id, text, gold_sentiment, slice, split, source`
- `gold_sentiment`: lowercase string, 3-way — `positive` / `negative` / `neutral`
- `slice`: `english` / `arabic` / `mixed`
- `split`: `dev` / `test`
- `id` format: `{slice}_{split}_{NNN}` (zero-padded, sequential per slice)
- Encoding: UTF-8, no BOM

## Per-source preprocessing

**English — `cardiffnlp/tweet_sentiment_multilingual` (english config)**
- Raw files: `data/raw/tweet_sentiment_multilingual/english_{validation,test}.csv`
- Columns raw: `text, label` (label ∈ {0, 1, 2})
- Map labels: `0 → negative`, `1 → neutral`, `2 → positive`
- Text untouched (Cardiff pre-normalised URLs as `http`, mentions as `@user` — XLM-R expects this)

**Arabic — same dataset (arabic config)**
- Raw files: `data/raw/tweet_sentiment_multilingual/arabic_{validation,test}.csv`
- Same label map as English
- No RTL/diacritic/Arabic normalisation — raw script preserved

**Mixed — EESA corpus**
- Raw files: `data/raw/eesa/EESA-{Dev,Test}.csv`
- No header row; parsed as `text, gold_sentiment`
- Labels already lowercase strings — safety `.lower().strip()` applied anyway
- Rows with embedded newlines in quoted fields are handled by pandas CSV parsing (file has ~1065 lines but 818 logical rows)

## Universal cleaning steps

- `text.strip()` applied; rows where text becomes empty are dropped
- Reject on any unexpected gold label (fail fast)
- Stratified sample using `sklearn.model_selection.train_test_split(stratify=gold_sentiment, random_state=42)` — exact N preserved, per-class ratios match the source
- Empties-first → sample-second (so dropping never breaks class balance)

## Explicitly NOT done (locked out per spec)

- No emoji stripping — passed through as text
- No lowercasing of text
- No URL / mention normalisation beyond Cardiff's native format
- No deduplication / near-dup detection
- No fine-tuning, no Arabic script normalisation, no Arabizi handling

## Config (`src/config.py`)

| Key | Value |
|---|---|
| `SEED` | `42` |
| `DEV_N` | `50` per slice |
| `TEST_N` | `100` per slice |
| `SENTIMENT_LABELS` | `("positive", "negative", "neutral")` |
| `CARDIFF_LABEL_MAP` | `{0: "negative", 1: "neutral", 2: "positive"}` |
| `RAW_DIR` | `data/raw/` |
| `PROCESSED_DIR` | `data/processed/` |

## Result

- `data/processed/dev.csv` — 150 rows (50 × 3 slices)
- `data/processed/test.csv` — 300 rows (100 × 3 slices)
- 0 empty-text rows dropped from any source
- Deterministic: same seed + same raw files → bit-identical output

**Label distribution after sampling**

| slice | split | negative | neutral | positive |
|---|---|---|---|---|
| english | dev | 17 | 17 | 16 |
| english | test | 34 | 33 | 33 |
| arabic | dev | 17 | 17 | 16 |
| arabic | test | 34 | 33 | 33 |
| mixed | dev | 12 | 16 | 22 |
| mixed | test | 24 | 32 | 44 |

- English + Arabic effectively balanced (Cardiff sets are near-uniform)
- EESA (mixed) is positive-skewed; stratification preserved the skew
- Consequence: negative recall on the mixed slice is noisier (dev has only 12 negatives)

# Pipeline Evaluation

Date: 2026-10-05
Split: **test** (300 rows)

## Headline

| Setup | EN F1 | AR F1 | Mixed F1 | Overall F1 | Neg Recall | Topic F1 | LLM Calls | Cost / 1k |
|---|---|---|---|---|---|---|---|---|
| A | 0.724 | 0.677 | 0.838 | 0.749 | 0.848 | 0.230 | 0% | $0.000 |
| B | 0.761 | 0.714 | 0.939 | 0.805 | 0.826 | 0.435 | 100% | $0.250 |
| C | 0.726 | 0.694 | 0.939 | 0.787 | 0.859 | 0.276 | 70% | $0.174 |

## Setup A

LLM call rate: **0.0%**  ·  Cost / 1k posts: **$0.000**

### Sentiment metrics

| scope   |   n |   accuracy |   macro_f1 |   neg_precision |   neg_recall |
|:--------|----:|-----------:|-----------:|----------------:|-------------:|
| english | 100 |      0.73  |      0.724 |           0.698 |        0.882 |
| arabic  | 100 |      0.68  |      0.677 |           0.7   |        0.824 |
| mixed   | 100 |      0.85  |      0.838 |           0.8   |        0.833 |
| overall | 300 |      0.753 |      0.749 |           0.722 |        0.848 |

### Confusion (gold rows × pred cols; `mixed` col kept)

**english**

|          |   negative |   neutral |   positive |   mixed |
|:---------|-----------:|----------:|-----------:|--------:|
| negative |         30 |         4 |          0 |       0 |
| neutral  |         10 |        18 |          5 |       0 |
| positive |          3 |         5 |         25 |       0 |

**arabic**

|          |   negative |   neutral |   positive |   mixed |
|:---------|-----------:|----------:|-----------:|--------:|
| negative |         28 |         5 |          1 |       0 |
| neutral  |         10 |        18 |          5 |       0 |
| positive |          2 |         9 |         22 |       0 |

**mixed**

|          |   negative |   neutral |   positive |   mixed |
|:---------|-----------:|----------:|-----------:|--------:|
| negative |         20 |         2 |          2 |       0 |
| neutral  |          5 |        24 |          3 |       0 |
| positive |          0 |         3 |         41 |       0 |

### Topic metrics

| topic            |     P |     R |    F1 |   support_gold |
|:-----------------|------:|------:|------:|---------------:|
| delivery         | 0.1   | 1     | 0.182 |              1 |
| sizing           | 0     | 0     | 0     |              0 |
| quality          | 0.031 | 0.25  | 0.056 |              8 |
| price            | 0.176 | 0.75  | 0.286 |              4 |
| returns_refunds  | 0     | 0     | 0     |              0 |
| customer_service | 0     | 0     | 0     |              2 |
| product_praise   | 0.571 | 0.377 | 0.455 |             53 |
| other            | 0.874 | 0.856 | 0.865 |            236 |
| macro            | 0.219 | 0.404 | 0.23  |            304 |

## Setup B

LLM call rate: **100.0%**  ·  Cost / 1k posts: **$0.250**

### Sentiment metrics

| scope   |   n |   accuracy |   macro_f1 |   neg_precision |   neg_recall |
|:--------|----:|-----------:|-----------:|----------------:|-------------:|
| english | 100 |       0.76 |      0.761 |           0.721 |        0.912 |
| arabic  | 100 |       0.71 |      0.714 |           1     |        0.647 |
| mixed   | 100 |       0.93 |      0.939 |           1     |        0.958 |
| overall | 300 |       0.8  |      0.805 |           0.864 |        0.826 |

### Confusion (gold rows × pred cols; `mixed` col kept)

**english**

|          |   negative |   neutral |   positive |   mixed |
|:---------|-----------:|----------:|-----------:|--------:|
| negative |         31 |         3 |          0 |       0 |
| neutral  |         10 |        23 |          0 |       0 |
| positive |          2 |         9 |         22 |       0 |

**arabic**

|          |   negative |   neutral |   positive |   mixed |
|:---------|-----------:|----------:|-----------:|--------:|
| negative |         22 |        12 |          0 |       0 |
| neutral  |          0 |        32 |          1 |       0 |
| positive |          0 |        16 |         17 |       0 |

**mixed**

|          |   negative |   neutral |   positive |   mixed |
|:---------|-----------:|----------:|-----------:|--------:|
| negative |         23 |         1 |          0 |       0 |
| neutral  |          0 |        29 |          3 |       0 |
| positive |          0 |         2 |         41 |       1 |

### Topic metrics

| topic            |     P |     R |    F1 |   support_gold |
|:-----------------|------:|------:|------:|---------------:|
| delivery         | 0     | 0     | 0     |              1 |
| sizing           | 0     | 0     | 0     |              0 |
| quality          | 1     | 0.25  | 0.4   |              8 |
| price            | 1     | 0.75  | 0.857 |              4 |
| returns_refunds  | 0     | 0     | 0     |              0 |
| customer_service | 0.333 | 1     | 0.5   |              2 |
| product_praise   | 0.827 | 0.811 | 0.819 |             53 |
| other            | 0.842 | 0.97  | 0.902 |            236 |
| macro            | 0.5   | 0.473 | 0.435 |            304 |

## Setup C

LLM call rate: **69.7%**  ·  Cost / 1k posts: **$0.174**

### Sentiment metrics

| scope   |   n |   accuracy |   macro_f1 |   neg_precision |   neg_recall |
|:--------|----:|-----------:|-----------:|----------------:|-------------:|
| english | 100 |      0.73  |      0.726 |           0.705 |        0.912 |
| arabic  | 100 |      0.69  |      0.694 |           0.833 |        0.735 |
| mixed   | 100 |      0.93  |      0.939 |           1     |        0.958 |
| overall | 300 |      0.783 |      0.787 |           0.814 |        0.859 |

### Confusion (gold rows × pred cols; `mixed` col kept)

**english**

|          |   negative |   neutral |   positive |   mixed |
|:---------|-----------:|----------:|-----------:|--------:|
| negative |         31 |         3 |          0 |       0 |
| neutral  |         11 |        19 |          3 |       0 |
| positive |          2 |         8 |         23 |       0 |

**arabic**

|          |   negative |   neutral |   positive |   mixed |
|:---------|-----------:|----------:|-----------:|--------:|
| negative |         25 |         9 |          0 |       0 |
| neutral  |          5 |        27 |          1 |       0 |
| positive |          0 |        16 |         17 |       0 |

**mixed**

|          |   negative |   neutral |   positive |   mixed |
|:---------|-----------:|----------:|-----------:|--------:|
| negative |         23 |         1 |          0 |       0 |
| neutral  |          0 |        29 |          3 |       0 |
| positive |          0 |         2 |         41 |       1 |

### Topic metrics

| topic            |     P |     R |    F1 |   support_gold |
|:-----------------|------:|------:|------:|---------------:|
| delivery         | 0     | 0     | 0     |              1 |
| sizing           | 0     | 0     | 0     |              0 |
| quality          | 0.039 | 0.25  | 0.068 |              8 |
| price            | 0.2   | 0.5   | 0.286 |              4 |
| returns_refunds  | 0     | 0     | 0     |              0 |
| customer_service | 0.167 | 1     | 0.286 |              2 |
| product_praise   | 0.7   | 0.66  | 0.68  |             53 |
| other            | 0.897 | 0.881 | 0.889 |            236 |
| macro            | 0.25  | 0.411 | 0.276 |            304 |

# Pipeline Evaluation

Date: 2026-10-05
Split: **test** (234 rows)

## Headline

| Setup | EN F1 | AR F1 | Mixed F1 | Overall F1 | Neg Recall | Topic F1 | LLM Calls | Cost / 1k |
|---|---|---|---|---|---|---|---|---|
| A | 0.463 | 0.334 | 0.556 | 0.441 | 0.615 | 0.347 | 0% | $0.000 |
| B | 0.945 | 0.899 | 0.715 | 0.909 | 0.975 | 0.904 | 100% | $0.250 |
| C | 0.813 | 0.800 | 0.715 | 0.807 | 0.951 | 0.681 | 76% | $0.190 |

## Setup A

LLM call rate: **0.0%**  ·  Cost / 1k posts: **$0.000**

### Sentiment metrics

| scope   |   n |   accuracy |   macro_f1 |   neg_precision |   neg_recall |
|:--------|----:|-----------:|-----------:|----------------:|-------------:|
| english | 100 |      0.65  |      0.463 |           0.74  |        0.755 |
| arabic  | 100 |      0.5   |      0.334 |           0.795 |        0.508 |
| mixed   |  34 |      0.735 |      0.556 |           0.875 |        0.583 |
| overall | 234 |      0.598 |      0.441 |           0.773 |        0.615 |

### Confusion (gold rows × pred cols; `mixed` col kept)

**english**

|          |   negative |   neutral |   positive |   mixed |
|:---------|-----------:|----------:|-----------:|--------:|
| negative |         37 |         6 |          6 |       0 |
| neutral  |          9 |        21 |          5 |       0 |
| positive |          2 |         2 |          7 |       0 |
| mixed    |          2 |         2 |          1 |       0 |

**arabic**

|          |   negative |   neutral |   positive |   mixed |
|:---------|-----------:|----------:|-----------:|--------:|
| negative |         31 |         3 |         27 |       0 |
| neutral  |          7 |         4 |          7 |       0 |
| positive |          0 |         3 |         15 |       0 |
| mixed    |          1 |         1 |          1 |       0 |

**mixed**

|          |   negative |   neutral |   positive |   mixed |
|:---------|-----------:|----------:|-----------:|--------:|
| negative |          7 |         5 |          0 |       0 |
| neutral  |          1 |         9 |          1 |       0 |
| positive |          0 |         2 |          9 |       0 |
| mixed    |          0 |         0 |          0 |       0 |

### Topic metrics

| topic            |     P |     R |    F1 |   support_gold |
|:-----------------|------:|------:|------:|---------------:|
| delivery         | 0.333 | 0.5   | 0.4   |              2 |
| sizing           | 0.6   | 0.5   | 0.545 |             24 |
| quality          | 0.226 | 0.214 | 0.22  |             56 |
| price            | 0.222 | 0.444 | 0.296 |              9 |
| returns_refunds  | 0.1   | 1     | 0.182 |              1 |
| customer_service | 0     | 0     | 0     |              4 |
| product_praise   | 0.256 | 0.345 | 0.294 |             29 |
| other            | 0.856 | 0.816 | 0.836 |            196 |
| macro            | 0.324 | 0.477 | 0.347 |            321 |

## Setup B

LLM call rate: **100.0%**  ·  Cost / 1k posts: **$0.250**

### Sentiment metrics

| scope   |   n |   accuracy |   macro_f1 |   neg_precision |   neg_recall |
|:--------|----:|-----------:|-----------:|----------------:|-------------:|
| english | 100 |      0.95  |      0.945 |           0.94  |        0.959 |
| arabic  | 100 |      0.97  |      0.899 |           0.984 |        0.984 |
| mixed   |  34 |      0.941 |      0.715 |           1     |        1     |
| overall | 234 |      0.957 |      0.909 |           0.967 |        0.975 |

### Confusion (gold rows × pred cols; `mixed` col kept)

**english**

|          |   negative |   neutral |   positive |   mixed |
|:---------|-----------:|----------:|-----------:|--------:|
| negative |         47 |         2 |          0 |       0 |
| neutral  |          2 |        33 |          0 |       0 |
| positive |          0 |         0 |         11 |       0 |
| mixed    |          1 |         0 |          0 |       4 |

**arabic**

|          |   negative |   neutral |   positive |   mixed |
|:---------|-----------:|----------:|-----------:|--------:|
| negative |         60 |         0 |          0 |       1 |
| neutral  |          1 |        17 |          0 |       0 |
| positive |          0 |         0 |         18 |       0 |
| mixed    |          0 |         0 |          1 |       2 |

**mixed**

|          |   negative |   neutral |   positive |   mixed |
|:---------|-----------:|----------:|-----------:|--------:|
| negative |         12 |         0 |          0 |       0 |
| neutral  |          0 |        10 |          1 |       0 |
| positive |          0 |         0 |         10 |       1 |
| mixed    |          0 |         0 |          0 |       0 |

### Topic metrics

| topic            |     P |     R |    F1 |   support_gold |
|:-----------------|------:|------:|------:|---------------:|
| delivery         | 1     | 1     | 1     |              2 |
| sizing           | 0.957 | 0.917 | 0.936 |             24 |
| quality          | 0.945 | 0.929 | 0.937 |             56 |
| price            | 0.818 | 1     | 0.9   |              9 |
| returns_refunds  | 1     | 1     | 1     |              1 |
| customer_service | 0.6   | 0.75  | 0.667 |              4 |
| product_praise   | 0.806 | 0.862 | 0.833 |             29 |
| other            | 0.969 | 0.944 | 0.956 |            196 |
| macro            | 0.887 | 0.925 | 0.904 |            321 |

## Setup C

LLM call rate: **76.1%**  ·  Cost / 1k posts: **$0.190**

### Sentiment metrics

| scope   |   n |   accuracy |   macro_f1 |   neg_precision |   neg_recall |
|:--------|----:|-----------:|-----------:|----------------:|-------------:|
| english | 100 |      0.9   |      0.813 |           0.902 |        0.939 |
| arabic  | 100 |      0.93  |      0.8   |           0.967 |        0.951 |
| mixed   |  34 |      0.941 |      0.715 |           1     |        1     |
| overall | 234 |      0.919 |      0.807 |           0.943 |        0.951 |

### Confusion (gold rows × pred cols; `mixed` col kept)

**english**

|          |   negative |   neutral |   positive |   mixed |
|:---------|-----------:|----------:|-----------:|--------:|
| negative |         46 |         1 |          2 |       0 |
| neutral  |          3 |        32 |          0 |       0 |
| positive |          0 |         1 |         10 |       0 |
| mixed    |          2 |         0 |          1 |       2 |

**arabic**

|          |   negative |   neutral |   positive |   mixed |
|:---------|-----------:|----------:|-----------:|--------:|
| negative |         58 |         0 |          2 |       1 |
| neutral  |          1 |        16 |          1 |       0 |
| positive |          0 |         0 |         18 |       0 |
| mixed    |          1 |         0 |          1 |       1 |

**mixed**

|          |   negative |   neutral |   positive |   mixed |
|:---------|-----------:|----------:|-----------:|--------:|
| negative |         12 |         0 |          0 |       0 |
| neutral  |          0 |        10 |          1 |       0 |
| positive |          0 |         0 |         10 |       1 |
| mixed    |          0 |         0 |          0 |       0 |

### Topic metrics

| topic            |     P |     R |    F1 |   support_gold |
|:-----------------|------:|------:|------:|---------------:|
| delivery         | 1     | 0.5   | 0.667 |              2 |
| sizing           | 0.815 | 0.917 | 0.863 |             24 |
| quality          | 0.667 | 0.679 | 0.673 |             56 |
| price            | 0.6   | 1     | 0.75  |              9 |
| returns_refunds  | 0.25  | 1     | 0.4   |              1 |
| customer_service | 0.375 | 0.75  | 0.5   |              4 |
| product_praise   | 0.636 | 0.724 | 0.677 |             29 |
| other            | 0.946 | 0.893 | 0.919 |            196 |
| macro            | 0.661 | 0.808 | 0.681 |            321 |

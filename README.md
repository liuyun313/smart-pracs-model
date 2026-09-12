# smart-pracs-model: pathway gene mutation combination optimization

This repository contains the code and data for predicting production-enhancing MCP enzyme (or variant) combinations from **binary combinations of 13 gene
mutations**. It includes:

- a **main deep-learning model** (1-D CNN + multi-head self-attention + a
  differentiable Spearman-correlation loss), and
- several **classical regression baselines** (SVR, Random Forest, Gradient
  Boosting, plain CNN, LSTM)

for comparison.

---

## 1. Repository Layout

```
main/
├── data/
│   ├── data.xlsx                  # Raw input: binary genotype matrix + RP value
│   ├── encoded_known.csv          # One-hot encoded KNOWN samples (with target)
│   ├── encoded_unknown.csv        # One-hot encoded UNKNOWN samples (to be predicted)
│   └── encoded_unknown_0_1.csv    # Binary 0/1 representation of UNKNOWN samples
├── models/
│   ├── best_model.keras           # Main model: CNN + Self-Attention (Spearman loss)
│   ├── cnn.keras                  # Baseline: plain 1-D CNN
│   ├── lstm.keras                 # Baseline: stacked LSTM
│   ├── model_gbr.joblib           # Baseline: Gradient Boosting Regressor
│   ├── model_rf.joblib            # Baseline: Random Forest Regressor
│   └── model_svr.joblib           # Baseline: Support Vector Regression
└── scripts/
    ├── generate_one_hot_encoder.py  # Build the one-hot encoded CSVs
    ├── training.py                  # Train the MAIN model (best_model.keras)
    ├── training_comparative.py      # Train the BASELINE models (currently commented)
    ├── prediction.py                # Run inference with the MAIN model
    └── prediction_comparative.py    # Run inference with the BASELINE models
```

---

## 2. File-by-File Description

### 2.1 `data/`

| File | Purpose |
|------|---------|
| `data.xlsx` | **Raw input**. Excel sheet named `data`. Columns: `[ID, gene_1, gene_2, …, gene_13, target]`. Each `gene_i` is binary (`0` = wild-type, `1` = mutated). `RP value` is the continuous production value. |
| `encoded_known.csv` | One-hot encoded version of the KNOWN samples (those with measured RP value). The last column is the normalised target. Produced by `generate_one_hot_encoder.py`. Used as the training set. |
| `encoded_unknown.csv` | One-hot encoded version of UNKNOWN samples (those to be predicted). Produced by `generate_one_hot_encoder.py`. Consumed by `prediction.py`. |
| `encoded_unknown_0_1.csv` | Binary `0/1` matrix of the UNKNOWN samples, after de-duplication against the known data. Consumed by `prediction_comparative.py` (the classical baselines expect raw `0/1` features, not one-hot indices). |

### 2.2 `models/`

Pre-trained model artefacts. The main model is `best_model.keras`; everything
else is a baseline for comparison.

| File | Description |
|------|-------------|
| `best_model.keras` | The proposed model: 1-D CNN → multi-head self-attention → flatten → dense. Trained with a **differentiable Spearman-correlation loss** (see `training.py`). |
| `cnn.keras` | Baseline CNN (3 stacked `Convolution1D` layers + dense head). |
| `lstm.keras` | Baseline stacked LSTM. |
| `model_rf.joblib` | Random Forest Regressor (with `oob_score=True`). |
| `model_gbr.joblib` | Gradient Boosting Regressor. |
| `model_svr.joblib` | Support Vector Regressor. |

### 2.3 `scripts/`

| Script | Role |
|--------|------|
| `generate_one_hot_encoder.py` | Reads `data/data.xlsx`, encodes each gene as a unique token (`<gene_name>` if mutated, `<gene_name_reversed>` if wild-type), runs Keras `one_hot` + `pad_sequences`, and writes `encoded_known.csv`. Then enumerates all `2^13` binary combinations, removes duplicates of the known set, and writes `encoded_unknown.csv` and `encoded_unknown_0_1.csv`. |
| `training.py` | Defines `spearman_loss`, `spearman_correlation`, the `PositionEmbedding` layer, and `build_regression_model`. Trains the main model with early-stopping on the validation Spearman correlation, plots train/test curves, and saves the best checkpoint as `models/best_model.keras`. |
| `training_comparative.py` | Trains the five baselines (CNN, LSTM, SVR, RF, GBR). Each baseline's training block is wrapped in `'''…'''` and toggled on by removing the triple quotes. |
| `prediction.py` | Loads `models/best_model.keras` (re-registering the custom `spearman_loss`, `spearman_correlation`, `PositionEmbedding` objects) and predicts `rank of RP value` for every row in `encoded_unknown.csv`. |
| `prediction_comparative.py` | Loads one of the classical / DL baselines and predicts `rank of RP value` for every row in `data/encoded_unknown_0_1.csv`. |

---

## 3. Quick Start

### 3.1 Environment

```bash
# Tested with Python 3.13
pip install numpy pandas scikit-learn scipy \
            tensorflow keras matplotlib openpyxl joblib
```

### 3.2 Reproduce the main pipeline

```bash
cd github/

# 1. (Re)build the encoded datasets
python scripts/generate_one_hot_encoder.py

# 2. (Re)train the main model  → models/best_model.keras
python scripts/training.py

# 3. Predict on the unknown combinations
python scripts/prediction.py
```

### 3.3 Reproduce a baseline

Open `scripts/training_comparative.py`, un-comment the block you want
(e.g. the RF block), then:

```bash
python scripts/training_comparative.py     # training
python scripts/prediction_comparative.py   # inference
```

The baseline inference script loads whichever model you point it at; edit
the `load(...)` / `load_model(...)` lines near the top of
`prediction_comparative.py` to switch between `model_rf.joblib`,
`model_gbr.joblib`, `model_svr.joblib`, `cnn.keras`, `lstm.keras`.

---

## 4. Migrating to a New Dataset

The pipeline is parameterised in only a few places, so adapting it to a
different experiment is straightforward.

### 4.1 Provide your own data

Replace `data/data.xlsx` with a file of the **same shape**:

```
| ID (optional) | gene_1 | gene_2 | … | gene_N | target |
|----------------|--------|--------|---|--------|--------|
| sample_001     |   0    |   1    | … |   1    |  3.21  |
| sample_002     |   1    |   0    | … |   0    |  4.87  |
```

- `gene_i` must be binary (`0` / `1`).
- `target` is a single continuous column at the end.
- The sheet name must be `data` (or change the `sheet_name='data'` argument).

### 4.2 Update the gene count `N`

The number of genes is currently hard-coded as **`13`** in three scripts. If
your pathway has a different number of genes `N`, search-and-replace:

| Script | What to change |
|--------|----------------|
| `generate_one_hot_encoder.py` | `for j in range(13):` (two occurrences); the `pad_sequences(..., maxlen=13, ...)` argument; `n = 13`. |
| `training.py` | `seq_len = 13`, `max_len = 13`; the LSTM baseline (if used) `input_shape=(13, 1)` in `training_comparative.py`. |
| `training_comparative.py` | `max_features=range(1,13,2)` in the RF grid (replace `13` with `N+1`). |

### 4.3 Re-encode and re-train

```bash
python scripts/generate_one_hot_encoder.py   # writes encoded_known.csv & encoded_unknown_*.csv
python scripts/training.py                   # trains the main model
```

`vocab_size = 100` in the encoder is large enough for any `N ≤ 13`; if you
go to a much larger pathway, increase it (must be `> 2N` because each gene
contributes two distinct tokens: mutated / wild-type).

### 4.4 Optional: re-target the model

- **Different target range**: `training.py` already normalises the target
  via `MinMaxScaler`; no change needed unless you want a different
  preprocessing.
- **Classification instead of regression**: change the final
  `layers.Dense(1, activation='relu')` to `layers.Dense(num_classes,
  activation='softmax')` and swap `spearman_loss` for
  `tf.keras.losses.SparseCategoricalCrossentropy()`.
- **More / fewer features per gene**: replace the binary input with
  multi-level categorical encoding (e.g. 0/1/2) and adjust the column count
  in `range(13)` accordingly.

### 4.5 Folder hygiene

Keep this layout on GitHub:

```
data/      # raw + encoded CSVs / XLSX (≤ a few MB each)
models/    # large .keras / .joblib files — use Git LFS
scripts/   # all .py files
README.md  # this file
```

---

## 5. Dependencies (versions used during development)

| Package | Version |
|---------|---------|
| Python | 3.13 |
| numpy | ≥ 1.26 |
| pandas | ≥ 2.2 |
| scikit-learn | ≥ 1.4 |
| scipy | ≥ 1.13 |
| tensorflow / keras | ≥ 2.15 |
| matplotlib | ≥ 3.9 |
| openpyxl | ≥ 3.1 (only needed for `.xlsx` input) |
| joblib | ≥ 1.3 |

---

## 6. Citation

If you use this code or data in academic work, please cite the accompanying
paper (details to be added upon publication).

---

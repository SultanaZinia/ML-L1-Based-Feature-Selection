# L1-Based Feature Selection

A two-phase machine learning experiment comparing baseline classification performance against L1-regularized feature selection across 16 datasets and 5 classifiers, using 10-fold stratified cross-validation.

## Project Structure

```
L1-feature-selection/
├── data/
│   ├── processed/
│   │   ├── Datasets/
│   │   │   ├── 1-16/               # 16 datasets (train.csv + test.csv each)
│   │   │   └── Results.xlsx        # Final results (both phases)
│   │   ├── baseline_results.json   # Phase 1 output
│   │   ├── l1_results.json         # Phase 2 output (aggregated)
│   │   └── phase2_results/         # Per-dataset Phase 2 JSONs
│   └── sample/
│       └── wdbc.csv
├── notebooks/
│   ├── 02_experiment_loop.ipynb         # Phase 1: Baseline
│   └── 03_phase2_logreg_l1.ipynb        # Phase 2: L1 Feature Selection
└── results/
```

## Datasets

| Group | Datasets | Samples | Features | Notes |
|-------|----------|---------|----------|-------|
| Low-dim | 1–8 | ~9,583 | 19 | Near-perfect baseline accuracy |
| High-dim | 9–16 | ~8,330 | 265 | Benefits significantly from feature selection |

All datasets share a `Label` target column.

## Methodology

### Phase 1 — Baseline (`02_experiment_loop.ipynb`)

- Trains 5 classifiers on all 16 datasets with all original features
- Evaluation: 10-fold `StratifiedKFold`, metrics = accuracy + macro F1
- Pipeline: `SimpleImputer → (StandardScaler) → Classifier`
- Results saved to `baseline_results.json` and `Results.xlsx` sheet **"Before FS-DR"**

### Phase 2 — L1 Feature Selection (`03_phase2_logreg_l1.ipynb`)

- Uses `LogisticRegression(penalty='l1', solver='saga')` as the feature selector
  - Chosen over LinearSVC for speed — LinearSVC was prohibitively slow on 265-feature datasets
- Regularization grid: `C ∈ {0.001, 0.01, 0.1, 1.0}`, best C selected via 80/20 holdout split
- Pipeline: `SimpleImputer → StandardScaler → SelectFromModel(L1 LogReg) → (StandardScaler) → Classifier`
- Per-dataset results saved immediately to `phase2_results/dataset_XX.json` for crash resilience
- Aggregated into `l1_results.json` and `Results.xlsx` sheet **"After FS-DR"**

## Classifiers

| Classifier | Notes |
|------------|-------|
| SVM | RBF kernel |
| kNN | k-Nearest Neighbors |
| DecisionTree | CART |
| RandomForest | Ensemble |
| MLP | Multi-layer Perceptron |

## Key Results

- **Datasets 1–8 (19 features):** ~12 features selected on average — baseline performance maintained
- **Datasets 9–16 (265 features):** ~87–126 features selected — dimensionality reduced by ~50–67% with maintained or improved performance
- Example: Dataset 12 RandomForest improved from 0.8131 acc (265 features) → 0.8140 acc (87 features)

Results are compared across both phases in `data/processed/Datasets/Results.xlsx`.

## Dependencies

```
scikit-learn
numpy
pandas
openpyxl
```

Recommended: use a conda environment (e.g. `ml-env`).

```bash
conda create -n ml-env python=3.10
conda activate ml-env
pip install scikit-learn numpy pandas openpyxl notebook
```

## Running the Experiments

1. **Phase 1 — Baseline:**
   Open and run `notebooks/02_experiment_loop.ipynb` top to bottom.

2. **Phase 2 — L1 Feature Selection:**
   Open and run `notebooks/03_phase2_logreg_l1.ipynb` top to bottom.
   Already-completed datasets are skipped automatically (checkpoint system).

## Output Files

| File | Description |
|------|-------------|
| `data/processed/baseline_results.json` | Phase 1: per-fold accuracy + macro F1 |
| `data/processed/l1_results.json` | Phase 2: per-fold accuracy, macro F1, n_selected features, selected feature names, best L1 C |
| `data/processed/phase2_results/dataset_XX.json` | Phase 2 per-dataset checkpoints |
| `data/processed/Datasets/Results.xlsx` | Summary Excel workbook with both phases |

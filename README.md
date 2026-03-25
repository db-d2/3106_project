# Here Be Dragons: When Does PU Learning Work for Vulnerability Detection?

A systematic investigation of positive-unlabeled learning on code representations, using software vulnerability detection as a testbed.

STAT 3106 Applied Machine Learning, Spring 2026.

## The question

Security teams confirm a handful of vulnerabilities but leave most of their codebase unreviewed. This creates a positive-unlabeled (PU) setting: a small set of known-vulnerable functions and a large pool of unlabeled code with no confirmed negatives. I investigate how many labeled positives PU learning actually needs, whether the two main families of PU methods (risk estimators and pseudo-labeling) break down at different points, and which vulnerability types are hardest to detect with limited labels.

## Project structure

```
src/
  model.py                 MLP classifier (768-d input, 256 hidden, 1 output)
  extract_embeddings.py    Extract CodeBERT [CLS] embeddings (run on Colab)
  train_supervised.py      Supervised baseline (upper bound)
  train_naive_pu.py        Naive PU baseline (lower bound)
  train_nnpu.py            nnPU, risk estimator family
  train_self_training.py   Self-training, pseudo-labeling family
  losses.py                nnPU loss function
  evaluate.py              F1, AUROC, AUPRC, VDS, pairwise accuracy, per-CWE

data/
  prepare_pu_splits.py     Generate PU splits at any labeling fraction
  raw/PrimeVul_v0.1/       Raw JSONL files (not in git, see below)
  processed/embeddings/    CodeBERT .npz files (not in git, see below)
  processed/frac*_seed*/   Generated PU splits

blog/
  index.html               The deliverable (self-contained HTML, no dependencies)

checkpoints/               Saved model weights (not in git)
experiments/logs/          Experiment result CSVs
figures/                   Generated figures for the blog
quality_reports/           Automated review reports
```

## Setup

### 1. Python environment

Requires Python 3.12 (PyTorch does not support 3.13 yet).

```bash
pyenv virtualenv 3.12.3 stat3106
pyenv local stat3106
pip install -r requirements.txt
```

### 2. Get the data

The raw PrimeVul data and pre-extracted CodeBERT embeddings are hosted on HuggingFace because they are too large for git (~1.2GB total).

```bash
pip install huggingface_hub
hf download db-d2/primevul-codebert-embeddings --repo-type dataset --local-dir data/hf_download

# Move into expected locations
mkdir -p data/processed/embeddings data/raw/PrimeVul_v0.1
cp data/hf_download/*.npz data/processed/embeddings/
cp data/hf_download/raw/*.jsonl data/raw/PrimeVul_v0.1/
rm -rf data/hf_download
```

If you want to re-extract embeddings yourself instead of using the pre-extracted ones, upload `src/extract_embeddings.py` to Google Colab with a GPU runtime and run it against the raw JSONL files. Takes about 23 minutes on an A100.

### 3. Generate PU splits

```bash
python data/prepare_pu_splits.py --labeled-frac 0.20 --seed 42
```

This creates `data/processed/frac0.20_seed42/` with P.jsonl (labeled positives), U.jsonl (unlabeled), and metadata.json. Change `--labeled-frac` and `--seed` for the labeling budget sweep.

## Running experiments

All training runs operate on the pre-extracted embeddings. Each takes about 30 seconds on CPU.

```bash
# Supervised baseline (upper bound, uses full labels)
python src/train_supervised.py --seed 42

# Naive PU (lower bound, treats U as negative)
python src/train_naive_pu.py --labeled-frac 0.20 --split-seed 42 --seed 42

# nnPU (risk estimator family, requires class prior)
python src/train_nnpu.py --labeled-frac 0.20 --split-seed 42 --seed 42 --prior 0.0277

# Self-training (pseudo-labeling family, prior-free)
python src/train_self_training.py --labeled-frac 0.20 --split-seed 42 --seed 42
```

`--split-seed` controls which positives go into P. `--seed` controls model weight initialization and training order. Separating them enables variance decomposition (3 splits x 3 seeds).

## Evaluation

```bash
# With threshold tuned per method on the validation set
python src/evaluate.py --checkpoint checkpoints/supervised_seed42.pt --tune-threshold

# With fixed threshold (not recommended for PU methods)
python src/evaluate.py --checkpoint checkpoints/supervised_seed42.pt
```

Reports F1 score (at tuned threshold), AUROC, AUPRC, VDS (miss rate at FPR <= 0.5%), pairwise accuracy on matched vuln/patched pairs, and per-CWE recall breakdown.

## Key design decisions

- CodeBERT embeddings are extracted once and frozen. All experiments run on these fixed vectors with lightweight MLPs. This makes the full experiment sweep (~100 configurations) feasible on a laptop in about an hour.
- All four methods use the same MLP architecture (768->256->1), the same optimizer (Adam, lr=1e-3), the same early stopping criterion (validation F1, patience=5), and the same max epochs (30).
- F1 threshold is tuned per method on the validation set because PU methods produce uncalibrated probabilities. AUROC, AUPRC, VDS, and pairwise accuracy are all threshold-free.
- The nnPU prior (0.0277) is the marginal class prior P(Y=1) in the full training set, which is the correct value for the nnPU formulation regardless of how much of P is labeled.

## Dataset

PrimeVul v0.1 (Ding et al., ICSE 2025). 175,797 training functions (4,862 vulnerable, 2.77% class prior), 23,948 validation, 24,788 test. Chronological splits. 100+ CWE types.

Original source: https://github.com/DLVulDet/PrimeVul

## License

Code: MIT. Data: MIT (same as PrimeVul).

# Here Be Dragons: When Does PU Learning Work for Vulnerability Detection?

A systematic investigation of positive-unlabeled learning on code representations, using software vulnerability detection as a testbed.

STAT 3106 Applied Machine Learning, Spring 2026.

## The question

Security teams confirm a handful of vulnerabilities but leave most of their codebase unreviewed. This creates a positive-unlabeled (PU) setting: a small set of known-vulnerable functions and a large pool of unlabeled code with no confirmed negatives. I investigate how many labeled positives PU learning actually needs, whether the two main families of PU methods (risk estimators and pseudo-labeling) break down at different points, and which vulnerability types are hardest to detect with limited labels.

## Project structure

```
src/
  model.py                       MLP classifier (768-d input, 256 hidden, 1 output)
  extract_embeddings.py          Extract CodeBERT [CLS] embeddings (run on Colab)
  extract_embeddings_vulberta.py Extract VulBERTa [CLS] embeddings (run on Colab)
  train_supervised.py            Supervised baseline (upper bound)
  train_naive_pu.py              Naive PU baseline (lower bound)
  train_nnpu.py                  nnPU, risk estimator family
  train_self_training.py         Self-training, pseudo-labeling family
  losses.py                      nnPU loss function
  evaluate.py                    F1, AUROC, AUPRC, VDS, pairwise accuracy, per-CWE
  training.py                    Shared training utilities used by notebooks

notebooks/                       Reproduces every blog result (run in order)
  01_data_exploration.ipynb
  02_baseline_comparison.ipynb
  03_labeling_sweep.ipynb
  04_prior_sensitivity.ipynb
  05_ablations.ipynb
  06_cwe_analysis.ipynb
  07_generate_blog_figures.ipynb

data/
  prepare_pu_splits.py             Generate PU splits at any labeling fraction
  raw/PrimeVul_v0.1/               Raw JSONL files (not in git, see below)
  processed/embeddings/            CodeBERT .npz files (not in git, see below)
  processed/embeddings_vulberta/   VulBERTa .npz files (not in git, see below)
  processed/frac*_seed*/           Generated PU splits

blog/
  index.html               The deliverable (self-contained HTML, no dependencies)

checkpoints/               Saved model weights (not in git)
experiments/logs/          Experiment result CSVs
figures/                   Generated figures for the blog
quality_reports/           Automated review reports
```

## Setup

### Quick start

If you just want to reproduce everything, do step 1 below to set up the Python environment, then open `notebooks/00_reproduce_all.ipynb` and run all cells. The first two cells download the data from HuggingFace into the expected layout and generate the PU splits; the rest of the notebook reproduces every figure and result in the blog. You can skip steps 2 and 3 entirely.

The manual setup below is for users who want to run the training scripts from the command line, or who want to understand the layout before running anything.

### 1. Python environment

Requires Python 3.12 (PyTorch does not support 3.13 yet).

```bash
pyenv virtualenv 3.12.3 stat3106
pyenv local stat3106
pip install -r requirements.txt
```

### 2. Get the data

The raw PrimeVul data and pre-extracted embeddings (both CodeBERT and VulBERTa) are hosted on HuggingFace because they are too large for git (~2.4GB total).

Dataset page: <https://huggingface.co/datasets/db-d2/primevul-codebert-embeddings> (the name reflects the original CodeBERT-only release; VulBERTa was added later in a `vulberta/` subfolder on the same dataset).

```bash
pip install huggingface_hub
hf download db-d2/primevul-codebert-embeddings --repo-type dataset --local-dir data/hf_download

# Move into expected locations
mkdir -p data/processed/embeddings data/processed/embeddings_vulberta data/raw/PrimeVul_v0.1
cp data/hf_download/*.npz data/processed/embeddings/
cp data/hf_download/vulberta/*.npz data/processed/embeddings_vulberta/
cp data/hf_download/raw/*.jsonl data/raw/PrimeVul_v0.1/
rm -rf data/hf_download
```

If you want to re-extract embeddings yourself, upload `src/extract_embeddings.py` (CodeBERT) or `src/extract_embeddings_vulberta.py` (VulBERTa) to Google Colab with a GPU runtime and run it against the raw JSONL files. Each takes about 23 minutes on an A100.

#### Switching between encoders

Training scripts read embeddings from the path in the `EMBEDDING_DIR` environment variable, defaulting to `data/processed/embeddings` (CodeBERT). To run any experiment on VulBERTa instead:

```bash
EMBEDDING_DIR=data/processed/embeddings_vulberta python src/train_nnpu.py --labeled-frac 0.20 --split-seed 42 --seed 42 --prior 0.0277
```

The blog reports CodeBERT as the primary representation and uses VulBERTa as a cross-encoder replication check.

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

## Reproducing the blog end-to-end

The notebooks in `notebooks/` reproduce every figure and result in the blog. Run them in numerical order; each writes its CSV outputs to `experiments/logs/` and figures to `blog/images/`.

```bash
# Run notebook 00 to handle data download + splits, or do Setup steps 2-3 manually first
jupyter lab notebooks/
```

| Notebook | What it does | Outputs |
|---|---|---|
| 00_reproduce_all | One-click reproduction. Downloads data, generates PU splits, then runs the contents of 01-07 inline | Everything below |
| 01_data_exploration | CWE distribution, embedding norms, UMAP comparison | 3 figures |
| 02_baseline_comparison | Supervised vs nnPU vs naive PU vs self-training at 20% labeling | exp1_baseline.csv, 3 figures |
| 03_labeling_sweep | Performance vs labeled fraction (2% to 80%) on CodeBERT and VulBERTa | exp2 CSVs, 2 figures |
| 04_prior_sensitivity | nnPU under prior misspecification (0.5x to 2x) | exp3_prior.csv, 1 figure |
| 05_ablations | Self-training iteration trace and naive-PU negative-selection ablation | exp4 + exp5 CSVs, 2 figures |
| 06_cwe_analysis | Per-CWE recall comparison (memory safety, logic/semantic, concurrency) | exp6_cwe.csv, 1 figure |
| 07_generate_blog_figures | Final figure styling and copy to `blog/images/` | 12 PNGs |

Notebooks 02-06 import shared training code from `src/training.py`. Notebooks 03 and 06 set `EMBEDDING_DIR=data/processed/embeddings_vulberta` partway through to add the VulBERTa replication runs.

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

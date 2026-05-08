# Data

The data files are too large for git. They are hosted on HuggingFace:

https://huggingface.co/datasets/db-d2/primevul-codebert-embeddings

## Quick setup

```bash
# Install the HF CLI if you don't have it
pip install huggingface_hub

# Download everything
hf download db-d2/primevul-codebert-embeddings --repo-type dataset --local-dir data/hf_download

# Move files into place
mkdir -p data/processed/embeddings data/processed/embeddings_vulberta data/raw/PrimeVul_v0.1
cp data/hf_download/*.npz data/processed/embeddings/
cp data/hf_download/vulberta/*.npz data/processed/embeddings_vulberta/
cp data/hf_download/raw/*.jsonl data/raw/PrimeVul_v0.1/
rm -rf data/hf_download
```

## What's in the dataset

**CodeBERT embeddings** (data/processed/embeddings/): Pre-extracted [CLS] token vectors (768-d) from microsoft/codebert-base for all PrimeVul v0.1 functions. General-purpose code understanding model.

**VulBERTa embeddings** (data/processed/embeddings_vulberta/): Same format, extracted from claudios/VulBERTa-mlm. RoBERTa model pretrained specifically on C/C++ vulnerability code. Same functions, same labels, same 768-d vectors, different representation space.

**Raw data** (data/raw/PrimeVul_v0.1/): The original PrimeVul v0.1 JSONL files. Each line is a JSON object with source code, labels, CWE types, and metadata.

All .npz files load with `np.load("train.npz")`. No special flags needed.

## Re-extracting embeddings from scratch

If you want to re-extract instead of using the pre-extracted files:

```bash
# CodeBERT (run on Colab with GPU, ~23 min)
python src/extract_embeddings.py --data-dir data/raw/PrimeVul_v0.1 --out-dir data/processed/embeddings

# VulBERTa (run on Colab with GPU, ~23 min)
python src/extract_embeddings_vulberta.py --data-dir data/raw/PrimeVul_v0.1 --out-dir data/processed/embeddings_vulberta
```

## Generating PU splits

Once the data is in place:

```bash
python data/prepare_pu_splits.py --labeled-frac 0.20 --seed 42
```

This creates `data/processed/frac0.20_seed42/` with P.jsonl, U.jsonl, and metadata.json. Change `--labeled-frac` and `--seed` for the labeling budget sweep.

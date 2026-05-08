"""Extract VulBERTa [CLS] embeddings from PrimeVul functions.

Same pipeline as extract_embeddings.py but uses claudios/VulBERTa-mlm,
a RoBERTa model pretrained on C/C++ code for vulnerability detection.

Usage (run on Colab with GPU):
    python src/extract_embeddings_vulberta.py \
        --data-dir data/raw/PrimeVul_v0.1 \
        --out-dir data/processed/embeddings_vulberta
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MODEL_NAME = "claudios/VulBERTa-mlm"
MAX_LENGTH = 512


def load_jsonl(path):
    records = []
    with open(path) as f:
        for line in f:
            records.append(json.loads(line.strip()))
    return records


def extract_embeddings(records, tokenizer, model, device, batch_size=64):
    model.eval()
    all_embeddings = []
    all_labels = []
    all_cwes = []
    all_idxs = []

    for i in tqdm(range(0, len(records), batch_size), desc="Extracting"):
        batch = records[i : i + batch_size]
        texts = [r["func"] for r in batch]

        tokens = tokenizer(
            texts,
            max_length=MAX_LENGTH,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            outputs = model(**tokens)
            cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()

        all_embeddings.append(cls_embeddings)
        all_labels.extend(r["target"] for r in batch)
        all_cwes.extend(
            r.get("cwe", [None])[0] if r.get("cwe") else "unknown"
            for r in batch
        )
        all_idxs.extend(r["idx"] for r in batch)

    return {
        "embeddings": np.concatenate(all_embeddings, axis=0),
        "labels": np.array(all_labels, dtype=np.int32),
        "cwe_types": np.array(all_cwes, dtype="U20"),
        "idxs": np.array(all_idxs, dtype=np.int64),
    }


def main():
    parser = argparse.ArgumentParser(description="Extract VulBERTa embeddings")
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw/PrimeVul_v0.1"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/processed/embeddings_vulberta"))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    logger.info("Using device: %s", device)

    logger.info("Loading %s", MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device)
    model.eval()

    hidden_size = model.config.hidden_size
    logger.info("Model loaded. Hidden size: %d", hidden_size)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    splits = {
        "train": "primevul_train.jsonl",
        "valid": "primevul_valid.jsonl",
        "test": "primevul_test.jsonl",
        "test_paired": "primevul_test_paired.jsonl",
    }

    for split_name, filename in splits.items():
        path = args.data_dir / filename
        if not path.exists():
            logger.warning("Skipping %s: %s not found", split_name, path)
            continue

        logger.info("Processing %s: %s", split_name, path)
        records = load_jsonl(path)
        logger.info("  Loaded %d records", len(records))

        data = extract_embeddings(records, tokenizer, model, device, batch_size=args.batch_size)

        out_path = args.out_dir / (split_name + ".npz")
        np.savez_compressed(
            out_path,
            embeddings=data["embeddings"],
            labels=data["labels"],
            cwe_types=data["cwe_types"],
            idxs=data["idxs"],
        )
        logger.info("  Saved %s embeddings to %s", data["embeddings"].shape, out_path)


if __name__ == "__main__":
    main()

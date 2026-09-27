from pathlib import Path
import csv
import json
import sqlite3
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, "pilot")
import run_pilot as rp

WORK = Path("artifacts/pilot_10k")
OUT_FILE = Path("output/matching_results.tsv")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    print("=" * 70)
    print("GPU RERANKING & INFERENCE (CAP-500 CANDIDATES -> THRESHOLD 0.99 -> TOP-5)")
    print("=" * 70)

    start_time = time.perf_counter()

    # 1. Load Training Features & Train New GPU Weights
    feat_db = WORK / "pilot_features.sqlite"
    con = sqlite3.connect(feat_db)
    cursor = con.execute("SELECT qrow, target_id, features FROM pair_features")

    queries = []
    with (WORK / "pilot_queries.tsv").open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            queries.append(
                rp.QueryRecord(
                    qrow=int(row["qrow"]),
                    entity_id=row["entity_id"],
                    business_name=row["business_name"],
                    business_address=row["business_address"],
                    country=row["country"],
                    split=row["split"],
                )
            )

    split = {query.qrow: query.split for query in queries}
    qrow_to_entity = {query.qrow: query.entity_id for query in queries}

    truth_by_entity = {}
    with (WORK / "pilot_labels.tsv").open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            truth_by_entity[row["source1_entity_id"]] = {x for x in row["matched_entity_ids"].split(",") if x}

    train_features, train_labels = [], []
    for raw_qrow, raw_tid, blob in cursor:
        qrow = int(raw_qrow)
        tid_str = str(raw_tid)
        s = split.get(qrow)
        feat = rp.unpack_features(blob).copy()

        if s == "train":
            entity_id = qrow_to_entity.get(qrow)
            is_match = 1 if (entity_id and tid_str in truth_by_entity.get(entity_id, set())) else 0
            train_features.append(feat)
            train_labels.append(is_match)

    con.close()

    # Standardize Features
    X = np.vstack(train_features).astype(np.float32)
    y = np.asarray(train_labels, dtype=np.float32)

    mean = X.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = X.std(axis=0, dtype=np.float64).astype(np.float32)
    std[std < 1e-6] = 1.0
    mean[0] = 0.0
    std[0] = 1.0

    pos_count = int(y.sum())
    neg_count = int(len(y) - pos_count)
    sample_weight = np.where(y == 1, neg_count / max(1, pos_count), 1.0).astype(np.float32)
    sample_weight /= sample_weight.mean()

    # Train PyTorch Model on GPU
    Z = (X - mean) / std
    Z[:, 0] = 1.0

    Z_tensor = torch.tensor(Z, dtype=torch.float32, device=DEVICE)
    y_tensor = torch.tensor(y, dtype=torch.float32, device=DEVICE).unsqueeze(1)
    w_tensor = torch.tensor(sample_weight, dtype=torch.float32, device=DEVICE).unsqueeze(1)

    model = nn.Linear(Z.shape[1], 1, bias=False).to(DEVICE)
    nn.init.zeros_(model.weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.03, weight_decay=1e-4)

    batch_size = 4096
    epochs = 25
    num_samples = len(Z)

    print(f"Training Logistic Reranker on {DEVICE} across {num_samples:,} samples...")
    for epoch in range(1, epochs + 1):
        permutation = torch.randperm(num_samples, device=DEVICE)
        for start in range(0, num_samples, batch_size):
            indices = permutation[start : start + batch_size]
            xb, yb, wb = Z_tensor[indices], y_tensor[indices], w_tensor[indices]
            optimizer.zero_grad()
            logits = model(xb)
            loss = (nn.functional.binary_cross_entropy_with_logits(logits, yb, reduction="none") * wb).mean()
            loss.backward()
            optimizer.step()

    weights = model.weight.detach().cpu().numpy().ravel().astype(np.float32)
    print("GPU Reranker training complete!")

    # 2. Score Candidates and Export matching_results.tsv
    THRESHOLD = 0.99
    CAP = 5

    candidate_db = WORK / "pilot_features.sqlite"
    con = sqlite3.connect(candidate_db)
    cursor = con.execute("SELECT qrow, target_id, features FROM pair_features")

    preds_by_query = {}
    for raw_qrow, raw_tid, blob in cursor:
        qrow = int(raw_qrow)
        tid_str = str(raw_tid)
        feat = rp.unpack_features(blob).copy()

        z_feat = (feat - mean) / std
        z_feat[0] = 1.0
        logit = float(np.clip(z_feat @ weights, -30.0, 30.0))
        score = 1.0 / (1.0 + np.exp(-logit))

        if score >= THRESHOLD:
            preds_by_query.setdefault(qrow, []).append((tid_str, score))

    con.close()

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with OUT_FILE.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["source1_entity_id", "matched_entity_ids"])
        
        for q in queries:
            entity_id = q.entity_id
            candidates = preds_by_query.get(q.qrow, [])
            candidates.sort(key=lambda x: x[1], reverse=True)
            top_matches = [c[0] for c in candidates[:CAP]]
            writer.writerow([entity_id, ",".join(top_matches)])

    elapsed = time.perf_counter() - start_time
    print(f"\nSuccessfully written predictions to {OUT_FILE} in {elapsed:.2f}s!")

if __name__ == "__main__":
    main()

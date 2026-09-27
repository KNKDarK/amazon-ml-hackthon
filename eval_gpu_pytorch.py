from pathlib import Path
import csv
import json
import sqlite3
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, "pilot")
import run_pilot as rp

WORK = Path("artifacts/pilot_10k")
SEED = 20260925
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_feature_table(db_path: Path) -> str:
    con = sqlite3.connect(db_path)
    tables = [r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table';")]
    con.close()
    if "pair_features" in tables:
        return "pair_features"
    raise RuntimeError(f"No 'pair_features' table found in {db_path}. Found tables: {tables}")

def main():
    print("=" * 70)
    print("FAST GPU PYTORCH EVALUATOR (PURE VECTORIZED IN-MEMORY)")
    print("=" * 70)

    start_time = time.perf_counter()

    feat_db = WORK / "pilot_features.sqlite"
    feat_table = get_feature_table(feat_db)

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
    qids = {q.entity_id for q in queries if q.split == "validation"}

    labels_path = WORK / "pilot_labels.tsv"
    truth = {}

    with labels_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            src = row["source1_entity_id"]
            targets = {x for x in row["matched_entity_ids"].split(",") if x}
            truth[src] = targets

    print(f"Extracting features from {feat_db.name} ('{feat_table}' table)...")
    con_feat = sqlite3.connect(feat_db)
    cursor = con_feat.execute(f"SELECT qrow, target_id, features FROM {feat_table}")

    train_features, train_labels = [], []
    val_records = []

    for raw_qrow, raw_tid, blob in cursor:
        qrow = int(raw_qrow)
        tid_str = str(raw_tid)
        s = split.get(qrow)
        
        feat = rp.unpack_features(blob).copy()

        if s == "train":
            entity_id = qrow_to_entity.get(qrow)
            is_match = 1 if (entity_id and tid_str in truth.get(entity_id, set())) else 0
            train_features.append(feat)
            train_labels.append(is_match)
        elif s == "validation":
            val_records.append((qrow, tid_str, feat))

    con_feat.close()

    X = np.vstack(train_features).astype(np.float32)
    y = np.asarray(train_labels, dtype=np.float32)

    mean = X.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = X.std(axis=0, dtype=np.float64).astype(np.float32)
    std[std < 1e-6] = 1.0
    mean[0] = 0.0
    std[0] = 1.0
    Z = (X - mean) / std
    Z[:, 0] = 1.0

    pos_count = int(y.sum())
    neg_count = int(len(y) - pos_count)
    sample_weight = np.where(y == 1, neg_count / max(1, pos_count), 1.0).astype(np.float32)
    sample_weight /= sample_weight.mean()

    print(f"Loaded {len(Z):,} training pairs on {DEVICE} ({pos_count:,} positive, {neg_count:,} negative)...")

    # GPU PyTorch Training
    Z_tensor = torch.tensor(Z, dtype=torch.float32, device=DEVICE)
    y_tensor = torch.tensor(y, dtype=torch.float32, device=DEVICE).unsqueeze(1)
    w_tensor = torch.tensor(sample_weight, dtype=torch.float32, device=DEVICE).unsqueeze(1)

    linear_layer = nn.Linear(Z.shape[1], 1, bias=False).to(DEVICE)
    nn.init.zeros_(linear_layer.weight)
    optimizer = optim.AdamW(linear_layer.parameters(), lr=0.03, weight_decay=1e-4)

    batch_size = 4096
    epochs = 25
    num_samples = len(Z)

    for epoch in range(1, epochs + 1):
        permutation = torch.randperm(num_samples, device=DEVICE)
        epoch_loss = 0.0
        for start in range(0, num_samples, batch_size):
            indices = permutation[start : start + batch_size]
            xb, yb, wb = Z_tensor[indices], y_tensor[indices], w_tensor[indices]
            optimizer.zero_grad()
            logits = linear_layer(xb)
            bce = nn.functional.binary_cross_entropy_with_logits(logits, yb, reduction="none")
            loss = (bce * wb).mean()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(xb)

        if epoch % 5 == 0 or epoch == epochs:
            print(f"  [GPU Epoch {epoch:02d}/{epochs}] Loss: {epoch_loss / num_samples:.6f}")

    weights = linear_layer.weight.detach().cpu().numpy().ravel().astype(np.float32)

    # In-memory validation candidate scoring
    print(f"\nScoring {len(val_records):,} validation candidates directly in memory...")
    validation_data = {q.entity_id: [] for q in queries if q.split == "validation"}

    for qrow, tid_str, feat in val_records:
        entity_id = qrow_to_entity[qrow]
        z_feat = (feat - mean) / std
        z_feat[0] = 1.0
        logit = float(np.clip(z_feat @ weights, -30.0, 30.0))
        score = 1.0 / (1.0 + np.exp(-logit))
        is_true = int(tid_str in truth.get(entity_id, set()))
        validation_data[entity_id].append((tid_str, score, is_true))

    for entity_id in validation_data:
        validation_data[entity_id].sort(key=lambda x: x[1], reverse=True)

    print("Running policy threshold/cap search...")
    best, trials = rp.optimize_policy(validation_data, truth, qids)

    OUT = Path("artifacts/gpu_linear_model_10k")
    OUT.mkdir(parents=True, exist_ok=True)

    artifact = {
        "model_type": "gpu_linear_logistic",
        "feature_names": list(rp.FEATURE_NAMES),
        "weights": weights.tolist(),
        "feature_mean": mean.tolist(),
        "feature_std": std.tolist(),
        "decision_policy": {
            "threshold": float(best["threshold"]),
            "max_predictions_per_query": int(best["max_predictions_per_query"]),
        },
        "validation": {
            "macro_f05": float(best["macro_f05"]),
            "micro_precision": float(best["micro_precision"]),
            "micro_recall": float(best["micro_recall"]),
        },
    }

    (OUT / "model.json").write_text(
        json.dumps(artifact, indent=2),
        encoding="utf-8",
    )

    print("Saved:", OUT / "model.json")

    elapsed = time.perf_counter() - start_time
    print("\n" + "=" * 70)
    print(f"GPU EVALUATION RESULTS (Completed in {elapsed:.2f}s)")
    print("=" * 70)
    print(json.dumps(best, indent=2))

if __name__ == "__main__":
    main()

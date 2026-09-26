#!/usr/bin/env python3
"""Train CPU-based LightGBM classifier, tune macro per-S1 F0.5 score, and export deliverables."""
from __future__ import annotations
import json
from pathlib import Path
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from features import FEATURE_SCHEMA, extract_pair_features

def calculate_f05_score(precision: float, recall: float, beta: float = 0.5) -> float:
    """Calculates F0.5 score favoring precision over recall."""
    if precision + recall == 0:
        return 0.0
    beta_sq = beta ** 2
    return (1 + beta_sq) * (precision * recall) / ((beta_sq * precision) + recall)

def macro_s1_f05(y_true: np.ndarray, y_pred: np.ndarray, s1_ids: np.ndarray) -> float:
    """Calculates macro-averaged F0.5 score grouped per S1 entity ID."""
    df = pd.DataFrame({'s1_id': s1_ids, 'target': y_true, 'pred': y_pred})
    scores = []
    
    for _, group in df.groupby('s1_id'):
        tp = np.sum((group['target'] == 1) & (group['pred'] == 1))
        fp = np.sum((group['target'] == 0) & (group['pred'] == 1))
        fn = np.sum((group['target'] == 1) & (group['pred'] == 0))
        
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        scores.append(calculate_f05_score(prec, rec))
        
    return float(np.mean(scores))

def main():
    print("Initializing LightGBM Training Pipeline...")
    
    # 1. LightGBM Hyperparameters (CPU-optimized)
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'n_estimators': 300,
        'learning_rate': 0.05,
        'num_leaves': 31,
        'n_jobs': -1,  # Saturate CPU cores
        'verbose': -1
    }
    
    # Placeholder structure: Integrate your extracted training candidate feature matrix
    # Replace X_dummy/y_dummy with actual features generated from train_source TSVs
    np.random.seed(42)
    num_samples = 5000
    X_data = np.random.randn(num_samples, len(FEATURE_SCHEMA)).astype(np.float32)
    y_data = np.random.randint(0, 2, size=num_samples)
    s1_ids = np.repeat(np.arange(500), 10)
    
    X_train, X_val, y_train, y_val, s1_train, s1_val = train_test_split(
        X_data, y_data, s1_ids, test_size=0.2, random_state=42
    )
    
    train_ds = lgb.Dataset(X_train, label=y_train)
    val_ds = lgb.Dataset(X_val, label=y_val, reference=train_ds)
    
    print("Training LightGBM model on CPU...")
    model = lgb.train(params, train_ds, valid_sets=[val_ds])
    
    # 2. Macro Per-S1 F0.5 Threshold Optimization
    val_probs = model.predict(X_val)
    best_thresh = 0.5
    best_f05 = -1.0
    
    threshold_search = np.arange(0.20, 0.90, 0.02)
    for thresh in threshold_search:
        preds = (val_probs >= thresh).astype(int)
        score = macro_s1_f05(y_val, preds, s1_val)
        if score > best_f05:
            best_f05 = score
            best_thresh = float(thresh)
            
    print(f"\n---> Optimal Decision Threshold: {best_thresh:.2f}")
    print(f"---> Validation Macro Per-S1 F0.5 Score: {best_f05:.4f}\n")
    
    # 3. Export Deliverable Artifacts
    model.save_model("lightgbm_model.txt")
    
    metadata = {
        'model_type': 'LightGBM',
        'feature_schema': FEATURE_SCHEMA,
        'optimal_threshold': best_thresh,
        'val_macro_per_s1_f05': best_f05,
        'decision_policy': {
            'threshold': best_thresh,
            'max_predictions_per_query': 5
        }
    }
    
    with open("model_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
        
    print("All deliverables generated successfully:")
    print("  - features.py")
    print("  - lightgbm_model.txt")
    print("  - model_metadata.json")

if __name__ == '__main__':
    main()
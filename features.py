#!/usr/bin/env python3
"""Feature engineering pipeline reusing pilot features and definitions."""
from __future__ import annotations
import numpy as np
from pilot.er_common import PairText, pair_features, FEATURE_NAMES

# Explicit feature schema order expected by model artifact
FEATURE_SCHEMA = list(FEATURE_NAMES)

def extract_pair_features(left_name: str, left_addr: str, left_country: str,
                          right_name: str, right_addr: str, right_country: str) -> np.ndarray:
    """Calculates name and address similarity features for a single entity pair."""
    left = PairText.make(left_name, left_addr, left_country)
    right = PairText.make(right_name, right_addr, right_country)
    return np.array(pair_features(left, right), dtype=np.float32)

def extract_batch_features(pairs_list: list[tuple]) -> np.ndarray:
    """Batch-computes features for a list of (l_name, l_addr, l_country, r_name, r_addr, r_country) tuples."""
    return np.array([
        extract_pair_features(*pair) for pair in pairs_list
    ], dtype=np.float32)

if __name__ == '__main__':
    print(f"Feature Schema ({len(FEATURE_SCHEMA)} features):")
    for idx, name in enumerate(FEATURE_SCHEMA):
        print(f"  [{idx:02d}] {name}")
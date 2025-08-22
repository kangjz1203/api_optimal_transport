# src/anchors.py
from __future__ import annotations
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
from scipy.linalg import svd

def load_anchors(csv_path: str) -> List[Tuple[str, str]]:
    """
    CSV with columns: old_id,new_id
    """
    df = pd.read_csv(csv_path)
    if not {"old_id", "new_id"} <= set(df.columns):
        raise ValueError("anchors CSV must have columns: old_id,new_id")
    return list(df[["old_id", "new_id"]].itertuples(index=False, name=None))

def build_id_index_map(ids_csv: str, id_col: str = "id") -> Dict[str, int]:
    df = pd.read_csv(ids_csv)
    if id_col not in df.columns:
        raise ValueError(f"{ids_csv} must contain column '{id_col}'")
    return {sid: i for i, sid in enumerate(df[id_col].tolist())}

def anchors_to_indices(anchors: List[Tuple[str, str]],
                       old_map: Dict[str, int],
                       new_map: Dict[str, int]) -> List[Tuple[int, int]]:
    idx_pairs = []
    for o, n in anchors:
        if o in old_map and n in new_map:
            idx_pairs.append((old_map[o], new_map[n]))
    return idx_pairs

def procrustes_rotation(E_old: np.ndarray,
                        E_new: np.ndarray,
                        idx_pairs: List[Tuple[int, int]]) -> np.ndarray:
    """
    Orthogonal Procrustes: compute rotation R (d x d)
    """
    if not idx_pairs:
        raise ValueError("No anchor index pairs for Procrustes.")
    X = np.stack([E_old[i] for i, _ in idx_pairs], axis=0)
    Y = np.stack([E_new[j] for _, j in idx_pairs], axis=0)
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)
    M = X.T @ Y
    U, _, Vt = svd(M, full_matrices=False)
    R = U @ Vt
    return R  # use E_old_aligned = E_old @ R

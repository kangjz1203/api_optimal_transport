# utils/metrics.py
from __future__ import annotations
from typing import Iterable, List, Tuple, Dict, Sequence
import numpy as np
from sklearn.metrics import roc_auc_score

def topk_hits(P: np.ndarray,
              pairs: List[Tuple[int, int]],
              ks: Sequence[int] = (1, 5, 10)) -> Dict[int, float]:
    """
    For each (i,j) in pairs, check if j is within top-k indices of row i (descending by P[i]).
    Returns {k: accuracy}.
    """
    m, n = P.shape
    # argsort descending per row can be expensive; compute ranks via argpartition + refine
    scores = {}
    for k in ks:
        k = min(k, n)
        hit = 0
        for i, j in pairs:
            # top-k indices for row i
            idx = np.argpartition(-P[i], kth=k-1)[:k]
            # exact set membership
            if j in idx:
                hit += 1
        scores[k] = hit / max(1, len(pairs))
    return scores

def auc_on_pairs(P: np.ndarray,
                 pairs: List[Tuple[int, int]],
                 negatives_per_row: int = 50,
                 rng: np.random.RandomState | None = None) -> float:
    """
    Build a binary classification set from each anchor row:
    - positive: P[i,j_true]
    - negatives: sample `negatives_per_row` columns j' != j_true
    Compute ROC-AUC over all collected scores.
    """
    if rng is None:
        rng = np.random.RandomState(0)
    m, n = P.shape
    pos_scores = []
    neg_scores = []
    for i, j_true in pairs:
        pos_scores.append(P[i, j_true])
        # sample negatives
        if n > 1:
            choices = np.setdiff1d(np.arange(n), np.array([j_true]), assume_unique=False)
            k = min(negatives_per_row, len(choices))
            js = rng.choice(choices, size=k, replace=False)
            neg_scores.extend(P[i, js].tolist())
    y_true = np.array([1]*len(pos_scores) + [0]*len(neg_scores))
    y_score = np.array(pos_scores + neg_scores)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))

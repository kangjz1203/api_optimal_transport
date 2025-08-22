# src/cost_matrix.py
from __future__ import annotations
import math
from typing import Optional, Tuple

import numpy as np
from scipy import sparse

DEFAULT_BLOCK = 2048  # block size for memory-friendly cosine

def _ensure_l2_normalized(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    nrm = np.linalg.norm(X, axis=1, keepdims=True)
    nrm = np.maximum(nrm, eps)
    return X / nrm

def cosine_cost_blockwise(E_old: np.ndarray,
                          E_new: np.ndarray,
                          block: int = DEFAULT_BLOCK,
                          normalize_inputs: bool = True) -> np.ndarray:
    """
    Compute cost matrix M = 1 - cosine(E_old, E_new) in blocks.
    Returns a dense (m x n) float32 matrix in [0, 2], later you may rescale to [0,1] if needed.
    """
    if normalize_inputs:
        E_old = _ensure_l2_normalized(E_old)
        E_new = _ensure_l2_normalized(E_new)

    m, d1 = E_old.shape
    n, d2 = E_new.shape
    assert d1 == d2, "Embedding dims must match."

    M = np.empty((m, n), dtype=np.float32)
    for i0 in range(0, m, block):
        i1 = min(i0 + block, m)
        # dot product equals cosine since vectors are L2-normalized
        sim = E_old[i0:i1].dot(E_new.T)  # (b, n)
        M[i0:i1] = 1.0 - sim  # [0, 2]
    # Optional clipping for numerical safety
    np.clip(M, 0.0, 2.0, out=M)
    return M

def minmax_scale01(M: np.ndarray) -> np.ndarray:
    """Scale cost to [0,1] per whole matrix (robust baseline)."""
    lo = float(M.min())
    hi = float(M.max())
    if hi <= lo + 1e-12:
        return np.zeros_like(M, dtype=np.float32)
    return ((M - lo) / (hi - lo)).astype(np.float32)

def topk_prune(M: np.ndarray, k: int) -> sparse.csr_matrix:
    """
    Keep per-row k smallest costs → CSR sparse cost matrix.
    """
    m, n = M.shape
    k = min(k, n)
    # Use argpartition to get indices of k smallest per row
    idx_part = np.argpartition(M, kth=k-1, axis=1)[:, :k]  # (m,k)
    rows = np.repeat(np.arange(m), k)
    cols = idx_part.reshape(-1)
    vals = M[np.arange(m)[:, None], idx_part].reshape(-1)
    # Build CSR; then per-row sort columns for nicer locality
    S = sparse.csr_matrix((vals, (rows, cols)), shape=(m, n))
    S.sum_duplicates()
    S.sort_indices()
    return S

def fuse_costs(base_cost: np.ndarray,
               signature_gap: Optional[np.ndarray] = None,
               contract_gap: Optional[np.ndarray] = None,
               incompat: Optional[np.ndarray] = None,
               weights: Tuple[float, float, float, float] = (1.0, 0.5, 0.5, 1.0),
               scale_each_to01: bool = True) -> np.ndarray:
    """
    Early fusion: M = alpha*base + beta*sig + gamma*contract + lambda*incompat
    All inputs must be same shape (m,n). Any missing term is skipped.
    """
    alpha, beta, gamma, lam = weights
    M = np.zeros_like(base_cost, dtype=np.float32)
    def _prep(X):
        if X is None:
            return None
        X = X.astype(np.float32)
        return minmax_scale01(X) if scale_each_to01 else X

    terms = []
    terms.append((alpha, _prep(base_cost)))
    terms.append((beta, _prep(signature_gap)))
    terms.append((gamma, _prep(contract_gap)))
    terms.append((lam,  _prep(incompat)))
    for w, arr in terms:
        if arr is not None and w != 0.0:
            M += w * arr
    return minmax_scale01(M)

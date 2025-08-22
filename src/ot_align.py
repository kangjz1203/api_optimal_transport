# src/ot_align.py
from __future__ import annotations
from typing import Optional, Tuple, Union

import numpy as np
from scipy import sparse
import ot  # POT

def _to_dense(M: Union[np.ndarray, sparse.spmatrix]) -> np.ndarray:
    if sparse.issparse(M):
        return M.toarray()
    return M

def uniform_marginals(m: int, n: int) -> Tuple[np.ndarray, np.ndarray]:
    a = np.full(m, 1.0 / m, dtype=np.float64)
    b = np.full(n, 1.0 / n, dtype=np.float64)
    return a, b

def sinkhorn_align(M: Union[np.ndarray, sparse.spmatrix],
                   a: Optional[np.ndarray] = None,
                   b: Optional[np.ndarray] = None,
                   reg: float = 0.1,
                   unbalanced_tau: Optional[float] = None,
                   num_iter_max: int = 10_000,
                   stop_thr: float = 1e-9) -> np.ndarray:
    """
    Compute transport matrix P using entropic (unbalanced) Sinkhorn.
    - If unbalanced_tau is None → balanced OT
    - Else → unbalanced OT with KL relaxation strength tau
    Returns: P (m,n) float64
    """
    M = _to_dense(M).astype(np.float64)
    m, n = M.shape
    if a is None or b is None:
        a, b = uniform_marginals(m, n)

    if unbalanced_tau is None:
        # Balanced Sinkhorn
        P = ot.sinkhorn(a, b, M, reg=reg, numItermax=num_iter_max, stopThr=stop_thr)
    else:
        # Unbalanced Sinkhorn (allow mass variation)
        P = ot.unbalanced.sinkhorn_unbalanced(
            a, b, M, reg=reg, tau=unbalanced_tau, numItermax=num_iter_max, stopThr=stop_thr
        )
    return P

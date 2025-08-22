# scripts/run_align.py

"""
Usage:
    Calculate cost Matrix M
    -- python scripts/run_align.py cost --block 2048 --scale01
    KNN prune retain top 100 cost
    -- python scripts/run_align.py prune -k 100
    Sinkhorn Optimal Transport
    -- python scripts/run_align.py ot --reg 0.1
    # or unbalanced:
    -- python scripts/run_align.py ot --reg 0.1 --tau 1.0
    Anchor evaluation
    -- python scripts/run_align.py eval --anchors data/code_migration/anchors/anchors_dev.csv --negs 50
"""




from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
from scipy import sparse

from utils.file_io import load_embeddings, save_matrix, load_matrix, load_ids, save_json
from src.cost_matrix import cosine_cost_blockwise, minmax_scale01, topk_prune, fuse_costs
from src.ot_align import sinkhorn_align, uniform_marginals
from src.anchors import load_anchors, build_id_index_map, anchors_to_indices
from utils.metrics import topk_hits, auc_on_pairs

DATA_DIR = Path("data/code_migration")
EMB_DIR = DATA_DIR / "embeddings"
MAP_DIR = DATA_DIR / "mappings"
REP_DIR = DATA_DIR / "reports"

def cmd_cost(args):
    E_old, E_new = load_embeddings(str(EMB_DIR))
    M = cosine_cost_blockwise(E_old, E_new, block=args.block, normalize_inputs=True)
    if args.scale01:
        M = minmax_scale01(M)
    save_matrix(EMB_DIR / "M.npy", M)
    print(f"Saved M.npy shape={M.shape}")

def cmd_prune(args):
    #M refers to the cost matrix

    M = load_matrix(EMB_DIR / "M.npy")
    if sparse.issparse(M):
        raise ValueError("Expected dense M.npy before pruning.")
    S = topk_prune(M, k=args.k)
    save_matrix(EMB_DIR / "M_knn.npz", S)
    print(f"Saved M_knn.npz shape={S.shape}, nnz={S.nnz}, k={args.k}")

def cmd_ot(args):
    # Prefer pruned matrix if exists
    M_path = EMB_DIR / ("M_knn.npz" if (EMB_DIR / "M_knn.npz").exists() else "M.npy")
    M = load_matrix(M_path)
    E_old, E_new = load_embeddings(str(EMB_DIR))
    a, b = uniform_marginals(E_old.shape[0], E_new.shape[0])
    P = sinkhorn_align(M, a=a, b=b, reg=args.reg,
                       unbalanced_tau=args.tau if args.tau > 0 else None)
    np.save(EMB_DIR / "P.npy", P)
    print(f"Saved P.npy shape={P.shape}")

def cmd_eval(args):
    P = np.load(EMB_DIR / "P.npy")
    old_df, new_df = load_ids(EMB_DIR / "old_ids.csv", EMB_DIR / "new_ids.csv")
    old_map = {sid: i for i, sid in enumerate(old_df["id"].tolist())}
    new_map = {sid: i for i, sid in enumerate(new_df["id"].tolist())}

    pairs = load_anchors(args.anchors)
    idx_pairs = anchors_to_indices(pairs, old_map, new_map)
    if not idx_pairs:
        raise ValueError("No valid anchor pairs matched your id tables.")

    # metrics
    hits = topk_hits(P, idx_pairs, ks=(1, 5, 10))
    auc = auc_on_pairs(P, idx_pairs, negatives_per_row=args.negs)

    REP_DIR.mkdir(parents=True, exist_ok=True)
    save_json(REP_DIR / "metrics.json", {"hits@k": hits, "auc": auc})
    print("Evaluation:")
    for k, v in hits.items():
        print(f"  Top-{k}: {v:.4f}")
    print(f"  AUC: {auc:.4f}")

def cmd_all(args):
    # cost
    args2 = argparse.Namespace(block=args.block, scale01=True)
    cmd_cost(args2)
    # prune
    args3 = argparse.Namespace(k=args.k)
    cmd_prune(args3)
    # ot
    args4 = argparse.Namespace(reg=args.reg, tau=args.tau)
    cmd_ot(args4)
    # eval (requires anchors path)
    if args.anchors:
        args5 = argparse.Namespace(anchors=args.anchors, negs=args.negs)
        cmd_eval(args5)
    else:
        print("Skip eval: --anchors not provided")

def main():
    parser = argparse.ArgumentParser(description="Alignment pipeline (Step 2).")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_cost = sub.add_parser("cost", help="build dense cost matrix M")
    p_cost.add_argument("--block", type=int, default=2048)
    p_cost.add_argument("--scale01", action="store_true")
    p_cost.set_defaults(func=cmd_cost)

    p_prune = sub.add_parser("prune", help="kNN prune M to sparse")
    p_prune.add_argument("-k", type=int, default=100)
    p_prune.set_defaults(func=cmd_prune)

    p_ot = sub.add_parser("ot", help="Sinkhorn OT to get P")
    p_ot.add_argument("--reg", type=float, default=0.1)
    p_ot.add_argument("--tau", type=float, default=0.0, help=">0 for unbalanced OT")
    p_ot.set_defaults(func=cmd_ot)

    p_eval = sub.add_parser("eval", help="evaluate P with anchors")
    p_eval.add_argument("--anchors", type=str, required=True)
    p_eval.add_argument("--negs", type=int, default=50)
    p_eval.set_defaults(func=cmd_eval)

    p_all = sub.add_parser("all", help="run cost→prune→ot→eval")
    p_all.add_argument("--block", type=int, default=2048)
    p_all.add_argument("-k", type=int, default=100)
    p_all.add_argument("--reg", type=float, default=0.1)
    p_all.add_argument("--tau", type=float, default=0.0)
    p_all.add_argument("--anchors", type=str, default=None)
    p_all.add_argument("--negs", type=int, default=50)
    p_all.set_defaults(func=cmd_all)

    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()

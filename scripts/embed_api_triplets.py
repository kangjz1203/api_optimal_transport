
# embed_api_triplets.py


"""
Usage:
    python scripts/embed_api_triplets.py \
  --input data/code_migration/triplets/old_to_new.json \
  --output-dir data/code_migration/embeddings \
  --model sentence-transformers/all-MiniLM-L6-v2 \
  --batch-size 16 \
  --io-batch-size 5000


"""


from __future__ import annotations
import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple, Iterable
import orjson
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5"

# project utils (fallbacks if absent)
try:
    from utils.file_io import load_json_or_jsonl
except Exception:
    load_json_or_jsonl = None

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

def safe_get(d: Dict[str, Any], keys: List[str], default: str = "") -> str:
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur if isinstance(cur, str) else default

def build_template(signature: str, description: str, code_slice: str) -> str:
    return "\n".join([
        f"[SIG] {signature.strip()}",
        f"[DESC] {description.strip()}",
        f"[CODE]",
        code_slice.strip()
    ]).strip()

def norm_name(s: str) -> str:
    return (s or "").strip().replace("-", "_").replace(" ", "_").lower()

def pick_representative(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    # Choose the one with the longest code_slice as representative
    best = None
    best_len = -1
    for it in items:
        l = len(it.get("code_slice") or "")
        if l > best_len:
            best = it
            best_len = l
    return best

def dedup_pair_level_incremental(grouped: Dict[Tuple[str,str,str], Dict[str, Any]],
                                 old_items: List[Dict[str, Any]],
                                 new_items: List[Dict[str, Any]]) -> None:
    # Merge a batch of items into global grouped dict
    from collections import defaultdict
    g_old = defaultdict(list)
    g_new = defaultdict(list)
    for it in old_items:
        g_old[it["pair_key"]].append(it)
    for it in new_items:
        g_new[it["pair_key"]].append(it)
    keys = set(g_old.keys()) & set(g_new.keys())
    for k in keys:
        dep, on, nn = k
        o = pick_representative(g_old[k])
        n = pick_representative(g_new[k])
        pair_id = f"{dep}::{on}=>{nn}"
        o = {**o, "id": f"{pair_id}::old_api"}
        n = {**n, "id": f"{pair_id}::new_api"}
        # keep best context if duplicate key already exists
        if k not in grouped:
            grouped[k] = {"old": o, "new": n}
        else:
            prev_o = grouped[k]["old"]
            prev_n = grouped[k]["new"]
            if len(o.get("code_slice") or "") > len(prev_o.get("code_slice") or ""):
                grouped[k]["old"] = o
            if len(n.get("code_slice") or "") > len(prev_n.get("code_slice") or ""):
                grouped[k]["new"] = n

def encode_texts(texts: List[str], model_name: str, batch_size: int, device: str | None = None) -> np.ndarray:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name, device=device)
    # sentence-transformers already supports batching and a tqdm bar
    emb = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=False,
        show_progress_bar=True
    )
    # L2 normalize
    n = np.linalg.norm(emb, axis=1, keepdims=True)
    n = np.maximum(n, 1e-12)
    return emb / n

def save_ids(items: List[Dict[str, Any]], path: Path) -> None:
    meta_cols = ["id", "api_name", "version", "signature", "source_file"]
    df = pd.DataFrame([{k: v for k, v in it.items() if k in meta_cols} for it in items])
    df.to_csv(path, index=False)

def iter_records_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield orjson.loads(line)

def main():
    parser = argparse.ArgumentParser(description="Embed API triples with batched IO and tqdm; pair-level dedup (K=1).")
    parser.add_argument("--input", required=True, help="Triplets JSON/JSONL with evolution_pair records.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--batch-size", type=int, default=64, help="encoder batch size")
    parser.add_argument("--io-batch-size", type=int, default=5000, help="records per IO batch")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--no-enrich-desc", action="store_true")
    args = parser.parse_args()

    in_path = Path(args.input)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    grouped: Dict[Tuple[str,str,str], Dict[str, Any]] = {}

    # streaming read in batches
    is_jsonl = in_path.suffix.lower() == ".jsonl"
    if is_jsonl:
        # unknown total → no total bar; show per-batch bars
        batch: List[Dict[str, Any]] = []
        for rec in iter_records_jsonl(in_path):
            batch.append(rec)
            if len(batch) >= args.io_batch_size:
                process_batch(batch, grouped, enrich_desc=not args.no_enrich_desc)
                batch.clear()
        if batch:
            process_batch(batch, grouped, enrich_desc=not args.no_enrich_desc)
    else:
        # JSON: could be list or {"data": [...]}
        raw = orjson.loads(in_path.read_bytes())
        records = raw["data"] if isinstance(raw, dict) and "data" in raw else raw
        total = len(records)
        for start in tqdm(range(0, total, args.io_batch_size), desc="IO Batches"):
            sub = records[start:start+args.io_batch_size]
            process_batch(sub, grouped, enrich_desc=not args.no_enrich_desc)

    # materialize representatives
    keys = sorted(grouped.keys())
    old_items, new_items = [], []
    for k in keys:
        o = grouped[k]["old"]
        n = grouped[k]["new"]
        old_items.append(o)
        new_items.append(n)

    print(f"[INFO] pair-level dedup done: old={len(old_items)}, new={len(new_items)}")

    # encode
    old_texts = [build_template(it.get("signature",""), it.get("description",""), it.get("code_slice","")) for it in old_items]
    new_texts = [build_template(it.get("signature",""), it.get("description",""), it.get("code_slice","")) for it in new_items]

    print(f"Encoding OLD with {args.model}")
    E_old = encode_texts(old_texts, args.model, args.batch_size, args.device)
    print(f"Encoding NEW with {args.model}")
    E_new = encode_texts(new_texts, args.model, args.batch_size, args.device)

    np.save(out_dir / "E_old.npy", E_old)
    np.save(out_dir / "E_new.npy", E_new)
    save_ids(old_items, out_dir / "old_ids.csv")
    save_ids(new_items, out_dir / "new_ids.csv")

    cfg = {
        "model": args.model,
        "batch_size": args.batch_size,
        "device": args.device,
        "input": str(in_path),
        "num_old": len(old_items),
        "num_new": len(new_items),
        "l2_normalized": True,
        "template": "[SIG] ... [DESC] ... [CODE] ...",
        "pair_level_dedup": "K=1 (representative by longest context)",
        "io_batch_size": args.io_batch_size
    }
    (out_dir / "embedding_config.json").write_bytes(orjson.dumps(cfg, option=orjson.OPT_INDENT_2))

    print(f"\nSaved:")
    print(f"  {out_dir / 'E_old.npy'}  shape={E_old.shape}")
    print(f"  {out_dir / 'E_new.npy'}  shape={E_new.shape}")
    print(f"  {out_dir / 'old_ids.csv'}")
    print(f"  {out_dir / 'new_ids.csv'}")
    print(f"  {out_dir / 'embedding_config.json'}")

def process_batch(records: List[Dict[str, Any]], grouped: Dict[Tuple[str,str,str], Dict[str, Any]], enrich_desc: bool):
    old_items: List[Dict[str, Any]] = []
    new_items: List[Dict[str, Any]] = []
    for rec in records:
        root_id = rec.get("id", "")
        dependency = rec.get("dependency", "")
        typ = rec.get("type", "")
        desc = rec.get("description", "")
        desc_full = (f"{desc}\nDependency: {dependency}\nEvolutionType: {typ}".strip()
                     if enrich_desc else desc)

        pair = rec.get("evolution_pair", {}) or {}
        o = pair.get("old_api", {}) or {}
        n = pair.get("new_api", {}) or {}

        o_sig = (safe_get(o, ["signature_hint"], "") or safe_get(o, ["match", "signature"], ""))
        o_ctx = safe_get(o, ["match", "context"], "")
        n_sig = (safe_get(n, ["signature_hint"], "") or safe_get(n, ["match", "signature"], ""))
        n_ctx = safe_get(n, ["match", "context"], "")

        o_name = o.get("api_name", "")
        n_name = n.get("api_name", "")
        o_ver = o.get("version", "")
        n_ver = n.get("version", "")
        o_src = safe_get(o, ["source_meta", "file_path"], "")
        n_src = safe_get(n, ["source_meta", "file_path"], "")

        key = (dependency, norm_name(o_name), norm_name(n_name))

        old_items.append({
            "id": f"{root_id}::old_api",
            "dependency": dependency,
            "pair_key": key,
            "api_name": o_name,
            "version": o_ver,
            "signature": o_sig,
            "description": desc_full,
            "code_slice": o_ctx,
            "source_file": o_src,
        })
        new_items.append({
            "id": f"{root_id}::new_api",
            "dependency": dependency,
            "pair_key": key,
            "api_name": n_name,
            "version": n_ver,
            "signature": n_sig,
            "description": desc_full,
            "code_slice": n_ctx,
            "source_file": n_src,
        })

    # Merge batch into global grouped dict
    dedup_pair_level_incremental(grouped, old_items, new_items)

if __name__ == "__main__":
    main()

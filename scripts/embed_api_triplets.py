"""
embed_api_triples.py

Purpose:
  Convert your API triples (signature, description, code_slice) for old/new versions
  into embedding matrices with L2 normalization, ready for downstream OT/FGW alignment.

Input format:
  A JSON or JSONL file consisting of one or more "evolution_pair" records like the example you shared.
  The script will extract per-side triples from:
    - signature:   record['evolution_pair'][side]['match']['signature'] or fallback to ['signature_hint']
    - description: record['description'] (optionally enriched with dependency/type)
    - code_slice:  record['evolution_pair'][side]['match']['context']

Output:
  - <output_dir>/E_old.npy  (m × d)
  - <output_dir>/E_new.npy  (n × d)
  - <output_dir>/old_ids.csv, <output_dir>/new_ids.csv  (row index ↔ API id metadata)
  - <output_dir>/embedding_config.json (model + settings)
  - prints matrix shapes

Dependencies:
  pip install -U sentence-transformers transformers numpy pandas tqdm orjson

Run:
  python embed_api_triplets.py \
  --input path/to/pairs.json \
  --output-dir outputs/emb_step1 \
  --model intfloat/e5-base-v2 \
  --batch-size 64 \
  --max-length 512
"""

from __future__ import annotations
import argparse
import json
import orjson
from pathlib import Path
from typing import Any, Dict, List, Tuple

from utils.file_io import load_json_or_jsonl

import numpy as np
import pandas as pd
from tqdm import tqdm

# You can swap to another sentence embedding model (text/code-friendly).
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"   # alternatives: "BAAI/bge-base-en-v1.5", "intfloat/e5-base-v2"



def safe_get(d: Dict[str, Any], keys: List[str], default: str = "") -> str:
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur if isinstance(cur, str) else default

def build_template(signature: str, description: str, code_slice: str) -> str:
    # Minimal, robust template for a single-encoder baseline
    # Keep it plain text (no markdown) to avoid token overhead.
    parts = [
        f"[SIG] {signature.strip()}",
        f"[DESC] {description.strip()}",
        f"[CODE]\n{code_slice.strip()}",
    ]
    return "\n".join(parts).strip()

def extract_triples(
    records: List[Dict[str, Any]],
    enrich_desc: bool = True
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Return two lists (old_items, new_items).
    Each item: {
        "id": f"{root_id}::old" or "::new",
        "api_name": str,
        "version": str,
        "signature": str,
        "description": str,
        "code_slice": str,
        "template": str,
        "source_file": str,
    }
    """
    old_items, new_items = [], []
    for rec in records:
        root_id = rec.get("id", "")
        dependency = rec.get("dependency", "")
        typ = rec.get("type", "")
        desc = rec.get("description", "")
        if enrich_desc:
            # Light enrichment for extra signal; safe, short, generalizable.
            desc_full = f"{desc}\nDependency: {dependency}\nEvolutionType: {typ}".strip()
        else:
            desc_full = desc

        pair = rec.get("evolution_pair", {})
        for side in ("old_api", "new_api"):
            node = pair.get(side, {}) or {}
            match = node.get("match", {}) or {}
            version = node.get("version", "")
            api_name = node.get("api_name", "")

            signature = (
                safe_get(node, ["signature_hint"], "")
                or safe_get(match, ["signature"], "")
            )
            code_slice = safe_get(match, ["context"], "")
            source_file = safe_get(node, ["source_meta", "file_path"], "")

            template = build_template(signature, desc_full, code_slice)
            item = {
                "id": f"{root_id}::{side}",
                "api_name": api_name,
                "version": version,
                "signature": signature,
                "description": desc_full,
                "code_slice": code_slice,
                "template": template,
                "source_file": source_file,
            }
            if side == "old_api":
                old_items.append(item)
            else:
                new_items.append(item)

    return old_items, new_items

def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    return x / norms

def encode_texts(
    texts: List[str],
    model_name: str,
    max_length: int,
    batch_size: int,
    device: str | None = None
) -> np.ndarray:
    """
    Encode texts with sentence-transformers (CLS pooling).
    """
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name, device=device)
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=False,   # we'll normalize ourselves for consistency
        show_progress_bar=True
    )
    # Some models support truncation config; sentence-transformers handles it internally.
    # If you need stricter control, you can switch to raw HF and handle tokenization manually.
    return l2_normalize(embeddings)

def save_ids(items: List[Dict[str, Any]], path: Path) -> None:
    meta_cols = ["id", "api_name", "version", "signature", "source_file"]
    df = pd.DataFrame([{k: v for k, v in it.items() if k in meta_cols} for it in items])
    df.to_csv(path, index=False)

def main():
    parser = argparse.ArgumentParser(description="Embed API triples (old/new) into matrices.")
    parser.add_argument("--input", type=str, required=True, help="Path to JSON or JSONL with evolution_pair records.")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save outputs.")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help=f"HuggingFace model name (default: {DEFAULT_MODEL})")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=512, help="Best-effort; sentence-transformers handles internally.")
    parser.add_argument("--device", type=str, default=None, help="e.g., 'cuda', 'cuda:0', or leave None for auto")
    parser.add_argument("--no-enrich-desc", action="store_true", help="Disable enrichment with dependency/type in description.")
    args = parser.parse_args()

    input_path = Path(args.input)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    records = load_json_or_jsonl(input_path)
    old_items, new_items = extract_triples(records, enrich_desc=(not args.no_enrich_desc))

    # Build text lists
    old_texts = [it["template"] for it in old_items]
    new_texts = [it["template"] for it in new_items]

    # Encode
    print(f"Encoding OLD side with model: {args.model}")
    E_old = encode_texts(old_texts, args.model, args.max_length, args.batch_size, args.device)
    print(f"Encoding NEW side with model: {args.model}")
    E_new = encode_texts(new_texts, args.model, args.max_length, args.batch_size, args.device)

    # Save embeddings
    np.save(out_dir / "E_old.npy", E_old)
    np.save(out_dir / "E_new.npy", E_new)

    # Save ids
    save_ids(old_items, out_dir / "old_ids.csv")
    save_ids(new_items, out_dir / "new_ids.csv")

    # Minimal config snapshot
    cfg = {
        "model": args.model,
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "device": args.device,
        "input": str(input_path),
        "num_old": len(old_items),
        "num_new": len(new_items),
        "l2_normalized": True,
        "template_format": "[SIG] ... [DESC] ... [CODE] ...",
        "desc_enriched": not args.no_enrich_desc,
    }
    (out_dir / "embedding_config.json").write_bytes(orjson.dumps(cfg, option=orjson.OPT_INDENT_2))

    print(f"\nSaved:")
    print(f"  {out_dir / 'E_old.npy'}  shape={E_old.shape}")
    print(f"  {out_dir / 'E_new.npy'}  shape={E_new.shape}")
    print(f"  {out_dir / 'old_ids.csv'}")
    print(f"  {out_dir / 'new_ids.csv'}")
    print(f"  {out_dir / 'embedding_config.json'}")

if __name__ == "__main__":
    main()

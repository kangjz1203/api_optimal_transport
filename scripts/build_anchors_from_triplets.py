# scripts/build_anchors_from_triplets.py
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List
from utils.file_io import load_json_or_jsonl

# def load_records(path: Path) -> List[Dict[str, Any]]:
#     """Load JSON or JSONL; return a list of dicts."""
#     if not path.exists():
#         raise FileNotFoundError(f"Input not found: {path}")
#     if path.suffix.lower() == ".jsonl":
#         records = []
#         with path.open("r", encoding="utf-8") as f:
#             for line in f:
#                 line = line.strip()
#                 if line:
#                     records.append(json.loads(line))
#         return records
#     # .json
#     obj = json.loads(path.read_text(encoding="utf-8"))
#     return obj if isinstance(obj, list) else [obj]

def ensure_dir(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)

def main():
    ap = argparse.ArgumentParser(
        description="Build anchors_dev.csv from triplet/evolution_pair JSON."
    )
    ap.add_argument(
        "--input",
        type=str,
        default="data/code_migration/triplets/old_to_new.json",
        help="Path to JSON or JSONL file with evolution_pair records.",
    )
    ap.add_argument(
        "--output",
        type=str,
        default="data/code_migration/anchors/anchors_dev.csv",
        help="Output CSV path (will be created).",
    )
    ap.add_argument(
        "--dedup",
        action="store_true",
        help="Deduplicate identical (old_id,new_id) pairs.",
    )
    args = ap.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)

    # records = load_records(in_path)
    records = load_json_or_jsonl(in_path)
    rows = []
    for rec in records:
        rid = rec.get("id")
        if not rid:
            # skip records without an id; our embedding/eval pipeline needs this field
            continue
        pair = (rec.get("evolution_pair") or {})
        old_api = (pair.get("old_api") or {})
        new_api = (pair.get("new_api") or {})

        old_id = f"{rid}::old_api"
        new_id = f"{rid}::new_api"

        rows.append({
            "old_id": old_id,
            "new_id": new_id,
            # extra columns for human audit (eval只用到old_id/new_id)
            "dependency": rec.get("dependency", ""),
            "type": rec.get("type", ""),
            "old_api_name": old_api.get("api_name", ""),
            "old_version": old_api.get("version", ""),
            "new_api_name": new_api.get("api_name", ""),
            "new_version": new_api.get("version", ""),
        })

    if args.dedup:
        # deduplicate by (old_id, new_id)
        seen = set()
        deduped = []
        for r in rows:
            key = (r["old_id"], r["new_id"])
            if key not in seen:
                seen.add(key)
                deduped.append(r)
        rows = deduped

    ensure_dir(out_path)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "old_id", "new_id",
                "dependency", "type",
                "old_api_name", "old_version",
                "new_api_name", "new_version",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} anchors to {out_path}")

if __name__ == "__main__":
    main()

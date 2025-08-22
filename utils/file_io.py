# utils/file_io.py
from __future__ import annotations

import orjson
from typing import Any, Dict, List, Tuple

import json
import numpy as np
import pandas as pd
from scipy import sparse
from pathlib import Path

def load_embeddings(dir_path: str) -> Tuple[np.ndarray, np.ndarray]:
    p = Path(dir_path)
    E_old = np.load(p / "E_old.npy")
    E_new = np.load(p / "E_new.npy")
    return E_old, E_new

def save_matrix(path, X):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if sparse.issparse(X):
        sparse.save_npz(path, X)
    else:
        np.save(path, X)

def load_matrix(path):
    p = Path(path)
    if p.suffix == ".npz":
        return sparse.load_npz(p)
    return np.load(p)

def load_ids(old_csv: str, new_csv: str):
    old_df = pd.read_csv(old_csv)
    new_df = pd.read_csv(new_csv)
    return old_df, new_df

def save_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def load_json_or_jsonl(path: Path) -> List[Dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        records = []
        with path.open("rb") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                records.append(orjson.loads(line))
        return records
    else:
        with path.open("rb") as f:
            data = orjson.loads(f.read())["data"]
        # Allow single-object or list
        return data if isinstance(data, list) else [data]
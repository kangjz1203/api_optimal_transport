# scripts/extract_code_context.py
# 可批量并行处理 + 成对去重（K=1 单代表）

import ast
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from functools import partial
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

# 输入/输出默认目录
INPUT_DIR = Path("data/code_migration/raw_code")
OUTPUT_DIR = Path("data/code_migration/triplets")

# ----------------- 工具函数 -----------------
def get_decorator_source(node: ast.expr) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Call):
        return get_decorator_source(node.func)
    if isinstance(node, ast.Attribute):
        return f"{get_decorator_source(node.value)}.{node.attr}"
    return "unknown_decorator"

class APIContextFinder(ast.NodeVisitor):
    def __init__(self, target_api_name: str):
        self.target_api_name = target_api_name
        self.matches = []

    def visit_FunctionDef(self, node: ast.FunctionDef):
        if node.name == self.target_api_name:
            precise_signature = format_signature_from_node(node)
            self.matches.append({
                "node": node,
                "kind": "function",
                "signature": precise_signature,
                "match_lineno": node.lineno,
                "def_lineno": node.lineno
            })
        for decorator in node.decorator_list:
            decorator_source = get_decorator_source(decorator)
            if decorator_source.endswith(self.target_api_name):
                self.matches.append({
                    "node": node,
                    "kind": "decorator",
                    "signature": f"@{decorator_source}",
                    "match_lineno": getattr(decorator, "lineno", node.lineno),
                    "def_lineno": node.lineno
                })
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        self.visit_FunctionDef(node)

def format_signature_from_node(node: ast.FunctionDef) -> str:
    args = []
    num_defaults = len(node.args.defaults)
    num_pos_args = len(node.args.args)
    for i, arg in enumerate(node.args.args):
        arg_str = arg.arg
        if arg.annotation:
            arg_str += f": {ast.unparse(arg.annotation)}"
        if i >= num_pos_args - num_defaults:
            default_index = i - (num_pos_args - num_defaults)
            default_node = node.args.defaults[default_index]
            arg_str += f"={ast.unparse(default_node)}"
        args.append(arg_str)
    if node.args.vararg:
        vararg_str = f"*{node.args.vararg.arg}"
        if node.args.vararg.annotation:
            vararg_str += f": {ast.unparse(node.args.vararg.annotation)}"
        args.append(vararg_str)
    if node.args.kwonlyargs:
        if not node.args.vararg:
            args.append("*")
        for i, kwarg in enumerate(node.args.kwonlyargs):
            kwarg_str = kwarg.arg
            if kwarg.annotation:
                kwarg_str += f": {ast.unparse(kwarg.annotation)}"
            default_node = node.args.kw_defaults[i]
            if default_node is not None:
                kwarg_str += f"={ast.unparse(default_node)}"
            args.append(kwarg_str)
    if node.args.kwarg:
        kwarg_str = f"**{node.args.kwarg.arg}"
        if node.args.kwarg.annotation:
            kwarg_str += f": {ast.unparse(node.args.kwarg.annotation)}"
        args.append(kwarg_str)
    return f"{node.name}({', '.join(args)})"

# 语料缓存（进程内）
_CORPUS_CACHE: Dict[str, Optional[List[Dict[str, Any]]]] = {}

def load_corpus_file(corpus_path: Path) -> Optional[List[Dict[str, Any]]]:
    key = str(corpus_path)
    if key in _CORPUS_CACHE:
        return _CORPUS_CACHE[key]
    if not corpus_path.exists():
        print(f"[WARN] corpus not found: {corpus_path}")
        _CORPUS_CACHE[key] = None
        return None
    try:
        data = []
        with open(corpus_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
        _CORPUS_CACHE[key] = data
        return data
    except Exception as e:
        print(f"[ERROR] read corpus {corpus_path}: {e}")
        _CORPUS_CACHE[key] = None
        return None

def find_api_context_in_corpus(corpus_data: Optional[List[Dict[str, Any]]], api_name: str) -> Optional[Dict[str, Any]]:
    if not corpus_data:
        return None
    for item in corpus_data:
        source_code = item.get("source_code")
        if not source_code:
            continue
        try:
            tree = ast.parse(source_code)
            finder = APIContextFinder(target_api_name=api_name)
            finder.visit(tree)
            if finder.matches:
                first_match = finder.matches[0]
                context_code = ast.get_source_segment(source_code, first_match["node"])
                return {
                    "match_info": first_match,
                    "context_code": context_code or "",
                    "file_path": item.get("file_path", "unknown_file")
                }
        except SyntaxError:
            continue
        except Exception as e:
            print(f"[WARN] parse error in {item.get('file_path','')}: {e}")
    return None

def process_api_side(entry: Dict[str, Any], version: str, api_name: str, corpus_base_path: Path) -> Optional[Dict[str, Any]]:
    if not all([version, api_name, entry.get("dependency")]):
        return None
    clean_version = str(version).strip("=<>~^ ")
    rel_path = Path(entry["dependency"]) / f"{clean_version}.jsonl"

    # primary path
    corpus_file_path = corpus_base_path / rel_path
    if not corpus_file_path.exists():
        # fallback path
        alt_base = Path("/Volumes/kjz-SSD/Datasets/VersiCode_Raw/VersiCode_Raw/Corpus/downstream_application_code/version_corpus")
        alt_path = alt_base / rel_path
        if alt_path.exists():
            corpus_file_path = alt_path
        else:
            print(f"[WARN] not found in both corpus dirs: {rel_path}")
            return None

    corpus_data = load_corpus_file(corpus_file_path)
    ctx = find_api_context_in_corpus(corpus_data, api_name)
    if ctx:
        match = ctx["match_info"]
        return {
            "version": clean_version,
            "api_name": api_name,
            "signature_hint": match["signature"],
            "match": {
                "signature": match["signature"],
                "kind": match["kind"],
                "context": ctx["context_code"],
                "match_lineno": match["match_lineno"],
                "def_lineno": match["def_lineno"]
            },
            "source_meta": {"file_path": ctx["file_path"]}
        }
    return None


def norm_name(s: str) -> str:
    s = (s or "").strip()
    return s.replace("-", "_").replace(" ", "_").lower()

def pick_better_side(a: Optional[Dict[str, Any]], b: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if a is None: return b
    if b is None: return a
    la = len(((a.get("match") or {}).get("context") or ""))
    lb = len(((b.get("match") or {}).get("context") or ""))
    return a if la >= lb else b

def group_key(entry: Dict[str, Any], old_api: Dict[str, Any], new_api: Dict[str, Any]) -> Tuple[str, str, str]:
    dep = entry.get("dependency", "")
    o = norm_name((old_api or {}).get("api_name", ""))
    n = norm_name((new_api or {}).get("api_name", ""))
    return (dep, o, n)

# ----------------- 并行 worker：处理一个 entry -----------------
def process_one_entry(entry: Dict[str, Any], corpus_base_path: Path) -> Optional[Dict[str, Any]]:
    old_api_info = process_api_side(entry, entry.get("old_version"), entry.get("old_name"), corpus_base_path)
    new_api_info = process_api_side(entry, entry.get("new_version"), entry.get("new_name"), corpus_base_path)
    if old_api_info and new_api_info:
        return {
            "id": entry.get("id"),
            "dependency": entry.get("dependency"),
            "description": entry.get("description", ""),
            "type": entry.get("type"),
            "evolution_pair": {
                "old_api": old_api_info,
                "new_api": new_api_info
            }
        }
    return None

# ----------------- 主流程 -----------------
def main():
    parser = argparse.ArgumentParser(description="Extract API contexts → triplets; pair-level dedup; parallel & batched.")
    parser.add_argument("--input_file", required=True, help="raw input JSON filename under INPUT_DIR")
    parser.add_argument("--output_file", required=True, help="output JSON filename under OUTPUT_DIR")
    parser.add_argument("--corpus_dir", default="/Volumes/kjz-SSD/Datasets/VersiCode_Raw/VersiCode_Raw/Corpus/library_source_code/version_corpus", help="root directory of versioned corpus")
    parser.add_argument("--batch-size", type=int, default=512, help="number of entries per parallel batch")
    parser.add_argument("--num-workers", type=int, default=max(1, cpu_count()//2), help="parallel processes")
    args = parser.parse_args()

    input_path = INPUT_DIR / args.input_file
    output_path = OUTPUT_DIR / args.output_file
    corpus_base_path = Path(args.corpus_dir)

    if not input_path.exists():
        print(f"[ERROR] input file not found: {input_path}")
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 读取原始数据（支持 {"data":[...]} 或 直接 list ）
    with open(input_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    data_to_process: List[Dict[str, Any]] = raw["data"] if isinstance(raw, dict) and "data" in raw else raw

    # 分批并行处理
    def chunks(lst, size):
        for i in range(0, len(lst), size):
            yield lst[i:i+size]

    all_results: List[Dict[str, Any]] = []
    worker_fn = partial(process_one_entry, corpus_base_path=corpus_base_path)

    for bidx, batch in enumerate(tqdm(chunks(data_to_process, args.batch_size), total=(len(data_to_process)//args.batch_size+1), desc="Batches")):
        print(f"[INFO] Batch {bidx}: size={len(batch)}  workers={args.num_workers}")
        if args.num_workers > 1:
            with Pool(processes=args.num_workers) as pool:
                with tqdm(total=len(batch), desc=f"Batch {bidx}") as pbar:
                    for rec in pool.imap_unordered(worker_fn, batch, chunksize=8):
                        if rec: all_results.append(rec)
                        pbar.update(1)
        else:
            for item in batch:
                rec = worker_fn(item)
                if rec: all_results.append(rec)

    print(f"[INFO] collected {len(all_results)} pairs before dedup")

    # 成对去重（K=1 单代表）
    grouped: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    for rec in all_results:
        pair = rec["evolution_pair"]
        k = group_key(rec, pair.get("old_api"), pair.get("new_api"))
        if k not in grouped:
            grouped[k] = rec
        else:
            a = grouped[k]["evolution_pair"]
            b = pair
            grouped[k]["evolution_pair"]["old_api"] = pick_better_side(a.get("old_api"), b.get("old_api"))
            grouped[k]["evolution_pair"]["new_api"] = pick_better_side(a.get("new_api"), b.get("new_api"))

    deduped = list(grouped.values())
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"count": len(deduped), "data": deduped}, f, indent=2, ensure_ascii=False)

    print(f"[OK] done. original={len(all_results)}  deduped={len(deduped)}  -> {output_path}")

if __name__ == "__main__":
    main()

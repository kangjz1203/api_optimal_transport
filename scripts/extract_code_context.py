# scripts/extract_code_context.py

import ast
import json
import argparse
from pathlib import Path
from tqdm import tqdm
# import os  # <-- 优化：不再需要 os 模块，因为 pathlib 已经处理了所有路径操作

# --- 核心逻辑：AST 解析器 (这部分没有变化) ---
# ... (APIContextFinder, get_decorator_source, etc. 省略以保持简洁) ...

# --- 核心逻辑：AST 解析器 ---
# (当项目变大时，这个类可以移动到 src/context_extractor/ast_parser.py)

# 优化：将目录常量也定义为 Path 对象
INPUT_DIR = Path("data/code_migration/raw_code")
OUTPUT_DIR = Path("data/code_migration/triplets")

def get_decorator_source(node: ast.expr) -> str:
    """从装饰器 AST 节点中获取其源代码字符串，例如 'click.group'"""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Call):
        return get_decorator_source(node.func)
    if isinstance(node, ast.Attribute):
        return f"{get_decorator_source(node.value)}.{node.attr}"
    return "unknown_decorator"


class APIContextFinder(ast.NodeVisitor):
    """
    一个 AST 访问者，用于在源代码中查找特定的 API 定义或用法。
    """

    def __init__(self, target_api_name: str):
        self.target_api_name = target_api_name
        self.matches = []

    def visit_FunctionDef(self, node: ast.FunctionDef):

        # 检查函数名是否匹配
        if node.name == self.target_api_name:
            precise_signature = format_signature_from_node(node)
            self.matches.append({
                "node": node,
                "kind": "function",
                "signature": precise_signature,  # 简化的签名
                "match_lineno": node.lineno,
                "def_lineno": node.lineno
            })

        # 检查装饰器是否匹配
        for decorator in node.decorator_list:
            decorator_source = get_decorator_source(decorator)
            # 检查装饰器是否以目标 API 名称结尾 (例如 a.b.target_api_name)
            if decorator_source.endswith(self.target_api_name):
                self.matches.append({
                    "node": node,  # 我们关心的是被装饰的整个函数
                    "kind": "decorator",
                    "signature": f"@{decorator_source}",
                    "match_lineno": decorator.lineno,
                    "def_lineno": node.lineno  # 被装饰函数的行号
                })

        self.generic_visit(node)  # 继续访问函数内部的嵌套函数

    # 支持异步函数
    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        self.visit_FunctionDef(node)


# --- 全局变量和帮助函数 ---

# 注意：这是一个全局缓存，用于存储已加载的语料库文件，避免重复读取
CORPUS_CACHE = {}


def load_corpus_file(corpus_path: Path):
    """加载并缓存 JSONL 语料库文件"""
    if str(corpus_path) in CORPUS_CACHE:
        return CORPUS_CACHE[str(corpus_path)]

    if not corpus_path.exists():
        print(f"警告：语料库文件不存在: {corpus_path}")
        return None

    try:
        with open(corpus_path, 'r', encoding='utf-8') as f:
            data = [json.loads(line) for line in f]
            CORPUS_CACHE[str(corpus_path)] = data
            return data
    except Exception as e:
        print(f"错误：无法读取或解析语料库文件 {corpus_path}: {e}")
        return None


def find_api_context_in_corpus(corpus_data: list, api_name: str):
    """在已加载的语料库数据中搜索 API 上下文"""
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
                # 简单起见，我们只取第一个匹配项
                first_match = finder.matches[0]
                # 需要 Python 3.8+
                context_code = ast.get_source_segment(source_code, first_match["node"])

                return {
                    "match_info": first_match,
                    "context_code": context_code,
                    "file_path": item.get("file_path", "unknown_file")
                }
        except SyntaxError:
            # 某些源代码片段可能有语法错误，直接跳过
            continue
        except Exception as e:
            print(f"在处理文件时发生未知错误 {item.get('file_path', '')}: {e}")

    return None


def process_api_side(entry, side, version, api_name, corpus_base_path):
    """一个辅助函数，用于处理单个 side (old 或 new) 的上下文提取"""
    if not all([version, api_name, entry.get("dependency")]):
        return None

    clean_version = version.strip("=<>~^ ")
    corpus_file_path = corpus_base_path / entry["dependency"] / f"{clean_version}.jsonl"
    corpus_data = load_corpus_file(corpus_file_path)
    context_result = find_api_context_in_corpus(corpus_data, api_name)

    if context_result:
        match = context_result["match_info"]
        return {
            "version": clean_version,
            "api_name": api_name,
            "signature_hint": match["signature"],
            "match": {
                "signature": match["signature"],
                "kind": match["kind"],
                "context": context_result["context_code"],
                "match_lineno": match["match_lineno"],
                "def_lineno": match["def_lineno"]
            },
            "source_meta": {
                "file_path": context_result["file_path"]
            }
        }
    else:
        print(f"警告：在 {corpus_file_path} 中未找到 API '{api_name}' 的上下文 (id: {entry['id']})")
        return None


def format_signature_from_node(node: ast.FunctionDef) -> str:
    """
    从一个 FunctionDef AST 节点中构建一个完全精确的函数签名字符串，
    包含真实的默认值。
    需要 Python 3.9+
    """
    args = []

    # 1. 处理位置参数和关键字参数
    num_defaults = len(node.args.defaults)
    num_pos_args = len(node.args.args)

    for i, arg in enumerate(node.args.args):
        arg_str = arg.arg
        if arg.annotation:
            arg_str += f": {ast.unparse(arg.annotation)}"

        # 检查并附加真实的默认值
        if i >= num_pos_args - num_defaults:
            default_index = i - (num_pos_args - num_defaults)
            default_node = node.args.defaults[default_index]
            default_val_str = ast.unparse(default_node)
            arg_str += f"={default_val_str}"
        args.append(arg_str)

    # 2. 处理 *args
    if node.args.vararg:
        vararg_str = f"*{node.args.vararg.arg}"
        if node.args.vararg.annotation:
            vararg_str += f": {ast.unparse(node.args.vararg.annotation)}"
        args.append(vararg_str)

    # 3. 处理仅关键字参数
    if node.args.kwonlyargs:
        if not node.args.vararg:
            args.append("*")
        for i, kwarg in enumerate(node.args.kwonlyargs):
            kwarg_str = kwarg.arg
            if kwarg.annotation:
                kwarg_str += f": {ast.unparse(kwarg.annotation)}"

            # 检查并附加真实的默认值
            default_node = node.args.kw_defaults[i]
            if default_node is not None:
                default_val_str = ast.unparse(default_node)
                kwarg_str += f"={default_val_str}"
            args.append(kwarg_str)

    # 4. 处理 **kwargs
    if node.args.kwarg:
        kwarg_str = f"**{node.args.kwarg.arg}"
        if node.args.kwarg.annotation:
            kwarg_str += f": {ast.unparse(node.args.kwarg.annotation)}"
        args.append(kwarg_str)

    return f"{node.name}({', '.join(args)})"

def main(args):
    input_path = INPUT_DIR / args.input_file
    output_path = OUTPUT_DIR / args.output_file
    corpus_base_path = Path(args.corpus_dir)

    if not input_path.exists():
        print(f"错误: 输入文件不存在 {input_path}")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(input_path, 'r', encoding='utf-8') as f:
        data_to_process = json.load(f).get("data", [])[:10000]

    if not data_to_process:
        print(f"警告：输入文件 {input_path} 中没有找到 'data' 字段或 'data' 列表为空。")

    results = []

    for entry in tqdm(data_to_process, desc=f"Processing {input_path.name}"):
        # 分别处理 old 和 new 的信息
        old_api_info = process_api_side(
            entry, "old", entry.get("old_version"), entry.get("old_name"), corpus_base_path
        )
        new_api_info = process_api_side(
            entry, "new", entry.get("new_version"), entry.get("new_name"), corpus_base_path
        )

        # 关键：只有当 old 和 new 两边的上下文都成功找到时，才创建这个配对的记录
        if old_api_info and new_api_info:
            output_entry = {
                "id": entry["id"],
                "dependency": entry["dependency"],
                "description": entry["description"],
                "type": entry.get("type"),
                "evolution_pair": {
                    "old_api": old_api_info,
                    "new_api": new_api_info
                }
            }
            results.append(output_entry)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({"count": len(results), "data": results}, f, indent=2)

    print(f"\n处理完成！{len(results)} 个完整的 API 对已保存到 {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="从源代码语料库中提取 API 上下文，生成三元组信息。")
    # 保持这里的参数名与 main 函数中的使用一致
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="输入的 raw_code JSON 文件名, 例如 'code_editing_old_to_new.json'"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="输出的 triplets JSON 文件名, 例如 'old_to_new.json'"
    )
    parser.add_argument(
        "--corpus_dir",
        type=str,
        default="/Volumes/kjz-SSD/Datasets/VersiCode_Raw/VersiCode_Raw/Corpus/library_source_code/version_corpus",
        help="包含版本化语料库的根目录"
    )

    args = parser.parse_args()
    main(args)
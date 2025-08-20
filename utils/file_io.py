import orjson
from pathlib import Path
from typing import Any, Dict, List, Tuple



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
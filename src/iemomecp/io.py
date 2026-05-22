from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any


def load_label_file(path: str | Path) -> dict[str, Any]:
    """Load one sanitized IEMO-MECP label overlay JSON file."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def iter_pairs(label_doc: dict[str, Any]) -> Iterator[dict[str, Any]]:
    """Yield pair records with split and dialogue identifiers attached."""
    split = str(label_doc["split"])
    for item in label_doc["splits"][split]:
        dialogue_id = str(item["dialogue_id"])
        for pair in item.get("pairs", []):
            row = dict(pair)
            row["split"] = split
            row["dialogue_id"] = dialogue_id
            row["num_utterances"] = int(item["num_utterances"])
            yield row


from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from .io import iter_pairs, load_label_file
from .schema import REQUIRED_PAIR_FIELDS, ROLE_NAMES, ROLE_TO_ID


def validate_label_doc(label_doc: dict[str, Any]) -> list[str]:
    """Return a list of validation errors for one label document."""
    errors: list[str] = []
    split = str(label_doc.get("split", ""))
    if split not in {"train", "valid", "test"}:
        errors.append(f"invalid split: {split!r}")
        return errors
    if split not in label_doc.get("splits", {}):
        errors.append(f"missing splits[{split!r}] payload")
        return errors

    seen: set[tuple[str, int, int]] = set()
    for row in iter_pairs(label_doc):
        missing = REQUIRED_PAIR_FIELDS.difference(row)
        if missing:
            errors.append(f"{row.get('dialogue_id')} missing fields: {sorted(missing)}")
            continue
        dialogue_id = str(row["dialogue_id"])
        target_turn = int(row["target_emotion_turn"])
        cause_turn = int(row["cause_turn"])
        key = (dialogue_id, target_turn, cause_turn)
        if key in seen:
            errors.append(f"{dialogue_id}: duplicate pair t={target_turn}, c={cause_turn}")
        seen.add(key)

        label = str(row["label"])
        label_id = int(row["label_id"])
        if label not in ROLE_TO_ID:
            errors.append(f"{dialogue_id}: unknown label {label!r}")
        elif ROLE_TO_ID[label] != label_id:
            errors.append(f"{dialogue_id}: label/id mismatch {label!r}/{label_id}")
    return errors


def summarize_label_dir(label_dir: str | Path) -> tuple[list[dict[str, int | str]], list[str]]:
    """Validate all split files and return split-level counts plus errors."""
    label_dir = Path(label_dir)
    rows: list[dict[str, int | str]] = []
    errors: list[str] = []
    for split in ("train", "valid", "test"):
        path = label_dir / f"{split}.json"
        doc = load_label_file(path)
        split_errors = validate_label_doc(doc)
        errors.extend(f"{path.name}: {err}" for err in split_errors)
        pairs = list(iter_pairs(doc))
        counts = Counter(row["label"] for row in pairs)
        temporal_exceptions = sum(
            1 for row in pairs if int(row["cause_turn"]) > int(row["target_emotion_turn"])
        )
        rows.append(
            {
                "split": split,
                "pairs": sum(counts.values()),
                **{role: int(counts[role]) for role in ROLE_NAMES},
                "temporal_exception_pairs": int(temporal_exceptions),
            }
        )
    return rows, errors

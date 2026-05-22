#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from iemomecp.validation import summarize_label_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the sanitized IEMO-MECP label overlay.")
    parser.add_argument("--label-dir", default="data/labels", help="Directory containing train/valid/test JSON labels.")
    parser.add_argument("--write-summary", default="", help="Optional CSV path for split-level counts.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows, errors = summarize_label_dir(args.label_dir)
    for row in rows:
        print(row)
    if args.write_summary:
        path = Path(args.write_summary)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "split",
                    "pairs",
                    "emo_cause",
                    "emo_context",
                    "non_pair",
                    "temporal_exception_pairs",
                ],
            )
            writer.writeheader()
            writer.writerows(rows)
    if errors:
        print("\nValidation errors:", file=sys.stderr)
        for err in errors[:50]:
            print(f"- {err}", file=sys.stderr)
        if len(errors) > 50:
            print(f"... {len(errors) - 50} more", file=sys.stderr)
        return 1
    print("OK: label overlay validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

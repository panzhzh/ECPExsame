#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Placeholder entry point for reconstructing full local split files from "
            "properly obtained IEMOCAP/ConvECPE sources and the public IEMO-MECP label overlay."
        )
    )
    parser.add_argument("--iemocap-root", required=True, help="Local IEMOCAP root obtained from USC SAIL.")
    parser.add_argument("--convecpe-root", required=True, help="Local ConvECPE/ECPEC source data root.")
    parser.add_argument("--label-dir", default="data/labels", help="Public IEMO-MECP label overlay directory.")
    parser.add_argument("--output-dir", default="local_data/iemomecp_full", help="Local output directory.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    for value in (args.iemocap_root, args.convecpe_root, args.label_dir):
        path = Path(value)
        if not path.exists():
            raise FileNotFoundError(path)

    raise NotImplementedError(
        "Full reconstruction is intentionally a local-only step. The public release "
        "currently provides the safe label overlay and diagnostics; this script marks "
        "the expected interface for adding source-data reconstruction without "
        "redistributing IEMOCAP text or media."
    )


if __name__ == "__main__":
    raise SystemExit(main())


#!/usr/bin/env python
from __future__ import annotations

import sys

from iemomecp.models.pair_role_baseline import main


if __name__ == "__main__":
    main(["--model-kind", "roberta", "--no-audio", "--no-video", *sys.argv[1:]])

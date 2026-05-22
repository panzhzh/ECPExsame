"""Utilities for the IEMO-MECP release artifact."""

from .schema import ROLE_NAMES, ROLE_TO_ID
from .io import load_label_file, iter_pairs

__all__ = ["ROLE_NAMES", "ROLE_TO_ID", "load_label_file", "iter_pairs"]


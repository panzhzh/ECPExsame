from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


FALLBACK_FEATURE_DIM = 768


@dataclass(slots=True)
class Utterance:
    turn: int
    speaker: str
    text: str
    emotion: str | None = None
    timecode: str | None = None
    utterance_name: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Dialogue:
    dataset: str
    split: str
    dialogue_id: str
    utterances: list[Utterance]
    emotion_cause_pairs: list[tuple[int, int]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class FeaturePaths:
    audio_pt: Path | None = None
    video_pt: Path | None = None


@dataclass(slots=True)
class LoadedSplit:
    dataset: str
    split: str
    root: Path
    dialogues: list[Dialogue]
    feature_paths: FeaturePaths = field(default_factory=FeaturePaths)
    metadata: dict[str, Any] = field(default_factory=dict)


def load_iemomecp_dataset(
    dataset_name: str | None = None,
    data_root: str | Path | None = None,
    splits: tuple[str, ...] = ("train", "valid", "test"),
) -> dict[str, LoadedSplit]:
    """Load locally reconstructed IEMO-MECP full splits.

    The public release ships label overlays only. This loader expects users to
    reconstruct full split JSON files locally from licensed IEMOCAP and
    ConvECPE/ECPEC resources, typically under ``local_data/iemomecp_full``.
    """
    dataset = dataset_name or "IemoMECP"
    root = _resolve_dataset_root(Path(data_root or "local_data/iemomecp_full"), dataset)
    return {split: load_iemomecp_split(root=root, dataset_name=dataset, split=split) for split in splits}


def load_iemomecp_split(*, root: Path, dataset_name: str, split: str) -> LoadedSplit:
    split = _canonical_split(split)
    path = _split_path(root, split)
    if not path.exists():
        raise FileNotFoundError(
            "Full IEMO-MECP split JSON is required for model training but was not found: "
            f"{path}. See README.md for the external-data preparation boundary."
        )
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError(f"Split JSON must contain a list of dialogues: {path}")

    dialogues: list[Dialogue] = []
    for fallback_index, row in enumerate(rows):
        utterances = [_load_utterance(item, root=root) for item in row.get("utterances", [])]
        source_pairs = [
            (int(target_turn), int(cause_turn))
            for target_turn, cause_turn in row.get("emotion_cause_pairs", [])
            if int(cause_turn) <= int(target_turn)
        ]
        metadata = {
            "doc_id": _maybe_int(row.get("doc_id"), fallback_index + 1),
            "raw_doc_id": str(row.get("raw_doc_id", row.get("dialogue_id", fallback_index + 1))),
        }
        for key in (
            "chunk_id",
            "session",
            "dialogue_type",
            "chunk_start_turn",
            "chunk_end_turn",
        ):
            if key in row:
                metadata[key] = row[key]
        dialogues.append(
            Dialogue(
                dataset=dataset_name,
                split=split,
                dialogue_id=str(row.get("dialogue_id", metadata["raw_doc_id"])),
                utterances=utterances,
                emotion_cause_pairs=source_pairs,
                metadata=metadata,
            )
        )

    return LoadedSplit(
        dataset=dataset_name,
        split=split,
        root=root,
        dialogues=dialogues,
        feature_paths=_resolve_feature_paths(root=root, split=split),
        metadata={"source_file": str(path)},
    )


def dialogue_doc_id(dialogue: Dialogue, fallback_index: int = 0) -> int:
    doc_id = dialogue.metadata.get("doc_id", dialogue.metadata.get("raw_doc_id", dialogue.dialogue_id))
    try:
        return int(doc_id)
    except (TypeError, ValueError):
        return fallback_index + 1


def build_feature_lookup(
    splits: dict[str, LoadedSplit],
    *,
    modality: str,
    default_dim: int = FALLBACK_FEATURE_DIM,
) -> dict[tuple[int, int], np.ndarray]:
    split_maps = {name: load_split_feature_map(split, modality=modality) for name, split in splits.items()}
    feature_dim = infer_feature_dim(split_maps.values(), default_dim=default_dim)
    features: dict[tuple[int, int], np.ndarray] = {}
    for split_name, split in splits.items():
        feature_map = split_maps[split_name]
        for dialogue_index, dialogue in enumerate(split.dialogues):
            doc_id = dialogue_doc_id(dialogue, dialogue_index)
            for utterance in dialogue.utterances:
                features[(doc_id, int(utterance.turn))] = lookup_feature(
                    feature_map,
                    dialogue=dialogue,
                    utterance=utterance,
                    fallback_index=dialogue_index,
                    default_dim=feature_dim,
                )
    return features


def load_split_feature_map(split: LoadedSplit, *, modality: str) -> dict[Any, Any]:
    if modality not in {"audio", "video"}:
        raise ValueError(f"Unsupported modality={modality!r}; expected 'audio' or 'video'.")
    path = split.feature_paths.audio_pt if modality == "audio" else split.feature_paths.video_pt
    if path is None or not path.exists():
        return {}
    import torch

    return torch.load(path, map_location="cpu", weights_only=False)


def lookup_feature(
    feature_map: dict[Any, Any],
    *,
    dialogue: Dialogue,
    utterance: Utterance,
    fallback_index: int,
    default_dim: int = FALLBACK_FEATURE_DIM,
) -> np.ndarray:
    doc_id = dialogue_doc_id(dialogue, fallback_index)
    raw_doc_id = dialogue.metadata.get("raw_doc_id", dialogue.dialogue_id)
    candidates = (
        (doc_id, int(utterance.turn)),
        (str(doc_id), int(utterance.turn)),
        (raw_doc_id, int(utterance.turn)),
        (str(raw_doc_id), int(utterance.turn)),
        (dialogue.dialogue_id, int(utterance.turn)),
        (str(dialogue.dialogue_id), int(utterance.turn)),
        utterance.utterance_name,
    )
    for key in candidates:
        if key in feature_map:
            return as_numpy_feature(feature_map[key], default_dim=default_dim)
    return np.zeros(default_dim, dtype=np.float32)


def as_numpy_feature(value: Any, *, default_dim: int = FALLBACK_FEATURE_DIM) -> np.ndarray:
    if value is None:
        return np.zeros(default_dim, dtype=np.float32)
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    array = np.asarray(value, dtype=np.float32)
    if array.ndim == 0:
        return np.zeros(default_dim, dtype=np.float32)
    return array.reshape(-1)


def infer_feature_dim(feature_maps: Any, *, default_dim: int = FALLBACK_FEATURE_DIM) -> int:
    for feature_map in feature_maps:
        if not feature_map:
            continue
        for value in feature_map.values():
            return int(as_numpy_feature(value, default_dim=default_dim).shape[-1])
    return int(default_dim)


def _load_utterance(item: dict[str, Any], *, root: Path) -> Utterance:
    metadata = dict(item.get("metadata", {}))
    for key in ("audio_path", "video_path"):
        if item.get(key):
            value = Path(str(item[key]))
            metadata[key] = str(value if value.is_absolute() else root / value)
    for key in ("start_sec", "end_sec"):
        if item.get(key) is not None:
            metadata[key] = float(item[key])
    if item.get("original_turn") is not None:
        metadata["original_turn"] = int(item["original_turn"])
    return Utterance(
        turn=int(item["turn"]),
        speaker=str(item.get("speaker", "")),
        text=str(item.get("text", "")),
        emotion=item.get("emotion"),
        timecode=item.get("timecode"),
        utterance_name=item.get("utterance_name"),
        metadata=metadata,
    )


def _canonical_split(split: str) -> str:
    split = str(split).lower()
    return "valid" if split == "dev" else split


def _split_path(root: Path, split: str) -> Path:
    candidates = (
        root / "splits" / f"{split}.json",
        root / f"{split}.json",
    )
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def _resolve_dataset_root(data_root: Path, dataset_name: str) -> Path:
    data_root = data_root.expanduser()
    if (data_root / "splits").exists():
        return data_root
    for candidate in (data_root / dataset_name, data_root / dataset_name.lower(), data_root / "IemoMECP"):
        if (candidate / "splits").exists():
            return candidate
    return data_root


def _resolve_feature_paths(*, root: Path, split: str) -> FeaturePaths:
    audio_path = _first_existing(
        root / "cache" / f"audio_features_{split}.pt",
        root / "features" / f"audio_features_{split}.pt",
        root / f"audio_features_{split}.pt",
    )
    video_path = _first_existing(
        root / "cache" / f"video_features_{split}.pt",
        root / "features" / f"video_features_{split}.pt",
        root / f"video_features_{split}.pt",
    )
    return FeaturePaths(audio_pt=audio_path, video_pt=video_path)


def _first_existing(*paths: Path) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def _maybe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)

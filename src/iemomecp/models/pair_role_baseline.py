from __future__ import annotations

import argparse
import json
import os
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


from iemomecp.data import build_feature_lookup, dialogue_doc_id, load_iemomecp_dataset
from iemomecp.pair_roles import (
    PAIR_ROLE_CAUSE_CLASS_ID,
    PAIR_ROLE_NON_PAIR_CLASS_ID,
    labels_for_dialogue,
    load_pair_role_label_index,
    map_pair_role_label_for_task,
    make_class_weights,
    normalize_pair_role_task,
    pair_role_class_names,
    pair_role_confusion_rows,
    pair_role_turn_metric_dict,
    write_pair_role_predictions_with_scores_csv,
    write_pair_role_predictions_csv,
)


DEFAULT_LABEL_ROOT = Path("data") / "labels"
DEFAULT_DATA_ROOT = Path("local_data/iemomecp_full")
DEFAULT_ROBERTA = "roberta-base"
FALLBACK_FEATURE_DIM = 768
ROBERTA_BASE_NAME = "roberta_base"
WAVLM_BASE_PLUS_NAME = "wavlm_base_plus"
CLIP_VIT_LARGE_PATCH14_NAME = "clip_vit_large_patch14"


@dataclass(frozen=True)
class PairRoleExample:
    text: str
    label: int
    audio: np.ndarray | None
    video: np.ndarray | None
    meta: dict[str, Any]


class PairRoleDataset(Dataset):
    def __init__(
        self,
        *,
        split_name: str,
        split,
        label_index: dict[str, dict[tuple[str, int, int], dict[str, Any]]],
        audio_lookup: dict[tuple[int, int], np.ndarray],
        video_lookup: dict[tuple[int, int], np.ndarray],
        use_audio: bool,
        use_video: bool,
        audio_dim: int,
        video_dim: int,
        pair_role_task: str,
        candidate_text_mode: str = "original",
        mask_candidate_features: bool = False,
    ) -> None:
        self.split_name = split_name
        self.examples: list[PairRoleExample] = []
        self.class_counts: Counter[int] = Counter()
        self.gold_emotion_turns: set[tuple[str, int]] = set()

        for dialogue_index, dialogue in enumerate(split.dialogues):
            doc_id = dialogue_doc_id(dialogue, dialogue_index)
            utterances = {int(item.turn): item for item in dialogue.utterances}
            source_pairs = {
                (int(target_turn), int(cause_turn))
                for target_turn, cause_turn in getattr(dialogue, "emotion_cause_pairs", [])
            }
            for utterance in dialogue.utterances:
                emotion = str(utterance.emotion or "").strip().lower()
                if emotion and emotion not in {"neu", "neutral"}:
                    self.gold_emotion_turns.add((str(dialogue.dialogue_id), int(utterance.turn)))
            for item in labels_for_dialogue(label_index, split=split_name, dialogue=dialogue):
                emotion_turn = int(item["emotion_turn"])
                cause_turn = int(item["cause_turn"])
                emotion_utterance = utterances.get(emotion_turn)
                cause_utterance = utterances.get(cause_turn)
                if emotion_utterance is None or cause_utterance is None:
                    continue
                if normalize_pair_role_task(pair_role_task) == "source_binary":
                    label = 0 if (emotion_turn, cause_turn) in source_pairs else 1
                else:
                    label = map_pair_role_label_for_task(int(item["label"]), pair_role_task)
                text = format_pair_text(emotion_utterance, cause_utterance, candidate_text_mode=candidate_text_mode)
                audio = None
                video = None
                if use_audio:
                    audio = concat_features(audio_lookup, doc_id, emotion_turn, cause_turn, audio_dim, mask_cause=mask_candidate_features)
                if use_video:
                    video = concat_features(video_lookup, doc_id, emotion_turn, cause_turn, video_dim, mask_cause=mask_candidate_features)
                self.examples.append(
                    PairRoleExample(
                        text=text,
                        label=label,
                        audio=audio,
                        video=video,
                        meta={
                            "split": split_name,
                            "dialogue_id": dialogue.dialogue_id,
                            "doc_id": doc_id,
                            "emotion_turn": emotion_turn,
                            "cause_turn": cause_turn,
                        },
                    )
                )
                self.class_counts[label] += 1

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> PairRoleExample:
        return self.examples[index]


class PairRoleCollator:
    def __init__(
        self,
        *,
        tokenizer,
        max_length: int,
        use_audio: bool,
        use_video: bool,
        audio_dim: int,
        video_dim: int,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_length = int(max_length)
        self.use_audio = bool(use_audio)
        self.use_video = bool(use_video)
        self.audio_dim = int(audio_dim)
        self.video_dim = int(video_dim)

    def __call__(self, examples: list[PairRoleExample]) -> dict[str, Any]:
        batch: dict[str, Any] = {}
        if self.tokenizer is not None:
            tokens = self.tokenizer(
                [example.text for example in examples],
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            batch.update(dict(tokens))
        batch["labels"] = torch.tensor([example.label for example in examples], dtype=torch.long)
        if self.use_audio:
            batch["audio"] = torch.tensor(
                np.stack([example.audio if example.audio is not None else np.zeros(self.audio_dim * 2, dtype=np.float32) for example in examples]),
                dtype=torch.float32,
            )
        if self.use_video:
            batch["video"] = torch.tensor(
                np.stack([example.video if example.video is not None else np.zeros(self.video_dim * 2, dtype=np.float32) for example in examples]),
                dtype=torch.float32,
            )
        batch["meta"] = [example.meta for example in examples]
        return batch


class PairRoleModel(nn.Module):
    def __init__(
        self,
        *,
        model_kind: str,
        backbone: str,
        use_audio: bool,
        use_video: bool,
        audio_dim: int,
        video_dim: int,
        feature_hidden_size: int,
        dropout: float,
        num_classes: int,
    ) -> None:
        super().__init__()
        self.model_kind = model_kind
        self.use_audio = bool(use_audio)
        self.use_video = bool(use_video)
        self.text_encoder = None

        if model_kind == "roberta":
            from transformers import AutoModel

            self.text_encoder = AutoModel.from_pretrained(backbone)
            hidden_size = int(self.text_encoder.config.hidden_size)
        else:
            hidden_size = int(feature_hidden_size)

        branches = [hidden_size] if self.text_encoder is not None else []
        self.audio_projector = None
        self.video_projector = None
        if self.use_audio:
            self.audio_projector = nn.Sequential(
                nn.Linear(audio_dim * 2, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            branches.append(hidden_size)
        if self.use_video:
            self.video_projector = nn.Sequential(
                nn.Linear(video_dim * 2, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            branches.append(hidden_size)
        if not branches:
            raise ValueError("At least one text/audio/video branch must be enabled.")

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(sum(branches), int(num_classes))

    def forward(self, *, input_ids=None, attention_mask=None, labels=None, audio=None, video=None, **_: Any) -> dict[str, torch.Tensor]:
        reps = []
        if self.text_encoder is not None:
            output = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
            pooled = getattr(output, "pooler_output", None)
            if pooled is None:
                pooled = output.last_hidden_state[:, 0]
            reps.append(pooled)
        if self.use_audio:
            reps.append(self.audio_projector(audio))
        if self.use_video:
            reps.append(self.video_projector(video))
        logits = self.classifier(self.dropout(torch.cat(reps, dim=-1)))
        return {"logits": logits}


def default_label_paths() -> str:
    return os.pathsep.join(str(DEFAULT_LABEL_ROOT / f"{split}.json") for split in ("train", "valid", "test"))


def baseline_model_name(args: argparse.Namespace) -> str:
    parts = []
    if args.model_kind == "roberta":
        parts.append(ROBERTA_BASE_NAME)
    if args.use_audio:
        parts.append(WAVLM_BASE_PLUS_NAME)
    if args.use_video:
        parts.append(CLIP_VIT_LARGE_PATCH14_NAME)
    return "_".join(parts) if parts else str(args.model_kind)


def infer_dim(feature_lookup: dict[tuple[int, int], np.ndarray]) -> int:
    for value in feature_lookup.values():
        return int(np.asarray(value).reshape(-1).shape[-1])
    return FALLBACK_FEATURE_DIM


def concat_features(
    feature_lookup: dict[tuple[int, int], np.ndarray],
    doc_id: int,
    emotion_turn: int,
    cause_turn: int,
    dim: int,
    *,
    mask_cause: bool = False,
) -> np.ndarray:
    empty = np.zeros(dim, dtype=np.float32)
    first = np.asarray(feature_lookup.get((doc_id, emotion_turn), empty), dtype=np.float32).reshape(-1)
    second = empty if mask_cause else np.asarray(feature_lookup.get((doc_id, cause_turn), empty), dtype=np.float32).reshape(-1)
    if first.shape[0] != dim:
        first = empty
    if second.shape[0] != dim:
        second = empty
    return np.concatenate([first, second]).astype(np.float32)


def format_pair_text(emotion_utterance, cause_utterance, *, candidate_text_mode: str = "original") -> str:
    if candidate_text_mode == "removed":
        candidate_text = "[REMOVED_CONTEXT]"
    elif candidate_text_mode == "empty":
        candidate_text = ""
    elif candidate_text_mode == "original":
        candidate_text = cause_utterance.text
    else:
        raise ValueError(f"Unsupported candidate_text_mode={candidate_text_mode!r}")
    return (
        f"Target utterance {emotion_utterance.speaker}: {emotion_utterance.text}\n"
        f"Candidate utterance {cause_utterance.speaker}: {candidate_text}"
    )


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def move_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}


def pair_role_metrics(golds: list[int], preds: list[int], class_names: list[str]) -> dict[str, Any]:
    labels = list(range(len(class_names)))
    gold_array = np.asarray(golds, dtype=np.int64)
    pred_array = np.asarray(preds, dtype=np.int64)
    tp = int(((gold_array == PAIR_ROLE_CAUSE_CLASS_ID) & (pred_array == PAIR_ROLE_CAUSE_CLASS_ID)).sum())
    fp = int(((gold_array != PAIR_ROLE_CAUSE_CLASS_ID) & (pred_array == PAIR_ROLE_CAUSE_CLASS_ID)).sum())
    fn = int(((gold_array == PAIR_ROLE_CAUSE_CLASS_ID) & (pred_array != PAIR_ROLE_CAUSE_CLASS_ID)).sum())
    pair_precision = tp / (tp + fp) if (tp + fp) else 0.0
    pair_recall = tp / (tp + fn) if (tp + fn) else 0.0
    pair_f1 = 2 * pair_precision * pair_recall / (pair_precision + pair_recall) if (pair_precision + pair_recall) else 0.0
    precision, recall, f1, support = precision_recall_fscore_support(golds, preds, labels=labels, zero_division=0)
    macro = precision_recall_fscore_support(golds, preds, labels=labels, average="macro", zero_division=0)
    weighted = precision_recall_fscore_support(golds, preds, labels=labels, average="weighted", zero_division=0)
    result: dict[str, Any] = {
        "pair_role_eval_scope": "full_pair",
        "pair_precision": float(pair_precision),
        "pair_recall": float(pair_recall),
        "pair_f1": float(pair_f1),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "pred_pairs": int(tp + fp),
        "gold_pairs": int(tp + fn),
        "pair_role_macro_precision": float(macro[0]),
        "pair_role_macro_recall": float(macro[1]),
        "pair_role_macro_f1": float(macro[2]),
        "pair_role_weighted_precision": float(weighted[0]),
        "pair_role_weighted_recall": float(weighted[1]),
        "pair_role_weighted_f1": float(weighted[2]),
        "pair_role_accuracy": float(np.mean(gold_array == pred_array)) if len(golds) else 0.0,
        "pair_role_support": int(len(golds)),
        "pair_role_confusion": pair_role_confusion_rows(golds, preds, class_names),
    }
    if "emo_context" in class_names:
        context_idx = class_names.index("emo_context")
        context_total = int((gold_array == context_idx).sum())
        result["gold_context_pred_as_cause_rate"] = (
            float(((gold_array == context_idx) & (pred_array == PAIR_ROLE_CAUSE_CLASS_ID)).sum() / context_total)
            if context_total
            else 0.0
        )
        result["gold_context_pred_as_non_pair_rate"] = (
            float(((gold_array == context_idx) & (pred_array == PAIR_ROLE_NON_PAIR_CLASS_ID)).sum() / context_total)
            if context_total
            else 0.0
        )
    for index, name in enumerate(class_names):
        result[f"pair_role_{name}_precision"] = float(precision[index])
        result[f"pair_role_{name}_recall"] = float(recall[index])
        result[f"pair_role_{name}_f1"] = float(f1[index])
        result[f"pair_role_{name}_support"] = int(support[index])
    return result


def has_required_result_metrics(payload: dict[str, Any]) -> bool:
    if payload.get("pair_role_task") == "source_binary":
        required = (
            "pair_f1",
            "pair_role_pair_f1",
            "pair_role_non_pair_f1",
            "pair_role_confusion",
        )
        return all(key in payload for key in required)
    required = (
        "emotion_f1",
        "cause_f1",
        "context_f1",
        "pair_role_emo_cause_f1",
        "pair_role_emo_context_f1",
        "pair_role_confusion",
    )
    return all(key in payload for key in required)


@torch.no_grad()
def evaluate(
    model: PairRoleModel,
    loader: DataLoader,
    device: torch.device,
    class_names: list[str],
    *,
    prediction_path: Path | None = None,
    score_prediction_path: Path | None = None,
    run_name: str = "",
    split_name: str = "test",
) -> dict[str, Any]:
    model.eval()
    golds: list[int] = []
    preds: list[int] = []
    pair_records: list[tuple[str, int, int, int, int]] = []
    score_records: list[dict[str, Any]] = []
    total_loss = 0.0
    total_items = 0
    criterion = nn.CrossEntropyLoss()
    for batch in loader:
        batch = move_to_device(batch, device)
        labels = batch["labels"]
        logits = model(**batch)["logits"]
        loss = criterion(logits, labels)
        total_loss += float(loss.item()) * int(labels.numel())
        total_items += int(labels.numel())
        batch_golds = labels.detach().cpu().tolist()
        probs = torch.softmax(logits, dim=-1)
        batch_preds = torch.argmax(logits, dim=-1).detach().cpu().tolist()
        batch_logits = logits.detach().cpu().tolist()
        batch_probs = probs.detach().cpu().tolist()
        golds.extend(batch_golds)
        preds.extend(batch_preds)
        for meta, gold_label, pred_label, item_logits, item_probs in zip(batch["meta"], batch_golds, batch_preds, batch_logits, batch_probs):
            pair_records.append(
                (
                    str(meta["dialogue_id"]),
                    int(meta["emotion_turn"]),
                    int(meta["cause_turn"]),
                    int(gold_label),
                    int(pred_label),
                )
            )
            score_records.append(
                {
                    "dialogue_id": str(meta["dialogue_id"]),
                    "emotion_turn": int(meta["emotion_turn"]),
                    "cause_turn": int(meta["cause_turn"]),
                    "gold_label": int(gold_label),
                    "pred_label": int(pred_label),
                    "logits": item_logits,
                    "probs": item_probs,
                    "scored_by_model": 1,
                }
            )
    metrics = pair_role_metrics(golds, preds, class_names)
    if len(class_names) == 3 and "emo_context" in class_names:
        turn_metrics = pair_role_turn_metric_dict(
            pair_records,
            gold_emotion_turns=getattr(loader.dataset, "gold_emotion_turns", None),
        )
        metrics.update(turn_metrics)
        for role in ("emotion", "cause", "context"):
            for name in ("precision", "recall", "f1", "support", "predicted", "tp", "fp", "fn"):
                metrics[f"{role}_{name}"] = metrics[f"turn_{role}_{name}"]
    else:
        metrics["emotion_f1"] = 0.0
        metrics["cause_f1"] = 0.0
        metrics["context_f1"] = 0.0
    metrics["loss"] = total_loss / max(1, total_items)
    if prediction_path is not None:
        write_pair_role_predictions_csv(prediction_path, pair_records, class_names, run_name=run_name, split=split_name)
        metrics["pair_predictions_csv"] = str(prediction_path)
    if score_prediction_path is not None:
        write_pair_role_predictions_with_scores_csv(score_prediction_path, score_records, class_names, run_name=run_name, split=split_name)
        metrics["pair_predictions_with_scores_csv"] = str(score_prediction_path)
    return metrics


def build_datasets(
    args: argparse.Namespace,
    *,
    candidate_text_mode: str = "original",
    mask_candidate_features: bool = False,
) -> tuple[PairRoleDataset, PairRoleDataset, PairRoleDataset, int, int]:
    splits = load_iemomecp_dataset(dataset_name=args.dataset_name, data_root=args.data_root, splits=("train", "valid", "test"))
    pair_role_task = normalize_pair_role_task(args.pair_role_task)
    label_index = load_pair_role_label_index(
        {
            "dataset_name": args.dataset_name,
            "data_root": args.data_root,
            "pair_role_task": pair_role_task,
            "pair_role_labels_path": args.pair_role_labels_path,
        }
    )
    audio_lookup = build_feature_lookup(splits, modality="audio") if args.use_audio else {}
    video_lookup = build_feature_lookup(splits, modality="video") if args.use_video else {}
    audio_dim = infer_dim(audio_lookup)
    video_dim = infer_dim(video_lookup)
    datasets = {
        split_name: PairRoleDataset(
            split_name=split_name,
            split=split,
            label_index=label_index,
            audio_lookup=audio_lookup,
            video_lookup=video_lookup,
            use_audio=args.use_audio,
            use_video=args.use_video,
            audio_dim=audio_dim,
            video_dim=video_dim,
            pair_role_task=pair_role_task,
            candidate_text_mode=candidate_text_mode,
            mask_candidate_features=mask_candidate_features,
        )
        for split_name, split in splits.items()
    }
    for split_name, dataset in datasets.items():
        if len(dataset) == 0:
            raise ValueError(f"No pair-role examples loaded for split={split_name}")
    return datasets["train"], datasets["valid"], datasets["test"], audio_dim, video_dim


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=True, indent=2, sort_keys=True)


def load_checkpoint(path: Path, model: nn.Module, optimizer, scheduler, device: torch.device) -> tuple[int, float, dict[str, Any] | None]:
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None and checkpoint.get("scheduler") is not None:
        scheduler.load_state_dict(checkpoint["scheduler"])
    return int(checkpoint.get("epoch", -1)) + 1, float(checkpoint.get("best_metric", -1.0)), checkpoint.get("best_valid_metrics")


def _score_file_for_split(base_path: Path, split_name: str) -> Path:
    if split_name == "test":
        return base_path
    return base_path.with_name(f"{base_path.stem}_{split_name}{base_path.suffix}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-kind", choices=("roberta", "audio_mlp", "video_mlp"), required=True)
    parser.add_argument("--backbone", default=None)
    parser.add_argument("--dataset-name", default="IemoMECP")
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--pair-role-labels-path", default=default_label_paths())
    parser.add_argument("--pair-role-task", default="3class")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cuda-index", type=int, default=0)
    parser.add_argument("--use-audio", dest="use_audio", action="store_true", default=True)
    parser.add_argument("--no-audio", dest="use_audio", action="store_false")
    parser.add_argument("--use-video", dest="use_video", action="store_true", default=True)
    parser.add_argument("--no-video", dest="use_video", action="store_false")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--feature-hidden-size", type=int, default=256)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument(
        "--export-scores-only",
        action="store_true",
        help="Load the saved best checkpoint and export probability/logit CSVs without training.",
    )
    parser.add_argument(
        "--context-removal-scores",
        action="store_true",
        help="Also export scores after replacing the candidate utterance/features with an explicit removal mask.",
    )
    parser.add_argument(
        "--score-splits",
        nargs="+",
        choices=("train", "valid", "test"),
        default=["test"],
        help="Splits to export when --export-scores-only is used. Non-test splits are written with a split suffix.",
    )
    parser.add_argument("--score-prediction-name", default="pair_predictions_with_scores.csv")
    parser.add_argument("--removed-score-prediction-name", default="pair_predictions_context_removed_with_scores.csv")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    output_dir = Path(args.output_dir).expanduser().resolve()
    result_path = output_dir / "test_results.json"
    best_path = output_dir / "best.pt"
    checkpoint_path = output_dir / "checkpoint.pt"
    score_path = output_dir / args.score_prediction_name
    removed_score_path = output_dir / args.removed_score_prediction_name
    if result_path.exists() and not args.force:
        with result_path.open("r", encoding="utf-8") as handle:
            existing_payload = json.load(handle)
        needs_score_export = bool(
            args.export_scores_only
            and any(not _score_file_for_split(score_path, split_name).exists() for split_name in args.score_splits)
        )
        needs_removed_export = bool(args.export_scores_only and args.context_removal_scores and not removed_score_path.exists())
        if (
            has_required_result_metrics(existing_payload)
            and (output_dir / "pair_predictions.csv").exists()
            and not needs_score_export
            and not needs_removed_export
        ):
            print(f"SKIP existing result: {result_path}", flush=True)
            return
        if not best_path.exists():
            print(f"RETRAIN missing required metrics and no best checkpoint: {result_path}", flush=True)
        else:
            print(f"UPDATE/export from checkpoint: {result_path}", flush=True)
    else:
        existing_payload = None
    if args.export_scores_only and not best_path.exists():
        raise FileNotFoundError(f"--export-scores-only requires saved best checkpoint: {best_path}")

    if not torch.cuda.is_available() and not args.allow_cpu:
        raise RuntimeError("CUDA is required for this baseline. Pass --allow-cpu only for debugging.")
    device = torch.device(f"cuda:{args.cuda_index}" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)

    if args.backbone is None and args.model_kind == "roberta":
        args.backbone = DEFAULT_ROBERTA
    if args.max_length is None:
        args.max_length = 160

    from transformers import AutoTokenizer, get_linear_schedule_with_warmup

    train_dataset, valid_dataset, test_dataset, audio_dim, video_dim = build_datasets(args)
    tokenizer = AutoTokenizer.from_pretrained(args.backbone, use_fast=False) if args.model_kind == "roberta" else None
    class_names = pair_role_class_names(args.pair_role_task)
    collator = PairRoleCollator(
        tokenizer=tokenizer,
        max_length=args.max_length,
        use_audio=args.use_audio,
        use_video=args.use_video,
        audio_dim=audio_dim,
        video_dim=video_dim,
    )
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=args.num_workers,
        generator=generator,
    )
    valid_loader = DataLoader(valid_dataset, batch_size=args.eval_batch_size, shuffle=False, collate_fn=collator, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.eval_batch_size, shuffle=False, collate_fn=collator, num_workers=args.num_workers)
    score_train_loader = DataLoader(train_dataset, batch_size=args.eval_batch_size, shuffle=False, collate_fn=collator, num_workers=args.num_workers)
    eval_loaders = {"train": score_train_loader, "valid": valid_loader, "test": test_loader}

    model = PairRoleModel(
        model_kind=args.model_kind,
        backbone=args.backbone,
        use_audio=args.use_audio,
        use_video=args.use_video,
        audio_dim=audio_dim,
        video_dim=video_dim,
        feature_hidden_size=args.feature_hidden_size,
        dropout=args.dropout,
        num_classes=len(class_names),
    ).to(device)

    weights = torch.tensor(make_class_weights(train_dataset.class_counts, len(class_names)), dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = max(1, len(train_loader) * args.epochs)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * args.warmup_ratio),
        num_training_steps=total_steps,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    start_epoch = 0
    best_metric = -1.0
    best_valid_metrics = None
    eval_only = bool(args.export_scores_only or (existing_payload is not None and best_path.exists() and not args.force))
    if args.resume and checkpoint_path.exists() and not args.force:
        start_epoch, best_metric, best_valid_metrics = load_checkpoint(checkpoint_path, model, optimizer, scheduler, device)
        print(f"RESUME checkpoint={checkpoint_path} start_epoch={start_epoch} best={best_metric:.4f}", flush=True)

    print(
        "DATA "
        f"model={baseline_model_name(args)} seed={args.seed} audio={args.use_audio} video={args.use_video} "
        f"train={len(train_dataset)} valid={len(valid_dataset)} test={len(test_dataset)} "
        f"counts={dict(train_dataset.class_counts)} class_weights={[round(float(x), 4) for x in weights.detach().cpu().tolist()]}",
        flush=True,
    )

    if not eval_only:
        for epoch in range(start_epoch, args.epochs):
            model.train()
            losses = []
            progress = tqdm(train_loader, desc=f"epoch {epoch + 1}/{args.epochs}", leave=False)
            for batch in progress:
                batch = move_to_device(batch, device)
                labels = batch["labels"]
                logits = model(**batch)["logits"]
                loss = criterion(logits, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                losses.append(float(loss.item()))
                progress.set_postfix(loss=f"{np.mean(losses):.4f}")

            valid_metrics = evaluate(model, valid_loader, device, class_names)
            metric = float(valid_metrics["pair_role_macro_f1"])
            if metric > best_metric:
                best_metric = metric
                best_valid_metrics = valid_metrics
                torch.save(model.state_dict(), best_path)
            torch.save(
                {
                    "epoch": epoch,
                    "best_metric": best_metric,
                    "best_valid_metrics": best_valid_metrics,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "args": vars(args),
                },
                checkpoint_path,
            )
            print(
                f"EPOCH {epoch + 1}/{args.epochs} train_loss={np.mean(losses):.4f} "
                f"valid_macro_f1={valid_metrics['pair_role_macro_f1'] * 100:.2f} "
                f"valid_pair_f1={valid_metrics['pair_f1'] * 100:.2f}",
                flush=True,
            )

    if best_path.exists():
        model.load_state_dict(torch.load(best_path, map_location=device))
    test_metrics = evaluate(
        model,
        test_loader,
        device,
        class_names,
        prediction_path=output_dir / "pair_predictions.csv",
        score_prediction_path=score_path if args.export_scores_only and "test" in args.score_splits else None,
        run_name=output_dir.name,
        split_name="test",
    )
    if args.export_scores_only:
        for split_name in args.score_splits:
            if split_name == "test":
                continue
            split_score_path = _score_file_for_split(score_path, split_name)
            split_metrics = evaluate(
                model,
                eval_loaders[split_name],
                device,
                class_names,
                prediction_path=None,
                score_prediction_path=split_score_path,
                run_name=output_dir.name,
                split_name=split_name,
            )
            test_metrics[f"{split_name}_pair_predictions_with_scores_csv"] = str(split_score_path)
            test_metrics[f"{split_name}_pair_role_macro_f1"] = float(split_metrics.get("pair_role_macro_f1", 0.0))
    if args.export_scores_only and args.context_removal_scores:
        _, _, removed_test_dataset, _, _ = build_datasets(
            args,
            candidate_text_mode="removed",
            mask_candidate_features=True,
        )
        removed_test_loader = DataLoader(
            removed_test_dataset,
            batch_size=args.eval_batch_size,
            shuffle=False,
            collate_fn=collator,
            num_workers=args.num_workers,
        )
        removed_metrics = evaluate(
            model,
            removed_test_loader,
            device,
            class_names,
            prediction_path=None,
            score_prediction_path=removed_score_path,
            run_name=output_dir.name,
            split_name="test_context_removed",
        )
        test_metrics["context_removed_pair_predictions_with_scores_csv"] = str(removed_score_path)
        test_metrics["context_removed_pair_role_macro_f1"] = float(removed_metrics.get("pair_role_macro_f1", 0.0))
        test_metrics["context_removed_pair_role_emo_context_f1"] = float(removed_metrics.get("pair_role_emo_context_f1", 0.0))
        test_metrics["context_removed_gold_context_pred_as_cause_rate"] = float(removed_metrics.get("gold_context_pred_as_cause_rate", 0.0))
        test_metrics["context_removed_gold_context_pred_as_non_pair_rate"] = float(removed_metrics.get("gold_context_pred_as_non_pair_rate", 0.0))
    payload = {
        **test_metrics,
        "model": baseline_model_name(args),
        "model_kind": args.model_kind,
        "backbone": args.backbone,
        "seed": int(args.seed),
        "use_audio": bool(args.use_audio),
        "use_video": bool(args.use_video),
        "best_valid_metrics": best_valid_metrics,
        "pair_role_task": normalize_pair_role_task(args.pair_role_task),
        "pair_role_class_names": list(class_names),
        "train_counts": {class_names[index]: int(train_dataset.class_counts.get(index, 0)) for index in range(len(class_names))},
        "valid_support": int(len(valid_dataset)),
        "test_support": int(len(test_dataset)),
    }
    save_json(result_path, payload)
    print(
        f"WRITE {result_path} macro_f1={test_metrics['pair_role_macro_f1'] * 100:.2f} "
        f"pair_f1={test_metrics['pair_f1'] * 100:.2f}",
        flush=True,
    )


if __name__ == "__main__":
    main()

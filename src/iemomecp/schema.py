from __future__ import annotations

ROLE_NAMES = ("emo_cause", "emo_context", "non_pair")
ROLE_TO_ID = {name: idx for idx, name in enumerate(ROLE_NAMES)}
ID_TO_ROLE = {idx: name for name, idx in ROLE_TO_ID.items()}

DISPLAY_ROLE_NAMES = {
    "emo_cause": "emo-cause",
    "emo_context": "emo-context",
    "non_pair": "non-pair",
}

REQUIRED_PAIR_FIELDS = {
    "target_emotion_turn",
    "cause_turn",
    "label_id",
    "label",
    "source",
    "taxonomy_label",
    "context_subtype_ids",
    "context_subtypes",
}


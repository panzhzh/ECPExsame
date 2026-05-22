# Third-Party Baseline Reproduction Guide

This artifact vendors only the lightweight baselines written for IEMO-MECP:
RoBERTa text-only, WavLM-feature audio-only, CLIP-feature video-only, and
RWC-Fusion. The full HiLo, M3HG, and MECPE-2step implementations are third-party
research codebases with separate dependencies and license terms, so they are not
copied into this release.

## Upstream Repositories

| Family in the paper | Upstream project | URL | Release boundary |
| --- | --- | --- | --- |
| HiLo | Multimodal Emotion-Cause Pair Extraction with Holistic Interaction and Label Constraint | https://github.com/whulacc/hilo | Clone separately; not vendored here. |
| M3HG | Multimodal, Multi-scale, and Multi-type Node Heterogeneous Graph | https://github.com/redifinition/M3HG | Clone separately; not vendored here. |
| MECPE-2step | Multimodal Emotion-Cause Pair Extraction in Conversations | https://github.com/NUSTM/MECPE | Clone separately; not vendored here. |

The paper tables in `data/paper_tables/` include the aggregate results reported
for these families. The instructions below document the data interface we used
to connect those families to IEMO-MECP labels.

## Shared Data Contract

All third-party baselines should be treated as local experiments over the same
IEMO-MECP data contract:

```text
local_data/
  iemomecp_full/
    splits/
      train.json
      valid.json
      test.json
    cache/
      audio_features_train.pt     # optional, if the model consumes audio features
      audio_features_valid.pt
      audio_features_test.pt
      video_features_train.pt     # optional, if the model consumes video features
      video_features_valid.pt
      video_features_test.pt
```

The public label overlay remains in this repository:

```text
data/labels/
  train.json
  valid.json
  test.json
```

The full split JSON files are not redistributed because they contain IEMOCAP
utterance text and media-derived paths. Reconstruct them locally from properly
obtained IEMOCAP and ConvECPE/ECPEC resources as described in `README.md` and
`DATA_TERMS.md`.

## Common Loader Entry Point

Use the release loader when writing adapters for third-party code:

```python
from iemomecp.data import load_iemomecp_dataset
from iemomecp.pair_roles import labels_for_dialogue, load_pair_role_label_index

splits = load_iemomecp_dataset(
    dataset_name="IemoMECP",
    data_root="local_data/iemomecp_full",
)

label_index = load_pair_role_label_index({
    "dataset_name": "IemoMECP",
    "pair_role_task": "3class",
    "pair_role_labels_path": ":".join([
        "data/labels/train.json",
        "data/labels/valid.json",
        "data/labels/test.json",
    ]),
})

for split_name, split in splits.items():
    for dialogue in split.dialogues:
        pair_role_rows = labels_for_dialogue(
            label_index,
            split=split_name,
            dialogue=dialogue,
        )
```

Each `pair_role_rows` item contains:

```text
emotion_turn, cause_turn, label, label_name, taxonomy_label, context_subtypes
```

The canonical three-class labels are:

```text
0 = emo_cause
1 = emo_context
2 = non_pair
```

For source-binary controls, use `pair_role_task="source_binary"` or map the
original `dialogue.emotion_cause_pairs` to `pair` and all other released
candidate pairs to `non_pair`.

## HiLo Adapter Guidance

Clone the upstream project outside this repository:

```bash
mkdir -p external_repos
git clone https://github.com/whulacc/hilo external_repos/hilo
```

HiLo has its own dataset and configuration layer. The cleanest integration is to
add a small adapter at the point where HiLo builds dialogue examples:

1. Load local full splits with `load_iemomecp_dataset(...)`.
2. Load role labels with `load_pair_role_label_index(...)`.
3. For each dialogue, expose utterance text, speakers, emotions, audio/video
   feature ids, and candidate pair labels to HiLo's expected batch format.
4. Evaluate predictions over the released candidate pairs, not over a newly
   generated candidate space.

Do not commit the cloned upstream repo, local split JSONs, feature caches, or
HiLo checkpoints back into this release.

## M3HG Adapter Guidance

Clone the upstream project outside this repository:

```bash
mkdir -p external_repos
git clone https://github.com/redifinition/M3HG external_repos/M3HG
```

M3HG is originally organized around its own multimodal triplet extraction data.
For IEMO-MECP reproduction, use the same adapter pattern:

1. Use `load_iemomecp_dataset(...)` as the dialogue source.
2. Use `labels_for_dialogue(...)` to construct the candidate pair supervision.
3. Keep the train/valid/test split fixed to the released label overlay.
4. If M3HG writes preprocessed pickle caches, keep them under an ignored local
   output directory such as `outputs/m3hg_cache/`.

The released aggregate tables report pair-role evaluation on IEMO-MECP labels;
they are not intended to imply that the upstream M3HG data format is redistributed
here.

## MECPE-2step Adapter Guidance

Clone the upstream project outside this repository:

```bash
mkdir -p external_repos
git clone https://github.com/NUSTM/MECPE external_repos/MECPE
```

MECPE-2step has a two-stage pipeline. For IEMO-MECP reproduction:

1. Convert `local_data/iemomecp_full/splits/*.json` into the upstream step-1
   dialogue input format.
2. Use the released source binary layer for the original pair/non-pair setting,
   or the IEMO-MECP `label_id` values for three-class pair-role evaluation.
3. Ensure step-2 candidate pairs are scored against the released candidate pairs.
4. Store intermediate step-1/step-2 outputs under `outputs/mecpe_2step/`.

Because the upstream project uses a distinct environment and legacy dependencies,
keep its environment separate from the lightweight IEMO-MECP package environment.

## Practical Rule

The reproducibility boundary is:

- This repository provides the public label overlay, aggregate paper tables,
  diagnostic figures, and lightweight baseline code.
- Upstream third-party model repositories must be cloned separately.
- Restricted IEMOCAP text/media and media-derived features must remain local.
- Adapters should use the shared loader contract above so all models evaluate the
  same fixed IEMO-MECP candidate-pair label space.

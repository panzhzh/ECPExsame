# Release Manifest

This file describes the intended public-release boundary for the IEMO-MECP
artifact.

## Public Files

| Path | Status | Notes |
| --- | --- | --- |
| `data/labels/train.json` | include | Sanitized pair-role label overlay. No utterance text or media paths. |
| `data/labels/valid.json` | include | Sanitized pair-role label overlay. No utterance text or media paths. |
| `data/labels/test.json` | include | Sanitized pair-role label overlay. No utterance text or media paths. |
| `data/metadata/label_schema.json` | include | Public role schema. |
| `data/metadata/split_manifest.csv` | include | Split-level label counts and temporal-exception counts. |
| `data/paper_tables/*.csv` | include | Aggregate paper-table/figure data with raw-text CSVs excluded. |
| `src/iemomecp/*` | include | Lightweight loading, validation, metrics, and released baseline utilities. |
| `src/iemomecp/models/pair_role_baseline.py` | include | Our released RoBERTa/WavLM/CLIP/RWC-Fusion pair-role training runner. |
| `scripts/validate_labels.py` | include | Public label-overlay validator. |
| `scripts/generate_figures.py` | include | Recreates paper figures from aggregate CSVs. |
| `scripts/run_roberta.py` | include | Text-only baseline wrapper. |
| `scripts/run_wavlm.py` | include | Audio-feature baseline wrapper. |
| `scripts/run_clip.py` | include | Video-feature baseline wrapper. |
| `scripts/run_rwc_fusion.py` | include | Text+audio+video fusion baseline wrapper. |
| `scripts/launch_baseline_matrix.py` | include | Multi-seed launcher for the released lightweight baseline families. |
| `scripts/build_from_sources.py` | include | Documents the local-only interface for combining external IEMOCAP/ConvECPE data with the label overlay. |

## Excluded Files

| Source | Reason |
| --- | --- |
| Full `splits/*.json` with utterance text | Contains IEMOCAP transcript text and local media paths. |
| IEMOCAP audio/video/motion-capture files | Must be obtained from USC SAIL under the original terms. |
| Feature caches (`*.pt`, extracted audio/video features) | Derived from restricted media and too large for GitHub. |
| Full third-party baseline repositories | Use their upstream releases and license terms; see `docs/THIRD_PARTY_BASELINES.md`. |
| `boundary_example_candidates.csv` | Contains raw target/candidate utterance text. |
| Local run outputs and checkpoints | Large, path-specific, and not needed for label/figure validation. |

## Release Checks Run

```bash
PYTHONPATH=src python scripts/validate_labels.py \
  --label-dir data/labels \
  --write-summary data/metadata/label_counts_check.csv

python scripts/generate_figures.py

rg -n "/scr/user|/home/user|ipanzhzh|target_text|cause_text|audio_path|video_path|raw_text|transcript" \
  release/iemomecp --glob '!paper/figures/**'
```

Expected scan hits are explanatory references, skipped-file metadata, and public
parser/loader field names. Released data files should not contain local absolute
paths or raw utterance text.

## Naming

- GitHub repository: `iemomecp`
- Python package: `iemomecp`
- Paper/dataset display name: `IEMO-MECP`

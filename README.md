# IEMO-MECP 
<a href="https://arxiv.org/abs/2605.25208" target="_blank">
  <img src="https://img.shields.io/badge/Paper-arXiv-red?style=for-the-badge&logo=arxiv" alt="Paper">
</a>

This is the official repo for:

> They Are Not the Same: Direct Causes Are Not Grounded Emotion Explanations

The repository provides a pair-role label overlay and diagnostic code for studying
evaluation validity in binary Emotion-Cause Pair Extraction (ECPE). IEMOCAP is the
underlying multimodal conversation corpus, while ConvECPE/ECPEC provides the
source binary emotion-cause layer. IEMO-MECP adds a role-aware label space over
the same lower-triangular target-candidate pair space:

- `emo-cause`: direct trigger, event, or appraisal evidence for the target emotion.
- `emo-context`: non-triggering discourse evidence that helps interpret the target emotion.
- `non-pair`: insufficient evidence for direct causality or contextual emotional support.

## What Is Included

```text
data/
  labels/                 Sanitized train/valid/test pair-role label overlays.
  metadata/               Label schema, split counts, and release notes.
  paper_tables/           Aggregated CSV files used to reproduce paper figures/tables.
src/iemomecp/             Lightweight loading, validation, and metric utilities.
src/iemomecp/models/      Released lightweight RoBERTa/WavLM/CLIP/RWC-Fusion baselines.
scripts/                  Validation, figure-generation, and baseline-training entry points.
paper/figures/            Output directory for regenerated figures.
```

The included label files are overlays keyed by `split`, `dialogue_id`,
`target_emotion_turn`, and `cause_turn`. They intentionally exclude original
IEMOCAP utterance text, audio, and video.

## What Is Not Included

This repository does not redistribute IEMOCAP text/audio/video or full derived
conversation JSON files containing original utterance text. To run full model
training, obtain the underlying resources from their original sources:

- IEMOCAP: https://sail.usc.edu/iemocap/
- ConvECPE/ECPEC source data/code: https://github.com/SenticNet/ECPEC

See `DATA_TERMS.md` for the data-release boundary.

## Preparing External Data

For label inspection and figure reproduction, no external data is required. The
included label overlay and aggregate CSV files are self-contained.

For full training/evaluation with utterance text or modalities, prepare the
underlying datasets locally:

```text
local_data/
  IEMOCAP/                 # Official IEMOCAP release from USC SAIL.
  ECPEC/                   # ConvECPE/ECPEC source files from the original paper repo.
```

Recommended source steps:

1. Request and download IEMOCAP from USC SAIL:
   https://sail.usc.edu/iemocap/
2. Clone or download the ConvECPE/ECPEC source resources:
   https://github.com/SenticNet/ECPEC
3. Keep both resources under `local_data/` or pass their paths explicitly to scripts.
4. Use `data/labels/*.json` as the IEMO-MECP pair-role overlay keyed by
   `split`, `dialogue_id`, `target_emotion_turn`, and `cause_turn`.

The intended path convention is:

```bash
python scripts/build_from_sources.py \
  --iemocap-root local_data/IEMOCAP \
  --convecpe-root local_data/ECPEC \
  --label-dir data/labels \
  --output-dir local_data/iemomecp_full
```

The anonymous review artifact currently provides the safe label overlay and diagnostics.
The reconstruction entry point documents the expected local interface without
redistributing IEMOCAP text or media.

## Quick Checks

Install in editable mode:

```bash
python -m pip install -e .
```

Validate the included label overlay:

```bash
python scripts/validate_labels.py --label-dir data/labels
```

Regenerate paper figures from the released aggregate CSV files:

```bash
python scripts/generate_figures.py
```

Figures are written to `paper/figures/`.

## Training Baselines

The release includes our lightweight pair-role baseline runner for:

- RoBERTa text-only (`scripts/run_roberta.py`)
- WavLM-feature audio-only (`scripts/run_wavlm.py`)
- CLIP-feature video-only (`scripts/run_clip.py`)
- RWC-Fusion text+audio+video (`scripts/run_rwc_fusion.py`)

Install the optional training dependencies:

```bash
python -m pip install -e ".[train]"
```

Training requires locally reconstructed full split JSON files under
`local_data/iemomecp_full/splits/{train,valid,test}.json`. Audio/video baselines
also expect optional feature caches such as
`local_data/iemomecp_full/cache/audio_features_train.pt` and
`local_data/iemomecp_full/cache/video_features_train.pt`.

Example commands:

```bash
python scripts/run_roberta.py \
  --data-root local_data/iemomecp_full \
  --pair-role-labels-path data/labels/train.json:data/labels/valid.json:data/labels/test.json \
  --pair-role-task 3class \
  --output-dir outputs/roberta_seed42 \
  --seed 42

python scripts/run_rwc_fusion.py \
  --data-root local_data/iemomecp_full \
  --pair-role-labels-path data/labels/train.json:data/labels/valid.json:data/labels/test.json \
  --pair-role-task 3class \
  --output-dir outputs/rwc_fusion_seed42 \
  --seed 42
```

For the original binary-control setting, use the same runner with
`--pair-role-task source_binary`. Full third-party MECPE-style repositories are
not vendored into this artifact; see `docs/THIRD_PARTY_BASELINES.md`.

To launch the included lightweight matrix over three seeds:

```bash
python scripts/launch_baseline_matrix.py \
  --gpus 0,1 \
  --per-gpu 1 \
  --pair-role-task 3class \
  --data-root local_data/iemomecp_full \
  --output-root outputs/pair_role_matrix
```

## Third-Party Baselines

HiLo, M3HG, and MECPE-2step are supported as reproduction targets through a
documented data-loading contract, but their full upstream repositories are not
vendored into this artifact. See `docs/THIRD_PARTY_BASELINES.md` for GitHub
links and adapter guidance.

## Reproducibility Scope

This artifact is designed to support three levels of reproducibility:

1. Label-level checks: verify split counts, role schema, and pair-space consistency.
2. Paper-table checks: inspect the aggregate CSV files used for reported figures/tables.
3. Diagnostic figures: regenerate the released PDF/SVG figures from aggregate CSV files.
4. Lightweight baselines: rerun the released RoBERTa/WavLM/CLIP/RWC-Fusion runner
   after preparing the licensed external data locally.

Full model training requires local access to the underlying IEMOCAP/ConvECPE data
and is intentionally not made possible from redistributed raw dialogue text.

## Citation

If you use this artifact, please cite the paper and the underlying IEMOCAP and
ConvECPE/ECPEC resources. A `CITATION.cff` file is included for the artifact.
```bibtex
@article{pan2026they,
  title={They Are Not the Same: Direct Causes Are Not Grounded Emotion Explanations},
  author={Pan, Zhuangzhuang and Xia, Yan and Chan, Chee Seng},
  journal={arXiv preprint arXiv:2605.25208},
  year={2026}
}
```

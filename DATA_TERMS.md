# Data Release Boundary

This repository releases IEMO-MECP as a pair-role label overlay. It does not
redistribute the underlying IEMOCAP utterance text, audio, video, or motion-capture
files.

## Included Data

- `data/labels/*.json`: pair-role labels keyed by split, dialogue id, target turn,
  and candidate turn.
- `data/metadata/*`: label schema, split-level counts, and release notes.
- `data/paper_tables/*.csv`: aggregate statistics used to reproduce paper figures
  and tables, excluding files with raw utterance text columns.

## Excluded Data

- Original IEMOCAP audio/video/text transcripts.
- Full split JSON files containing utterance text or local audio/video paths.
- Feature caches derived from original media.
- Human-audit process files that contain raw utterance text.

## Required External Resources

To reconstruct full training/evaluation examples with text or modalities, users
must obtain the underlying resources from their original providers:

- IEMOCAP from USC SAIL: https://sail.usc.edu/iemocap/
- ConvECPE/ECPEC source data/code: https://github.com/SenticNet/ECPEC

Users are responsible for complying with the original dataset licenses and terms.
The label overlay in this repository is intended for non-commercial research use
with properly obtained source data.

## Recommended Local Layout

Place external resources outside version control, for example:

```text
local_data/
  IEMOCAP/
  ECPEC/
```

Then combine them with the public IEMO-MECP label overlay:

```bash
python scripts/build_from_sources.py \
  --iemocap-root local_data/IEMOCAP \
  --convecpe-root local_data/ECPEC \
  --label-dir data/labels \
  --output-dir local_data/iemomecp_full
```

Do not commit `local_data/` or any reconstructed split files that contain
IEMOCAP utterance text, audio paths, video paths, timestamps, or media-derived
feature caches.

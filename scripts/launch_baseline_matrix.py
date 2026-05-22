#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
LABEL_PATH = os.pathsep.join(str(Path("data") / "labels" / f"{split}.json") for split in ("train", "valid", "test"))
MODEL_GROUPS = ("roberta", "wavlm", "clip", "rw_fusion", "rc_fusion", "rwc_fusion")


@dataclass(frozen=True)
class Job:
    name: str
    group: str
    cmd: list[str]


def modality_flags(tag: str) -> list[str]:
    return [
        "--use-audio" if "a" in tag else "--no-audio",
        "--use-video" if "v" in tag else "--no-video",
    ]


def model_kind(group: str) -> str:
    if group == "wavlm":
        return "audio_mlp"
    if group == "clip":
        return "video_mlp"
    return "roberta"


def model_name(group: str) -> str:
    names = {
        "roberta": "roberta_base",
        "wavlm": "wavlm_base_plus",
        "clip": "clip_vit_large_patch14",
        "rw_fusion": "roberta_base_wavlm_base_plus",
        "rc_fusion": "roberta_base_clip_vit_large_patch14",
        "rwc_fusion": "roberta_base_wavlm_base_plus_clip_vit_large_patch14",
    }
    return names[group]


def model_tag(group: str) -> str:
    tags = {
        "roberta": "t",
        "wavlm": "a",
        "clip": "v",
        "rw_fusion": "ta",
        "rc_fusion": "tv",
        "rwc_fusion": "tav",
    }
    return tags[group]


def result_exists(job: Job) -> bool:
    result_path = _output_dir_from_cmd(job.cmd) / "test_results.json"
    prediction_path = result_path.with_name("pair_predictions.csv")
    if not result_path.exists() or not prediction_path.exists():
        return False
    try:
        payload = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return str(payload.get("pair_role_task", "")) in {"3class", "source_binary"} and "pair_f1" in payload


def build_jobs(args: argparse.Namespace) -> list[Job]:
    jobs: list[Job] = []
    for seed in args.seeds:
        for group in args.groups:
            name = f"{model_name(group)}_{args.pair_role_task}_seed{seed}"
            output_dir = Path(args.output_root) / name
            cmd = [
                sys.executable,
                "scripts/train_pair_role_baseline.py",
                "--model-kind",
                model_kind(group),
                "--pair-role-task",
                args.pair_role_task,
                "--pair-role-labels-path",
                args.pair_role_labels_path,
                "--dataset-name",
                "IemoMECP",
                "--data-root",
                args.data_root,
                "--seed",
                str(seed),
                "--cuda-index",
                "0",
                "--output-dir",
                str(output_dir),
                "--resume",
                *modality_flags(model_tag(group)),
            ]
            jobs.append(Job(name=name, group=group, cmd=cmd))
    return jobs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch released IEMO-MECP lightweight baseline jobs.")
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--per-gpu", type=int, default=1)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456])
    parser.add_argument("--groups", nargs="+", choices=MODEL_GROUPS, default=list(MODEL_GROUPS))
    parser.add_argument("--pair-role-task", choices=("3class", "source_binary"), default="3class")
    parser.add_argument("--data-root", default="local_data/iemomecp_full")
    parser.add_argument("--pair-role-labels-path", default=LABEL_PATH)
    parser.add_argument("--output-root", default="outputs/baseline_matrix")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root).expanduser().resolve()
    logs_dir = output_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    jobs = build_jobs(args)
    pending = [job for job in jobs if args.force or not result_exists(job)]
    print(f"TOTAL jobs={len(jobs)} pending={len(pending)} output={output_root}", flush=True)
    for job in pending:
        print(f"JOB {job.name}: {shlex.join(job.cmd)}", flush=True)
    if args.dry_run:
        return

    gpus = [item.strip() for item in args.gpus.split(",") if item.strip()]
    running: dict[subprocess.Popen, tuple[str, str, Path]] = {}
    queue = list(pending)
    failures: list[str] = []

    while queue or running:
        for gpu in gpus:
            active = sum(1 for _, used_gpu, _ in running.values() if used_gpu == gpu)
            while active < args.per_gpu and queue:
                job = queue.pop(0)
                log_path = logs_dir / f"{job.name}_gpu{gpu}.log"
                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = gpu
                env.setdefault("TOKENIZERS_PARALLELISM", "false")
                with log_path.open("ab") as log:
                    proc = subprocess.Popen(job.cmd, cwd=REPO_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
                running[proc] = (job.name, gpu, log_path)
                active += 1
                print(f"START gpu={gpu} pid={proc.pid} job={job.name} log={log_path}", flush=True)

        time.sleep(20)
        for proc in list(running):
            status = proc.poll()
            if status is None:
                continue
            name, gpu, log_path = running.pop(proc)
            if status == 0:
                print(f"DONE gpu={gpu} pid={proc.pid} job={name}", flush=True)
            else:
                failures.append(name)
                print(f"FAIL gpu={gpu} pid={proc.pid} job={name} code={status} log={log_path}", flush=True)
    if failures:
        raise SystemExit(f"failed jobs: {', '.join(failures)}")
    print("ALL_DONE", flush=True)


def _output_dir_from_cmd(cmd: list[str]) -> Path:
    for index, item in enumerate(cmd[:-1]):
        if item == "--output-dir":
            return Path(cmd[index + 1])
    raise ValueError(f"Missing --output-dir in command: {cmd}")


if __name__ == "__main__":
    main()

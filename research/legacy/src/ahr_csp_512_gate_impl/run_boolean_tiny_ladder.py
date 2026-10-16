#!/usr/bin/env python3
"""Run a Boolean-only curriculum ladder after the Boolean data audit passes."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.ahr512_gate_runner import (  # noqa: E402
    REPORT_DIR,
    build_unique_controlled_dataset,
    load_notebook_runtime,
    train_gate_fast,
    write_json,
)


def _stage_dataset_name(size: int) -> str:
    return f"phase5p7_bool_tiny_ladder_unique_{size}"


def _stage_train_name(size: int) -> str:
    return f"phase5p7_bool_tiny_ladder_unique{size}_power35_b16"


def _compact_metrics(report: Dict[str, Any]) -> Dict[str, Any]:
    split_reports = report.get("split_reports", {})
    compact: Dict[str, Any] = {"passed": bool(report.get("passed", False))}
    for split in ("train", "val", "test"):
        metrics = split_reports.get(split, {})
        compact[split] = {
            "cpu_valid_rate": metrics.get("cpu_valid_rate"),
            "cpu_mean_energy": metrics.get("cpu_mean_energy"),
            "hard_valid_rate": metrics.get("hard_valid_rate"),
            "mean_hard_energy": metrics.get("mean_hard_energy"),
        }
    return compact


def run_ladder(
    stages: Sequence[int],
    epochs: int,
    batch_size: int,
    eval_every: int,
    patience: int,
    min_epochs: int,
    domain_cap: int,
    seed_base: int,
    fresh_each_stage: bool,
) -> Dict[str, Any]:
    ns = load_notebook_runtime()
    stage_reports: List[Dict[str, Any]] = []
    previous_train_name: str | None = None

    for size in stages:
        dataset_name = _stage_dataset_name(size)
        build_report = build_unique_controlled_dataset(
            ns=ns,
            name=dataset_name,
            train=size,
            val=max(16, size // 4),
            test=max(16, size // 4),
            seed=seed_base + size,
            domain_cap=domain_cap,
            require_unique="all",
            max_attempts_multiplier=200,
            families=("boolean_sat",),
        )
        dataset_path = Path(build_report["dataset_path"])
        train_name = _stage_train_name(size)
        resume_names: Sequence[str]
        if fresh_each_stage or previous_train_name is None:
            resume_names = ()
        else:
            resume_names = (previous_train_name,)

        print(
            json.dumps(
                {
                    "event": "stage_start",
                    "size": size,
                    "dataset": str(dataset_path),
                    "train_name": train_name,
                    "resume_names": list(resume_names),
                },
                sort_keys=True,
            ),
            flush=True,
        )

        report = train_gate_fast(
            ns=ns,
            dataset_path=dataset_path,
            name=train_name,
            epochs=epochs,
            batch_size=batch_size,
            eval_every=eval_every,
            patience=patience,
            min_epochs=min_epochs,
            resume_names=resume_names,
            full_only=False,
        )
        report["stage_size"] = size
        report["build_report"] = build_report
        report["initialized_from"] = list(resume_names)
        stage_reports.append(report)

        compact = _compact_metrics(report)
        print(json.dumps({"event": "stage_done", "size": size, **compact}, sort_keys=True), flush=True)

        if not bool(report.get("passed", False)):
            print(json.dumps({"event": "ladder_stop", "reason": "stage_failed", "size": size}, sort_keys=True), flush=True)
            break
        previous_train_name = train_name

    out = {
        "created_at_unix": time.time(),
        "mode": "boolean_tiny_ladder",
        "stages_requested": list(stages),
        "fresh_each_stage": fresh_each_stage,
        "config": {
            "epochs": epochs,
            "batch_size": batch_size,
            "eval_every": eval_every,
            "patience": patience,
            "min_epochs": min_epochs,
            "domain_cap": domain_cap,
            "seed_base": seed_base,
        },
        "stage_summaries": [
            {
                "stage_size": int(stage.get("stage_size", -1)),
                "name": stage.get("name"),
                **_compact_metrics(stage),
            }
            for stage in stage_reports
        ],
        "stages": stage_reports,
    }
    write_json(REPORT_DIR / "phase5p7_bool_tiny_ladder_power35_b16_report.json", out)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Boolean-only tiny curriculum ladder.")
    parser.add_argument("--stages", nargs="+", type=int, default=[16, 32, 64, 128])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--min-epochs", type=int, default=8)
    parser.add_argument("--domain-cap", type=int, default=1024)
    parser.add_argument("--seed-base", type=int, default=20260503)
    parser.add_argument("--fresh-each-stage", action="store_true")
    args = parser.parse_args()

    report = run_ladder(
        stages=args.stages,
        epochs=args.epochs,
        batch_size=args.batch_size,
        eval_every=args.eval_every,
        patience=args.patience,
        min_epochs=args.min_epochs,
        domain_cap=args.domain_cap,
        seed_base=args.seed_base,
        fresh_each_stage=args.fresh_each_stage,
    )
    print(json.dumps({"report": str(REPORT_DIR / "phase5p7_bool_tiny_ladder_power35_b16_report.json"), "stage_summaries": report["stage_summaries"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Boolean SAT overfit diagnostics for AHR-CSP.

Creates controlled datasets from one certified Boolean record:
- same1: the exact same formula appears in train/val/test.
- repeat16: the same formula is repeated 16 times in train and reused for val/test.

The goal is to distinguish a mechanics bug from a sample-efficiency problem.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.ahr512_gate_runner import (  # noqa: E402
    DATA_DIR,
    REPORT_DIR,
    build_unique_controlled_dataset,
    load_notebook_runtime,
    train_gate_fast,
    write_json,
    write_jsonl,
)


def load_first_record(source: Path) -> Dict[str, Any]:
    with source.open("r", encoding="utf-8") as handle:
        return json.loads(next(line for line in handle if line.strip()))


def make_record(base: Dict[str, Any], split: str, idx: int, name: str) -> Dict[str, Any]:
    rec = copy.deepcopy(base)
    rec["split"] = split
    rec["stage"] = name
    rec["micro_dataset"] = name
    rec["debug_duplicate_index"] = idx
    return rec


def make_same_record_dataset(base: Dict[str, Any], name: str, train_count: int) -> Path:
    records: List[Dict[str, Any]] = []
    for idx in range(train_count):
        records.append(make_record(base, "train", idx, name))
    records.append(make_record(base, "val", 0, name))
    records.append(make_record(base, "test", 0, name))
    path = DATA_DIR / f"{name}.jsonl"
    write_jsonl(path, records)
    return path


def metric_summary(report: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {"passed": bool(report.get("passed"))}
    for split, obj in (report.get("split_reports") or {}).items():
        metrics = obj.get("metrics", obj)
        out[split] = {
            "cpu_valid_rate": metrics.get("cpu_valid_rate"),
            "cpu_mean_energy": metrics.get("cpu_mean_energy"),
            "hard_valid_rate": metrics.get("hard_valid_rate"),
            "mean_hard_energy": metrics.get("mean_hard_energy"),
            "assignment_acc": metrics.get("assignment_acc"),
            "exact_reference_match_rate": metrics.get("exact_reference_match_rate"),
        }
    return out


def run_case(ns: Dict[str, Any], dataset: Path, name: str, epochs: int, batch_size: int, patience: int) -> Dict[str, Any]:
    report = train_gate_fast(
        ns=ns,
        dataset_path=dataset,
        name=name,
        epochs=epochs,
        batch_size=batch_size,
        eval_every=1,
        patience=patience,
        min_epochs=epochs,
        resume_names=(),
        full_only=False,
    )
    print(json.dumps({"case": name, **metric_summary(report)}, sort_keys=True), flush=True)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DATA_DIR / "phase5p7_bool_tiny_ladder_unique_16.jsonl")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=999)
    args = parser.parse_args()

    ns = load_notebook_runtime()
    if not args.source.exists():
        build_unique_controlled_dataset(
            ns=ns,
            name="phase5p7_bool_tiny_ladder_unique_16",
            train=16,
            val=16,
            test=16,
            seed=20260519,
            domain_cap=1024,
            require_unique="all",
            max_attempts_multiplier=200,
            families=("boolean_sat",),
        )

    base = load_first_record(args.source)
    same1_path = make_same_record_dataset(base, "phase5p7_bool_overfit_same1", train_count=1)
    repeat16_path = make_same_record_dataset(base, "phase5p7_bool_overfit_repeat16", train_count=16)

    start = time.time()
    same1 = run_case(
        ns=ns,
        dataset=same1_path,
        name="phase5p7_bool_overfit_same1_power35",
        epochs=args.epochs,
        batch_size=1,
        patience=args.patience,
    )
    repeat16 = run_case(
        ns=ns,
        dataset=repeat16_path,
        name="phase5p7_bool_overfit_repeat16_power35",
        epochs=args.epochs,
        batch_size=16,
        patience=args.patience,
    )
    out = {
        "created_at_unix": time.time(),
        "elapsed_s": time.time() - start,
        "source": str(args.source),
        "cases": {
            "same1": metric_summary(same1),
            "repeat16": metric_summary(repeat16),
        },
    }
    report_path = REPORT_DIR / "phase5p7_bool_overfit_debug_report.json"
    write_json(report_path, out)
    print(json.dumps({"report": str(report_path), **out}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

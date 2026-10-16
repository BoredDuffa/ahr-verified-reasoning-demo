#!/usr/bin/env python3
"""Run Boolean filtered-by-variable-count debug gates."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.ahr512_gate_runner import DATA_DIR, REPORT_DIR, load_notebook_runtime, train_gate_fast, write_json, write_jsonl  # noqa: E402


def filter_dataset(source: Path, max_vars: int, name: str) -> Path:
    records: List[Dict[str, Any]] = []
    with source.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rec = json.loads(line)
            diff = rec["problem"]["metadata"]["difficulty"]
            if int(diff["n_vars"]) <= max_vars:
                rec = dict(rec)
                rec["stage"] = name
                rec["micro_dataset"] = name
                rec["debug_filter"] = {"max_n_vars": max_vars, "source": str(source)}
                records.append(rec)
    path = DATA_DIR / f"{name}.jsonl"
    write_jsonl(path, records)
    return path


def split_counts(path: Path) -> Dict[str, int]:
    counts = {"train": 0, "val": 0, "test": 0}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                split = json.loads(line).get("split")
                counts[split] = counts.get(split, 0) + 1
    return counts


def metrics(report: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {"passed": bool(report.get("passed"))}
    for split, obj in (report.get("split_reports") or {}).items():
        m = obj.get("metrics", obj)
        out[split] = {
            "cpu_valid_rate": m.get("cpu_valid_rate"),
            "cpu_mean_energy": m.get("cpu_mean_energy"),
            "assignment_acc": m.get("assignment_acc"),
            "exact_reference_match_rate": m.get("exact_reference_match_rate"),
        }
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DATA_DIR / "phase5p7_bool_micro_unique_128.jsonl")
    parser.add_argument("--max-vars", nargs="+", type=int, default=[3, 4, 5])
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--min-epochs", type=int, default=20)
    args = parser.parse_args()

    ns = load_notebook_runtime()
    results: List[Dict[str, Any]] = []
    start = time.time()
    for max_vars in args.max_vars:
        dataset_name = f"phase5p7_bool_nvars_le{max_vars}"
        dataset_path = filter_dataset(args.source, max_vars, dataset_name)
        counts = split_counts(dataset_path)
        train_name = f"{dataset_name}_power35_b16"
        print(json.dumps({"event": "stage_start", "max_vars": max_vars, "dataset": str(dataset_path), "counts": counts}), flush=True)
        report = train_gate_fast(
            ns=ns,
            dataset_path=dataset_path,
            name=train_name,
            epochs=args.epochs,
            batch_size=args.batch_size,
            eval_every=1,
            patience=args.patience,
            min_epochs=args.min_epochs,
            resume_names=(),
            full_only=False,
        )
        summary = {"max_vars": max_vars, "dataset": str(dataset_path), "counts": counts, "name": train_name, **metrics(report)}
        results.append(summary)
        print(json.dumps({"event": "stage_done", **summary}, sort_keys=True), flush=True)
    out = {"created_at_unix": time.time(), "elapsed_s": time.time() - start, "source": str(args.source), "results": results}
    path = REPORT_DIR / "phase5p7_bool_nvars_debug_report.json"
    write_json(path, out)
    print(json.dumps({"report": str(path), **out}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

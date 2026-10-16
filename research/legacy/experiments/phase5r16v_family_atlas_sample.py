#!/usr/bin/env python3
"""Sampled v16v operation-synthesis family atlas.

v16v passed the 55% unhinted length-3 curriculum. This atlas checks the
retained length-2 and unhinted length-3 families before spending GPU time on a
harder curriculum.
"""

from __future__ import annotations

import dataclasses
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent.parent
EXP_DIR = ROOT / "experiments"
REPORT_DIR = ROOT / "reports"
DATA_DIR = ROOT / "data" / "reasoning_challenge_v0" / "compiled_ahr_csp_v0"
SUBSET_DIR = REPORT_DIR / "v16v_family_atlas_subsets"

if str(EXP_DIR) not in sys.path:
    sys.path.insert(0, str(EXP_DIR))

import ahr512_gate_runner as runner  # type: ignore

RUN = "phase5r16v_operation_synthesis_len3_unhinted55_fullgpu_b64"
DATASET = "reasoning_curriculum_v16v_operation_synthesis_len3_unhinted55_4096_1024_1024.jsonl"
FAMILIES = [
    "operation_synthesis_len2_factored",
    "operation_synthesis_len3_factored",
]
SPLITS = ["train", "val", "test"]
MAX_PER_SPLIT = {"train": 512, "val": 512, "test": 512}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def slim(metrics: Dict[str, Any]) -> Dict[str, Any]:
    keep = [
        "cpu_valid_rate",
        "hard_valid_rate",
        "mean_hard_energy",
        "cpu_mean_energy",
        "cpu_verified_count",
        "assignment_acc",
        "loss_total",
    ]
    return {key: metrics.get(key) for key in keep if key in metrics}


def row_stats(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    ops: Counter[str] = Counter()
    lengths: Counter[str] = Counter()
    states: List[int] = []
    for row in rows:
        meta = row.get("problem", {}).get("metadata", {}) or {}
        ops.update(meta.get("witness_program") or [])
        lengths[str(meta.get("program_length"))] += 1
        if meta.get("state_count") is not None:
            states.append(int(meta["state_count"]))
    states_sorted = sorted(states)
    return {
        "rows": len(rows),
        "program_lengths": dict(lengths),
        "witness_operation_counts": dict(ops),
        "state_count_min": min(states) if states else None,
        "state_count_median": states_sorted[len(states_sorted) // 2] if states_sorted else None,
        "state_count_max": max(states) if states else None,
    }


def sampled_subset(family: str, rows: List[Dict[str, Any]]) -> tuple[Path, List[Dict[str, Any]], Dict[str, int], Dict[str, int]]:
    rng = random.Random(f"20260604:v16v:{family}")
    out: List[Dict[str, Any]] = []
    counts: Dict[str, int] = {}
    total_counts: Dict[str, int] = {}
    for split in SPLITS:
        candidates = [row for row in rows if row.get("family") == family and row.get("split") == split]
        total_counts[split] = len(candidates)
        take = min(MAX_PER_SPLIT[split], len(candidates))
        picked = rng.sample(candidates, take) if take < len(candidates) else candidates
        counts[split] = len(picked)
        out.extend(picked)
    path = SUBSET_DIR / f"v16v_{family}_sampled.jsonl"
    write_jsonl(path, out)
    return path, out, counts, total_counts


def eval_split(ns: Dict[str, Any], model: Any, path: Path, split: str, cfg: Any, device: Any) -> Dict[str, Any]:
    loader, data_report = ns["make_phase6_loader"](
        dataset_source=path,
        split=split,
        sat_only=True,
        batch_size=64,
        limit=None,
    )
    if int(data_report.get("records", 0)) == 0:
        return {"data_report": data_report, "metrics": {}}
    metrics = ns["evaluate_ahr_model"](model=model, val_loader=loader, device=device, train_config=cfg)
    return {"data_report": data_report, "metrics": slim(metrics)}


def history_summary(run: str) -> Dict[str, Any]:
    path = REPORT_DIR / f"{run}_history.json"
    if not path.exists():
        return {"available": False}
    history = read_json(path)
    return {"available": True, "epochs": len(history), "tail": history[-5:]}


def main() -> int:
    start = time.time()
    SUBSET_DIR.mkdir(parents=True, exist_ok=True)
    print("[v16v-atlas] loading runtime", flush=True)
    ns = runner.load_notebook_runtime()
    runner.install_repaired_loss(ns)
    runner.install_vectorized_factor_messages(ns)

    torch = ns["torch"]
    device = torch.device("cpu" if ns["get_device"]() == "no-torch" else ns["get_device"]())
    arch = runner.make_debug_arch(ns)
    cfg = runner.make_repaired_train_config(
        ns=ns,
        name="v16v_family_atlas_sample",
        epochs=1,
        batch_size=64,
        train_limit=None,
        val_limit=None,
    )
    cfg = dataclasses.replace(cfg, cpu_verify_items=64, max_val_batches=None)

    report = read_json(REPORT_DIR / f"{RUN}_report.json")
    rows = read_jsonl(DATA_DIR / DATASET)
    model = runner._load_best_checkpoint_into_model(ns, RUN, arch, device)
    atlas: Dict[str, Any] = {
        "clinical_use": "forbidden",
        "kind": "phase5r16v_family_atlas_sampled",
        "sample_policy": MAX_PER_SPLIT,
        "targets": runner.PASS_TARGETS,
        "run": RUN,
        "dataset": str(DATA_DIR / DATASET),
        "final_passed": report.get("passed"),
        "final_split_metrics": {
            split: slim(split_report.get("metrics", {}))
            for split, split_report in report.get("split_reports", {}).items()
        },
        "history_summary": history_summary(RUN),
        "families": {},
    }

    for family in FAMILIES:
        print(f"[v16v-atlas] {family}", flush=True)
        path, subset, counts, total = sampled_subset(family, rows)
        family_out = {
            "subset_path": str(path),
            "sample_counts": counts,
            "total_counts": total,
            "row_stats": row_stats(subset),
            "splits": {},
        }
        for split in SPLITS:
            family_out["splits"][split] = eval_split(ns, model, path, split, cfg, device)
        atlas["families"][family] = family_out

    atlas["elapsed_s"] = time.time() - start
    out = REPORT_DIR / "phase5r16v_operation_synthesis_family_atlas_sampled_20260604.json"
    write_json(out, atlas)
    print(json.dumps({"atlas_path": str(out), "elapsed_s": atlas["elapsed_s"]}, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

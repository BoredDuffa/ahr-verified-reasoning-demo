#!/usr/bin/env python3
"""Evaluate verifier-guided Hamming repair for Boolean AHR-CSP checkpoints."""

from __future__ import annotations

import argparse
import dataclasses
import itertools
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.ahr512_gate_runner import (  # noqa: E402
    REPORT_DIR,
    install_repaired_loss,
    install_vectorized_factor_messages,
    load_notebook_runtime,
    make_debug_arch,
    write_json,
)


def hamming_masks(names: Sequence[str], max_radius: int) -> Iterable[Tuple[str, ...]]:
    for radius in range(1, int(max_radius) + 1):
        for combo in itertools.combinations(names, radius):
            yield combo


def flip_assignment(assignment: Dict[str, Any], names: Sequence[str]) -> Dict[str, Any]:
    out = dict(assignment)
    for name in names:
        out[name] = not bool(out[name])
    return out


def bool_assignment_from_indices(batch: Any, item_index: int, index_tensor: Any, ns: Dict[str, Any]) -> Dict[str, bool]:
    assignment = ns["decode_batch_indices_to_assignment"](batch, item_index, index_tensor)
    return {str(k): bool(v) for k, v in assignment.items()}


def try_repair(
    ns: Dict[str, Any],
    raw_problem_json: Dict[str, Any],
    base_assignment: Dict[str, bool],
    max_radius: int,
) -> Dict[str, Any]:
    problem = ns["problem_from_jsonable"](raw_problem_json)
    raw_check = ns["verify_assignment"](problem, base_assignment)
    raw_valid = bool(raw_check.get("valid", False))
    if raw_valid:
        return {
            "raw_valid": True,
            "raw_energy": float(raw_check.get("total_energy", 0.0)),
            "repaired_valid": True,
            "repaired_energy": 0.0,
            "repair_radius": 0,
            "repair_flips": [],
            "attempts": 0,
            "assignment": base_assignment,
        }

    names = list(base_assignment.keys())
    attempts = 0
    best = {
        "energy": float(raw_check.get("total_energy", 999.0)),
        "radius": None,
        "flips": [],
        "assignment": base_assignment,
    }
    for combo in hamming_masks(names, max_radius):
        attempts += 1
        candidate = flip_assignment(base_assignment, combo)
        check = ns["verify_assignment"](problem, candidate)
        energy = float(check.get("total_energy", 999.0))
        if energy < best["energy"]:
            best = {"energy": energy, "radius": len(combo), "flips": list(combo), "assignment": candidate}
        if bool(check.get("valid", False)):
            return {
                "raw_valid": False,
                "raw_energy": float(raw_check.get("total_energy", 999.0)),
                "repaired_valid": True,
                "repaired_energy": 0.0,
                "repair_radius": len(combo),
                "repair_flips": list(combo),
                "attempts": attempts,
                "assignment": candidate,
            }

    return {
        "raw_valid": False,
        "raw_energy": float(raw_check.get("total_energy", 999.0)),
        "repaired_valid": False,
        "repaired_energy": best["energy"],
        "repair_radius": best["radius"],
        "repair_flips": best["flips"],
        "attempts": attempts,
        "assignment": best["assignment"],
    }


def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {"count": 0}
    count = len(rows)
    raw_valid = sum(1 for r in rows if r["raw_valid"])
    repaired_valid = sum(1 for r in rows if r["repaired_valid"])
    raw_energy = sum(float(r["raw_energy"]) for r in rows) / count
    repaired_energy = sum(float(r["repaired_energy"]) for r in rows) / count
    repairable = [r for r in rows if (not r["raw_valid"]) and r["repaired_valid"]]
    by_radius: Dict[str, int] = {}
    for r in rows:
        key = str(r["repair_radius"])
        by_radius[key] = by_radius.get(key, 0) + 1
    return {
        "count": count,
        "raw_valid": raw_valid,
        "raw_valid_rate": raw_valid / count,
        "raw_mean_energy": raw_energy,
        "repaired_valid": repaired_valid,
        "repaired_valid_rate": repaired_valid / count,
        "repaired_mean_energy": repaired_energy,
        "repair_lift": (repaired_valid - raw_valid) / count,
        "newly_repaired": len(repairable),
        "by_repair_radius": by_radius,
    }


def evaluate_checkpoint(
    ns: Dict[str, Any],
    dataset_path: Path,
    checkpoint_name: str,
    batch_size: int,
    max_radius: int,
    device: Any,
) -> Dict[str, Any]:
    torch = ns["torch"]
    arch = make_debug_arch(ns)
    model = ns["AHRReasoner"](tensor_config=ns["TENSOR_CFG"], arch_config=arch).to(device)
    ckpt_path = PROJECT_ROOT / "checkpoints" / f"{checkpoint_name}_best.pt"
    ns["load_ahr_checkpoint"](ckpt_path, model=model, optimizer=None, map_location=device)
    model.eval()

    all_rows: List[Dict[str, Any]] = []
    by_split_rows: Dict[str, List[Dict[str, Any]]] = {}
    examples: List[Dict[str, Any]] = []

    for split in ("train", "val", "test"):
        loader, data_report = ns["make_phase6_loader"](
            dataset_source=dataset_path,
            split=split,
            sat_only=True,
            batch_size=batch_size,
            limit=None,
        )
        rows: List[Dict[str, Any]] = []
        for batch0 in loader:
            batch = ns["batch_to_device"](batch0, device)
            with torch.no_grad():
                output = model(batch)
            hard_index = output.hard_index.detach().cpu()
            hard_energy = output.hard_energy.detach().cpu().tolist()
            for i, problem_id in enumerate(batch0.problem_ids):
                raw_json = batch0.raw_problem_jsons[i]
                base = bool_assignment_from_indices(batch0, i, hard_index[i], ns)
                repair = try_repair(ns, raw_json, base, max_radius=max_radius)
                row = {
                    "split": split,
                    "problem_id": problem_id,
                    "model_hard_energy": float(hard_energy[i]),
                    **{k: v for k, v in repair.items() if k != "assignment"},
                    "n_vars": len(base),
                }
                rows.append(row)
                all_rows.append(row)
                if len(examples) < 20 and (not row["raw_valid"]):
                    examples.append(
                        {
                            **row,
                            "base_assignment": base,
                            "repaired_assignment": repair["assignment"],
                        }
                    )
        by_split_rows[split] = rows

    report = {
        "created_at_unix": time.time(),
        "dataset_path": str(dataset_path),
        "checkpoint_name": checkpoint_name,
        "checkpoint_path": str(ckpt_path),
        "device": str(device),
        "max_radius": max_radius,
        "batch_size": batch_size,
        "overall": summarize(all_rows),
        "by_split": {split: summarize(rows) for split, rows in by_split_rows.items()},
        "failed_or_repaired_examples": examples,
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-radius", type=int, default=2)
    parser.add_argument("--case", action="append", nargs=3, metavar=("NAME", "DATASET", "CHECKPOINT"))
    args = parser.parse_args()

    cases = args.case or [
        (
            "le5_assign2_nll1",
            str(PROJECT_ROOT / "data" / "phase5p7_bool_nvars_le5.jsonl"),
            "phase5p7_bool_le5_sweep_assign2_nll1_power35_b16",
        ),
        (
            "bool128_fresh_resume",
            str(PROJECT_ROOT / "data" / "phase5p7_bool_micro_unique_128.jsonl"),
            "phase5p7_bool_micro_128_fresh_resume_power35_b16",
        ),
    ]

    ns = load_notebook_runtime()
    install_repaired_loss(ns)
    install_vectorized_factor_messages(ns)
    torch = ns["torch"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    reports = {}
    for label, dataset, checkpoint in cases:
        print(json.dumps({"event": "repair_eval_start", "label": label, "dataset": dataset, "checkpoint": checkpoint}), flush=True)
        report = evaluate_checkpoint(
            ns=ns,
            dataset_path=Path(dataset),
            checkpoint_name=checkpoint,
            batch_size=args.batch_size,
            max_radius=args.max_radius,
            device=device,
        )
        reports[label] = report
        out_path = REPORT_DIR / f"phase5p7_boolean_repair_{label}_r{args.max_radius}_report.json"
        write_json(out_path, report)
        print(json.dumps({"event": "repair_eval_done", "label": label, "report": str(out_path), "overall": report["overall"], "by_split": report["by_split"]}, indent=2), flush=True)

    combined = {
        "created_at_unix": time.time(),
        "max_radius": args.max_radius,
        "reports": reports,
    }
    combined_path = REPORT_DIR / f"phase5p7_boolean_repair_r{args.max_radius}_combined_report.json"
    write_json(combined_path, combined)
    print(json.dumps({"combined_report": str(combined_path), "summary": {k: v["overall"] for k, v in reports.items()}}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

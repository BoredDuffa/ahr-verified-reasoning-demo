#!/usr/bin/env python3
"""Failure atlas for a trained Boolean AHR-CSP checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.ahr512_gate_runner import (  # noqa: E402
    REPORT_DIR,
    install_repaired_loss,
    install_vectorized_factor_messages,
    load_notebook_runtime,
    make_debug_arch,
    make_repaired_train_config,
    write_json,
)


def lit_pattern(constraint: Dict[str, Any]) -> str:
    args = constraint.get("args") or []
    return "".join("N" if bool(lit.get("negated")) else "P" for lit in args)


def summarize_group(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {"count": 0}
    count = len(rows)
    valid = sum(1 for row in rows if row["valid"])
    energies = [float(row["energy"]) for row in rows]
    return {
        "count": count,
        "valid": valid,
        "valid_rate": valid / count,
        "mean_energy": sum(energies) / count,
        "max_energy": max(energies),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=PROJECT_ROOT / "data" / "phase5p7_bool_tiny_ladder_unique_16.jsonl")
    parser.add_argument("--checkpoint-name", default="phase5p7_bool_tiny_ladder_unique16_power35_b16")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--out", type=Path, default=REPORT_DIR / "phase5p7_bool_tiny_ladder_unique16_failure_atlas.json")
    args = parser.parse_args()

    ns = load_notebook_runtime()
    install_repaired_loss(ns)
    install_vectorized_factor_messages(ns)
    torch = ns["torch"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    arch = make_debug_arch(ns)
    model = ns["AHRReasoner"](tensor_config=ns["TENSOR_CFG"], arch_config=arch).to(device)
    ns["load_ahr_checkpoint"](
        PROJECT_ROOT / "checkpoints" / f"{args.checkpoint_name}_best.pt",
        model=model,
        optimizer=None,
        map_location=device,
    )
    model.eval()

    all_rows: List[Dict[str, Any]] = []
    violated_ops = Counter()
    violated_patterns = Counter()
    violated_by_arity = Counter()

    for split in ("train", "val", "test"):
        loader, _ = ns["make_phase6_loader"](
            dataset_source=args.dataset,
            split=split,
            sat_only=True,
            batch_size=args.batch_size,
            limit=None,
        )
        for batch0 in loader:
            batch = ns["batch_to_device"](batch0, device)
            with torch.no_grad():
                output = model(batch)
            checks = ns["verify_hard_assignments_cpu"](batch0, output, max_items=args.batch_size)
            hard_energy = output.hard_energy.detach().cpu().tolist()
            hard_index = output.hard_index.detach().cpu()
            target_index = batch0.tensors["target_index"]
            target_mask = batch0.tensors["target_mask"]

            for i, check in enumerate(checks):
                raw = batch0.raw_problem_jsons[i]
                meta = raw.get("metadata", {})
                diff = meta.get("difficulty", {})
                valid = bool(check.get("valid", False))
                pred = ns["decode_batch_indices_to_assignment"](batch0, i, hard_index[i])
                target = raw.get("certificate", {}).get("assignment", {})
                mask = target_mask[i].bool()
                var_acc = float(((hard_index[i] == target_index[i]) & mask).sum().item()) / float(mask.sum().item())
                wrong_vars = [
                    name
                    for j, name in enumerate(batch0.variable_names[i])
                    if bool(mask[j].item()) and int(hard_index[i, j].item()) != int(target_index[i, j].item())
                ]
                violated: List[Dict[str, Any]] = []
                for cr in check.get("constraint_results", []):
                    if cr.get("satisfied") is False:
                        cid = cr.get("id")
                        constraint = next((c for c in raw.get("constraints", []) if c.get("id") == cid), None)
                        pattern = lit_pattern(constraint or {})
                        arity = len((constraint or {}).get("args") or [])
                        violated_ops[(constraint or {}).get("op", cr.get("op"))] += 1
                        violated_patterns[pattern] += 1
                        violated_by_arity[str(arity)] += 1
                        violated.append(
                            {
                                "id": cid,
                                "op": (constraint or {}).get("op", cr.get("op")),
                                "arity": arity,
                                "pattern": pattern,
                                "scope": (constraint or {}).get("scope"),
                                "args": (constraint or {}).get("args"),
                            }
                        )
                all_rows.append(
                    {
                        "split": split,
                        "problem_id": batch0.problem_ids[i],
                        "valid": valid,
                        "energy": float(hard_energy[i]),
                        "var_acc": var_acc,
                        "wrong_vars": wrong_vars,
                        "n_wrong_vars": len(wrong_vars),
                        "n_vars": int(diff.get("n_vars", len(batch0.variable_names[i]))),
                        "n_clauses": int(diff.get("n_clauses", len(raw.get("constraints", [])))),
                        "k": int(diff.get("k", 0) or 0),
                        "pred_assignment": pred,
                        "target_assignment": target,
                        "violated": violated[:20],
                        "n_violated": len(violated),
                    }
                )

    by_split = {split: summarize_group([r for r in all_rows if r["split"] == split]) for split in ("train", "val", "test")}
    by_n_vars = {str(k): summarize_group([r for r in all_rows if r["n_vars"] == k]) for k in sorted({r["n_vars"] for r in all_rows})}
    by_k = {str(k): summarize_group([r for r in all_rows if r["k"] == k]) for k in sorted({r["k"] for r in all_rows})}
    by_wrong_vars = {
        str(k): summarize_group([r for r in all_rows if r["n_wrong_vars"] == k])
        for k in sorted({r["n_wrong_vars"] for r in all_rows})
    }
    failed = [r for r in all_rows if not r["valid"]]
    report = {
        "dataset": str(args.dataset),
        "checkpoint_name": args.checkpoint_name,
        "device": str(device),
        "overall": summarize_group(all_rows),
        "by_split": by_split,
        "by_n_vars": by_n_vars,
        "by_k": by_k,
        "by_wrong_vars": by_wrong_vars,
        "violated_ops": dict(violated_ops),
        "violated_patterns": dict(violated_patterns),
        "violated_by_arity": dict(violated_by_arity),
        "failed_examples": failed[:20],
    }
    write_json(args.out, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Interactive playground for AHR operation-synthesis checkpoints.

This is a research/debugging interface, not a clinical or production surface.
It samples existing AHR-CSP operation-synthesis records, runs a checkpoint,
decodes the hard assignment, verifies it with the CPU verifier, and optionally
tries a small local verifier-guided repair.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import random
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

ROOT = Path(__file__).resolve().parent.parent
EXP_DIR = ROOT / "experiments"
DATA_DIR = ROOT / "data" / "reasoning_challenge_v0" / "compiled_ahr_csp_v0"

if str(EXP_DIR) not in sys.path:
    sys.path.insert(0, str(EXP_DIR))

import ahr512_gate_runner as runner  # type: ignore

DEFAULT_RUN = "phase5r16x_operation_synthesis_len3_unhinted65_fullgpu_b64"
DEFAULT_DATASET = "reasoning_curriculum_v16x_operation_synthesis_len3_unhinted65_4096_1024_1024.jsonl"


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def resolve_dataset(value: str) -> Path:
    path = Path(value)
    if path.exists():
        return path
    candidate = DATA_DIR / value
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"dataset not found: {value}")


def choose_rows(args: argparse.Namespace, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    selected = rows
    if args.split != "any":
        selected = [row for row in selected if row.get("split") == args.split]
    if args.family != "any":
        selected = [row for row in selected if row.get("family") == args.family]
    if args.problem_id:
        selected = [row for row in selected if row.get("id") == args.problem_id]
    if not selected:
        raise SystemExit("No records matched the requested filters.")
    if args.problem_id:
        return selected[:1]
    rng = random.Random(args.seed)
    if args.offset is not None:
        return [selected[int(args.offset) % len(selected)]]
    count = min(int(args.count), len(selected))
    return rng.sample(selected, count) if args.random else selected[:count]


def op_names(row: Dict[str, Any]) -> List[str]:
    meta = row.get("problem", {}).get("metadata", {}) or {}
    return list(meta.get("operation_names") or [])


def assignment_program(row: Dict[str, Any], assignment: Dict[str, Any]) -> List[str]:
    names = op_names(row)
    out: List[str] = []
    for key in sorted((name for name in assignment if name.startswith("op")), key=lambda x: int(x[2:])):
        value = assignment.get(key)
        if isinstance(value, int) and 0 <= value < len(names):
            out.append(names[value])
        else:
            out.append(str(value))
    return out


def compact_check(check: Dict[str, Any]) -> Dict[str, Any]:
    failed = [
        {
            "id": item.get("id"),
            "op": item.get("op"),
            "energy": item.get("energy"),
        }
        for item in check.get("constraint_results", [])
        if not item.get("satisfied", False)
    ]
    return {
        "valid": bool(check.get("valid", False)),
        "energy": check.get("total_energy", check.get("energy", check.get("hard_energy"))),
        "failed_constraints": failed,
    }


def load_model(ns: Dict[str, Any], checkpoint: str, device: Any) -> Any:
    runner.install_repaired_loss(ns)
    runner.install_vectorized_factor_messages(ns)
    arch = runner.make_debug_arch(ns)
    model = runner._load_best_checkpoint_into_model(ns, checkpoint, arch, device)
    model.eval()
    return model


def run_rows(args: argparse.Namespace, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    ns = runner.load_notebook_runtime()
    torch = ns["torch"]
    device_name = "cpu" if args.cpu else ns["get_device"]()
    device = torch.device("cpu" if device_name == "no-torch" else device_name)
    model = load_model(ns, args.checkpoint, device)
    cfg = runner.make_repaired_train_config(
        ns=ns,
        name="operation_synthesis_playground",
        epochs=1,
        batch_size=max(1, min(int(args.batch_size), len(rows))),
        train_limit=None,
        val_limit=None,
    )
    cfg = dataclasses.replace(cfg, cpu_verify_items=int(args.cpu_verify_items), max_val_batches=None)

    by_split: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        by_split.setdefault(str(row.get("split", "train")), []).append(row)

    results: List[Dict[str, Any]] = []
    for split, split_rows in by_split.items():
        with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        try:
            write_jsonl(tmp_path, split_rows)
            loader, data_report = ns["make_phase6_loader"](
                dataset_source=tmp_path,
                split=split,
                sat_only=True,
                batch_size=max(1, min(int(args.batch_size), len(split_rows))),
                limit=None,
            )
            for batch0 in loader:
                batch = ns["batch_to_device"](batch0, device)
                with torch.no_grad():
                    output = model(batch)
                hard_index = output.hard_index.detach().cpu()
                hard_energy = output.hard_energy.detach().cpu().tolist()
                for item_index, problem_id in enumerate(batch0.problem_ids):
                    row = split_rows[len(results) % len(split_rows)] if len(split_rows) == 1 else None
                    raw_problem = batch0.raw_problem_jsons[item_index]
                    assignment = ns["decode_batch_indices_to_assignment"](batch0, item_index, hard_index[item_index])
                    check = ns["verify_assignment"](ns["problem_from_jsonable"](raw_problem), assignment)
                    repair = runner.repair_assignment_with_verifier(
                        ns=ns,
                        raw_problem_json=raw_problem,
                        base_assignment=assignment,
                        max_radius=int(args.repair_radius),
                        max_attempts=int(args.repair_attempts),
                    )
                    # Match back by full problem fingerprint when possible.
                    source_row = row
                    fingerprint = raw_problem.get("metadata", {}).get("problem_fingerprint")
                    if source_row is None:
                        source_row = next(
                            (
                                candidate
                                for candidate in split_rows
                                if candidate.get("id") == fingerprint
                                or candidate.get("problem", {}).get("metadata", {}).get("problem_fingerprint") == fingerprint
                            ),
                            {},
                        )
                    meta = raw_problem.get("metadata", {}) or {}
                    result = {
                        "id": source_row.get("id") or fingerprint or problem_id,
                        "split": split,
                        "family": source_row.get("family") or meta.get("family"),
                        "program_length": meta.get("program_length"),
                        "start_sequence": meta.get("start_sequence"),
                        "target_sequence": meta.get("target_sequence"),
                        "operation_names": meta.get("operation_names"),
                        "witness_program": meta.get("witness_program"),
                        "model_program": assignment_program(source_row or {"problem": {"metadata": meta}}, assignment),
                        "model_assignment": assignment,
                        "reference_assignment": raw_problem.get("certificate", {}).get("assignment"),
                        "model_hard_energy_tensor": float(hard_energy[item_index]),
                        "cpu_check": compact_check(check),
                        "repair": {
                            key: value
                            for key, value in repair.items()
                            if key != "assignment"
                        },
                        "repaired_assignment": repair.get("assignment"),
                        "data_report": data_report,
                    }
                    results.append(result)
        finally:
            try:
                tmp_path.unlink()
            except FileNotFoundError:
                pass
    return results


def print_human(results: Sequence[Dict[str, Any]]) -> None:
    for i, result in enumerate(results, 1):
        check = result["cpu_check"]
        repair = result["repair"]
        print("=" * 88)
        print(f"[{i}] {result['id']}")
        print(f"split={result['split']} family={result['family']} length={result['program_length']}")
        print(f"start={result['start_sequence']} target={result['target_sequence']}")
        print(f"operations={result['operation_names']}")
        print(f"witness_program={result['witness_program']}")
        print(f"model_program={result['model_program']}")
        print(f"model_assignment={result['model_assignment']}")
        print(f"reference_assignment={result['reference_assignment']}")
        print(
            "cpu_valid={valid} cpu_energy={energy} tensor_energy={tensor}".format(
                valid=check["valid"],
                energy=check["energy"],
                tensor=result["model_hard_energy_tensor"],
            )
        )
        if check["failed_constraints"]:
            print(f"failed_constraints={check['failed_constraints']}")
        print(
            "repair: raw_valid={raw_valid} repaired_valid={repaired_valid} "
            "radius={repair_radius} variables={repair_variables} attempts={attempts}".format(**repair)
        )
        if repair.get("repair_radius") not in (None, 0):
            print(f"repaired_assignment={result['repaired_assignment']}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=DEFAULT_RUN)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--split", default="test", choices=["train", "val", "test", "any"])
    parser.add_argument("--family", default="operation_synthesis_len3_factored", choices=["operation_synthesis_len2_factored", "operation_synthesis_len3_factored", "any"])
    parser.add_argument("--problem-id", default="")
    parser.add_argument("--offset", type=int, default=None, help="Deterministic offset within filtered rows.")
    parser.add_argument("--count", type=int, default=3)
    parser.add_argument("--random", action="store_true")
    parser.add_argument("--seed", default="playground")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--cpu", action="store_true", help="Force CPU inference.")
    parser.add_argument("--cpu-verify-items", type=int, default=64)
    parser.add_argument("--repair-radius", type=int, default=2)
    parser.add_argument("--repair-attempts", type=int, default=5000)
    parser.add_argument("--json", action="store_true", help="Print JSON instead of human-readable output.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    dataset = resolve_dataset(args.dataset)
    rows = choose_rows(args, read_jsonl(dataset))
    results = run_rows(args, rows)
    if args.json:
        print(json.dumps({"clinical_use": "forbidden", "checkpoint": args.checkpoint, "dataset": str(dataset), "results": results}, indent=2, sort_keys=True))
    else:
        print_human(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

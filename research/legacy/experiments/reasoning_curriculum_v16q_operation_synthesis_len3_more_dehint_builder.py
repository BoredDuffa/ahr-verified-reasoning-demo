#!/usr/bin/env python3
"""Build v16q length-3 more-dehint bridge for operation synthesis.

v16p passed the prior 30% unhinted length-3 mix and its sampled family atlas
was clean. v16q makes the next conservative de-hint move: 5% path-hinted
length-3 and 35% unhinted length-3, while retaining 60% length-2 rows.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent.parent
EXP_DIR = ROOT / "experiments"
DATA_DIR = ROOT / "data" / "reasoning_challenge_v0" / "compiled_ahr_csp_v0"
REPORT_DIR = ROOT / "reports"

if str(EXP_DIR) not in sys.path:
    sys.path.insert(0, str(EXP_DIR))

import reasoning_curriculum_v16i_operation_synthesis_narrow_len23_bridge_builder as base  # type: ignore

DATASET_NAME = "reasoning_curriculum_v16q_operation_synthesis_len3_more_dehint_4096_1024_1024"
DATASET_PATH = DATA_DIR / f"{DATASET_NAME}.jsonl"
DATASET_REPORT_PATH = REPORT_DIR / f"{DATASET_NAME}_dataset_report.json"
VERIFIER_REPORT_PATH = REPORT_DIR / f"{DATASET_NAME}_verifier_report.json"
PROOF_REPORT_PATH = REPORT_DIR / "phase5r16q_operation_synthesis_len5_no_solution_proof_report.json"
SPLIT_COUNTS = {"train": 4096, "val": 1024, "test": 1024}
GENERATOR = "reasoning_curriculum_v16q_operation_synthesis_len3_more_dehint_builder"


def configure_base_paths() -> None:
    base.DATASET_NAME = DATASET_NAME
    base.DATASET_PATH = DATASET_PATH
    base.DATASET_REPORT_PATH = DATASET_REPORT_PATH
    base.VERIFIER_REPORT_PATH = VERIFIER_REPORT_PATH
    base.PROOF_REPORT_PATH = PROOF_REPORT_PATH


def retag(row: Dict[str, Any], family: str, note: str) -> Dict[str, Any]:
    row["stage"] = DATASET_NAME
    row["generator"] = GENERATOR
    row["family"] = family
    row["num_constraints"] = len(row["problem"]["constraints"])
    meta = row["problem"]["metadata"]
    meta["family"] = family
    meta["source"] = GENERATOR
    meta["clinical_use"] = "forbidden"
    meta["v16q_bridge_note"] = note
    return row


def add_path_hints(row: Dict[str, Any]) -> Dict[str, Any]:
    assignment = row["problem"]["certificate"]["assignment"]
    hinted_states: List[str] = ["s1", "s2"]
    for state_var in hinted_states:
        row["problem"]["constraints"].append(
            base.eq(f"path_hint_{state_var}", state_var, int(assignment[state_var]), [state_var])
        )
    row["problem"]["metadata"]["path_hint_states"] = hinted_states
    row["problem"]["metadata"]["transition_encoding"] = "local_state_op_state_tables_plus_intermediate_state_hints"
    return row


def build_dataset() -> List[Dict[str, Any]]:
    configure_base_paths()
    rng = random.Random(2026060201)
    rows: List[Dict[str, Any]] = []
    for split, split_count in SPLIT_COUNTS.items():
        len2_cut = (3 * split_count) // 5
        hinted_cut = len2_cut + split_count // 20
        for i in range(split_count):
            if i < len2_cut:
                row_id = f"{DATASET_NAME}_{split}_operation_synthesis_len2_retention_{i:05d}"
                row = base.compile_operation_synthesis_factored(row_id, split, rng, i, 2)
                rows.append(
                    retag(
                        row,
                        "operation_synthesis_len2_factored",
                        "60% len2 retention after v16p 30% unhinted len3 pass and clean sampled atlas",
                    )
                )
            elif i < hinted_cut:
                row_id = f"{DATASET_NAME}_{split}_operation_synthesis_len3_path_hinted_{i:05d}"
                row = base.compile_operation_synthesis_factored(row_id, split, rng, i, 3)
                row = add_path_hints(row)
                rows.append(
                    retag(
                        row,
                        "operation_synthesis_len3_factored_path_hinted",
                        "5% hinted len3 scaffold retained while increasing unhinted len3 pressure",
                    )
                )
            else:
                row_id = f"{DATASET_NAME}_{split}_operation_synthesis_len3_unhinted_probe_{i:05d}"
                row = base.compile_operation_synthesis_factored(row_id, split, rng, i, 3)
                rows.append(
                    retag(
                        row,
                        "operation_synthesis_len3_factored",
                        "35% unhinted len3 after v16p passed 30% unhinted with clean sampled family atlas",
                    )
                )
    return rows


def main() -> int:
    configure_base_paths()
    rows = build_dataset()
    report = base.summarize(rows)
    base.write_jsonl(DATASET_PATH, rows)
    base.write_json(DATASET_REPORT_PATH, report)
    verifier = base.verify_dataset(rows) if report["passed"] else {"passed": False}
    base.write_json(VERIFIER_REPORT_PATH, verifier)
    proof = base.no_solution_proof_report()
    print(
        json.dumps(
            {
                "dataset_path": str(DATASET_PATH),
                "dataset_report": str(DATASET_REPORT_PATH),
                "verifier_report": str(VERIFIER_REPORT_PATH),
                "proof_report": str(PROOF_REPORT_PATH),
                "dataset_passed": report["passed"],
                "verifier_passed": verifier["passed"],
                "proof_passed": proof["passed"],
                "record_count": report["record_count"],
                "family_counts": report["family_counts"],
                "max_finite_domain_size": report["max_finite_domain_size"],
                "max_constraint_scope": report["max_constraint_scope"],
                "max_table_rows": report["max_table_rows"],
                "clinical_use": "forbidden",
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if report["passed"] and verifier["passed"] and proof["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

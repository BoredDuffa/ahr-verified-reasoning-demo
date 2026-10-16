#!/usr/bin/env python3
"""Build v16v length-3 unhinted-55 bridge for operation synthesis.

v16u passed the 50% unhinted length-3 curriculum, and the sampled v16u family
atlas was clean. v16v makes the next small increase by moving to 55% unhinted
length-3 while keeping 45% length-2 retention.
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

DATASET_NAME = "reasoning_curriculum_v16v_operation_synthesis_len3_unhinted55_4096_1024_1024"
DATASET_PATH = DATA_DIR / f"{DATASET_NAME}.jsonl"
DATASET_REPORT_PATH = REPORT_DIR / f"{DATASET_NAME}_dataset_report.json"
VERIFIER_REPORT_PATH = REPORT_DIR / f"{DATASET_NAME}_verifier_report.json"
PROOF_REPORT_PATH = REPORT_DIR / "phase5r16v_operation_synthesis_len5_no_solution_proof_report.json"
SPLIT_COUNTS = {"train": 4096, "val": 1024, "test": 1024}
GENERATOR = "reasoning_curriculum_v16v_operation_synthesis_len3_unhinted55_builder"


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
    meta["v16v_bridge_note"] = note
    return row


def build_dataset() -> List[Dict[str, Any]]:
    configure_base_paths()
    rng = random.Random(2026060401)
    rows: List[Dict[str, Any]] = []
    for split, split_count in SPLIT_COUNTS.items():
        len2_cut = (9 * split_count) // 20
        for i in range(split_count):
            if i < len2_cut:
                row_id = f"{DATASET_NAME}_{split}_operation_synthesis_len2_retention_{i:05d}"
                row = base.compile_operation_synthesis_factored(row_id, split, rng, i, 2)
                rows.append(
                    retag(
                        row,
                        "operation_synthesis_len2_factored",
                        "45% len2 retention after v16u 50% unhinted pass and clean sampled atlas",
                    )
                )
            else:
                row_id = f"{DATASET_NAME}_{split}_operation_synthesis_len3_unhinted55_{i:05d}"
                row = base.compile_operation_synthesis_factored(row_id, split, rng, i, 3)
                rows.append(
                    retag(
                        row,
                        "operation_synthesis_len3_factored",
                        "55% unhinted len3 after v16u passed 50% unhinted with clean sampled family atlas",
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

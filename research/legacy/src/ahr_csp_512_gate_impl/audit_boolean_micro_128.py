#!/usr/bin/env python3
"""Audit the unique Boolean micro gate data and tensorization.

This script intentionally runs on CPU and does not start or stop training jobs.
It checks the current Boolean 128 dataset at the schema/tensorizer/verifier
boundary before further architecture or training changes.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.ahr512_gate_runner import load_notebook_runtime  # noqa: E402


def json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Counter):
        return dict(value)
    if isinstance(value, defaultdict):
        return dict(value)
    if isinstance(value, set):
        return sorted(value)
    return str(value)


def pct(numer: int, denom: int) -> float:
    return float(numer) / float(denom) if denom else 0.0


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            rec["_line_no"] = line_no
            records.append(rec)
    return records


def variable_names(problem: Any) -> List[str]:
    return [v.name for v in problem.variables]


def brute_force_solution_count(problem: Any, verify_assignment: Any) -> Tuple[int, List[Dict[str, bool]]]:
    names = variable_names(problem)
    solutions: List[Dict[str, bool]] = []
    for bits in itertools.product([False, True], repeat=len(names)):
        assignment = dict(zip(names, bits))
        result = verify_assignment(problem, assignment)
        if bool(result.get("valid", False)):
            solutions.append(assignment)
    return len(solutions), solutions[:5]


def clause_signature(constraint: Any) -> Tuple[int, str, bool, bool]:
    literals = constraint.args or []
    arity = len(literals)
    neg_count = sum(1 for lit in literals if bool(lit.get("negated", False)))
    sign_pattern = "".join("N" if bool(lit.get("negated", False)) else "P" for lit in literals)
    vars_in_args = [str(lit.get("var")) for lit in literals]
    has_duplicate_vars = len(set(vars_in_args)) != len(vars_in_args)
    has_scope_mismatch = tuple(vars_in_args) != tuple(constraint.scope)
    return arity, f"{arity}:{sign_pattern}:{neg_count}neg", has_duplicate_vars, has_scope_mismatch


def expected_boolean_clause_allowed_count(constraint: Any) -> int | None:
    literals = constraint.args or []
    vars_in_args = [str(lit.get("var")) for lit in literals]
    if len(set(vars_in_args)) != len(vars_in_args):
        return None
    return (2 ** len(literals)) - 1


def summarize_numbers(values: Iterable[int]) -> Dict[str, Any]:
    vals = list(values)
    if not vals:
        return {"count": 0}
    return {
        "count": len(vals),
        "min": min(vals),
        "max": max(vals),
        "mean": statistics.fmean(vals),
        "median": statistics.median(vals),
    }


def audit_records(records: List[Dict[str, Any]], ns: Dict[str, Any]) -> Dict[str, Any]:
    problem_from_jsonable = ns["problem_from_jsonable"]
    verify_assignment = ns["verify_assignment"]
    tensorize_problem = ns["tensorize_problem"]
    verify_tensorized_target = ns["verify_tensorized_target"]
    tensor_cfg = ns["TENSOR_CFG"]

    totals = Counter()
    splits = Counter()
    by_n_vars = Counter()
    by_n_clauses = Counter()
    by_k = Counter()
    by_arity = Counter()
    by_sign_pattern = Counter()
    duplicate_clause_count_by_problem = Counter()
    duplicate_literals_by_problem = Counter()
    scope_mismatch_by_problem = Counter()
    factor_exact_by_problem = Counter()

    n_vars_values: List[int] = []
    n_clause_values: List[int] = []
    domain_products: List[int] = []
    solution_counts: List[int] = []
    duplicate_clause_totals: List[int] = []

    examples: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    for rec in records:
        totals["records"] += 1
        split = str(rec.get("split", "unknown"))
        splits[split] += 1

        problem = problem_from_jsonable(rec["problem"])
        problem_id = rec.get("problem_id") or problem.metadata.get("problem_fingerprint") or f"line_{rec['_line_no']}"
        metadata = dict(problem.metadata or {})
        difficulty = dict(metadata.get("difficulty") or {})

        n_vars = len(problem.variables)
        n_clauses = len(problem.constraints)
        k = int(difficulty.get("k", 0) or 0)
        domain_product = 2 ** n_vars

        n_vars_values.append(n_vars)
        n_clause_values.append(n_clauses)
        domain_products.append(domain_product)
        by_n_vars[str(n_vars)] += 1
        by_n_clauses[str(n_clauses)] += 1
        by_k[str(k)] += 1

        if problem.problem_type != "boolean_sat":
            totals["non_boolean_problem_type"] += 1
            examples["non_boolean_problem_type"].append({"problem_id": problem_id, "type": problem.problem_type})

        if rec.get("unique_solution") is not True:
            totals["record_unique_solution_not_true"] += 1
            examples["record_unique_solution_not_true"].append({"problem_id": problem_id, "value": rec.get("unique_solution")})

        cert_assignment = dict(problem.certificate.assignment or {})
        cert_result = verify_assignment(problem, cert_assignment)
        if not bool(cert_result.get("valid", False)):
            totals["certificate_invalid"] += 1
            examples["certificate_invalid"].append(
                {"problem_id": problem_id, "errors": cert_result.get("errors"), "energy": cert_result.get("total_energy")}
            )

        brute_count, first_solutions = brute_force_solution_count(problem, verify_assignment)
        solution_counts.append(brute_count)
        declared_counts = [
            ("record_solution_count", rec.get("solution_count")),
            ("certificate_solution_count", getattr(problem.certificate, "solution_count", None)),
            ("metadata_solution_count", metadata.get("solution_count")),
        ]
        for name, declared in declared_counts:
            if declared is not None and int(declared) != int(brute_count):
                totals[f"{name}_mismatch"] += 1
                examples[f"{name}_mismatch"].append(
                    {"problem_id": problem_id, "declared": declared, "brute_force": brute_count, "first_solutions": first_solutions}
                )
        if brute_count != 1:
            totals["not_unique_by_bruteforce"] += 1
            examples["not_unique_by_bruteforce"].append(
                {"problem_id": problem_id, "brute_force": brute_count, "first_solutions": first_solutions}
            )

        seen_clauses = Counter()
        duplicate_clauses = 0
        duplicate_literal_constraints = 0
        scope_mismatch_constraints = 0
        for constraint in problem.constraints:
            arity, sign_key, has_duplicate_vars, has_scope_mismatch = clause_signature(constraint)
            by_arity[str(arity)] += 1
            by_sign_pattern[sign_key] += 1
            if has_duplicate_vars:
                duplicate_literal_constraints += 1
            if has_scope_mismatch:
                scope_mismatch_constraints += 1

            normalized_lits = tuple((str(lit.get("var")), bool(lit.get("negated", False))) for lit in constraint.args)
            seen_clauses[normalized_lits] += 1

        duplicate_clauses = sum(count - 1 for count in seen_clauses.values() if count > 1)
        duplicate_clause_totals.append(duplicate_clauses)
        if duplicate_clauses:
            duplicate_clause_count_by_problem[str(duplicate_clauses)] += 1
        if duplicate_literal_constraints:
            duplicate_literals_by_problem[str(duplicate_literal_constraints)] += 1
            totals["problems_with_duplicate_literal_constraints"] += 1
        if scope_mismatch_constraints:
            scope_mismatch_by_problem[str(scope_mismatch_constraints)] += 1
            totals["problems_with_scope_mismatch_constraints"] += 1

        try:
            item = tensorize_problem(problem, tensor_cfg)
        except Exception as exc:  # noqa: BLE001
            totals["tensorize_exception"] += 1
            examples["tensorize_exception"].append({"problem_id": problem_id, "error": repr(exc)})
            continue

        tensor_verify = verify_tensorized_target(item)
        if not bool(tensor_verify.get("valid", False)):
            totals["tensorized_target_invalid"] += 1
            examples["tensorized_target_invalid"].append(
                {"problem_id": problem_id, "errors": tensor_verify.get("errors"), "energy": tensor_verify.get("total_energy")}
            )

        tensors = item.tensors
        exact_flags = tensors["factor_table_exact"].detach().cpu().tolist()
        total_counts = tensors["factor_allowed_total_count"].detach().cpu().tolist()
        stored_counts = tensors["factor_allowed_stored_count"].detach().cpu().tolist()
        candidate_counts = tensors["factor_candidate_tuple_count"].detach().cpu().tolist()
        factor_arities = tensors["constraint_arity"].detach().cpu().tolist()
        factor_exact_by_problem[str(sum(1 for flag in exact_flags if bool(flag)))] += 1

        if not all(bool(flag) for flag in exact_flags):
            totals["non_exact_factor_table_problem"] += 1
            examples["non_exact_factor_table_problem"].append({"problem_id": problem_id})

        for idx, constraint in enumerate(problem.constraints):
            expected_allowed = expected_boolean_clause_allowed_count(constraint)
            expected_candidates = 2 ** int(factor_arities[idx])
            if expected_allowed is not None and int(total_counts[idx]) != expected_allowed:
                totals["factor_allowed_count_anomaly"] += 1
                examples["factor_allowed_count_anomaly"].append(
                    {
                        "problem_id": problem_id,
                        "constraint_id": constraint.id,
                        "arity": int(factor_arities[idx]),
                        "expected_allowed": expected_allowed,
                        "actual_allowed": int(total_counts[idx]),
                    }
                )
            if int(stored_counts[idx]) != int(total_counts[idx]):
                totals["factor_stored_total_mismatch"] += 1
            if int(candidate_counts[idx]) != expected_candidates:
                totals["factor_candidate_count_anomaly"] += 1
                examples["factor_candidate_count_anomaly"].append(
                    {
                        "problem_id": problem_id,
                        "constraint_id": constraint.id,
                        "arity": int(factor_arities[idx]),
                        "expected_candidates": expected_candidates,
                        "actual_candidates": int(candidate_counts[idx]),
                    }
                )

    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "records": totals["records"],
            "splits": dict(sorted(splits.items())),
            "n_vars": summarize_numbers(n_vars_values),
            "n_clauses": summarize_numbers(n_clause_values),
            "domain_product": summarize_numbers(domain_products),
            "solution_count": summarize_numbers(solution_counts),
            "duplicate_clauses_per_problem": summarize_numbers(duplicate_clause_totals),
        },
        "distributions": {
            "by_n_vars": dict(sorted(by_n_vars.items(), key=lambda kv: int(kv[0]))),
            "by_n_clauses": dict(sorted(by_n_clauses.items(), key=lambda kv: int(kv[0]))),
            "by_k": dict(sorted(by_k.items(), key=lambda kv: int(kv[0]))),
            "by_arity": dict(sorted(by_arity.items(), key=lambda kv: int(kv[0]))),
            "by_sign_pattern": dict(sorted(by_sign_pattern.items())),
            "duplicate_clause_count_by_problem": dict(sorted(duplicate_clause_count_by_problem.items(), key=lambda kv: int(kv[0]))),
            "duplicate_literals_by_problem": dict(sorted(duplicate_literals_by_problem.items(), key=lambda kv: int(kv[0]))),
            "scope_mismatch_by_problem": dict(sorted(scope_mismatch_by_problem.items(), key=lambda kv: int(kv[0]))),
            "exact_factor_tables_per_problem": dict(sorted(factor_exact_by_problem.items(), key=lambda kv: int(kv[0]))),
        },
        "checks": {
            "certificate_invalid": totals["certificate_invalid"],
            "not_unique_by_bruteforce": totals["not_unique_by_bruteforce"],
            "record_solution_count_mismatch": totals["record_solution_count_mismatch"],
            "certificate_solution_count_mismatch": totals["certificate_solution_count_mismatch"],
            "tensorize_exception": totals["tensorize_exception"],
            "tensorized_target_invalid": totals["tensorized_target_invalid"],
            "non_exact_factor_table_problem": totals["non_exact_factor_table_problem"],
            "factor_allowed_count_anomaly": totals["factor_allowed_count_anomaly"],
            "factor_candidate_count_anomaly": totals["factor_candidate_count_anomaly"],
            "factor_stored_total_mismatch": totals["factor_stored_total_mismatch"],
            "problems_with_duplicate_literal_constraints": totals["problems_with_duplicate_literal_constraints"],
            "problems_with_scope_mismatch_constraints": totals["problems_with_scope_mismatch_constraints"],
        },
        "examples": {k: v[:10] for k, v in sorted(examples.items())},
    }


def write_summary(report: Dict[str, Any], path: Path) -> None:
    checks = report["checks"]
    dataset = report["dataset"]
    total_records = int(dataset["records"])
    hard_failures = [
        "certificate_invalid",
        "not_unique_by_bruteforce",
        "record_solution_count_mismatch",
        "certificate_solution_count_mismatch",
        "tensorize_exception",
        "tensorized_target_invalid",
        "non_exact_factor_table_problem",
        "factor_allowed_count_anomaly",
        "factor_candidate_count_anomaly",
        "factor_stored_total_mismatch",
        "problems_with_duplicate_literal_constraints",
        "problems_with_scope_mismatch_constraints",
    ]
    clean = all(int(checks.get(key, 0)) == 0 for key in hard_failures)

    lines = [
        "AHR-CSP Boolean Micro 128 Data Audit",
        f"Created: {report['created_at']}",
        "",
        f"Overall result: {'PASS' if clean else 'FAIL'}",
        f"Records audited: {total_records}",
        f"Splits: {dataset['splits']}",
        f"Variables: min={dataset['n_vars']['min']} max={dataset['n_vars']['max']} mean={dataset['n_vars']['mean']:.2f}",
        f"Clauses: min={dataset['n_clauses']['min']} max={dataset['n_clauses']['max']} mean={dataset['n_clauses']['mean']:.2f}",
        f"Domain product: min={dataset['domain_product']['min']} max={dataset['domain_product']['max']} mean={dataset['domain_product']['mean']:.2f}",
        f"Brute-force solution count: min={dataset['solution_count']['min']} max={dataset['solution_count']['max']} mean={dataset['solution_count']['mean']:.2f}",
        "",
        "Primary checks:",
    ]
    for key in hard_failures:
        value = int(checks.get(key, 0))
        rate = pct(value, total_records)
        lines.append(f"- {key}: {value} ({rate:.1%} of records, where record-scoped)")

    lines.extend(
        [
            "",
            "Distributions:",
            f"- by_n_vars: {report['distributions']['by_n_vars']}",
            f"- by_n_clauses: {report['distributions']['by_n_clauses']}",
            f"- by_k: {report['distributions']['by_k']}",
            f"- by_arity: {report['distributions']['by_arity']}",
            f"- by_sign_pattern: {report['distributions']['by_sign_pattern']}",
            f"- duplicate_clauses_per_problem: {dataset['duplicate_clauses_per_problem']}",
            "",
            "Interpretation:",
        ]
    )
    if clean:
        lines.append(
            "The Boolean 128 data contract looks clean: certificates verify, brute force confirms uniqueness, "
            "tensorized targets verify, and exact factor tables match Boolean clause expectations. The remaining "
            "blocker is therefore more likely optimization/curriculum/model dynamics than corrupt data."
        )
    else:
        lines.append(
            "The data/tensorization contract has failures. Fix these before running more GPU training, because "
            "the model would be learning from inconsistent targets or verifier semantics."
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=PROJECT_ROOT / "data" / "phase5p7_bool_micro_unique_128.jsonl")
    parser.add_argument("--report-json", type=Path, default=PROJECT_ROOT / "reports" / "phase5p7_bool_micro_128_data_audit_report.json")
    parser.add_argument("--summary", type=Path, default=PROJECT_ROOT / "reports" / "phase5p7_bool_micro_128_data_audit_summary.txt")
    args = parser.parse_args()

    ns = load_notebook_runtime()
    records = load_jsonl(args.dataset)
    report = audit_records(records, ns)

    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(json.dumps(report, indent=2, sort_keys=True, default=json_default) + "\n", encoding="utf-8")
    write_summary(report, args.summary)

    print(args.summary.read_text(encoding="utf-8"))
    return 0 if all(int(v) == 0 for v in report["checks"].values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())

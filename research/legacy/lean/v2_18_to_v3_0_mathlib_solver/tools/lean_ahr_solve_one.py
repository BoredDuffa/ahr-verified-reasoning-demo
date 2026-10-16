#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from lean_ahr_core import add_common_args, solve_user_spec, spec_from_statement
from lean_ahr_mathlib import DEFAULT_INDEX, DEFAULT_WORKSPACE, retrieve, verify_theorem


def solve_mathlib(args: argparse.Namespace, run_name: str) -> dict:
    imports = [x for item in args.imports for x in item.split(",") if x] if args.imports else []
    query = args.query or args.statement
    candidates = retrieve(query, args.retrieve_top_k, Path(args.index), Path(args.workspace), verify=True, timeout_s=args.timeout_s)
    proof_attempts: list[tuple[list[str], list[str], list[dict]]] = []
    for cand in candidates:
        imps = sorted(set(imports + [cand["import"]]))
        name = cand["full_name"]
        proof_attempts.extend([
            (imps, [f"exact {name}"], [cand]),
            (imps, [f"simpa using {name}"], [cand]),
            (imps, [f"apply {name}"], [cand]),
        ])
    if not proof_attempts and imports:
        proof_attempts.append((imports, ["simpa"], []))

    first_check = None
    for idx, (imps, proof, used) in enumerate(proof_attempts, start=1):
        check = verify_theorem(Path(args.workspace), imps, args.name, args.statement, proof, args.timeout_s)
        if first_check is None:
            first_check = check
        if check.get("accepted"):
            return {
                "schema": "ahr_lean_solve_one_v2_19_result",
                "run_name": run_name,
                "theorem": args.name,
                "family": args.family,
                "solved": True,
                "verified": True,
                "proof_lines": proof,
                "proof_depth": len(proof),
                "lean_calls": idx,
                "strategy": "mathlib_retrieved_theorem",
                "imports": imps,
                "used_theorems": used,
                "final_check": check,
                "elapsed_s": check.get("elapsed_s"),
                "failure_reason": None,
                "clinical_use": "forbidden",
            }
    return {
        "schema": "ahr_lean_solve_one_v2_19_result",
        "run_name": run_name,
        "theorem": args.name,
        "family": args.family,
        "solved": False,
        "verified": False,
        "proof_lines": [],
        "proof_depth": 0,
        "lean_calls": len(proof_attempts),
        "strategy": "mathlib_retrieval_failed",
        "imports": imports,
        "used_theorems": candidates,
        "final_check": first_check,
        "failure_reason": "no_verified_mathlib_proof_found",
        "clinical_use": "forbidden",
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Solve and verify one small Lean theorem statement.")
    ap.add_argument("--name", required=True)
    ap.add_argument("--statement", required=True)
    ap.add_argument("--family", default="user_supplied")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--mathlib", action="store_true")
    ap.add_argument("--imports", nargs="*", default=None)
    ap.add_argument("--query", default=None)
    ap.add_argument("--workspace", default=str(DEFAULT_WORKSPACE))
    ap.add_argument("--index", default=str(DEFAULT_INDEX))
    ap.add_argument("--retrieve-top-k", type=int, default=20)
    add_common_args(ap)
    args = ap.parse_args()

    run_name = args.run_name or f"lean_ahr_v2_15_solve_one_{int(time.time())}"
    out_dir = Path("lean_generated") / run_name
    try:
        if args.mathlib:
            row = solve_mathlib(args, run_name)
            print(json.dumps(row, ensure_ascii=False, sort_keys=True))
            sys.exit(0 if row.get("solved") else 2)
        spec = spec_from_statement(args.name, args.statement, args.family)
        row = solve_user_spec(
            spec=spec,
            out_dir=out_dir,
            lean_bin=args.lean_bin,
            timeout_s=args.timeout_s,
            max_direct=args.max_direct,
            checkpoint=args.checkpoint,
            cpu=args.cpu,
            max_k=args.max_k,
            max_depth=args.max_depth,
            max_nodes=args.max_nodes,
            state_timeout_s=args.state_timeout_s,
            state_capture_retries=args.state_capture_retries,
            fallback=not args.no_fallback,
        )
        row["schema"] = "ahr_lean_solve_one_v2_15_result"
        row["run_name"] = run_name
        print(json.dumps(row, ensure_ascii=False, sort_keys=True))
        sys.exit(0 if row.get("solved") else 2)
    except Exception as exc:
        print(json.dumps({
            "schema": "ahr_lean_solve_one_v2_15_result",
            "run_name": run_name,
            "theorem": args.name,
            "solved": False,
            "verified": False,
            "proof_lines": [],
            "failure_reason": "infrastructure_error",
            "error": str(exc),
        }, ensure_ascii=False, sort_keys=True))
        sys.exit(1)


if __name__ == "__main__":
    main()

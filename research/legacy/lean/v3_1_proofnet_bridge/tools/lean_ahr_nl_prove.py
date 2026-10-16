#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from lean_ahr_mathlib import DEFAULT_INDEX, DEFAULT_WORKSPACE, retrieve, verify_theorem


def statement_from_check_output(name: str, stdout_excerpt: str | None) -> str | None:
    text = stdout_excerpt or ""
    if " : " not in text:
        return None
    typ = text.split(" : ", 1)[1].strip()
    if not typ:
        return None
    return f"theorem THEOREM_NAME : {typ} := by"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", required=True)
    ap.add_argument("--name", default="nl_theorem")
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--workspace", default=str(DEFAULT_WORKSPACE))
    ap.add_argument("--index", default=str(DEFAULT_INDEX))
    ap.add_argument("--timeout-s", type=int, default=30)
    args = ap.parse_args()

    candidates = retrieve(args.query, args.top_k, Path(args.index), Path(args.workspace), verify=True, timeout_s=args.timeout_s)
    attempts = []
    for cand in candidates:
        stmt = statement_from_check_output(args.name, cand.get("check", {}).get("stdout_excerpt"))
        if not stmt:
            continue
        proof = [f"exact {cand['full_name']}"]
        check = verify_theorem(Path(args.workspace), [cand["import"]], args.name, stmt, proof, args.timeout_s)
        attempts.append({"candidate": cand, "statement": stmt, "proof_lines": proof, "check": check})
        if check.get("accepted"):
            print(json.dumps({
                "schema": "ahr_lean_nl_prove_v3_0_result",
                "query": args.query,
                "solved": True,
                "verified": True,
                "statement": stmt,
                "proof_lines": proof,
                "imports": [cand["import"]],
                "used_theorems": [cand],
                "final_check": check,
                "failure_reason": None,
                "clinical_use": "forbidden",
            }, ensure_ascii=False, sort_keys=True))
            return
    print(json.dumps({
        "schema": "ahr_lean_nl_prove_v3_0_result",
        "query": args.query,
        "solved": False,
        "verified": False,
        "attempts": attempts[:5],
        "failure_reason": "no_verified_known_theorem_wrapper_found",
        "clinical_use": "forbidden",
    }, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()

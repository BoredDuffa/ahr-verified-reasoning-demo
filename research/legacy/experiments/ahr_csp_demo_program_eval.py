#!/usr/bin/env python3
"""Gate checks for the verifier-gated AHR-CSP demo program backend."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from ahr_csp_demo_program_backend import solve  # type: ignore
from ahr_csp_nl_pipeline import QwenJsonTranslator, validate_dsl  # type: ignore


ROOT = Path(__file__).resolve().parents[1]
REPORT_JSON = ROOT / "reports" / "ahr_csp_demo_program_eval_20260612.json"
REPORT_CSV = ROOT / "reports" / "ahr_csp_demo_program_eval_20260612.csv"
REPORT_MD = ROOT / "reports" / "ahr_csp_demo_program_eval_20260612.md"


CASES = [
    ("What is the probability of rolling a sum of 7 with two dice?", "1/6"),
    ("What is the probability of rolling a sum of seven with two dice?", "1/6"),
    ("A fair coin is tossed 3 times. What is the probability of getting exactly 2 heads?", "3/8"),
    ("If I flip 50 coins, what is the chance there is a sequence of 5 heads in a row?", "155339093675555/281474976710656"),
    ("In a room of 23 people, what is the probability at least two share a birthday?", "0.507297"),
    ("What is the mean of 4, 8, 10, and 12?", "17/2"),
    ("Find the mean and variance of 2, 4, 6, and 8.", "5"),
    ("Alice is taller than Bob. Bob is taller than Cara. Who is tallest?", "Alice"),
    ("Maria is older than John. John is older than Lee. Who is oldest?", "Maria"),
    ("Add 12 and 19.", "31"),
    ("24 cookies are split equally among 6 kids. How many per kid?", "4"),
    ("Write me a poem about the model.", None),
    ("What is the weather tomorrow?", None),
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["rules", "qwen"], default="rules")
    args = parser.parse_args()
    translator = QwenJsonTranslator(mode=args.mode)
    rows = []
    for question, expected_fragment in CASES:
        translation = translator.translate(question)
        valid, reason = validate_dsl(translation.dsl)
        result = None
        if reason == "demo_program_backend":
            result = solve(question, translation.dsl)
        status = "abstained"
        answer = None
        verified = False
        if result is not None:
            verified = bool(result.get("verified"))
            answer = result.get("answer")
            status = "correct" if verified and expected_fragment and expected_fragment in str(answer) else "wrong" if verified else "abstained"
        elif expected_fragment is None and reason in {"unsupported", "known_but_untrained"}:
            status = "abstained"
        elif expected_fragment is not None:
            status = "wrong"
        if expected_fragment is None and verified:
            status = "wrong"
        rows.append(
            {
                "question": question,
                "expected_fragment": expected_fragment,
                "family": translation.dsl.get("family"),
                "dsl_reason": reason,
                "verified": verified,
                "status": status,
                "answer": answer,
                "abstain_reason": None if result is None else result.get("abstain_reason"),
                "backend_type": None if result is None else result.get("backend_type"),
                "trace": [] if result is None else result.get("trace", []),
            }
        )

    summary = {
        "clinical_use": "forbidden",
        "mode": args.mode,
        "created_at_unix": time.time(),
        "total": len(rows),
        "correct": sum(1 for row in rows if row["status"] == "correct"),
        "wrong": sum(1 for row in rows if row["status"] == "wrong"),
        "abstained": sum(1 for row in rows if row["status"] == "abstained"),
        "verified": sum(1 for row in rows if row["verified"]),
    }
    report = {"summary": summary, "rows": rows}
    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    with REPORT_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["question", "expected_fragment", "family", "dsl_reason", "verified", "status", "answer", "abstain_reason", "backend_type"])
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in writer.fieldnames})
    lines = [
        "# AHR-CSP Demo Program Eval",
        "",
        "- clinical_use: forbidden",
        f"- mode: {args.mode}",
        f"- total: {summary['total']}",
        f"- correct: {summary['correct']}",
        f"- wrong: {summary['wrong']}",
        f"- abstained: {summary['abstained']}",
        f"- verified: {summary['verified']}",
        "",
        "## Cases",
        "",
    ]
    for row in rows:
        lines.append(f"- {row['status']}: {row['question']} -> {row['answer'] or row['abstain_reason'] or row['dsl_reason']}")
    REPORT_MD.write_text("\n".join(lines) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 1 if summary["wrong"] else 0


if __name__ == "__main__":
    raise SystemExit(main())

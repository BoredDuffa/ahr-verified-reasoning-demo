#!/usr/bin/env python3
"""Evaluate the AHR-CSP-only natural-language pipeline contract.

This gate intentionally distinguishes three cases:
- published operation_synthesis requests must execute through AHR-CSP;
- known but untrained families must parse and abstain;
- unsupported prompts must abstain.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
EXP_DIR = ROOT / "experiments"
if str(EXP_DIR) not in sys.path:
    sys.path.insert(0, str(EXP_DIR))

from ahr_csp_nl_pipeline import QwenJsonTranslator, paraphrase_rows, validate_dsl  # type: ignore

REPORT_JSON = ROOT / "reports" / "ahr_csp_nl_pipeline_eval_20260609.json"
REPORT_CSV = ROOT / "reports" / "ahr_csp_nl_pipeline_eval_20260609.csv"
REPORT_MD = ROOT / "reports" / "ahr_csp_nl_pipeline_eval_20260609.md"


def evaluate_translator(mode: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    os.environ["AHR_CSP_TRANSLATOR_MODE"] = mode
    translator = QwenJsonTranslator(mode=mode)
    records: list[dict[str, Any]] = []
    summary: dict[str, Any] = {
        "total": 0,
        "valid_json": 0,
        "correct_family": 0,
        "wrong_family": 0,
        "invalid_dsl": 0,
        "by_expected_family": {},
    }
    for question, expected_family in paraphrase_rows():
        translation = translator.translate(question)
        valid, reason = validate_dsl(translation.dsl)
        got_family = translation.dsl.get("family")
        correct = got_family == expected_family
        summary["total"] += 1
        summary["valid_json"] += int(translation.valid_json)
        summary["correct_family"] += int(correct)
        summary["wrong_family"] += int(not correct)
        summary["invalid_dsl"] += int(not valid and got_family != "unsupported")
        stats = summary["by_expected_family"].setdefault(expected_family, {"total": 0, "correct": 0, "wrong": 0})
        stats["total"] += 1
        stats["correct"] += int(correct)
        stats["wrong"] += int(not correct)
        records.append(
            {
                "question": question,
                "expected_family": expected_family,
                "predicted_family": got_family,
                "valid_json": translation.valid_json,
                "valid_dsl": valid,
                "dsl_reason": reason,
                "translator_model": translation.translator_model,
                "translator_source": translation.source,
                "raw_output": translation.raw_output,
            }
        )
    return summary, records


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="rules", choices=["rules", "qwen"], help="Use rules for deterministic CI or qwen for live HF translator gate.")
    args = parser.parse_args()

    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    summary, records = evaluate_translator(args.mode)
    report = {
        "clinical_use": "forbidden",
        "kind": "ahr_csp_nl_pipeline_eval",
        "mode": args.mode,
        "translator_policy": "translator emits DSL only; AHR-CSP/verifier owns answers",
        "published_answer_families": ["operation_synthesis"],
        "known_untrained_families": [
            "short_arithmetic_program",
            "probability_counting_program",
            "statistics_program",
            "finite_logic_constraints",
            "sequence_graph_reasoning",
        ],
        "summary": summary,
    }
    REPORT_JSON.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with REPORT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)
    lines = [
        "# AHR-CSP Natural-Language Pipeline Eval",
        "",
        "- clinical_use: forbidden",
        f"- mode: {args.mode}",
        "- answer policy: Qwen/rules emit DSL only; AHR-CSP plus CPU verifier gates answers",
        f"- total: {summary['total']}",
        f"- valid_json: {summary['valid_json']}",
        f"- correct_family: {summary['correct_family']}",
        f"- wrong_family: {summary['wrong_family']}",
        f"- invalid_dsl: {summary['invalid_dsl']}",
        "",
        "## By Family",
        "",
    ]
    for family, stats in sorted(summary["by_expected_family"].items()):
        lines.append(f"- {family}: {stats['correct']}/{stats['total']} correct, {stats['wrong']} wrong")
    lines.extend(
        [
            "",
            "## Publication Status",
            "",
            "- `operation_synthesis`: published through current AHR-CSP checkpoint gate.",
            "- math/stats/probability/logic/sequence families: schema-known but not answer-published until AHR-CSP checkpoints are trained and gated.",
        ]
    )
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0 if summary["wrong_family"] == 0 and summary["invalid_dsl"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())

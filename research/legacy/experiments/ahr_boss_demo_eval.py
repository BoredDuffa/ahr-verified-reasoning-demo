#!/usr/bin/env python3
"""Boss-demo evaluation set for the AHR-only natural-language demo."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from ahr_csp_demo_program_backend import solve as solve_demo_program  # type: ignore
from ahr_csp_nl_pipeline import QwenJsonTranslator, validate_dsl  # type: ignore


ROOT = Path(__file__).resolve().parents[1]
REPORT_JSON = ROOT / "reports" / "ahr_boss_demo_eval_20260612.json"
REPORT_CSV = ROOT / "reports" / "ahr_boss_demo_eval_20260612.csv"
REPORT_MD = ROOT / "reports" / "ahr_boss_demo_questions_20260612.md"


@dataclass
class Case:
    category: str
    family: str
    question: str
    expectation: str


def build_cases() -> list[Case]:
    cases: list[Case] = []

    probability_questions = [
        "What is the probability of rolling a sum of 7 with two dice?",
        "What are the odds of rolling a total of 9 with two dice?",
        "What is the chance the sum is 4 when rolling two dice?",
        "A fair coin is tossed 3 times. What is the probability of exactly 2 heads?",
        "A fair coin is flipped 5 times. What is the probability of at least 3 heads?",
        "A fair coin is flipped 6 times. What is the probability of at most 2 heads?",
        "A fair coin is tossed 8 times. What is the probability of between 2 and 5 heads?",
        "If I flip 50 coins, what is the chance there is a sequence of 5 heads in a row?",
        "If I flip 20 coins, what is the probability of a run of 4 heads?",
        "In a room of 23 people, what is the probability at least two share a birthday?",
        "In a room of 35 people, what is the chance at least two share a birthday?",
        "A card is drawn from a standard deck. What is the probability of drawing a king or a heart?",
        "A card is drawn from a standard deck. What is the probability of drawing a queen?",
        "A card is drawn from a standard deck. What is the probability of drawing a spade?",
        "A bag has 5 red, 3 blue, and 2 green balls. Two are drawn without replacement. What is the probability both are red?",
        "A bag has 5 red, 3 blue, and 2 green balls. Two are drawn without replacement. What is the probability both are blue?",
        "A bag has 4 yellow, 6 black, and 5 white balls. Two are drawn without replacement. What is the probability both are black?",
        "What is the probability of rolling a sum of 12 with two dice?",
        "What is the probability of rolling a total of 2 with two dice?",
        "A fair coin is tossed 10 times. What is the probability of exactly 5 heads?",
        "A fair coin is tossed 10 times. What is the probability of no more than 4 heads?",
        "A fair coin is tossed 9 times. What is the probability of no fewer than 6 heads?",
        "If I flip 30 coins, what is the chance of 6 heads in a row?",
        "A card is drawn from a deck. What is the probability of drawing an ace or a diamond?",
        "A bag has 7 red and 3 blue balls. Two are drawn without replacement. What is the probability both red?",
    ]
    for q in probability_questions:
        cases.append(Case("green", "probability_counting_program", q, "answer"))

    stats_questions = [
        "What is the mean of 4, 8, 10, and 12?",
        "What is the average of 3, 9, 12, and 16?",
        "What is the mean and population variance of 2, 4, 6, and 8?",
        "What is the mean and sample variance of 4, 8, 10, and 12?",
        "Find the sample variance of 5, 7, 9, and 11.",
        "What is the median of 4, 8, 10, and 12?",
        "What is the median of 3, 9, 1, 7, and 5?",
        "What is the mode of 2, 4, 4, 7, and 9?",
        "What is the mode of 1, 1, 2, 3, and 5?",
        "Compute the mean of 10, 20, 30, 40, and 50.",
        "Compute the population variance of 1, 2, 3, 4, and 5.",
        "What is the average and sample variance of 6, 8, 10, and 12?",
        "Find the median of 11, 4, 9, 2, and 8.",
        "Find the mode of 6, 6, 8, 10, and 12.",
        "What is the mean of 100, 120, and 140?",
        "What is the population variance of 10, 10, 20, and 20?",
        "What is the sample variance of 10, 10, 20, and 20?",
        "What is the median of 1, 2, 3, 4?",
        "What is the mean and population variance of 7, 7, 7, and 7?",
        "What is the average of 15, 25, 35, and 45?",
    ]
    for q in stats_questions:
        cases.append(Case("green", "statistics_program", q, "answer"))

    arithmetic_questions = [
        "Add 12 and 19.",
        "What is 15 plus 27?",
        "Maria had 50 dollars and spent 18. How much remains?",
        "Tom has 30 dollars and spends 12. How much remains?",
        "A box has 6 rows of 8 pencils. How many pencils are there?",
        "A crate has 7 rows of 9 bottles. How many bottles are there?",
        "24 cookies are split equally among 6 kids. How many per kid?",
        "45 candies are shared equally among 9 people. How many per person?",
        "Subtract 14 from 39.",
        "Multiply 11 and 12.",
        "Divide 81 by 9.",
        "Add 5, 6, and 7.",
        "A runner covers 8 miles each day for 5 days. How many miles?",
        "There are 36 cards split equally into 4 piles. How many in each pile?",
        "A shop had 90 items and sold 35. How many are left?",
        "What is the product of 13 and 4?",
        "What is 100 minus 42?",
        "60 tickets are shared equally among 12 groups. How many each?",
        "Add 21 and 34.",
        "A classroom has 4 tables with 6 students at each table. How many students?",
    ]
    for q in arithmetic_questions:
        cases.append(Case("green", "short_arithmetic_program", q, "answer"))

    logic_questions = [
        "Alice is taller than Bob. Bob is taller than Cara. Who is tallest?",
        "Maria is older than John. John is older than Lee. Who is oldest?",
        "Dina is taller than Eli. Eli is taller than Faye. Who is tallest?",
        "Hannah is older than Isaac. Isaac is older than June. Who is oldest?",
        "Liam is taller than Noah. Noah is taller than Owen. Who is tallest?",
        "Ava is older than Mia. Mia is older than Zoe. Who is oldest?",
        "Red is higher than Blue. Blue is higher than Green. Who is highest?",
        "Alpha is greater than Beta. Beta is greater than Gamma. Which is greatest?",
        "Sam is taller than Tim. Tim is taller than Uma. Who is tallest?",
        "Nora is older than Omar. Omar is older than Pia. Who is oldest?",
        "Iris is taller than Jade. Jade is taller than Kira. Who is tallest?",
        "Ben is older than Cal. Cal is older than Dan. Who is oldest?",
        "East is higher than West. West is higher than South. Which is highest?",
        "Rita is taller than Sia. Sia is taller than Tia. Who is tallest?",
        "A is greater than B. B is greater than C. Which is greatest?",
        "Gina is older than Hari. Hari is older than Imani. Who is oldest?",
        "Olive is taller than Pete. Pete is taller than Quinn. Who is tallest?",
        "Yara is older than Xena. Xena is older than Willa. Who is oldest?",
        "Leo is taller than Max. Max is taller than Nia. Who is tallest?",
        "Uma is older than Val. Val is older than Wes. Who is oldest?",
    ]
    for q in logic_questions:
        cases.append(Case("green", "finite_logic_constraints", q, "answer"))

    unsupported_questions = [
        "Write me a poem about probability.",
        "What is the weather tomorrow?",
        "A loaded die has unknown probabilities. What is the probability of rolling a 6?",
        "Prove the Riemann hypothesis.",
        "Solve this full Sudoku puzzle with no board given.",
        "What is the probability the stock market goes up tomorrow?",
        "Who will win the match tonight?",
        "Tell me a story about the model.",
        "What is the capital of France?",
        "A card is drawn from a custom deck. What is the probability of a red card?",
        "What is the mean of the numbers I am thinking of?",
        "A bag has some red balls and some blue balls. What is the probability both are red?",
        "Solve this logic puzzle: everyone is lying and telling the truth.",
        "What is the chance of rain in Dublin tomorrow?",
        "Make up a hard math problem and solve it.",
    ]
    for q in unsupported_questions:
        cases.append(Case("red", "unsupported", q, "abstain"))

    return cases


def direct_answer(question: str, translator: QwenJsonTranslator) -> dict[str, Any]:
    translation = translator.translate(question)
    valid, reason = validate_dsl(translation.dsl)
    family = translation.dsl.get("family")
    if valid and reason == "demo_program_backend":
        result = solve_demo_program(question, translation.dsl)
        return {
            "verified": bool(result.get("verified")),
            "answer": result.get("answer"),
            "abstain_reason": result.get("abstain_reason"),
            "backend_type": result.get("backend_type"),
            "family": family,
            "dsl_reason": reason,
            "trace": result.get("trace", []),
        }
    return {
        "verified": False,
        "answer": None,
        "abstain_reason": reason if valid else "invalid_dsl",
        "backend_type": "abstain",
        "family": family,
        "dsl_reason": reason,
        "trace": [],
    }


def http_answer(question: str, url: str) -> dict[str, Any]:
    req = urllib.request.Request(
        url.rstrip("/") + "/api/ask",
        data=json.dumps({"question": question}).encode("utf-8"),
        headers={"content-type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=90) as response:
        data = json.load(response)
    return {
        "verified": bool(data.get("verified")),
        "answer": data.get("answer"),
        "abstain_reason": data.get("abstain_reason"),
        "backend_type": data.get("backend_type"),
        "family": data.get("dsl", {}).get("family"),
        "dsl_reason": None,
        "trace": data.get("trace", []),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["rules", "qwen"], default="rules")
    parser.add_argument("--url", default="", help="Optional live server base URL, e.g. http://127.0.0.1:7860")
    args = parser.parse_args()

    cases = build_cases()
    translator = QwenJsonTranslator(mode=args.mode)
    rows: list[dict[str, Any]] = []
    for idx, case in enumerate(cases, 1):
        result = http_answer(case.question, args.url) if args.url else direct_answer(case.question, translator)
        if case.expectation == "answer":
            status = "correct" if result["verified"] and result["answer"] and result["backend_type"] in {"trained_ahr_ranker", "verified_program_backend"} else "wrong"
        else:
            status = "correct" if not result["verified"] and result["answer"] is None else "wrong"
        rows.append(
            {
                "id": idx,
                "category": case.category,
                "expected": case.expectation,
                "question": case.question,
                "family": result["family"],
                "verified": result["verified"],
                "backend_type": result["backend_type"],
                "answer": result["answer"],
                "abstain_reason": result["abstain_reason"],
                "status": status,
            }
        )

    summary = {
        "clinical_use": "forbidden",
        "created_at_unix": time.time(),
        "mode": args.mode,
        "url": args.url or None,
        "total": len(rows),
        "correct": sum(1 for row in rows if row["status"] == "correct"),
        "wrong": sum(1 for row in rows if row["status"] == "wrong"),
        "verified": sum(1 for row in rows if row["verified"]),
        "abstained": sum(1 for row in rows if not row["verified"]),
    }
    report = {"summary": summary, "rows": rows}
    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with REPORT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    green = [row for row in rows if row["category"] == "green"]
    red = [row for row in rows if row["category"] == "red"]
    lines = [
        "# AHR Boss Demo Question Sheet",
        "",
        "- clinical_use: forbidden",
        f"- mode: {args.mode}",
        f"- total: {summary['total']}",
        f"- correct: {summary['correct']}",
        f"- wrong: {summary['wrong']}",
        f"- verified answers: {summary['verified']}",
        f"- abstentions: {summary['abstained']}",
        "",
        "## Green: Safe To Ask Live",
        "",
    ]
    lines.extend(f"- {row['question']}" for row in green)
    lines.extend(["", "## Red: Should Abstain", ""])
    lines.extend(f"- {row['question']}" for row in red)
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["wrong"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

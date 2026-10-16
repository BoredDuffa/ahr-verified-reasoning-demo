#!/usr/bin/env python3
"""Evaluate the live QA demo against a 1,000-question Hugging Face math set.

This benchmark is deliberately external to the hand-built boss/probability demo:
it samples answerable questions from GSM8K and MATH, routes each question through
the current live demo, and reports correct/wrong/not_answered coverage.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import re
import time
import urllib.parse
import urllib.request
from urllib.error import HTTPError
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

ROOT = Path(__file__).resolve().parent.parent
REPORT_DIR = ROOT / "reports"
HF_API = "https://datasets-server.huggingface.co"
DEFAULT_JSON = REPORT_DIR / "ahr_hf_math_benchmark_1000_20260608.json"


@dataclass(frozen=True)
class Case:
    case_id: str
    source_dataset: str
    source_config: str
    source_split: str
    source_row_idx: int
    difficulty: str
    category: str
    question: str
    expected_answer: str
    expected_solution: str


def load_demo_module() -> Any:
    path = ROOT / "experiments" / "ahr_question_answer_demo.py"
    spec = importlib.util.spec_from_file_location("ahr_question_answer_demo", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def hf_get(endpoint: str, params: Dict[str, Any]) -> Dict[str, Any]:
    url = f"{HF_API}/{endpoint}?" + urllib.parse.urlencode(params)
    for attempt in range(5):
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            if exc.code != 429 or attempt == 4:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError("unreachable")


def hf_rows(dataset: str, config: str, split: str, limit: int) -> Iterable[Dict[str, Any]]:
    offset = 0
    while offset < limit:
        length = min(100, limit - offset)
        payload = hf_get("rows", {
            "dataset": dataset,
            "config": config,
            "split": split,
            "offset": offset,
            "length": length,
        })
        rows = payload.get("rows", [])
        if not rows:
            break
        yield from rows
        offset += len(rows)


def extract_gsm8k_answer(answer: str) -> str:
    match = re.search(r"####\s*([^\n]+)", answer)
    return match.group(1).strip().replace(",", "") if match else answer.strip()


def extract_boxed_answer(solution: str) -> str:
    starts = [m.start() for m in re.finditer(r"\\boxed\{", solution)]
    if not starts:
        return solution.strip()
    i = starts[-1] + len(r"\boxed{")
    depth = 1
    out: List[str] = []
    while i < len(solution) and depth:
        ch = solution[i]
        if ch == "{":
            depth += 1
            out.append(ch)
        elif ch == "}":
            depth -= 1
            if depth:
                out.append(ch)
        else:
            out.append(ch)
        i += 1
    return "".join(out).strip()


def clean_answer(value: str) -> str:
    s = str(value).strip()
    s = s.replace("$", "").replace(",", "")
    s = s.replace("\\dfrac", "\\frac")
    s = s.replace("\\!", "").replace("\\,", "")
    s = s.replace("\\left", "").replace("\\right", "")
    s = re.sub(r"\\text\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def answer_variants(value: str) -> set[str]:
    s = clean_answer(value)
    variants = {s, s.lower()}
    frac = re.fullmatch(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", s)
    if frac:
        num, den = frac.group(1), frac.group(2)
        variants.add(f"{num}/{den}")
        try:
            f = Fraction(int(num), int(den))
            variants.add(str(f))
            variants.add(f"{float(f):.6f}")
        except Exception:
            pass
    simple_frac = re.fullmatch(r"(-?\d+)\s*/\s*(-?\d+)", s)
    if simple_frac:
        try:
            f = Fraction(int(simple_frac.group(1)), int(simple_frac.group(2)))
            variants.add(str(f))
            variants.add(f"{float(f):.6f}")
        except Exception:
            pass
    number = re.fullmatch(r"-?\d+(?:\.\d+)?", s)
    if number:
        variants.add(str(float(s)).rstrip("0").rstrip("."))
    return {v for v in variants if v}


def numeric_tokens(value: str) -> set[str]:
    s = clean_answer(value)
    out = set(re.findall(r"\\frac\{[^{}]+\}\{[^{}]+\}|-?\d+\s*/\s*-?\d+|-?\d+(?:\.\d+)?%?", s))
    expanded = set(out)
    for token in list(out):
        for variant in answer_variants(token.rstrip("%")):
            expanded.add(variant)
    return {token.strip().lower() for token in expanded if token.strip()}


def answer_matches(observed: str, expected: str) -> bool:
    observed_clean = clean_answer(observed).lower()
    expected_variants = {variant.lower() for variant in answer_variants(expected)}
    expected_numeric = numeric_tokens(expected)
    if expected_numeric:
        return bool(expected_numeric & numeric_tokens(observed))
    for variant in expected_variants:
        if variant and variant in observed_clean:
            return True
    return False


def build_cases() -> List[Case]:
    cases: List[Case] = []

    for item in hf_rows("openai/gsm8k", "main", "test", 500):
        row = item["row"]
        cases.append(Case(
            case_id=f"gsm8k_test_{item['row_idx']:04d}",
            source_dataset="openai/gsm8k",
            source_config="main",
            source_split="test",
            source_row_idx=int(item["row_idx"]),
            difficulty="grade_school",
            category="gsm8k",
            question=row["question"],
            expected_answer=extract_gsm8k_answer(row["answer"]),
            expected_solution=row["answer"],
        ))

    math_plan = [
        ("algebra", 100),
        ("counting_and_probability", 100),
        ("geometry", 50),
        ("intermediate_algebra", 100),
        ("number_theory", 50),
        ("prealgebra", 75),
        ("precalculus", 25),
    ]
    for config, limit in math_plan:
        for item in hf_rows("EleutherAI/hendrycks_math", config, "test", limit):
            row = item["row"]
            cases.append(Case(
                case_id=f"math_{config}_{item['row_idx']:04d}",
                source_dataset="EleutherAI/hendrycks_math",
                source_config=config,
                source_split="test",
                source_row_idx=int(item["row_idx"]),
                difficulty=row.get("level", "unknown"),
                category=row.get("type", config),
                question=row["problem"],
                expected_answer=extract_boxed_answer(row["solution"]),
                expected_solution=row["solution"],
            ))

    return cases


def cases_from_existing_report(path: Path) -> List[Case]:
    data = json.loads(path.read_text(encoding="utf-8"))
    cases: List[Case] = []
    for row in data.get("results", []):
        cases.append(Case(
            case_id=row["case_id"],
            source_dataset=row["source_dataset"],
            source_config=row["source_config"],
            source_split=row["source_split"],
            source_row_idx=int(row["source_row_idx"]),
            difficulty=row["difficulty"],
            category=row["category"],
            question=row["question"],
            expected_answer=row["expected_answer"],
            expected_solution=row.get("expected_solution", ""),
        ))
    if not cases:
        raise RuntimeError(f"No cached cases found in {path}")
    return cases


def score_case(demo: Any, case: Case) -> Dict[str, Any]:
    routed = demo.live_demo_from_text(case.question)
    item = routed[0] if routed else {"id": "unsupported_question", "answer": ""}
    observed_id = str(item.get("id", "unsupported_question"))
    observed_answer = str(item.get("answer", ""))
    if observed_id == "unsupported_question":
        status = "not_answered"
    elif answer_matches(observed_answer, case.expected_answer):
        status = "correct"
    else:
        status = "wrong"
    verification = item.get("verification", {})
    return {
        **asdict(case),
        "observed_id": observed_id,
        "observed_backend_type": item.get("backend_type"),
        "observed_answer": observed_answer,
        "status": status,
        "reasoning_notes": verification if isinstance(verification, dict) else {},
    }


def rollup(results: List[Dict[str, Any]], key: str) -> Dict[str, Dict[str, int]]:
    out: Dict[str, Counter[str]] = defaultdict(Counter)
    for row in results:
        name = str(row[key])
        out[name]["total"] += 1
        out[name][row["status"]] += 1
    return {
        name: {
            "total": counts["total"],
            "correct": counts["correct"],
            "wrong": counts["wrong"],
            "not_answered": counts["not_answered"],
        }
        for name, counts in sorted(out.items())
    }


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def markdown_report(report: Dict[str, Any]) -> str:
    s = report["summary"]
    lines = [
        "# AHR Hugging Face Math Benchmark",
        "",
        f"Created UTC: {report['created_at_utc']}",
        "",
        "## Summary",
        "",
        f"- Total cases: {s['total']}",
        f"- Correct: {s['correct']}",
        f"- Wrong: {s['wrong']}",
        f"- Not answered: {s['not_answered']}",
        f"- Answered accuracy: {s['answered_accuracy']:.4f}",
        f"- Coverage: {s['coverage']:.4f}",
        f"- Strict score: {s['strict_score']:.4f}",
        "",
        "## Sources",
        "",
        "| source | total | correct | wrong | not_answered |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name, row in report["by_source"].items():
        lines.append(f"| {name} | {row['total']} | {row['correct']} | {row['wrong']} | {row['not_answered']} |")
    lines += [
        "",
        "## Categories",
        "",
        "| category | total | correct | wrong | not_answered |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name, row in report["by_category"].items():
        lines.append(f"| {name} | {row['total']} | {row['correct']} | {row['wrong']} | {row['not_answered']} |")
    lines += [
        "",
        "## Answered Examples",
        "",
    ]
    answered = [r for r in report["results"] if r["status"] != "not_answered"][:20]
    if not answered:
        lines.append("No answered cases.")
    for row in answered:
        lines.append(f"- `{row['status']}` `{row['case_id']}` expected `{row['expected_answer']}`, observed `{row['observed_answer']}`")
    lines += [
        "",
        "## Notes",
        "",
        "- This is an external coverage benchmark against Hugging Face datasets.",
        "- The current live QA demo is a router over verified symbolic backends, not a general math-language model.",
        "- `not_answered` means the router abstained instead of fabricating an answer.",
        "",
    ]
    return "\n".join(lines)


def write_csv(path: Path, results: List[Dict[str, Any]]) -> None:
    fields = [
        "case_id",
        "source_dataset",
        "source_config",
        "source_row_idx",
        "difficulty",
        "category",
        "status",
        "question",
        "expected_answer",
        "observed_id",
        "observed_answer",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in results:
            writer.writerow({field: row.get(field, "") for field in fields})


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    demo = load_demo_module()
    try:
        cases = build_cases()
    except HTTPError as exc:
        if exc.code != 429 or not DEFAULT_JSON.exists():
            raise
        cases = cases_from_existing_report(DEFAULT_JSON)
    if args.limit is not None:
        cases = cases[: args.limit]

    results = [score_case(demo, case) for case in cases]
    counts = Counter(row["status"] for row in results)
    answered = counts["correct"] + counts["wrong"]
    total = len(results)
    summary = {
        "total": total,
        "correct": counts["correct"],
        "wrong": counts["wrong"],
        "not_answered": counts["not_answered"],
        "answered_accuracy": counts["correct"] / answered if answered else 0.0,
        "coverage": answered / total if total else 0.0,
        "strict_score": counts["correct"] / total if total else 0.0,
    }
    report = {
        "kind": "ahr_hf_math_benchmark",
        "created_at_unix": time.time(),
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "clinical_use": "forbidden",
        "sources": [
            {"dataset": "openai/gsm8k", "config": "main", "split": "test", "count": 500},
            {"dataset": "EleutherAI/hendrycks_math", "config": "mixed_test", "split": "test", "count": 500},
        ],
        "summary": summary,
        "by_source": rollup(results, "source_dataset"),
        "by_category": rollup(results, "category"),
        "by_difficulty": rollup(results, "difficulty"),
        "results": results,
    }

    json_path = DEFAULT_JSON
    md_path = REPORT_DIR / "ahr_hf_math_benchmark_1000_20260608.md"
    csv_path = REPORT_DIR / "ahr_hf_math_benchmark_1000_20260608.csv"
    write_json(json_path, report)
    md_path.write_text(markdown_report(report), encoding="utf-8")
    write_csv(csv_path, results)
    print(json.dumps({
        "json": str(json_path),
        "markdown": str(md_path),
        "csv": str(csv_path),
        "summary": summary,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

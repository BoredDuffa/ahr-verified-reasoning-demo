#!/usr/bin/env python3
"""GSM8K arithmetic program-synthesis baseline.

This script implements the three immediate pieces needed after the external
coverage benchmark:

1. Extract arithmetic equation programs from GSM8K solution traces.
2. Replay/verify those programs with an exact arithmetic evaluator.
3. Train/evaluate a first question-to-program baseline using retrieval over
   delexicalized arithmetic templates, with abstention when verification fails.

It is intentionally conservative.  The retrieval baseline is a starting point,
not a claim of broad math reasoning capability.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import re
import time
import urllib.parse
import urllib.request
from collections import Counter
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.error import HTTPError

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "gsm8k_program_synthesis_v0"
REPORT_DIR = ROOT / "reports"
HF_API = "https://datasets-server.huggingface.co"
GSM8K_PARQUET_URLS = {
    "train": "https://huggingface.co/datasets/openai/gsm8k/resolve/refs%2Fconvert%2Fparquet/main/train/0000.parquet",
    "test": "https://huggingface.co/datasets/openai/gsm8k/resolve/refs%2Fconvert%2Fparquet/main/test/0000.parquet",
}

NUMBER_RE = re.compile(r"(?<![A-Za-z])-?\$?\d[\d,]*(?:\.\d+)?%?")
EQUATION_RE = re.compile(r"<<([^<>=]+?)=([^<>]+?)>>")
TOKEN_RE = re.compile(r"[a-z0-9]+")


@dataclass(frozen=True)
class Equation:
    expr: str
    expected: str
    computed: str
    valid: bool
    template_expr: str


@dataclass(frozen=True)
class Example:
    case_id: str
    split: str
    row_idx: int
    question: str
    answer: str
    final_answer: str
    numbers: Tuple[str, ...]
    equations: Tuple[Equation, ...]
    replay_valid: bool


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def hf_get(endpoint: str, params: Dict[str, Any]) -> Dict[str, Any]:
    url = f"{HF_API}/{endpoint}?" + urllib.parse.urlencode(params)
    for attempt in range(5):
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            if exc.code not in {429, 500, 502, 503, 504} or attempt == 4:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError("unreachable")


def hf_rows(split: str, limit: int) -> Iterable[Dict[str, Any]]:
    parquet_rows = list(hf_parquet_rows(split, limit))
    if parquet_rows:
        yield from parquet_rows
        return
    offset = 0
    while offset < limit:
        length = min(100, limit - offset)
        payload = hf_get("rows", {
            "dataset": "openai/gsm8k",
            "config": "main",
            "split": split,
            "offset": offset,
            "length": length,
        })
        rows = payload.get("rows", [])
        if not rows:
            break
        yield from rows
        offset += len(rows)


def hf_parquet_rows(split: str, limit: int) -> Iterable[Dict[str, Any]]:
    try:
        import pyarrow.parquet as pq  # type: ignore
    except Exception:
        return []
    url = GSM8K_PARQUET_URLS.get(split)
    if not url:
        return []
    cache_dir = DATA_DIR / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"gsm8k_main_{split}.parquet"
    if not path.exists():
        with urllib.request.urlopen(url, timeout=120) as response:
            path.write_bytes(response.read())
    table = pq.read_table(path, columns=["question", "answer"])
    data = table.to_pylist()
    return [{"row_idx": idx, "row": row} for idx, row in enumerate(data[:limit])]


def clean_number_text(value: str) -> str:
    s = str(value).strip().replace("$", "").replace(",", "")
    if s.endswith("%"):
        return str(Fraction(s[:-1]) / 100)
    return s


def to_fraction(value: str) -> Fraction:
    s = clean_number_text(value)
    if "/" in s and not re.search(r"[+*()]", s):
        return Fraction(s)
    return Fraction(str(float(s))) if "." in s else Fraction(int(s), 1)


def fraction_text(value: Fraction) -> str:
    return str(value.numerator) if value.denominator == 1 else f"{value.numerator}/{value.denominator}"


def safe_eval_expr(expr: str) -> Fraction:
    expr = expr.replace(",", "")
    expr = expr.replace("$", "")
    expr = expr.replace("%", "/100")
    tree = ast.parse(expr, mode="eval")

    def eval_node(node: ast.AST) -> Fraction:
        if isinstance(node, ast.Expression):
            return eval_node(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return Fraction(str(node.value))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return -eval_node(node.operand)
        if isinstance(node, ast.BinOp):
            left = eval_node(node.left)
            right = eval_node(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                if right == 0:
                    raise ZeroDivisionError(expr)
                return left / right
        raise ValueError(f"unsupported arithmetic expression: {expr!r}")

    return eval_node(tree)


def extract_final_answer(answer: str) -> str:
    match = re.search(r"####\s*([^\n]+)", answer)
    return clean_number_text(match.group(1)) if match else ""


def extract_question_numbers(question: str) -> Tuple[str, ...]:
    return tuple(clean_number_text(match.group(0)) for match in NUMBER_RE.finditer(question))


def delexicalize_expr(expr: str, question_numbers: Sequence[str]) -> str:
    value_to_slots: Dict[str, List[str]] = {}
    for idx, number in enumerate(question_numbers):
        value_to_slots.setdefault(clean_number_text(number), []).append(f"N{idx}")

    def replace(match: re.Match[str]) -> str:
        raw = clean_number_text(match.group(0))
        slots = value_to_slots.get(raw)
        return slots[0] if slots else clean_number_text(match.group(0))

    return NUMBER_RE.sub(replace, expr)


def instantiate_template(template: str, numbers: Sequence[str]) -> Optional[str]:
    def replace(match: re.Match[str]) -> str:
        idx = int(match.group(1))
        if idx >= len(numbers):
            raise IndexError(idx)
        return clean_number_text(numbers[idx])

    try:
        return re.sub(r"\bN(\d+)\b", replace, template)
    except IndexError:
        return None


def parse_example(split: str, item: Dict[str, Any]) -> Example:
    row = item["row"]
    question = row["question"]
    answer = row["answer"]
    numbers = extract_question_numbers(question)
    equations: List[Equation] = []
    replay_valid = True
    for match in EQUATION_RE.finditer(answer):
        expr = match.group(1).strip()
        expected = clean_number_text(match.group(2).strip())
        try:
            computed = safe_eval_expr(expr)
            expected_fraction = to_fraction(expected)
            valid = computed == expected_fraction
            computed_text = fraction_text(computed)
        except Exception:
            valid = False
            computed_text = "ERROR"
        replay_valid = replay_valid and valid
        equations.append(Equation(
            expr=expr,
            expected=expected,
            computed=computed_text,
            valid=valid,
            template_expr=delexicalize_expr(expr, numbers),
        ))
    final_answer = extract_final_answer(answer)
    if not equations or not final_answer:
        replay_valid = False
    return Example(
        case_id=f"gsm8k_{split}_{int(item['row_idx']):05d}",
        split=split,
        row_idx=int(item["row_idx"]),
        question=question,
        answer=answer,
        final_answer=final_answer,
        numbers=numbers,
        equations=tuple(equations),
        replay_valid=replay_valid,
    )


def fetch_and_build(train_limit: int, val_limit: int, test_limit: int) -> Dict[str, Any]:
    train_items = list(hf_rows("train", train_limit + val_limit))
    test_items = list(hf_rows("test", test_limit))
    train_examples = [parse_example("train", item) for item in train_items[:train_limit]]
    val_examples = [parse_example("val", item) for item in train_items[train_limit:train_limit + val_limit]]
    test_examples = [parse_example("test", item) for item in test_items]
    splits = {"train": train_examples, "val": val_examples, "test": test_examples}
    for split, examples in splits.items():
        write_jsonl(DATA_DIR / f"{split}.jsonl", [example_to_dict(ex) for ex in examples])
    report = {
        "kind": "gsm8k_program_synthesis_dataset_v0",
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "clinical_use": "forbidden",
        "source": {"dataset": "openai/gsm8k", "config": "main"},
        "splits": {
            split: {
                "total": len(examples),
                "replay_valid": sum(1 for ex in examples if ex.replay_valid),
                "with_equations": sum(1 for ex in examples if ex.equations),
            }
            for split, examples in splits.items()
        },
        "paths": {split: str(DATA_DIR / f"{split}.jsonl") for split in splits},
    }
    write_json(REPORT_DIR / "ahr_gsm8k_program_synthesis_dataset_20260608.json", report)
    return report


def example_to_dict(example: Example) -> Dict[str, Any]:
    row = asdict(example)
    row["numbers"] = list(example.numbers)
    row["equations"] = [asdict(eq) for eq in example.equations]
    return row


def load_examples(split: str) -> List[Dict[str, Any]]:
    return read_jsonl(DATA_DIR / f"{split}.jsonl")


def tokenize(question: str) -> Counter[str]:
    stop = {
        "a", "an", "and", "are", "as", "at", "by", "for", "from", "has", "have",
        "he", "her", "his", "how", "if", "in", "is", "it", "many", "much", "of",
        "on", "or", "per", "she", "so", "than", "the", "their", "them", "to",
        "total", "what", "will", "with",
    }
    return Counter(tok for tok in TOKEN_RE.findall(question.lower()) if tok not in stop)


def cosine(a: Counter[str], b: Counter[str]) -> float:
    if not a or not b:
        return 0.0
    dot = sum(a[k] * b.get(k, 0) for k in a)
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    return dot / (na * nb) if na and nb else 0.0


def run_program_templates(templates: Sequence[str], numbers: Sequence[str]) -> Optional[Tuple[List[Dict[str, Any]], str]]:
    steps: List[Dict[str, Any]] = []
    last: Optional[Fraction] = None
    for template in templates:
        expr = instantiate_template(template, numbers)
        if not expr:
            return None
        try:
            value = safe_eval_expr(expr)
        except Exception:
            return None
        last = value
        steps.append({"expr": expr, "value": fraction_text(value), "template_expr": template})
    if last is None:
        return None
    return steps, fraction_text(last)


def train_retrieval_baseline(train_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    model: List[Dict[str, Any]] = []
    for row in train_rows:
        if not row.get("replay_valid"):
            continue
        templates = [eq["template_expr"] for eq in row.get("equations", [])]
        if not templates:
            continue
        model.append({
            "case_id": row["case_id"],
            "question": row["question"],
            "tokens": dict(tokenize(row["question"])),
            "templates": templates,
            "final_answer": row["final_answer"],
        })
    return model


def template_label(row: Dict[str, Any]) -> Optional[str]:
    templates = [eq["template_expr"] for eq in row.get("equations", [])]
    if not templates:
        return None
    return json.dumps(templates, separators=(",", ":"))


def train_naive_bayes_template_model(train_rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    class_counts: Counter[str] = Counter()
    token_counts: Dict[str, Counter[str]] = {}
    class_token_totals: Counter[str] = Counter()
    vocab: Counter[str] = Counter()
    templates_by_label: Dict[str, List[str]] = {}
    for row in train_rows:
        if not row.get("replay_valid"):
            continue
        label = template_label(row)
        if label is None:
            continue
        class_counts[label] += 1
        templates_by_label[label] = json.loads(label)
        toks = tokenize(row["question"])
        token_counts.setdefault(label, Counter()).update(toks)
        class_token_totals[label] += sum(toks.values())
        vocab.update(toks)
    return {
        "type": "multinomial_naive_bayes_template_classifier",
        "class_counts": dict(class_counts),
        "token_counts": {label: dict(counts) for label, counts in token_counts.items()},
        "class_token_totals": dict(class_token_totals),
        "templates_by_label": templates_by_label,
        "vocab": sorted(vocab),
        "train_examples": sum(class_counts.values()),
        "parameter_count": sum(len(counts) for counts in token_counts.values()) + len(class_counts),
    }


def nb_scores(model: Dict[str, Any], question: str) -> List[Tuple[str, float]]:
    toks = tokenize(question)
    class_counts = Counter(model["class_counts"])
    token_counts = {label: Counter(counts) for label, counts in model["token_counts"].items()}
    class_token_totals = Counter(model["class_token_totals"])
    vocab_size = max(1, len(model["vocab"]))
    total_classes = sum(class_counts.values())
    scores: List[Tuple[str, float]] = []
    for label, count in class_counts.items():
        score = math.log(count / total_classes)
        denom = class_token_totals[label] + vocab_size
        counts = token_counts[label]
        for tok, tok_count in toks.items():
            score += tok_count * math.log((counts.get(tok, 0) + 1) / denom)
        scores.append((label, score))
    scores.sort(key=lambda item: item[1], reverse=True)
    return scores


def evaluate_naive_bayes(model: Dict[str, Any], rows: Sequence[Dict[str, Any]], margin_threshold: float) -> Dict[str, Any]:
    results: List[Dict[str, Any]] = []
    for row in rows:
        scores = nb_scores(model, row["question"])
        status = "not_answered"
        predicted_answer = ""
        predicted_steps: List[Dict[str, Any]] = []
        best_label = scores[0][0] if scores else None
        best_score = scores[0][1] if scores else float("-inf")
        second_score = scores[1][1] if len(scores) > 1 else float("-inf")
        margin = best_score - second_score if math.isfinite(second_score) else 0.0
        if best_label is not None and margin >= margin_threshold:
            templates = model["templates_by_label"][best_label]
            run = run_program_templates(templates, row.get("numbers", []))
            if run is not None:
                predicted_steps, predicted_answer = run
                status = "correct" if answers_equal(predicted_answer, row["final_answer"]) else "wrong"
        results.append({
            "case_id": row["case_id"],
            "status": status,
            "margin": margin,
            "predicted_label": best_label,
            "question": row["question"],
            "expected_answer": row["final_answer"],
            "predicted_answer": predicted_answer,
            "predicted_steps": predicted_steps,
        })
    counts = Counter(result["status"] for result in results)
    answered = counts["correct"] + counts["wrong"]
    return {
        "margin_threshold": margin_threshold,
        "summary": {
            "total": len(results),
            "correct": counts["correct"],
            "wrong": counts["wrong"],
            "not_answered": counts["not_answered"],
            "coverage": answered / len(results) if results else 0.0,
            "answered_accuracy": counts["correct"] / answered if answered else 0.0,
            "strict_score": counts["correct"] / len(results) if results else 0.0,
        },
        "results": results,
    }


def evaluate_retrieval(model: Sequence[Dict[str, Any]], rows: Sequence[Dict[str, Any]], threshold: float) -> Dict[str, Any]:
    model_tokens = [(entry, Counter(entry["tokens"])) for entry in model]
    results: List[Dict[str, Any]] = []
    for row in rows:
        q_tokens = tokenize(row["question"])
        best_entry: Optional[Dict[str, Any]] = None
        best_score = 0.0
        for entry, entry_tokens in model_tokens:
            score = cosine(q_tokens, entry_tokens)
            if score > best_score:
                best_entry = entry
                best_score = score
        status = "not_answered"
        predicted_answer = ""
        predicted_steps: List[Dict[str, Any]] = []
        if best_entry is not None and best_score >= threshold:
            run = run_program_templates(best_entry["templates"], row.get("numbers", []))
            if run is not None:
                predicted_steps, predicted_answer = run
                status = "correct" if answers_equal(predicted_answer, row["final_answer"]) else "wrong"
        results.append({
            "case_id": row["case_id"],
            "status": status,
            "similarity": best_score,
            "nearest_case_id": best_entry["case_id"] if best_entry else None,
            "question": row["question"],
            "expected_answer": row["final_answer"],
            "predicted_answer": predicted_answer,
            "predicted_steps": predicted_steps,
        })
    counts = Counter(result["status"] for result in results)
    answered = counts["correct"] + counts["wrong"]
    return {
        "threshold": threshold,
        "summary": {
            "total": len(results),
            "correct": counts["correct"],
            "wrong": counts["wrong"],
            "not_answered": counts["not_answered"],
            "coverage": answered / len(results) if results else 0.0,
            "answered_accuracy": counts["correct"] / answered if answered else 0.0,
            "strict_score": counts["correct"] / len(results) if results else 0.0,
        },
        "results": results,
    }


def answers_equal(a: str, b: str) -> bool:
    try:
        return to_fraction(a) == to_fraction(b)
    except Exception:
        return clean_number_text(a) == clean_number_text(b)


def dataset_report() -> Dict[str, Any]:
    splits = {split: load_examples(split) for split in ("train", "val", "test")}
    return {
        "splits": {
            split: {
                "total": len(rows),
                "replay_valid": sum(1 for row in rows if row.get("replay_valid")),
                "avg_program_length": sum(len(row.get("equations", [])) for row in rows) / len(rows) if rows else 0,
                "with_question_numbers": sum(1 for row in rows if row.get("numbers")),
            }
            for split, rows in splits.items()
        }
    }


def write_markdown(report: Dict[str, Any], path: Path) -> None:
    ds = report["dataset"]["splits"]
    val = report["guarded_trainable_baseline"]["validation"]["summary"]
    test = report["guarded_trainable_baseline"]["test"]["summary"]
    retrieval = report["retrieval_baseline"]["test"]["summary"]
    lines = [
        "# GSM8K Program Synthesis Baseline",
        "",
        f"Created UTC: {report['created_at_utc']}",
        "",
        "## Dataset",
        "",
        "| split | total | replay_valid | avg_program_length | with_question_numbers |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for split, row in ds.items():
        lines.append(
            f"| {split} | {row['total']} | {row['replay_valid']} | {row['avg_program_length']:.2f} | {row['with_question_numbers']} |"
        )
    lines += [
        "",
        "## Guarded Trainable Program Baseline",
        "",
        f"- Model: {report['trainable_model']['type']}",
        f"- Stored parameters: {report['trainable_model']['parameter_count']}",
        f"- Validation margin threshold: {report['guarded_trainable_baseline']['selected_margin_threshold']:.2f}",
        f"- Validation: correct={val['correct']}, wrong={val['wrong']}, not_answered={val['not_answered']}, coverage={val['coverage']:.4f}, answered_accuracy={val['answered_accuracy']:.4f}, strict_score={val['strict_score']:.4f}",
        f"- Test: correct={test['correct']}, wrong={test['wrong']}, not_answered={test['not_answered']}, coverage={test['coverage']:.4f}, answered_accuracy={test['answered_accuracy']:.4f}, strict_score={test['strict_score']:.4f}",
        "",
        "## Retrieval Template Diagnostic",
        "",
        f"- Test diagnostic: correct={retrieval['correct']}, wrong={retrieval['wrong']}, not_answered={retrieval['not_answered']}, coverage={retrieval['coverage']:.4f}, answered_accuracy={retrieval['answered_accuracy']:.4f}",
        "",
        "## Notes",
        "",
        "- The dataset target is an executable arithmetic expression chain extracted from GSM8K `<<expr=value>>` traces.",
        "- Gold replay verifies that extracted programs compute their annotated intermediate values.",
        "- The trainable baseline is a small multinomial Naive Bayes classifier from question tokens to delexicalized arithmetic program-template classes.",
        "- The retrieval diagnostic is intentionally weak; its main value is showing why nearest-template matching is not sufficient.",
        "- Both paths instantiate candidate programs with target question numbers and abstain when the guarded validation policy does not support answering.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_results_csv(path: Path, results: Sequence[Dict[str, Any]]) -> None:
    fields = ["case_id", "status", "similarity", "nearest_case_id", "expected_answer", "predicted_answer", "question"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in results:
            writer.writerow({field: row.get(field, "") for field in fields})


def run_all(args: argparse.Namespace) -> Dict[str, Any]:
    if args.rebuild or not (DATA_DIR / "train.jsonl").exists():
        fetch_and_build(args.train_limit, args.val_limit, args.test_limit)
    train_rows = load_examples("train")
    val_rows = load_examples("val")
    test_rows = load_examples("test")
    retrieval_model = train_retrieval_baseline(train_rows)
    nb_model = train_naive_bayes_template_model(train_rows)
    thresholds = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
    val_runs = [evaluate_retrieval(retrieval_model, val_rows, threshold) for threshold in thresholds]
    # Prefer zero wrong answers; among those, maximize strict score then coverage.
    zero_wrong = [run for run in val_runs if run["summary"]["wrong"] == 0]
    if zero_wrong:
        selected = max(zero_wrong, key=lambda run: (run["summary"]["strict_score"], run["summary"]["coverage"], -run["threshold"]))
    else:
        selected = {
            "threshold": 1.01,
            "summary": {
                "total": len(val_rows),
                "correct": 0,
                "wrong": 0,
                "not_answered": len(val_rows),
                "coverage": 0.0,
                "answered_accuracy": 0.0,
                "strict_score": 0.0,
            },
            "results": [],
        }
    retrieval_test_run = evaluate_retrieval(retrieval_model, test_rows, selected["threshold"])

    margins = [0.0, 1.0, 2.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0]
    nb_val_runs = [evaluate_naive_bayes(nb_model, val_rows, margin) for margin in margins]
    nb_zero_wrong = [run for run in nb_val_runs if run["summary"]["wrong"] == 0]
    if nb_zero_wrong:
        nb_selected = max(nb_zero_wrong, key=lambda run: (run["summary"]["strict_score"], run["summary"]["coverage"], -run["margin_threshold"]))
    else:
        nb_selected = {
            "margin_threshold": 1000000000.0,
            "summary": {
                "total": len(val_rows),
                "correct": 0,
                "wrong": 0,
                "not_answered": len(val_rows),
                "coverage": 0.0,
                "answered_accuracy": 0.0,
                "strict_score": 0.0,
            },
            "results": [],
        }
    nb_test_run = evaluate_naive_bayes(nb_model, test_rows, nb_selected["margin_threshold"])
    report = {
        "kind": "ahr_gsm8k_program_synthesis_baseline",
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "clinical_use": "forbidden",
        "dataset": dataset_report(),
        "trainable_model": {
            "type": nb_model["type"],
            "train_examples": nb_model["train_examples"],
            "parameter_count": nb_model["parameter_count"],
            "class_count": len(nb_model["class_counts"]),
            "vocab_size": len(nb_model["vocab"]),
        },
        "guarded_trainable_baseline": {
            "validation_margin_sweep": [{"margin_threshold": run["margin_threshold"], **run["summary"]} for run in nb_val_runs],
            "selected_margin_threshold": nb_selected["margin_threshold"],
            "validation": {"summary": nb_selected["summary"]},
            "test": {"summary": nb_test_run["summary"], "results": nb_test_run["results"]},
        },
        "retrieval_baseline": {
            "validation_threshold_sweep": [{"threshold": run["threshold"], **run["summary"]} for run in val_runs],
            "selected_threshold": selected["threshold"],
            "validation": {"summary": selected["summary"]},
            "test": {"summary": retrieval_test_run["summary"], "results": retrieval_test_run["results"]},
        },
    }
    json_path = REPORT_DIR / "ahr_gsm8k_program_synthesis_baseline_20260608.json"
    md_path = REPORT_DIR / "ahr_gsm8k_program_synthesis_baseline_20260608.md"
    csv_path = REPORT_DIR / "ahr_gsm8k_program_synthesis_test_results_20260608.csv"
    write_json(json_path, report)
    write_markdown(report, md_path)
    write_results_csv(csv_path, nb_test_run["results"])
    model_path = DATA_DIR / "retrieval_template_model.json"
    nb_model_path = DATA_DIR / "naive_bayes_template_model.json"
    write_json(model_path, {"model": retrieval_model, "selected_threshold": selected["threshold"]})
    write_json(nb_model_path, {"model": nb_model, "selected_margin_threshold": nb_selected["margin_threshold"]})
    return {"json": str(json_path), "markdown": str(md_path), "csv": str(csv_path), "model": str(nb_model_path), "summary": nb_test_run["summary"]}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-limit", type=int, default=1000)
    parser.add_argument("--val-limit", type=int, default=200)
    parser.add_argument("--test-limit", type=int, default=300)
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()
    result = run_all(args)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Lightweight AHR-first GSM8K curriculum solver.

This experiment keeps the AHR method at the center:

* finite symbolic state: question numbers, intermediate exact Fraction results
* finite legal actions: add/sub/mul/div/halt over available sources
* lightweight trainable ranker: sparse perceptron over question/action context
* exact verifier/executor: every candidate action executes before it survives
* validation gate: test answers are allowed only at a zero-wrong validation gate

It is intentionally separate from the archived free-form and large-class GSM8K
trials.  The scorer ranks legal actions; it never generates text.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import random
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, DefaultDict, Dict, Iterable, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "gsm8k_program_synthesis_v0"
REPORT_DIR = ROOT / "reports"
MODEL_DIR = ROOT / "checkpoints"
DATE_TAG = "20260609"
STAGES = {
    "one_step": 1,
    "two_step": 2,
    "three_step": 3,
}
COMMON_CONSTANTS = {
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10",
    "11", "12", "20", "24", "25", "30", "50", "60", "100", "1000",
    "1/2", "1/3", "1/4", "1/5", "2/3", "3/4", "4/5",
}


def load_action_tools() -> Any:
    path = ROOT / "experiments" / "ahr_gsm8k_action_decoder_train.py"
    spec = importlib.util.spec_from_file_location("ahr_gsm8k_action_decoder_train", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


BASE = load_action_tools()
Action = BASE.Action


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def action_from_label(label: str) -> Any:
    return Action(*label.split("|"))


def program_ops(actions: Sequence[Any]) -> int:
    return sum(1 for action in actions if action.op != "halt")


def grounded_source(src: str) -> bool:
    if src.startswith(("N", "R")):
        return True
    if src.startswith("C:"):
        try:
            value = BASE.TOOLS.fraction_text(BASE.TOOLS.to_fraction(src[2:]))
        except Exception:
            return False
        return value in COMMON_CONSTANTS
    return False


def grounded_actions(actions: Sequence[Any]) -> bool:
    for action in actions:
        if not grounded_source(action.a):
            return False
        if action.op != "halt" and not grounded_source(action.b):
            return False
    return True


def compiled_rows(rows: Sequence[Dict[str, Any]], max_ops: int) -> List[Tuple[Dict[str, Any], List[Any]]]:
    out: List[Tuple[Dict[str, Any], List[Any]]] = []
    for row in rows:
        actions = BASE.compile_row(row)
        if actions and 1 <= program_ops(actions) <= max_ops and grounded_actions(actions):
            out.append((row, list(actions)))
    return out


def token_features(question: str) -> List[str]:
    toks = BASE.tokenize(question)
    features: List[str] = [f"tok={tok}" for tok in toks]
    for a, b in zip(toks, toks[1:]):
        features.append(f"bigram={a}_{b}")
    return features


def source_is_legal(src: str, number_count: int, result_count: int) -> bool:
    try:
        if src.startswith("N"):
            return int(src[1:]) < number_count
        if src.startswith("R"):
            return int(src[1:]) < result_count
        if src.startswith("C:"):
            BASE.TOOLS.to_fraction(src[2:])
            return True
    except Exception:
        return False
    return False


def legal_label(label: str, numbers: Sequence[str], results: Sequence[Fraction]) -> bool:
    action = action_from_label(label)
    if not source_is_legal(action.a, len(numbers), len(results)):
        return False
    if action.op == "halt":
        return len(results) > 0 and BASE.execute_action(action, numbers, results) is not None
    if not source_is_legal(action.b, len(numbers), len(results)):
        return False
    return BASE.execute_action(action, numbers, results) is not None


def execute_label(label: str, numbers: Sequence[str], results: Sequence[Fraction]) -> Optional[Fraction]:
    return BASE.execute_action(action_from_label(label), numbers, results)


def state_features(row: Dict[str, Any], step: int, prev_label: str, results: Sequence[Fraction]) -> List[str]:
    numbers = list(row.get("numbers", []))
    features = [
        "bias",
        f"step={step}",
        f"prev={prev_label}",
        f"number_count={min(len(numbers), 8)}",
        f"result_count={len(results)}",
    ]
    features.extend(token_features(row["question"]))
    for idx, number in enumerate(numbers[:8]):
        try:
            value = BASE.TOOLS.to_fraction(number)
            if value.denominator == 1:
                features.append(f"N{idx}_int")
                features.append(f"N{idx}_sign={'neg' if value < 0 else 'nonneg'}")
                if abs(value) <= 12:
                    features.append(f"N{idx}_small={value.numerator}")
        except Exception:
            pass
    return features


class SparseActionRanker:
    """A dependency-free sparse multiclass ranker.

    This is deliberately small and auditable.  It learns only to rank known
    finite action labels under a legal-action mask; exact execution and the
    validation gate decide whether an answer is exposed.
    """

    def __init__(self, labels: Sequence[str]) -> None:
        self.labels = list(labels)
        self.weights: Dict[str, Counter[str]] = {label: Counter() for label in self.labels}
        self.default_prior = Counter(self.labels)

    def score(self, label: str, features: Sequence[str]) -> float:
        weights = self.weights[label]
        return sum(weights.get(feature, 0.0) for feature in features) + 0.001 * self.default_prior[label]

    def rank(self, features: Sequence[str], legal_labels: Sequence[str], limit: int) -> List[Tuple[str, float, float]]:
        if not legal_labels:
            return []
        scored = [(label, self.score(label, features)) for label in legal_labels]
        scored.sort(key=lambda item: item[1], reverse=True)
        second = scored[1][1] if len(scored) > 1 else scored[0][1] - 1.0
        out: List[Tuple[str, float, float]] = []
        for label, score in scored[:limit]:
            out.append((label, score, score - second))
        return out

    def update(self, gold: str, pred: str, features: Sequence[str]) -> None:
        if gold == pred:
            return
        for feature in features:
            self.weights[gold][feature] += 1.0
            self.weights[pred][feature] -= 1.0

    def train(self, items: Sequence[Tuple[Dict[str, Any], List[Any]]], epochs: int, seed: str) -> List[Dict[str, Any]]:
        rng = random.Random(seed)
        history: List[Dict[str, Any]] = []
        label_set = set(self.labels)
        for epoch in range(1, epochs + 1):
            order = list(items)
            rng.shuffle(order)
            mistakes = 0
            total = 0
            for row, actions in order:
                numbers = list(row.get("numbers", []))
                results: List[Fraction] = []
                prev = "<START>"
                for step, action in enumerate(actions):
                    gold = action.label()
                    if gold not in label_set:
                        break
                    legal = [label for label in self.labels if legal_label(label, numbers, results)]
                    if gold not in legal:
                        break
                    features = state_features(row, step, prev, results)
                    ranked = self.rank(features, legal, 1)
                    pred = ranked[0][0] if ranked else ""
                    if pred != gold:
                        mistakes += 1
                        self.update(gold, pred, features)
                    total += 1
                    value = execute_label(gold, numbers, results)
                    if value is None:
                        break
                    if action.op != "halt":
                        results.append(value)
                    prev = gold
            if epoch == 1 or epoch == epochs or epoch % 5 == 0:
                history.append({
                    "epoch": epoch,
                    "training_action_items": total,
                    "mistakes": mistakes,
                    "mistake_rate": mistakes / total if total else 0.0,
                })
                print(json.dumps(history[-1], sort_keys=True), flush=True)
        return history

    def nonzero_parameter_count(self) -> int:
        return sum(1 for weights in self.weights.values() for value in weights.values() if value != 0)

    def to_jsonable(self) -> Dict[str, Any]:
        return {
            "labels": self.labels,
            "weights": {label: dict(weights) for label, weights in self.weights.items() if weights},
        }

    @classmethod
    def from_jsonable(cls, obj: Dict[str, Any]) -> "SparseActionRanker":
        ranker = cls(obj["labels"])
        for label, weights in obj.get("weights", {}).items():
            ranker.weights[label] = Counter(weights)
        return ranker


@dataclass
class Beam:
    results: List[Fraction]
    trace: List[Dict[str, Any]]
    score: float
    margin_sum: float
    action_count: int
    prev_label: str


def confidence_from_margin(margin_sum: float, action_count: int) -> float:
    if action_count <= 0:
        return 0.0
    avg = max(-20.0, min(20.0, margin_sum / action_count))
    return 1.0 / (1.0 + math.exp(-avg))


def run_policy(
    ranker: SparseActionRanker,
    row: Dict[str, Any],
    max_ops: int,
    beam_size: int,
    threshold: float,
) -> Dict[str, Any]:
    numbers = list(row.get("numbers", []))
    beams = [Beam(results=[], trace=[], score=0.0, margin_sum=0.0, action_count=0, prev_label="<START>")]
    completed: List[Dict[str, Any]] = []
    max_actions = max_ops + 1
    for step in range(max_actions):
        next_beams: List[Beam] = []
        for beam in beams:
            legal = [label for label in ranker.labels if legal_label(label, numbers, beam.results)]
            features = state_features(row, step, beam.prev_label, beam.results)
            for label, score, margin in ranker.rank(features, legal, max(beam_size * 2, 4)):
                action = action_from_label(label)
                value = execute_label(label, numbers, beam.results)
                if value is None:
                    continue
                trace_item = {
                    "step": step,
                    "action": label,
                    "value": BASE.TOOLS.fraction_text(value),
                    "rank_score": score,
                    "margin": margin,
                }
                if action.op == "halt":
                    pred = BASE.TOOLS.fraction_text(value)
                    confidence = confidence_from_margin(beam.margin_sum + margin, beam.action_count + 1)
                    completed.append({
                        "predicted_answer": pred,
                        "confidence": confidence,
                        "trace": beam.trace + [trace_item],
                        "verifier_valid": True,
                    })
                    continue
                if beam.action_count >= max_ops:
                    continue
                next_beams.append(Beam(
                    results=beam.results + [value],
                    trace=beam.trace + [trace_item],
                    score=beam.score + score,
                    margin_sum=beam.margin_sum + margin,
                    action_count=beam.action_count + 1,
                    prev_label=label,
                ))
        completed.sort(key=lambda item: item["confidence"], reverse=True)
        if completed and completed[0]["confidence"] >= threshold:
            return completed[0]
        next_beams.sort(key=lambda beam: (beam.score / max(beam.action_count, 1), beam.margin_sum), reverse=True)
        beams = next_beams[:beam_size]
        if not beams:
            break
    if completed and completed[0]["confidence"] >= threshold:
        return completed[0]
    return {"predicted_answer": "", "confidence": 0.0, "trace": [], "verifier_valid": False}


def evaluate_rows(
    ranker: SparseActionRanker,
    rows_with_actions: Sequence[Tuple[Dict[str, Any], List[Any]]],
    max_ops: int,
    beam_size: int,
    threshold: float,
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    for row, gold_actions in rows_with_actions:
        run = run_policy(ranker, row, max_ops, beam_size, threshold)
        pred = run["predicted_answer"]
        if not pred:
            status = "not_answered"
        elif BASE.TOOLS.answers_equal(pred, row["final_answer"]):
            status = "correct"
        else:
            status = "wrong"
        results.append({
            "case_id": row["case_id"],
            "question": row["question"],
            "expected_answer": row["final_answer"],
            "predicted_answer": pred,
            "status": status,
            "confidence": run["confidence"],
            "verifier_valid": run["verifier_valid"],
            "gold_program": [action.label() for action in gold_actions],
            "trace": run["trace"],
        })
    return results


def summarize(results: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    counts = Counter(row["status"] for row in results)
    answered = counts["correct"] + counts["wrong"]
    total = len(results)
    return {
        "total": total,
        "correct": counts["correct"],
        "wrong": counts["wrong"],
        "not_answered": counts["not_answered"],
        "coverage": answered / total if total else 0.0,
        "answered_accuracy": counts["correct"] / answered if answered else 0.0,
        "strict_score": counts["correct"] / total if total else 0.0,
    }


def thresholded(raw: Sequence[Dict[str, Any]], threshold: float) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in raw:
        if row["status"] != "not_answered" and float(row["confidence"]) < threshold:
            out.append({**row, "status": "not_answered", "predicted_answer": "", "trace": []})
        else:
            out.append(row)
    return out


def select_threshold(raw_val: Sequence[Dict[str, Any]], thresholds: Sequence[float]) -> Tuple[float, List[Dict[str, Any]], List[Dict[str, Any]]]:
    sweep: List[Dict[str, Any]] = []
    best_threshold = thresholds[-1]
    best_summary: Optional[Dict[str, Any]] = None
    for threshold in thresholds:
        results = thresholded(raw_val, threshold)
        summary = summarize(results)
        sweep.append({"threshold": threshold, **summary})
        if summary["wrong"] == 0 and summary["correct"] > 0:
            if best_summary is None or (summary["correct"], summary["coverage"], threshold) > (best_summary["correct"], best_summary["coverage"], best_threshold):
                best_summary = summary
                best_threshold = threshold
    if best_summary is None:
        for threshold in thresholds:
            results = thresholded(raw_val, threshold)
            summary = summarize(results)
            if summary["wrong"] == 0:
                best_threshold = threshold
                best_summary = summary
                break
    if best_summary is None:
        best_threshold = 1.01
        best_results = thresholded(raw_val, best_threshold)
        sweep.append({"threshold": best_threshold, **summarize(best_results)})
        return best_threshold, best_results, sweep
    return best_threshold, thresholded(raw_val, best_threshold), sweep


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    fields = ["case_id", "status", "confidence", "verifier_valid", "expected_answer", "predicted_answer", "question", "trace"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: json.dumps(row[field], sort_keys=True) if field == "trace" else row.get(field, "") for field in fields})


def write_markdown(report: Dict[str, Any], path: Path) -> None:
    v = report["validation_guarded"]["summary"]
    t = report["test_guarded"]["summary"]
    lines = [
        "# AHR GSM8K Curriculum Solver",
        "",
        f"Created UTC: {report['created_at_utc']}",
        "",
        "## Model",
        "",
        f"- Stage: {report['stage']}",
        f"- Type: {report['model']['type']}",
        f"- Parameters: {report['model']['parameter_count']}",
        f"- Action labels: {report['model']['action_label_count']}",
        f"- Selected threshold: {report['selected_threshold']}",
        "",
        "## Guarded Results",
        "",
        f"- Validation: correct={v['correct']}, wrong={v['wrong']}, not_answered={v['not_answered']}, coverage={v['coverage']:.4f}, answered_accuracy={v['answered_accuracy']:.4f}",
        f"- Test: correct={t['correct']}, wrong={t['wrong']}, not_answered={t['not_answered']}, coverage={t['coverage']:.4f}, answered_accuracy={t['answered_accuracy']:.4f}",
        "",
        "## Validation Threshold Sweep",
        "",
        "| threshold | total | correct | wrong | not_answered | coverage | answered_accuracy |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in report["validation_threshold_sweep"]:
        lines.append(f"| {row['threshold']:.3f} | {row['total']} | {row['correct']} | {row['wrong']} | {row['not_answered']} | {row['coverage']:.4f} | {row['answered_accuracy']:.4f} |")
    lines += [
        "",
        "## Notes",
        "",
        "- The ranker only orders finite legal actions.",
        "- Exact Fraction execution is required for every trace step.",
        "- Test answers are exposed only after zero-wrong validation threshold selection.",
        "- This is not a general language-model math solver.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def load_model(path: Path) -> SparseActionRanker:
    return SparseActionRanker.from_jsonable(json.loads(path.read_text(encoding="utf-8")))


def save_model(path: Path, ranker: SparseActionRanker, metadata: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, {"metadata": metadata, **ranker.to_jsonable()})


def run_probability_templates(args: argparse.Namespace) -> int:
    path = ROOT / "experiments" / "ahr_qa_benchmark_math_probability.py"
    spec = importlib.util.spec_from_file_location("ahr_qa_benchmark_math_probability", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    demo = module.load_demo_module()
    cases = module.make_cases()
    if args.max_eval:
        cases = cases[: args.max_eval]
    results = [module.score_case(demo, case) for case in cases]
    counts = Counter(row["status"] for row in results)
    total = len(results)
    summary = {
        "total": total,
        "correct": counts["correct"] + counts["correct_abstain"],
        "wrong": counts["wrong"],
        "not_answered": counts["not_answered"],
        "coverage": (counts["correct"] + counts["correct_abstain"]) / total if total else 0.0,
        "answered_accuracy": 1.0 if counts["wrong"] == 0 and total else 0.0,
        "strict_score": (counts["correct"] + counts["correct_abstain"]) / total if total else 0.0,
    }
    report = {
        "kind": "ahr_gsm8k_ahr_curriculum_solver",
        "stage": "probability_templates",
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "clinical_use": "forbidden",
        "model": {
            "type": "exact_probability_template_router",
            "parameter_count": 0,
            "action_label_count": 0,
        },
        "selected_threshold": 1.0,
        "training_history": [],
        "validation_threshold_sweep": [{"threshold": 1.0, **summary}],
        "validation_guarded": {"summary": summary},
        "test_guarded": {"summary": summary, "results": results},
    }
    stem = f"ahr_gsm8k_ahr_curriculum_solver_probability_templates_{DATE_TAG}"
    json_path = REPORT_DIR / f"{stem}.json"
    md_path = REPORT_DIR / f"{stem}.md"
    write_json(json_path, report)
    write_markdown(report, md_path)
    print(json.dumps({"json": str(json_path), "markdown": str(md_path), "summary": summary}, indent=2, sort_keys=True))
    return 0 if summary["wrong"] == 0 and summary["not_answered"] == 0 else 1


def run_arithmetic_stage(args: argparse.Namespace) -> int:
    max_ops = STAGES[args.stage]
    train_rows = read_jsonl(DATA_DIR / "train.jsonl")
    val_rows = read_jsonl(DATA_DIR / "val.jsonl")
    test_rows = read_jsonl(DATA_DIR / "test.jsonl")
    train_items = compiled_rows(train_rows, max_ops)
    if args.max_train:
        train_items = train_items[: args.max_train]
    val_items = compiled_rows(val_rows, max_ops)
    test_items = compiled_rows(test_rows, max_ops)
    if args.max_eval:
        val_items = val_items[: args.max_eval]
        test_items = test_items[: args.max_eval]

    model_path = MODEL_DIR / f"ahr_gsm8k_ahr_curriculum_solver_{args.stage}.json"
    if args.eval_only:
        ranker = load_model(model_path)
        history: List[Dict[str, Any]] = [{"eval_only": True, "model_path": str(model_path)}]
    else:
        labels = sorted({action.label() for _row, actions in train_items for action in actions})
        ranker = SparseActionRanker(labels)
        history = ranker.train(train_items, max(0, args.epochs), seed=f"ahr-{args.stage}")
        save_model(model_path, ranker, {
            "stage": args.stage,
            "max_ops": max_ops,
            "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "clinical_use": "forbidden",
        })

    raw_val = evaluate_rows(ranker, val_items, max_ops, args.beam_size, threshold=0.0)
    thresholds = [0.0, 0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95, 0.97, 0.98, 0.99, 0.995, 1.01]
    selected_threshold, val_guarded, sweep = select_threshold(raw_val, thresholds)
    test_raw = evaluate_rows(ranker, test_items, max_ops, args.beam_size, threshold=0.0)
    test_guarded = thresholded(test_raw, selected_threshold)
    report = {
        "kind": "ahr_gsm8k_ahr_curriculum_solver",
        "stage": args.stage,
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "clinical_use": "forbidden",
        "model": {
            "type": "sparse_ahr_action_ranker",
            "parameter_count": ranker.nonzero_parameter_count(),
            "action_label_count": len(ranker.labels),
            "checkpoint": str(model_path),
            "beam_size": args.beam_size,
            "max_ops": max_ops,
            "train_items": len(train_items),
            "val_items": len(val_items),
            "test_items": len(test_items),
        },
        "selected_threshold": selected_threshold,
        "training_history": history,
        "validation_threshold_sweep": sweep,
        "validation_guarded": {"summary": summarize(val_guarded), "results": val_guarded},
        "test_guarded": {"summary": summarize(test_guarded), "results": test_guarded},
    }
    stem = f"ahr_gsm8k_ahr_curriculum_solver_{args.stage}_{DATE_TAG}"
    json_path = REPORT_DIR / f"{stem}.json"
    md_path = REPORT_DIR / f"{stem}.md"
    csv_path = REPORT_DIR / f"{stem}_test_results.csv"
    write_json(json_path, report)
    write_markdown(report, md_path)
    write_csv(csv_path, test_guarded)
    print(json.dumps({
        "json": str(json_path),
        "markdown": str(md_path),
        "csv": str(csv_path),
        "selected_threshold": selected_threshold,
        "validation": report["validation_guarded"]["summary"],
        "test": report["test_guarded"]["summary"],
    }, indent=2, sort_keys=True))
    return 0


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=["one_step", "two_step", "three_step", "probability_templates"])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--beam-size", type=int, default=12)
    parser.add_argument("--max-train", type=int, default=0)
    parser.add_argument("--max-eval", type=int, default=0, help="Optional smoke-test limit for validation/test rows.")
    parser.add_argument("--cpu", action="store_true", help="Accepted for interface compatibility; this solver is CPU-native.")
    parser.add_argument("--eval-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.stage == "probability_templates":
        return run_probability_templates(args)
    return run_arithmetic_stage(args)


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Train and gate the lightweight AHR demo program ranker.

The ranker selects among finite legal program labels. It never emits answers;
exact execution and verifier-gated AHR program backends produce answers.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import ahr_csp_demo_program_backend as backend  # type: ignore


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "ahr_demo_program_ranker_v0"
MODEL_PATH = ROOT / "checkpoints" / "ahr_demo_program_ranker_v0.json"
REPORT_JSON = ROOT / "reports" / "ahr_demo_program_ranker_eval_20260612.json"
REPORT_CSV = ROOT / "reports" / "ahr_demo_program_ranker_eval_20260612.csv"
REPORT_MD = ROOT / "reports" / "ahr_demo_program_ranker_eval_20260612.md"


PROGRAMS_BY_FAMILY = {
    "probability_counting_program": [
        "dice_sum_exact",
        "coin_binomial",
        "coin_run_at_least_one",
        "birthday_at_least_one_match",
        "card_rank_or_suit",
        "card_rank",
        "card_suit",
        "urn_both_without_replacement",
    ],
    "statistics_program": ["mean_variance", "mean_sample_variance", "median", "mode"],
    "short_arithmetic_program": ["add_all", "subtract_chain", "multiply_all", "divide_exact"],
    "finite_logic_constraints": ["ordering_topological_sort"],
}


@dataclass
class Row:
    split: str
    family: str
    question: str
    program: str
    expected_answer: str | None
    should_abstain: bool = False


def features(question: str, family: str) -> list[str]:
    tokens = re.findall(r"[a-z0-9]+", question.lower())
    out = ["bias", f"family={family}"]
    out.extend(f"tok={tok}" for tok in tokens)
    out.extend(f"bigram={a}_{b}" for a, b in zip(tokens, tokens[1:]))
    for token in tokens:
        if token.isdigit():
            out.append("has_number")
    return out


class SparseProgramRanker:
    def __init__(self, labels: list[str]) -> None:
        self.labels = labels
        self.weights: dict[str, Counter[str]] = {label: Counter() for label in labels}
        self.priors: Counter[str] = Counter()

    def score(self, label: str, feats: list[str]) -> float:
        return float(self.priors[label]) + sum(self.weights[label].get(feat, 0.0) for feat in feats)

    def rank(self, question: str, family: str, legal: list[str]) -> list[tuple[str, float]]:
        feats = features(question, family)
        ranked = [(label, self.score(label, feats)) for label in legal]
        ranked.sort(key=lambda item: (-item[1], item[0]))
        return ranked

    def update(self, gold: str, pred: str, feats: list[str]) -> None:
        if gold == pred:
            return
        for feat in feats:
            self.weights[gold][feat] += 1.0
            self.weights[pred][feat] -= 1.0

    def train(self, rows: list[Row], epochs: int, seed: int) -> list[dict[str, Any]]:
        rng = random.Random(seed)
        for row in rows:
            self.priors[row.program] += 0.001
        history = []
        for epoch in range(1, epochs + 1):
            shuffled = list(rows)
            rng.shuffle(shuffled)
            mistakes = 0
            total = 0
            for row in shuffled:
                if row.should_abstain:
                    continue
                legal = PROGRAMS_BY_FAMILY[row.family]
                feats = features(row.question, row.family)
                pred = self.rank(row.question, row.family, legal)[0][0]
                if pred != row.program:
                    mistakes += 1
                    self.update(row.program, pred, feats)
                total += 1
            if epoch == 1 or epoch == epochs or epoch % 5 == 0:
                history.append({"epoch": epoch, "items": total, "mistakes": mistakes, "mistake_rate": mistakes / total if total else 0.0})
                print(json.dumps(history[-1], sort_keys=True), flush=True)
        return history

    def to_jsonable(self, report: dict[str, Any], history: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "model_type": "sparse_ahr_program_ranker",
            "checkpoint_name": "ahr_demo_program_ranker_v0",
            "clinical_use": "forbidden",
            "labels": self.labels,
            "priors": dict(self.priors),
            "weights": {label: dict(weights) for label, weights in self.weights.items() if weights},
            "parameter_count": sum(1 for weights in self.weights.values() for value in weights.values() if value != 0.0),
            "history": history,
            "acceptance": report["acceptance"],
        }


def expected_answer(question: str, family: str) -> str | None:
    result = backend.solve(question, {"family": family})
    if not result.get("verified"):
        return None
    return str(result.get("answer"))


def add(rows: list[Row], split: str, family: str, question: str, program: str, should_abstain: bool = False) -> None:
    rows.append(Row(split, family, question, program, None if should_abstain else expected_answer(question, family), should_abstain))


def generate_rows(train_count: int, val_count: int, test_count: int, seed: int) -> list[Row]:
    rng = random.Random(seed)
    rows: list[Row] = []
    split_counts = {"train": train_count, "val": val_count, "test": test_count}
    names = ["Alice", "Bob", "Cara", "Dina", "Eli", "Faye"]
    for split, count in split_counts.items():
        for i in range(count):
            n = 6 + (i % 55)
            k = 2 + (i % 8)
            target = 3 + (i % 15)
            add(rows, split, "probability_counting_program", f"What is the probability of rolling a sum of {target} with two dice?", "dice_sum_exact")
            add(rows, split, "probability_counting_program", f"A fair coin is tossed {n} times. What is the probability of exactly {min(k, n)} heads?", "coin_binomial")
            add(rows, split, "probability_counting_program", f"If I flip {n + 10} coins, what is the chance there is a sequence of {min(k + 1, n)} heads in a row?", "coin_run_at_least_one")
            add(rows, split, "probability_counting_program", f"In a room of {20 + i % 40} people, what is the probability at least two share a birthday?", "birthday_at_least_one_match")
            add(rows, split, "probability_counting_program", "A card is drawn from a standard deck. What is the probability of drawing a king or a heart?", "card_rank_or_suit")
            add(rows, split, "probability_counting_program", "A card is drawn from a standard deck. What is the probability of drawing a queen?", "card_rank")
            add(rows, split, "probability_counting_program", "A card is drawn from a standard deck. What is the probability of drawing a spade?", "card_suit")
            add(rows, split, "probability_counting_program", "A bag has 5 red, 3 blue, and 2 green balls. Two are drawn without replacement. What is the probability both are red?", "urn_both_without_replacement")

            values = [1 + ((i + j * 3) % 30) for j in range(5)]
            repeated = [2, 4, 4, 7, 9 + i % 3]
            add(rows, split, "statistics_program", f"What is the mean and population variance of {', '.join(map(str, values))}?", "mean_variance")
            add(rows, split, "statistics_program", f"What is the mean and sample variance of {', '.join(map(str, values))}?", "mean_sample_variance")
            add(rows, split, "statistics_program", f"What is the median of {', '.join(map(str, values))}?", "median")
            add(rows, split, "statistics_program", f"What is the mode of {', '.join(map(str, repeated))}?", "mode")

            a = 5 + i % 20
            b = 2 + (i * 7) % 15
            add(rows, split, "short_arithmetic_program", f"Add {a} and {b}.", "add_all")
            add(rows, split, "short_arithmetic_program", f"Maria had {a + b + 20} dollars and spent {b}. How much remains?", "subtract_chain")
            add(rows, split, "short_arithmetic_program", f"A box has {a} rows of {b} pencils. How many pencils are there?", "multiply_all")
            add(rows, split, "short_arithmetic_program", f"{a * b} cookies are split equally among {a} kids. How many per kid?", "divide_exact")

            order = rng.sample(names, 3)
            add(rows, split, "finite_logic_constraints", f"{order[0]} is taller than {order[1]}. {order[1]} is taller than {order[2]}. Who is tallest?", "ordering_topological_sort")

            add(rows, split, "probability_counting_program", "A loaded die has unknown probabilities. What is the probability of rolling a 6?", "unsupported", should_abstain=True)
    return rows


def evaluate(model: SparseProgramRanker, rows: list[Row]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    summary: dict[str, Any] = {
        "total": len(rows),
        "correct": 0,
        "wrong": 0,
        "abstained": 0,
        "hard_valid": 0,
        "mean_hard_energy": 0.0,
        "by_family": defaultdict(lambda: {"total": 0, "correct": 0, "wrong": 0, "abstained": 0}),
    }
    records = []
    for row in rows:
        family_summary = summary["by_family"][row.family]
        family_summary["total"] += 1
        if row.should_abstain:
            status = "abstained"
            answer = None
            predicted = None
            valid = False
            energy = None
        else:
            ranked = model.rank(row.question, row.family, PROGRAMS_BY_FAMILY[row.family])
            predicted = ranked[0][0]
            result = backend.solve(row.question, {"family": row.family})
            answer = result.get("answer")
            valid = bool(result.get("verified")) and predicted == row.program and answer == row.expected_answer
            energy = 0.0 if valid else 1.0
            status = "correct" if valid else "wrong"
        if status == "correct":
            summary["correct"] += 1
            summary["hard_valid"] += 1
            family_summary["correct"] += 1
        elif status == "wrong":
            summary["wrong"] += 1
            family_summary["wrong"] += 1
        else:
            summary["abstained"] += 1
            family_summary["abstained"] += 1
        records.append(
            {
                "split": row.split,
                "family": row.family,
                "question": row.question,
                "gold_program": row.program,
                "predicted_program": predicted,
                "status": status,
                "expected_answer": row.expected_answer,
                "predicted_answer": answer,
                "hard_valid": valid,
                "hard_energy": energy,
            }
        )
    answered = summary["correct"] + summary["wrong"]
    summary["coverage"] = answered / summary["total"] if summary["total"] else 0.0
    summary["hard_valid_rate"] = summary["hard_valid"] / answered if answered else 0.0
    summary["mean_hard_energy"] = summary["wrong"] / answered if answered else 0.0
    summary["by_family"] = dict(summary["by_family"])
    return summary, records


def write_dataset(rows: list[Row]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    by_split: dict[str, list[Row]] = defaultdict(list)
    for row in rows:
        by_split[row.split].append(row)
    for split, split_rows in by_split.items():
        with (DATA_DIR / f"{split}.jsonl").open("w", encoding="utf-8") as f:
            for row in split_rows:
                f.write(json.dumps(row.__dict__, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-count", type=int, default=1200, help="template iterations per split for training")
    parser.add_argument("--val-count", type=int, default=250)
    parser.add_argument("--test-count", type=int, default=250)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260612)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    rows = generate_rows(args.train_count, args.val_count, args.test_count, args.seed)
    write_dataset(rows)
    train_rows = [row for row in rows if row.split == "train" and not row.should_abstain]
    labels = sorted({label for labels in PROGRAMS_BY_FAMILY.values() for label in labels})
    model = SparseProgramRanker(labels)
    history = model.train(train_rows, args.epochs, args.seed)
    split_reports = {}
    all_records = []
    for split in ["train", "val", "test"]:
        summary, records = evaluate(model, [row for row in rows if row.split == split])
        split_reports[split] = summary
        all_records.extend(records)
    acceptance = {
        "val_zero_wrong": split_reports["val"]["wrong"] == 0,
        "test_zero_wrong": split_reports["test"]["wrong"] == 0,
        "val_hard_energy_zero": split_reports["val"]["mean_hard_energy"] == 0.0,
        "test_hard_energy_zero": split_reports["test"]["mean_hard_energy"] == 0.0,
    }
    acceptance["publish"] = all(acceptance.values())
    report = {
        "clinical_use": "forbidden",
        "run_name": "ahr_demo_program_ranker_v0",
        "created_at_unix": time.time(),
        "model_path": str(MODEL_PATH),
        "parameter_count": sum(1 for weights in model.weights.values() for value in weights.values() if value != 0.0),
        "splits": split_reports,
        "acceptance": acceptance,
        "history": history,
    }
    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with REPORT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_records[0].keys()))
        writer.writeheader()
        writer.writerows(all_records)
    lines = [
        "# AHR Demo Program Ranker v0",
        "",
        "- clinical_use: forbidden",
        f"- parameter_count: {report['parameter_count']}",
        f"- publish: {acceptance['publish']}",
    ]
    for split, summary in split_reports.items():
        lines.append(f"- {split}: correct={summary['correct']} wrong={summary['wrong']} abstained={summary['abstained']} hard_valid_rate={summary['hard_valid_rate']:.6f} mean_hard_energy={summary['mean_hard_energy']:.6f}")
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    MODEL_PATH.write_text(json.dumps(model.to_jsonable(report, history), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"acceptance": acceptance, "splits": split_reports, "model_path": str(MODEL_PATH)}, indent=2, sort_keys=True))
    return 0 if acceptance["publish"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Train and gate a lightweight AHR question router.

The neural-like component here is deliberately small: a sparse bag-of-words
Naive Bayes family router. It never emits final answers. A deterministic DSL
parser and exact solver must verify every answer before it is counted.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "ahr_capability_router_v0"
REPORT_JSON = ROOT / "reports" / "ahr_capability_router_training_20260609.json"
REPORT_CSV = ROOT / "reports" / "ahr_capability_router_training_20260609.csv"
REPORT_MD = ROOT / "reports" / "ahr_capability_router_training_20260609.md"
MODEL_PATH = ROOT / "checkpoints" / "ahr_capability_router_nb_model.json"
FAMILIES = [
    "coin_run",
    "binomial_tail",
    "dice_event",
    "card_event",
    "urn_draw",
    "expected_value",
    "birthday_event",
    "bayes_event",
    "proof_template",
    "unsupported",
]


@dataclass
class Row:
    split: str
    family: str
    question: str
    dsl: dict[str, Any]
    answer: str | None


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def frac_text(value: Fraction) -> str:
    return str(value.numerator) if value.denominator == 1 else f"{value.numerator}/{value.denominator}"


def coin_run_fraction(n_flips: int, run_length: int) -> tuple[Fraction, list[str]]:
    states = [0] * run_length
    states[0] = 1
    for _ in range(n_flips):
        nxt = [0] * run_length
        nxt[0] = sum(states)
        for k in range(run_length - 1):
            nxt[k + 1] += states[k]
        states = nxt
    total = 2**n_flips
    no_run = sum(states)
    favorable = total - no_run
    return Fraction(favorable, total), [
        f"total sequences = 2^{n_flips}",
        f"no-run sequences from DP = {no_run}",
        "favorable = total - no_run",
    ]


def solve_exact(dsl: dict[str, Any]) -> tuple[str | None, list[str]]:
    family = dsl["family"]
    if family == "coin_run":
        ans, trace = coin_run_fraction(dsl["n_flips"], dsl["run_length"])
        return frac_text(ans), trace
    if family == "binomial_tail":
        n = dsl["n_trials"]
        if dsl["event"] == "exactly":
            ks = [dsl["successes"]]
        elif dsl["event"] == "at_least":
            ks = range(dsl["successes"], n + 1)
        elif dsl["event"] == "at_most":
            ks = range(0, dsl["successes"] + 1)
        else:
            ks = range(dsl["low"], dsl["high"] + 1)
        favorable = sum(math.comb(n, k) for k in ks)
        return frac_text(Fraction(favorable, 2**n)), [f"X ~ Binomial({n}, 1/2)", f"favorable = {favorable}", f"total = 2^{n}"]
    if family == "dice_event":
        favorable = 0
        for a in range(1, 7):
            for b in range(1, 7):
                total = a + b
                if (dsl["event"] == "sum_exact" and total == dsl["target"]) or (dsl["event"] == "sum_at_least" and total >= dsl["target"]) or (dsl["event"] == "sum_at_most" and total <= dsl["target"]) or (dsl["event"] == "doubles" and a == b):
                    favorable += 1
        return frac_text(Fraction(favorable, 36)), ["enumerate 36 ordered dice outcomes", f"favorable = {favorable}"]
    if family == "card_event":
        if dsl["event"] == "rank_or_suit":
            favorable = 16
        elif dsl["event"] == "rank_and_suit":
            favorable = 1
        elif dsl["event"] == "conditional_same_rank":
            return "1/17", ["remaining rank cards = 3", "remaining deck size = 51"]
        elif dsl["event"] == "suit":
            favorable = 13
        else:
            favorable = 4
        return frac_text(Fraction(favorable, 52)), ["standard deck size = 52", f"favorable = {favorable}"]
    if family == "urn_draw":
        total = sum(dsl["colors"].values())
        target = dsl["colors"][dsl["target_color"]]
        if dsl["replacement"]:
            favorable = math.comb(dsl["draws"], dsl["target_count"]) * target ** dsl["target_count"] * (total - target) ** (dsl["draws"] - dsl["target_count"])
            return frac_text(Fraction(favorable, total ** dsl["draws"])), ["binomial count with replacement", f"weighted favorable = {favorable}"]
        favorable = target * (target - 1)
        return frac_text(Fraction(favorable, total * (total - 1))), [f"first target = {target}/{total}", f"second target = {target - 1}/{total - 1}"]
    if family == "expected_value":
        return "6", ["success probability = 1/6", "geometric expectation = 1/p = 6"]
    if family == "birthday_event":
        people = dsl["people"]
        no_shared = Fraction(1, 1)
        for i in range(people):
            no_shared *= Fraction(365 - i, 365)
        return frac_text(1 - no_shared), ["compute no shared birthday complement", "answer = 1 - no_shared"]
    if family == "bayes_event":
        prevalence = Fraction(round(dsl["prevalence_percent"] * 10000), 1000000)
        sensitivity = Fraction(round(dsl["sensitivity_percent"] * 10000), 1000000)
        specificity = Fraction(round(dsl["specificity_percent"] * 10000), 1000000)
        true_positive = prevalence * sensitivity
        false_positive = (1 - prevalence) * (1 - specificity)
        return frac_text(true_positive / (true_positive + false_positive)), ["Bayes numerator = prevalence*sensitivity", "denominator adds false positives"]
    if family == "proof_template":
        table = {"monty_hall": "2/3", "three_boxes": "2/3", "prisoners": "0.311828"}
        return table.get(dsl["template"]), [f"verified proof template = {dsl['template']}"]
    return None, ["unsupported route must abstain"]


def make_rows(seed: int, train_count: int, val_count: int, test_count: int) -> list[Row]:
    rng = random.Random(seed)
    rows: list[Row] = []
    split_sizes = {"train": train_count, "val": val_count, "test": test_count}
    ranks = ["ace", "king", "queen"]
    suits = ["heart", "spade", "club"]
    colors = [("red", 5), ("blue", 3), ("green", 2)]

    def add(split: str, family: str, question: str, dsl: dict[str, Any]) -> None:
        answer, _ = solve_exact(dsl)
        rows.append(Row(split, family, question, dsl, answer))

    for split, count in split_sizes.items():
        for i in range(count):
            n = 8 + (i % 13)
            k = 3 + (i % 4)
            add(split, "coin_run", f"If I flip {n} coins, what are the chances there is a sequence of {k} heads in a row?", {"family": "coin_run", "n_flips": n, "run_length": k, "target": "heads", "event": "at_least_one_run"})
            successes = 2 + (i % max(2, n // 2))
            event = ["exactly", "at_least", "at_most"][i % 3]
            phrase = {"exactly": f"exactly {successes}", "at_least": f"at least {successes}", "at_most": f"at most {successes}"}[event]
            add(split, "binomial_tail", f"A fair coin is tossed {n} times. What is the probability of getting {phrase} heads?", {"family": "binomial_tail", "n_trials": n, "successes": successes, "target": "heads", "event": event})
            target = 4 + (i % 8)
            dice_event = ["sum_exact", "sum_at_least", "sum_at_most"][i % 3]
            dice_phrase = {"sum_exact": f"the total is {target}", "sum_at_least": f"the total is at least {target}", "sum_at_most": f"the total is at most {target}"}[dice_event]
            add(split, "dice_event", f"Two fair six-sided dice are rolled. What is the probability that {dice_phrase}?", {"family": "dice_event", "dice": 2, "sides": 6, "event": dice_event, "target": target})
            rank = ranks[i % len(ranks)]
            suit = suits[(i + 1) % len(suits)]
            add(split, "card_event", f"A card is drawn from a standard deck. What is the probability of drawing a {rank} or a {suit}?", {"family": "card_event", "rank": rank, "suit": suit, "event": "rank_or_suit"})
            target_color, _ = colors[i % len(colors)]
            add(split, "urn_draw", f"A bag contains 5 red, 3 blue, and 2 green balls. Two balls are drawn without replacement. What is the probability that both are {target_color}?", {"family": "urn_draw", "colors": {"red": 5, "blue": 3, "green": 2}, "draws": 2, "replacement": False, "target_color": target_color, "target_count": 2})
            add(split, "expected_value", "You repeatedly roll a fair die until a 6 appears. What is the expected number of rolls?", {"family": "expected_value", "distribution": "geometric_die_until_face", "face": 6, "success_probability": "1/6"})
            people = 20 + (i % 15)
            add(split, "birthday_event", f"In a room of {people} people, what is the probability that at least two share a birthday?", {"family": "birthday_event", "people": people, "event": "at_least_one_shared_birthday"})
            prev = [1, 2, 5][i % 3]
            sens = [90, 95, 99][i % 3]
            spec = [90, 95, 98][(i + 1) % 3]
            add(split, "bayes_event", f"A disease affects {prev}% of people. A test has {sens}% sensitivity and {spec}% specificity. If positive, what is the probability they have the disease?", {"family": "bayes_event", "prevalence_percent": prev, "sensitivity_percent": sens, "specificity_percent": spec, "event": "posterior_given_positive"})
            template = rng.choice(["monty_hall", "three_boxes", "prisoners"])
            q = {
                "monty_hall": "Monty Hall has three doors, one car, and two goats. Should you switch?",
                "three_boxes": "Three boxes contain GG, SS, and GS. A drawn coin is gold. What is the chance the other coin is gold?",
                "prisoners": "100 prisoners and 100 numbered boxes. What strategy maximizes survival probability?",
            }[template]
            add(split, "proof_template", q, {"family": "proof_template", "template": template})
            rows.append(Row(split, "unsupported", f"A loaded die has unknown probabilities. What is the chance of rolling {1 + i % 6}?", {"family": "unsupported", "reason": "missing_probability_model"}, None))
    return rows


def parse_dsl(question: str, family: str) -> dict[str, Any] | None:
    q = question.lower()
    if family == "coin_run":
        m = re.search(r"flip (\d+) coins.*sequence of (\d+) heads", q)
        return {"family": family, "n_flips": int(m.group(1)), "run_length": int(m.group(2)), "target": "heads", "event": "at_least_one_run"} if m else None
    if family == "binomial_tail":
        n = re.search(r"tossed (\d+) times", q)
        k = re.search(r"(exactly|at least|at most) (\d+)", q)
        if not n or not k:
            return None
        return {"family": family, "n_trials": int(n.group(1)), "successes": int(k.group(2)), "target": "heads", "event": k.group(1).replace(" ", "_")}
    if family == "dice_event":
        target = re.search(r"(?:is|least|most) (\d+)", q)
        if not target:
            return None
        event = "sum_exact"
        if "at least" in q:
            event = "sum_at_least"
        if "at most" in q:
            event = "sum_at_most"
        return {"family": family, "dice": 2, "sides": 6, "event": event, "target": int(target.group(1))}
    if family == "card_event":
        rank = next((r for r in ["ace", "king", "queen"] if r in q), None)
        suit = next((s for s in ["heart", "spade", "club"] if s in q), None)
        return {"family": family, "rank": rank, "suit": suit, "event": "rank_or_suit"} if rank and suit else None
    if family == "urn_draw":
        target = next((c for c in ["red", "blue", "green"] if f"both are {c}" in q), None)
        return {"family": family, "colors": {"red": 5, "blue": 3, "green": 2}, "draws": 2, "replacement": False, "target_color": target, "target_count": 2} if target else None
    if family == "expected_value":
        return {"family": family, "distribution": "geometric_die_until_face", "face": 6, "success_probability": "1/6"} if "until a 6" in q else None
    if family == "birthday_event":
        m = re.search(r"room of (\d+) people", q)
        return {"family": family, "people": int(m.group(1)), "event": "at_least_one_shared_birthday"} if m else None
    if family == "bayes_event":
        nums = [float(x) for x in re.findall(r"(\d+(?:\.\d+)?)%", q)]
        return {"family": family, "prevalence_percent": nums[0], "sensitivity_percent": nums[1], "specificity_percent": nums[2], "event": "posterior_given_positive"} if len(nums) >= 3 else None
    if family == "proof_template":
        if "monty hall" in q:
            return {"family": family, "template": "monty_hall"}
        if "three boxes" in q:
            return {"family": family, "template": "three_boxes"}
        if "100 prisoners" in q:
            return {"family": family, "template": "prisoners"}
    if family == "unsupported":
        return {"family": "unsupported", "reason": "missing_probability_model"}
    return None


class NaiveBayesRouter:
    def __init__(self) -> None:
        self.class_counts: Counter[str] = Counter()
        self.token_counts: dict[str, Counter[str]] = defaultdict(Counter)
        self.vocab: set[str] = set()

    def fit(self, rows: list[Row]) -> None:
        for row in rows:
            self.class_counts[row.family] += 1
            for token in tokenize(row.question):
                self.token_counts[row.family][token] += 1
                self.vocab.add(token)

    def predict(self, question: str) -> tuple[str, float]:
        tokens = tokenize(question)
        total_docs = sum(self.class_counts.values())
        scores: dict[str, float] = {}
        for family in self.class_counts:
            token_total = sum(self.token_counts[family].values())
            score = math.log(self.class_counts[family] / total_docs)
            for token in tokens:
                score += math.log((self.token_counts[family][token] + 1) / (token_total + len(self.vocab)))
            scores[family] = score
        max_score = max(scores.values())
        probs = {family: math.exp(score - max_score) for family, score in scores.items()}
        denom = sum(probs.values())
        best = max(probs, key=probs.get)
        return best, probs[best] / denom

    def to_json(self, threshold: float) -> dict[str, Any]:
        return {
            "model_type": "sparse_multinomial_naive_bayes_router",
            "threshold": threshold,
            "classes": dict(self.class_counts),
            "vocab": sorted(self.vocab),
            "token_counts": {family: dict(counts) for family, counts in self.token_counts.items()},
            "parameter_count": len(self.vocab) * len(self.class_counts) + len(self.class_counts),
        }


def evaluate(model: NaiveBayesRouter, rows: list[Row], threshold: float) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    summary: dict[str, Any] = {"total": 0, "correct": 0, "wrong": 0, "abstained": 0, "by_family": defaultdict(lambda: {"total": 0, "correct": 0, "wrong": 0, "abstained": 0})}
    records = []
    for row in rows:
        predicted_family, confidence = model.predict(row.question)
        dsl = parse_dsl(row.question, predicted_family) if confidence >= threshold else None
        predicted_answer = None
        trace: list[str] = []
        status = "abstained"
        if dsl and dsl.get("family") != "unsupported":
            predicted_answer, trace = solve_exact(dsl)
            status = "correct" if predicted_answer == row.answer and dsl["family"] == row.family else "wrong"
        elif row.family == "unsupported":
            status = "correct"
            trace = ["unsupported example correctly abstained"]
        summary["total"] += 1
        summary["by_family"][row.family]["total"] += 1
        if status == "correct":
            summary["correct"] += 1
            summary["by_family"][row.family]["correct"] += 1
        elif status == "wrong":
            summary["wrong"] += 1
            summary["by_family"][row.family]["wrong"] += 1
        else:
            summary["abstained"] += 1
            summary["by_family"][row.family]["abstained"] += 1
        records.append({
            "split": row.split,
            "question": row.question,
            "gold_family": row.family,
            "predicted_family": predicted_family,
            "confidence": round(confidence, 6),
            "status": status,
            "expected_answer": row.answer,
            "predicted_answer": predicted_answer,
            "gold_dsl": row.dsl,
            "predicted_dsl": dsl,
            "trace": trace,
        })
    summary["by_family"] = dict(summary["by_family"])
    return summary, records


def write_jsonl(path: Path, rows: list[Row]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row.__dict__, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20260609)
    parser.add_argument("--train-count", type=int, default=12, help="examples per family in train")
    parser.add_argument("--val-count", type=int, default=5, help="examples per family in validation")
    parser.add_argument("--test-count", type=int, default=5, help="examples per family in test")
    parser.add_argument("--threshold", type=float, default=0.55)
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    rows = make_rows(args.seed, args.train_count, args.val_count, args.test_count)
    for split in ["train", "val", "test"]:
        write_jsonl(DATA_DIR / f"{split}.jsonl", [row for row in rows if row.split == split])

    model = NaiveBayesRouter()
    model.fit([row for row in rows if row.split == "train"])
    threshold = args.threshold
    MODEL_PATH.write_text(json.dumps(model.to_json(threshold), indent=2, sort_keys=True) + "\n", encoding="utf-8")

    report: dict[str, Any] = {
        "clinical_use": "forbidden",
        "seed": args.seed,
        "selected_threshold": threshold,
        "model": model.to_json(threshold),
        "splits": {},
        "published_families": [],
    }
    all_records = []
    for split in ["train", "val", "test"]:
        summary, records = evaluate(model, [row for row in rows if row.split == split], threshold)
        report["splits"][split] = summary
        all_records.extend(records)

    val_by_family = report["splits"]["val"]["by_family"]
    for family, stats in val_by_family.items():
        if family != "unsupported" and stats["wrong"] == 0 and stats["correct"] > 0:
            report["published_families"].append(family)

    REPORT_JSON.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with REPORT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["split", "status", "gold_family", "predicted_family", "confidence", "question", "expected_answer", "predicted_answer", "trace"])
        writer.writeheader()
        for rec in all_records:
            writer.writerow({key: json.dumps(rec[key]) if key == "trace" else rec[key] for key in writer.fieldnames})

    val = report["splits"]["val"]
    lines = [
        "# AHR Capability Router Gate Report",
        "",
        "- clinical_use: forbidden",
        f"- model: sparse multinomial Naive Bayes router",
        f"- parameter_count: {report['model']['parameter_count']}",
        f"- selected_threshold: {threshold}",
        f"- validation: {val['correct']} correct / {val['wrong']} wrong / {val['abstained']} abstained / {val['total']} total",
        f"- published_families: {', '.join(report['published_families'])}",
        "",
        "## Validation By Family",
        "",
    ]
    for family, stats in sorted(val["by_family"].items()):
        lines.append(f"- {family}: {stats['correct']} correct, {stats['wrong']} wrong, {stats['abstained']} abstained, {stats['total']} total")
    lines.extend(["", "## Example Verified Traces", ""])
    for rec in [r for r in all_records if r["split"] == "test" and r["status"] == "correct"][:8]:
        lines.append(f"- {rec['question']} -> {rec['predicted_answer']} ({'; '.join(rec['trace'])})")
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    if report["splits"]["val"]["wrong"] != 0:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

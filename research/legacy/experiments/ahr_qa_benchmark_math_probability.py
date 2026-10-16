#!/usr/bin/env python3
"""Benchmark the live QA router on math/probability-style prompts.

The benchmark is deterministic and intentionally transparent: every generated
question has an expected backend id and one or more answer substrings that must
appear in the routed answer. Unsupported-control prompts are counted separately
as correct abstentions when the router declines them.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent.parent
REPORT_DIR = ROOT / "reports"


def load_demo_module() -> Any:
    path = ROOT / "experiments" / "ahr_question_answer_demo.py"
    spec = importlib.util.spec_from_file_location("ahr_question_answer_demo", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@dataclass(frozen=True)
class Case:
    case_id: str
    category: str
    difficulty: str
    question: str
    expected_id: str
    expected_answer_contains: tuple[str, ...]


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def add_variants(cases: List[Case], category: str, difficulty: str, expected_id: str, expected: tuple[str, ...], variants: List[str]) -> None:
    for idx, question in enumerate(variants, start=1):
        cases.append(Case(
            case_id=f"{category}_{idx:03d}",
            category=category,
            difficulty=difficulty,
            question=question,
            expected_id=expected_id,
            expected_answer_contains=expected,
        ))


def make_cases() -> List[Case]:
    cases: List[Case] = []

    add_variants(cases, "coin_exactly_two_heads", "basic", "probability_coin_exactly_two_heads", ("3/8",), [
        "A fair coin is tossed 3 times. What is the probability of getting exactly 2 heads?",
        "If I flip a fair coin three times, what is P(exactly two heads)?",
        "Three fair coin tosses are made. Find the chance of seeing exactly two H outcomes.",
        "In 3 independent tosses of a fair coin, compute the probability of exactly 2 heads.",
        "A fair coin is flipped three times; what fraction of outcomes have precisely two heads?",
        "For three tosses of an unbiased coin, how likely are exactly two heads?",
        "Coin toss question: out of 3 fair flips, probability of exactly two heads?",
        "What is C(3,2)/2^3 as the probability for exactly two heads in three fair coin tosses?",
        "A fair coin lands heads/tails with equal chance. Toss it 3 times. Probability of exactly 2 heads?",
        "Basic probability: fair coin tossed three times, exactly two heads.",
    ])

    add_variants(cases, "dice_sum_seven", "basic", "probability_dice_sum_seven", ("1/6",), [
        "A fair six-sided die is rolled twice. What is the probability that the sum is 7?",
        "Roll two fair d6 dice. What is P(total = 7)?",
        "Two six-sided dice are rolled; calculate the chance the sum equals seven.",
        "If a fair die is rolled twice, how likely is a total of 7?",
        "Find the probability that two independent fair die rolls add to 7.",
        "What fraction of ordered pairs on two dice sum to 7?",
        "Dice roll: fair six-sided die twice, sum is seven.",
        "For two fair dice, what is the chance of one of the six pairs that total 7?",
        "A d6 is rolled two times. Probability the two results sum to 7?",
        "Compute P(sum 7) for two rolls of a fair six-sided die.",
    ])

    add_variants(cases, "card_king_or_heart", "basic", "probability_card_king_or_heart", ("4/13",), [
        "A card is drawn from a standard 52-card deck. What is the probability of drawing a king or a heart?",
        "From a 52-card deck, what is P(king or heart)?",
        "One standard deck card is selected. Chance it is either a king or a heart?",
        "Draw one card from 52. What fraction are kings or hearts?",
        "Probability that a single card is a heart or one of the kings?",
        "In a normal deck, what is the chance of drawing a king or a heart?",
        "Cards: standard 52-card deck, event is king union heart.",
        "Use inclusion-exclusion for one card: king or heart from a 52-card deck.",
        "A random card is drawn. How likely is it to be a king or belong to hearts?",
        "Find P(king OR heart) for a standard deck.",
    ])

    add_variants(cases, "balls_both_red", "intermediate", "probability_balls_both_red", ("2/9",), [
        "A bag contains 5 red, 3 blue, and 2 green balls. Two balls are drawn without replacement. What is the probability that both are red?",
        "From a bag with 5 red, 3 blue, 2 green balls, draw two without replacement. Chance both red?",
        "There are ten balls: 5 red, 3 blue, and 2 green. What is P(two red in two draws without replacement)?",
        "If I sample two balls without replacement from 5R/3B/2G, probability both are red?",
        "Bag problem: 5 red, 3 blue, 2 green; two draws, no replacement; both red.",
        "Compute (5/10)*(4/9) for drawing two red balls from a 5 red, 3 blue, 2 green bag.",
        "A bag has 5 red balls plus 5 non-red balls. Two are drawn without replacement. Probability both are red?",
        "What fraction of unordered pairs from 5 red, 3 blue, and 2 green balls are both red?",
        "Two balls are drawn without replacement from 5 red, 3 blue, 2 green. Find chance both red.",
        "Intermediate probability: both drawn balls are red from a 5/3/2 colored bag.",
    ])

    add_variants(cases, "conditional_aces", "intermediate", "probability_second_ace_given_first_ace", ("1/17",), [
        "Two cards are drawn from a deck without replacement. Given that the first card is an ace, what is the probability that the second card is also an ace?",
        "If the first card drawn is an ace, what is P(second ace) when drawing two cards without replacement?",
        "Two cards, no replacement. Condition on first being ace. Chance second is ace?",
        "A deck has 52 cards. After drawing an ace first, probability the next card is another ace?",
        "Given first card is an ace, how many remaining aces over remaining cards for the second draw?",
        "Conditional probability: first card ace, second card ace without replacement.",
        "Draw two cards. Knowing the first is ace, find probability second is also ace.",
        "After one ace is removed from the deck, what is the probability the next card is ace?",
        "Cards without replacement: P(second ace | first ace).",
        "Given an ace appeared first, compute the probability of an ace on draw two.",
    ])

    add_variants(cases, "bayes_disease", "intermediate", "probability_bayes_disease_positive", ("1/6", "16.6667%"), [
        "A disease affects 1% of a population. A test has 99% sensitivity and 95% specificity. If a person tests positive, what is the probability they actually have the disease?",
        "Disease prevalence is 1%, sensitivity 99%, specificity 95%. What is P(disease | positive test)?",
        "Bayes theorem: 1% base rate, 99% true positive, 95% true negative; after a positive, probability sick?",
        "A medical test has 99% sensitivity and 95% specificity for a disease with 1% prevalence. Find posterior probability given positive.",
        "If 1 in 100 have a disease, test sensitivity 99 and specificity 95 percent, what fraction of positives are true cases?",
        "Bayesian screening: prevalence 0.01, sensitivity 0.99, specificity 0.95, compute P(disease given positive).",
        "Positive test probability problem with 1% disease rate, 99% sensitivity, 95% specificity.",
        "What is the chance someone actually has the disease after a positive test when prevalence is 1%, sensitivity 99%, specificity 95%?",
        "Use Bayes: disease affects 1%, true positive rate 99%, true negative rate 95%, observed positive.",
        "Compute posterior disease probability for a positive result under 1% prevalence, 99% sensitivity, 95% specificity.",
    ])

    add_variants(cases, "birthday_23", "advanced", "probability_birthday_23", ("0.507297",), [
        "In a room of 23 people, what is the probability that at least two people share the same birthday?",
        "Birthday problem: for 23 people, chance of a shared birthday?",
        "Assuming 365 equally likely birthdays, compute P(at least one match) among 23 people.",
        "What is the probability of at least two same birthdays in a group of 23?",
        "For 23 independent birthdays, no leap years, chance some pair shares a birthday?",
        "Classic birthday paradox with 23 people: probability at least one duplicate birthday.",
        "Room of twenty-three people: calculate probability two or more share a birthday.",
        "Use 1 - product formula for 23 birthdays to get at least one shared birthday.",
        "In 23 people, what is the approximate shared-birthday probability?",
        "Advanced probability: birthday match in a room of 23.",
    ])

    add_variants(cases, "monty_hall", "advanced", "probability_monty_hall_switch", ("2/3",), [
        "There are 3 doors. One contains a car and two contain goats. You pick one door. The host opens another door showing a goat. Should you switch? What is the probability of winning if you switch?",
        "Monty Hall: 3 doors, one car, host reveals a goat. Should I switch and what is my win probability?",
        "In the three-door Monty Hall problem, after a goat door is opened, what is P(win by switching)?",
        "You choose one of 3 doors, host opens a goat door. Is switching better? Give probability.",
        "Car behind one of three doors, host shows goat after your pick. Probability of winning if you switch?",
        "Monty Hall probability: stay versus switch after host reveals goat.",
        "Should the contestant switch doors in Monty Hall? What chance of car if switching?",
        "Three doors, two goats, one car; host opens non-car door. What is switch-win probability?",
        "What is the optimal Monty Hall strategy and its probability?",
        "Advanced probability: Monty Hall with 3 doors, host opens goat, calculate switching odds.",
    ])

    add_variants(cases, "coin_random_variable", "advanced", "probability_coin_random_variable", ("5/16", "2.5", "1.25"), [
        "Let X be the number of heads obtained when flipping 5 fair coins. Find P(X=3), E[X], and Var(X).",
        "For X ~ Binomial(5, 1/2), compute P(X=3), expectation, and variance.",
        "Flip five fair coins. If X counts heads, what are P(X=3), E[X], Var(X)?",
        "Random variable X is heads in 5 coin flips. Find probability of 3 heads, mean, and variance.",
        "Five unbiased coin flips: compute P exactly 3 heads plus E and Var of head count.",
        "Let X count successes in 5 fair Bernoulli trials. Give P(X=3), expected value, variance.",
        "Binomial random variable for 5 fair coins: P(X equals 3), E[X], Var(X).",
        "What are P(X=3), E[X], and Var(X) if X is number of heads in 5 fair tosses?",
        "For five fair coin tosses, head-count distribution question: P3, mean, variance.",
        "Random variables: heads from flipping 5 fair coins; calculate P(X=3), E, Var.",
    ])

    add_variants(cases, "three_boxes", "very_challenging", "probability_three_boxes", ("2/3",), [
        "Box A contains 2 gold coins. Box B contains 2 silver coins. Box C contains 1 gold and 1 silver coin. A box is chosen at random, and one coin drawn is gold. What is the probability the other coin in the box is also gold?",
        "Three boxes puzzle: GG, SS, GS. Pick a box and draw a gold coin. Probability the other coin is gold?",
        "Boxes contain two gold, two silver, and mixed gold/silver. Given the observed coin is gold, chance the partner coin is gold?",
        "A box is selected randomly from GG, SS, GS; a random coin from it is gold. What is P(box is GG)?",
        "Bertrand box paradox: after drawing a gold coin, probability remaining coin in box is gold.",
        "One of three boxes is chosen: gold-gold, silver-silver, gold-silver. Drawn coin is gold; probability other is gold?",
        "Conditional probability with three boxes A=GG, B=SS, C=GS; observed gold coin. Find chance other coin gold.",
        "The drawn coin is gold from the classic three boxes puzzle. Is the other coin gold with what probability?",
        "Three boxes: 2G, 2S, 1G1S. Given one coin drawn is gold, probability same box contains another gold coin?",
        "Very challenging probability: solve the three boxes gold coin puzzle.",
    ])

    add_variants(cases, "prisoners", "very_challenging", "probability_100_prisoners_strategy", ("0.311828",), [
        "100 prisoners and 100 numbered boxes. Each prisoner may open 50 boxes. What strategy maximizes their survival probability?",
        "In the 100 prisoners 100 boxes problem, each opens 50 boxes. State the optimal strategy and survival probability.",
        "Prisoners problem: 100 prisoners, 100 boxes, 50 opens each. What is the best strategy?",
        "What strategy gives the highest chance in the 100 prisoners and boxes puzzle, and what is that chance approximately?",
        "For 100 prisoners each allowed to open half the boxes, explain cycle-following strategy and probability of success.",
        "How do 100 prisoners maximize survival when each can inspect 50 of 100 boxes?",
        "Classic 100 prisoners puzzle: give optimal cycle strategy and group success probability.",
        "100 numbered boxes and 100 prisoners; each may open 50. What survival strategy maximizes probability?",
        "What is the probability of all prisoners surviving using the best known cycle strategy in the 100 boxes problem?",
        "Very challenging probability: solve the 100 prisoners / 100 boxes strategy problem.",
    ])

    add_variants(cases, "die_until_six", "very_challenging", "probability_expected_die_until_six", ("6",), [
        "You repeatedly roll a fair die until a 6 appears. What is the expected number of rolls?",
        "Expected value: roll a fair die until the first six. How many rolls on average?",
        "For a geometric process with success rolling 6 on a fair die, what is the expected waiting time?",
        "A fair d6 is rolled until a 6 appears. Find E[number of rolls].",
        "How many rolls should we expect before the first 6 with a fair six-sided die?",
        "Repeatedly roll a die until success is face 6. Compute expected rolls.",
        "Expected number of trials to get a 6 on a fair die?",
        "Fair die, stop at first six. What is the expectation?",
        "Geometric distribution p=1/6: expected count of rolls until first success?",
        "Very challenging expected value: die rolls until a 6 appears.",
    ])

    # Unsupported controls: close in style, but intentionally outside the current verified router.
    add_variants(cases, "unsupported_controls", "control", "unsupported_question", ("does not have a verified backend",), [
        "A biased coin has unknown probability p of heads. It is tossed 10 times and lands heads 7 times. What is p exactly?",
        "A bag has red and blue balls in an unknown ratio. After seeing one red, give the exact probability of red next without any prior.",
        "What is the probability that a random real number selected uniformly from all real numbers is positive?",
        "A six-sided die is loaded but the loading is not specified. What is the probability of rolling a 6?",
        "A deck has some cards removed, but we do not know which. What is the probability the next card is an ace?",
        "Choose the simplest sequence rule for 2, 4, 8, 15, 26 and predict the next term.",
        "A test has unknown sensitivity and specificity. If someone tests positive, what is the probability of disease?",
        "A box contains an unknown number of gold and silver coins. One drawn coin is gold. What is the probability the next is gold?",
        "A random graph is generated by an unspecified distribution. What is the probability it is connected?",
        "A person may or may not be lying with unknown probability. What is the probability their statement is true?",
        "A coin is fair unless it is biased, but no prior is given. What is the probability of heads on the next toss?",
        "A puzzle asks for the most elegant proof of an unstated theorem. Give the exact answer.",
        "A die has faces with unknown probabilities. It was rolled twice and summed to 7. What is the exact probability of sum 7 next?",
        "A disease prevalence is unknown and the test accuracy is unknown. What is P(disease | positive)?",
        "A card is drawn from a nonstandard deck whose composition is not given. What is P(king or heart)?",
        "A process stops eventually with unknown stopping distribution. What is the expected stopping time?",
        "A random variable X has unknown distribution but mean appears near 5 from samples. What is Var(X) exactly?",
        "There are many possible generating functions through these points. Which is the true one without assumptions?",
        "A host opens a door but may follow an unknown strategy. What is the exact Monty Hall switching probability?",
        "An urn has 100 balls with colors assigned by an unknown adversary. After observations, give a unique exact Bayesian posterior without a prior.",
        "A black-box model answers yes 60% on a hidden dataset. What is its true accuracy exactly?",
        "A shuffled deck is missing an unknown card. What is the probability of drawing a heart?",
        "A probability problem omits whether replacement occurs. Give the exact numerical answer.",
        "A fair-looking coin may be double-headed with unknown chance. What is the probability of heads?",
    ])

    return cases


def score_case(demo: Any, case: Case) -> Dict[str, Any]:
    routed = demo.live_demo_from_text(case.question)
    item = routed[0] if routed else {"id": "unsupported_question", "answer": ""}
    answer = str(item.get("answer", ""))
    expected_id_ok = item.get("id") == case.expected_id
    answer_ok = all(fragment in answer for fragment in case.expected_answer_contains)
    status = "correct" if expected_id_ok and answer_ok else "wrong"
    if item.get("id") == "unsupported_question":
        status = "correct_abstain" if case.expected_id == "unsupported_question" else "not_answered"
    return {
        **asdict(case),
        "observed_id": item.get("id"),
        "observed_backend_type": item.get("backend_type"),
        "observed_answer": answer,
        "expected_id_ok": expected_id_ok,
        "answer_ok": answer_ok,
        "status": status,
    }


def markdown_report(report: Dict[str, Any]) -> str:
    s = report["summary"]
    lines = [
        "# AHR QA Math/Probability Benchmark",
        "",
        f"Created UTC: {report['created_at_utc']}",
        "",
        "## Summary",
        "",
        f"- Total cases: {s['total']}",
        f"- Correct answered: {s['correct']}",
        f"- Correct abstentions: {s['correct_abstain']}",
        f"- Wrong: {s['wrong']}",
        f"- Not answered: {s['not_answered']}",
        f"- Answered accuracy: {s['answered_accuracy']:.4f}",
        f"- Overall strict score: {s['overall_strict_score']:.4f}",
        "",
        "## By Difficulty",
        "",
        "| difficulty | total | correct | correct_abstain | wrong | not_answered |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for difficulty, row in sorted(report["by_difficulty"].items()):
        lines.append(
            f"| {difficulty} | {row['total']} | {row['correct']} | {row['correct_abstain']} | {row['wrong']} | {row['not_answered']} |"
        )
    lines += [
        "",
        "## By Category",
        "",
        "| category | total | correct | correct_abstain | wrong | not_answered |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for category, row in sorted(report["by_category"].items()):
        lines.append(
            f"| {category} | {row['total']} | {row['correct']} | {row['correct_abstain']} | {row['wrong']} | {row['not_answered']} |"
        )
    if report["failures"]:
        lines += ["", "## Failures", ""]
        for case in report["failures"][:50]:
            lines += [
                f"- `{case['case_id']}` expected `{case['expected_id']}`, observed `{case['observed_id']}`: {case['question']}",
            ]
    else:
        lines += ["", "## Failures", "", "No wrong or missed cases in this benchmark run."]
    lines += [
        "",
        "## Notes",
        "",
        "- This benchmark tests the live QA router plus exact symbolic/probability backends.",
        "- It is not a claim that the neural AHR checkpoint is a general probability-language model.",
        "- Unsupported controls are scored as correct only when the system abstains.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    demo = load_demo_module()
    cases = make_cases()
    if args.limit is not None:
        cases = cases[: args.limit]
    results = [score_case(demo, case) for case in cases]
    counts = Counter(r["status"] for r in results)
    answered = counts["correct"] + counts["wrong"]
    total = len(results)
    summary = {
        "total": total,
        "correct": counts["correct"],
        "correct_abstain": counts["correct_abstain"],
        "wrong": counts["wrong"],
        "not_answered": counts["not_answered"],
        "answered_accuracy": (counts["correct"] / answered) if answered else 0.0,
        "overall_strict_score": ((counts["correct"] + counts["correct_abstain"]) / total) if total else 0.0,
    }

    def rollup(key: str) -> Dict[str, Dict[str, int]]:
        out: Dict[str, Counter[str]] = defaultdict(Counter)
        for r in results:
            out[r[key]]["total"] += 1
            out[r[key]][r["status"]] += 1
        return {k: {
            "total": v["total"],
            "correct": v["correct"],
            "correct_abstain": v["correct_abstain"],
            "wrong": v["wrong"],
            "not_answered": v["not_answered"],
        } for k, v in out.items()}

    report = {
        "kind": "ahr_qa_math_probability_benchmark",
        "created_at_unix": time.time(),
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "clinical_use": "forbidden",
        "summary": summary,
        "by_difficulty": rollup("difficulty"),
        "by_category": rollup("category"),
        "results": results,
        "failures": [r for r in results if r["status"] in {"wrong", "not_answered"}],
    }
    json_path = REPORT_DIR / "ahr_qa_math_probability_benchmark_20260608.json"
    md_path = REPORT_DIR / "ahr_qa_math_probability_benchmark_20260608.md"
    write_json(json_path, report)
    md_path.write_text(markdown_report(report), encoding="utf-8")
    print(json.dumps({
        "json": str(json_path),
        "markdown": str(md_path),
        "summary": summary,
    }, indent=2, sort_keys=True))
    return 0 if not report["failures"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

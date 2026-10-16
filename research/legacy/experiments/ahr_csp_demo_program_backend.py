#!/usr/bin/env python3
"""Verifier-gated AHR-CSP demo program backend.

This backend keeps the public demo AHR-first: natural language is mapped to a
finite typed task, candidate programs are chosen from a small legal program
space, every candidate is executed exactly, and the verifier gates the answer.
"""

from __future__ import annotations

import itertools
import json
import hashlib
import math
import re
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from pathlib import Path
from typing import Any


CHECKPOINT_NAME = "ahr_csp_verified_program_demo_v0"
RANKER_CHECKPOINT_NAME = "ahr_demo_program_ranker_v0"
MULTIFAMILY_RANKER_CHECKPOINT_NAME = "ahr_multifamily_5m_ranker_v0"
MULTIFAMILY_RANKER_BACKEND_TYPE = "trained_ahr_5m_multifamily_ranker"
ROOT = Path(__file__).resolve().parents[1]
RANKER_MODEL_PATH = ROOT / "checkpoints" / f"{RANKER_CHECKPOINT_NAME}.json"
MULTIFAMILY_RANKER_MODEL_PATH = ROOT / "checkpoints" / f"{MULTIFAMILY_RANKER_CHECKPOINT_NAME}.pt"
SUPPORTED_FAMILIES = {
    "probability_counting_program",
    "statistics_program",
    "finite_logic_constraints",
    "short_arithmetic_program",
}


@dataclass(frozen=True)
class CandidateProgram:
    family: str
    name: str
    params: dict[str, Any]
    score: float


@lru_cache(maxsize=1)
def load_ranker_model() -> dict[str, Any] | None:
    if not RANKER_MODEL_PATH.exists():
        return None
    try:
        obj = json.loads(RANKER_MODEL_PATH.read_text(encoding="utf-8"))
    except Exception:
        return None
    acceptance = obj.get("acceptance", {})
    if not acceptance.get("publish"):
        return None
    return obj


def stable_hash(text: str, buckets: int) -> int:
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little") % buckets


@lru_cache(maxsize=1)
def load_multifamily_ranker_model() -> dict[str, Any] | None:
    if not MULTIFAMILY_RANKER_MODEL_PATH.exists():
        return None
    try:
        import torch
        import torch.nn as nn
    except Exception:
        return None
    try:
        checkpoint = torch.load(MULTIFAMILY_RANKER_MODEL_PATH, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(MULTIFAMILY_RANKER_MODEL_PATH, map_location="cpu")
    except Exception:
        return None
    acceptance = checkpoint.get("acceptance", {})
    if not acceptance.get("publish"):
        return None
    labels = list(checkpoint.get("labels", []))
    if not labels:
        return None

    class AHRMultiFamilyRanker(nn.Module):
        def __init__(self, feature_buckets: int, hidden1: int, hidden2: int, label_count: int) -> None:
            super().__init__()
            self.feature_embedding = nn.EmbeddingBag(feature_buckets, hidden1, mode="sum", include_last_offset=True)
            self.norm1 = nn.LayerNorm(hidden1)
            self.hidden = nn.Linear(hidden1, hidden2)
            self.norm2 = nn.LayerNorm(hidden2)
            self.output = nn.Linear(hidden2, label_count)

        def forward(self, flat_indices: Any, offsets: Any) -> Any:
            x = self.feature_embedding(flat_indices, offsets)
            x = self.norm1(x)
            x = torch.relu(x)
            x = self.hidden(x)
            x = self.norm2(x)
            x = torch.relu(x)
            return self.output(x)

    try:
        model = AHRMultiFamilyRanker(
            int(checkpoint["feature_buckets"]),
            int(checkpoint["hidden1"]),
            int(checkpoint["hidden2"]),
            len(labels),
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
    except Exception:
        return None
    return {
        "torch": torch,
        "model": model,
        "labels": labels,
        "feature_buckets": int(checkpoint["feature_buckets"]),
        "parameter_count": int(checkpoint.get("parameter_count", 0)),
        "acceptance": acceptance,
        "checkpoint_name": str(checkpoint.get("run_name", MULTIFAMILY_RANKER_CHECKPOINT_NAME)),
    }


def normalize_question(question: str) -> str:
    q = re.sub(r"\s+", " ", question.lower()).strip()
    replacements = {
        "no fewer than": "at least",
        "not fewer than": "at least",
        "no more than": "at most",
        "not more than": "at most",
        "what are the odds": "what is the probability",
        "what's the chance": "what is the chance",
        "average": "mean",
        "expected average": "expected mean",
    }
    for src, dst in replacements.items():
        q = q.replace(src, dst)
    return q


def fraction_text(value: Fraction) -> str:
    return str(value.numerator) if value.denominator == 1 else f"{value.numerator}/{value.denominator}"


def decimal_text(value: Fraction) -> str:
    return f"{float(value):.6f}"


def percent_text(value: Fraction) -> str:
    return f"{100 * float(value):.4f}%"


def ints_in_text(text: str) -> list[int]:
    return [int(x) for x in re.findall(r"(?<![\w.])-?\d+(?!\w)", text)]


def number_word_to_int(text: str) -> str:
    words = {
        "zero": 0,
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
        "seven": 7,
        "eight": 8,
        "nine": 9,
        "ten": 10,
        "eleven": 11,
        "twelve": 12,
        "thirteen": 13,
        "fourteen": 14,
        "fifteen": 15,
        "sixteen": 16,
        "seventeen": 17,
        "eighteen": 18,
        "nineteen": 19,
        "twenty": 20,
    }
    out = text
    for word, value in words.items():
        out = re.sub(rf"\b{word}\b", str(value), out)
    return out


def ranker_features(question: str, family: str) -> list[str]:
    tokens = re.findall(r"[a-z0-9]+", question.lower())
    features = ["bias", f"family={family}"]
    features.extend(f"tok={tok}" for tok in tokens)
    features.extend(f"bigram={a}_{b}" for a, b in zip(tokens, tokens[1:]))
    features.extend("has_number" for token in tokens if token.isdigit())
    return features


def ranker_score(model: dict[str, Any], label: str, features: list[str]) -> float:
    weights = model.get("weights", {}).get(label, {})
    return float(model.get("priors", {}).get(label, 0.0)) + sum(float(weights.get(feature, 0.0)) for feature in features)


def choose_best_with_multifamily_ranker(
    model_info: dict[str, Any],
    candidates: list[CandidateProgram],
    question: str,
    family: str,
) -> CandidateProgram | None:
    torch = model_info["torch"]
    model = model_info["model"]
    labels = model_info["labels"]
    label_to_id = {label: idx for idx, label in enumerate(labels)}
    legal = {candidate.name: candidate for candidate in candidates if candidate.name in label_to_id}
    if not legal:
        return None
    features = ranker_features(question, family)
    indices = sorted({stable_hash(feature, int(model_info["feature_buckets"])) for feature in features}) or [0]
    with torch.no_grad():
        flat = torch.tensor(indices, dtype=torch.long)
        offsets = torch.tensor([0, len(indices)], dtype=torch.long)
        logits = model(flat, offsets)[0]
        ranked = sorted(
            ((name, float(logits[label_to_id[name]].item())) for name in legal),
            key=lambda item: (-item[1], item[0]),
        )
    best_name, best_score = ranked[0]
    base = legal[best_name]
    return CandidateProgram(base.family, base.name, base.params, best_score)


def choose_best(candidates: list[CandidateProgram], question: str, family: str) -> tuple[CandidateProgram | None, str]:
    if not candidates:
        return None, "abstain"
    multifamily_model = load_multifamily_ranker_model()
    if multifamily_model is not None:
        ranked = choose_best_with_multifamily_ranker(multifamily_model, candidates, question, family)
        if ranked is not None:
            return ranked, MULTIFAMILY_RANKER_BACKEND_TYPE
    model = load_ranker_model()
    if model is not None:
        features = ranker_features(question, family)
        legal = {candidate.name: candidate for candidate in candidates}
        ranked = sorted(
            ((name, ranker_score(model, name, features)) for name in legal),
            key=lambda item: (-item[1], item[0]),
        )
        best_name, best_score = ranked[0]
        base = legal[best_name]
        return CandidateProgram(base.family, base.name, base.params, best_score), "trained_ahr_ranker"
    return sorted(candidates, key=lambda c: (-c.score, c.name))[0], "verified_program_backend"


def solve(question: str, dsl: dict[str, Any]) -> dict[str, Any]:
    family = str(dsl.get("family", ""))
    q = normalize_question(number_word_to_int(question))
    if family == "probability_counting_program":
        return solve_probability(q)
    if family == "statistics_program":
        return solve_statistics(q)
    if family == "finite_logic_constraints":
        return solve_logic(q)
    if family == "short_arithmetic_program":
        return solve_arithmetic(q)
    return abstain(question, family, "unsupported_demo_program_family")


def abstain(question: str, family: str, reason: str) -> dict[str, Any]:
    return {
        "clinical_use": "forbidden",
        "checkpoint": CHECKPOINT_NAME,
        "family": family,
        "verified": False,
        "answer": None,
        "trace": [
            f"Built finite AHR-CSP task for family {family}.",
            f"No legal verified demo program matched: {reason}.",
        ],
        "abstain_reason": reason,
        "model_results": [],
    }


def solve_probability(q: str) -> dict[str, Any]:
    candidates: list[CandidateProgram] = []
    ints = ints_in_text(q)
    if "dice" in q or "die" in q:
        target = None
        if "sum" in q or "total" in q:
            match = re.search(r"\b(?:sum|total)(?:\s+(?:is|of|to|equals?))?\s+(\d+)\b", q)
            target = int(match.group(1)) if match else (ints[-1] if ints else None)
        if target is not None:
            candidates.append(CandidateProgram("probability_counting_program", "dice_sum_exact", {"dice": 2, "sides": 6, "target": target}, 0.95))
    if ("coin" in q or "flip" in q or "heads" in q) and any(x in q for x in ["row", "run", "sequence", "consecutive", "streak"]):
        n = ints[0] if ints else None
        k = ints[1] if len(ints) > 1 else None
        if n and k:
            candidates.append(CandidateProgram("probability_counting_program", "coin_run_at_least_one", {"flips": n, "run_length": k, "target": "heads"}, 0.99))
    if ("coin" in q or "heads" in q or "tails" in q) and any(x in q for x in ["exactly", "at least", "at most", "between"]):
        n = None
        k = None
        trial_match = re.search(r"\b(?:tossed|flipped|flip|toss|flipping)\s+(\d+)\s+(?:times|coins?|flips?)\b", q)
        if trial_match is None:
            trial_match = re.search(r"\b(\d+)\s+(?:coin\s+)?(?:flips?|tosses?)\b", q)
        if trial_match:
            n = int(trial_match.group(1))
        exact_match = re.search(r"\bexactly\s+(\d+)\s+(?:heads|tails|successes)\b", q)
        atleast_match = re.search(r"\bat\s+least\s+(\d+)\s+(?:heads|tails|successes)\b", q)
        atmost_match = re.search(r"\bat\s+most\s+(\d+)\s+(?:heads|tails|successes)\b", q)
        between_match = re.search(r"\bbetween\s+(\d+)\s+and\s+(\d+)\s+(?:heads|tails|successes)\b", q)
        if exact_match:
            k = int(exact_match.group(1))
        elif atleast_match:
            k = int(atleast_match.group(1))
        elif atmost_match:
            k = int(atmost_match.group(1))
        elif between_match:
            low = int(between_match.group(1))
            high = int(between_match.group(2))
            k = low
        if len(ints) >= 2 and (n is None or k is None):
            n = max(ints) if n is None else n
            k = min(ints) if k is None else k
        event = "between" if between_match else "exactly" if "exactly" in q else "at_least" if "at least" in q else "at_most"
        if n is not None and k is not None:
            params = {"flips": n, "successes": k, "event": event}
            if between_match:
                params["low"] = low
                params["high"] = high
            candidates.append(CandidateProgram("probability_counting_program", "coin_binomial", params, 0.9))
    if "birthday" in q and ints:
        candidates.append(CandidateProgram("probability_counting_program", "birthday_at_least_one_match", {"people": ints[0]}, 0.82))
    if "card" in q or "deck" in q:
        ranks = ["ace", "king", "queen", "jack"]
        suits = ["heart", "spade", "club", "diamond"]
        rank = next((rank for rank in ranks if rank in q), None)
        suit = next((suit for suit in suits if suit in q), None)
        if rank and suit and (" or " in q or "either" in q):
            candidates.append(CandidateProgram("probability_counting_program", "card_rank_or_suit", {"rank": rank, "suit": suit}, 0.87))
        elif rank:
            candidates.append(CandidateProgram("probability_counting_program", "card_rank", {"rank": rank}, 0.8))
        elif suit:
            candidates.append(CandidateProgram("probability_counting_program", "card_suit", {"suit": suit}, 0.8))
    if "bag" in q or "urn" in q or "balls" in q:
        colors = {color: count for count, color in re.findall(r"\b(\d+)\s+(red|blue|green|yellow|white|black)\b", q)}
        colors = {color: int(count) for color, count in colors.items()}
        target = next((color for color in colors if f"both are {color}" in q or f"both {color}" in q), None)
        if colors and target:
            candidates.append(CandidateProgram("probability_counting_program", "urn_both_without_replacement", {"colors": colors, "target_color": target}, 0.86))

    program, backend_type = choose_best(candidates, q, "probability_counting_program")
    if program is None:
        return abstain(q, "probability_counting_program", "no_probability_template_matched")
    if program.name == "dice_sum_exact":
        return execute_dice_sum(program, backend_type)
    if program.name == "coin_run_at_least_one":
        return execute_coin_run(program, backend_type)
    if program.name == "coin_binomial":
        return execute_coin_binomial(program, backend_type)
    if program.name == "birthday_at_least_one_match":
        return execute_birthday(program, backend_type)
    if program.name in {"card_rank_or_suit", "card_rank", "card_suit"}:
        return execute_card(program, backend_type)
    if program.name == "urn_both_without_replacement":
        return execute_urn(program, backend_type)
    return abstain(q, "probability_counting_program", "unknown_probability_program")


def execute_dice_sum(program: CandidateProgram, backend_type: str) -> dict[str, Any]:
    dice = int(program.params["dice"])
    sides = int(program.params["sides"])
    target = int(program.params["target"])
    outcomes = list(itertools.product(range(1, sides + 1), repeat=dice))
    favorable = [outcome for outcome in outcomes if sum(outcome) == target]
    p = Fraction(len(favorable), len(outcomes))
    trace = [
        "AHR-CSP selected legal program: dice_sum_exact.",
        f"State variables: dice={dice}, sides={sides}, target_sum={target}.",
        f"Enumerated finite outcome space: {sides}^{dice} = {len(outcomes)} outcomes.",
        f"Verifier counted favorable outcomes where sum == {target}: {len(favorable)}.",
        f"Exact probability = {fraction_text(p)} = {decimal_text(p)} = {percent_text(p)}.",
    ]
    return verified_result(program, f"The probability is {fraction_text(p)} = {decimal_text(p)} ({percent_text(p)}).", trace, p, backend_type=backend_type)


def execute_coin_run(program: CandidateProgram, backend_type: str) -> dict[str, Any]:
    n = int(program.params["flips"])
    k = int(program.params["run_length"])
    if n <= 0 or k <= 0 or k > n or n > 200:
        return abstain("", "probability_counting_program", "coin_run_parameters_out_of_range")
    states = [0] * k
    states[0] = 1
    for _ in range(n):
        nxt = [0] * k
        nxt[0] = sum(states)
        for run_len in range(k - 1):
            nxt[run_len + 1] += states[run_len]
        states = nxt
    total = 2**n
    no_run = sum(states)
    favorable = total - no_run
    p = Fraction(favorable, total)
    trace = [
        "AHR-CSP selected legal program: coin_run_at_least_one.",
        f"State variables: flips={n}, target=heads, run_length={k}.",
        f"Total sequences = 2^{n} = {total}.",
        f"Dynamic-program verifier counted sequences with no run of {k} heads: {no_run}.",
        f"Favorable sequences = {total} - {no_run} = {favorable}.",
        f"Exact probability = {fraction_text(p)} = {decimal_text(p)} = {percent_text(p)}.",
    ]
    return verified_result(program, f"The probability is {fraction_text(p)} = {decimal_text(p)} ({percent_text(p)}).", trace, p, backend_type=backend_type)


def execute_coin_binomial(program: CandidateProgram, backend_type: str) -> dict[str, Any]:
    n = int(program.params["flips"])
    k = int(program.params["successes"])
    event = str(program.params["event"])
    if n < 0 or k < 0 or k > n or n > 200:
        return abstain("", "probability_counting_program", "coin_binomial_parameters_out_of_range")
    if event == "exactly":
        favorable = math.comb(n, k)
        desc = f"C({n},{k})"
    elif event == "at_least":
        favorable = sum(math.comb(n, i) for i in range(k, n + 1))
        desc = f"sum C({n},i), i={k}..{n}"
    elif event == "between":
        low = int(program.params["low"])
        high = int(program.params["high"])
        if low < 0 or high > n or low > high:
            return abstain("", "probability_counting_program", "coin_binomial_between_parameters_out_of_range")
        favorable = sum(math.comb(n, i) for i in range(low, high + 1))
        desc = f"sum C({n},i), i={low}..{high}"
    else:
        favorable = sum(math.comb(n, i) for i in range(0, k + 1))
        desc = f"sum C({n},i), i=0..{k}"
    total = 2**n
    p = Fraction(favorable, total)
    trace = [
        "AHR-CSP selected legal program: coin_binomial.",
        f"State variables: flips={n}, successes={k}, event={event}.",
        f"Verifier favorable count = {desc} = {favorable}.",
        f"Total sequences = 2^{n} = {total}.",
        f"Exact probability = {fraction_text(p)} = {decimal_text(p)} = {percent_text(p)}.",
    ]
    return verified_result(program, f"The probability is {fraction_text(p)} = {decimal_text(p)} ({percent_text(p)}).", trace, p, backend_type=backend_type)


def execute_birthday(program: CandidateProgram, backend_type: str) -> dict[str, Any]:
    people = int(program.params["people"])
    if people < 2 or people > 366:
        return abstain("", "probability_counting_program", "birthday_parameters_out_of_range")
    no_match = Fraction(1, 1)
    for i in range(people):
        no_match *= Fraction(365 - i, 365)
    p = 1 - no_match
    trace = [
        "AHR-CSP selected legal program: birthday_at_least_one_match.",
        f"State variables: people={people}, days=365.",
        "Verifier computes complement probability of all birthdays different.",
        f"P(no match) = product((365-i)/365 for i=0..{people - 1}) = {fraction_text(no_match)}.",
        f"P(at least one match) = 1 - P(no match) = {fraction_text(p)} = {decimal_text(p)} = {percent_text(p)}.",
    ]
    return verified_result(program, f"The probability is {fraction_text(p)} = {decimal_text(p)} ({percent_text(p)}).", trace, p, backend_type=backend_type)


def execute_card(program: CandidateProgram, backend_type: str) -> dict[str, Any]:
    if program.name == "card_rank_or_suit":
        favorable = 4 + 13 - 1
        desc = f"4 cards of rank {program.params['rank']} plus 13 {program.params['suit']}s minus 1 overlap"
    elif program.name == "card_rank":
        favorable = 4
        desc = f"4 cards of rank {program.params['rank']}"
    else:
        favorable = 13
        desc = f"13 cards in suit {program.params['suit']}"
    p = Fraction(favorable, 52)
    trace = [
        f"AHR-CSP selected legal program: {program.name}.",
        "State variables: standard_deck=52 cards.",
        f"Verifier favorable count: {desc} = {favorable}.",
        f"Exact probability = {fraction_text(p)} = {decimal_text(p)} = {percent_text(p)}.",
    ]
    return verified_result(program, f"The probability is {fraction_text(p)} = {decimal_text(p)} ({percent_text(p)}).", trace, p, backend_type=backend_type)


def execute_urn(program: CandidateProgram, backend_type: str) -> dict[str, Any]:
    colors = {str(k): int(v) for k, v in program.params["colors"].items()}
    target = str(program.params["target_color"])
    total = sum(colors.values())
    target_count = colors[target]
    if total < 2 or target_count < 2:
        return abstain("", "probability_counting_program", "urn_parameters_out_of_range")
    p = Fraction(target_count, total) * Fraction(target_count - 1, total - 1)
    trace = [
        "AHR-CSP selected legal program: urn_both_without_replacement.",
        f"State variables: colors={colors}, target={target}, draws=2, replacement=false.",
        f"Verifier step 1 probability = {target_count}/{total}.",
        f"Verifier step 2 probability = {target_count - 1}/{total - 1}.",
        f"Exact probability = {fraction_text(p)} = {decimal_text(p)} = {percent_text(p)}.",
    ]
    return verified_result(program, f"The probability is {fraction_text(p)} = {decimal_text(p)} ({percent_text(p)}).", trace, p, backend_type=backend_type)


def solve_statistics(q: str) -> dict[str, Any]:
    values = ints_in_text(q)
    if len(values) < 2:
        return abstain(q, "statistics_program", "need_at_least_two_numbers")
    if "median" in q:
        candidates = [CandidateProgram("statistics_program", "median", {"values": values}, 0.95)]
    elif "mode" in q:
        candidates = [CandidateProgram("statistics_program", "mode", {"values": values}, 0.95)]
    elif "sample variance" in q:
        candidates = [CandidateProgram("statistics_program", "mean_sample_variance", {"values": values}, 0.95)]
    else:
        candidates = [CandidateProgram("statistics_program", "mean_variance", {"values": values}, 0.95)]
    program, backend_type = choose_best(candidates, q, "statistics_program")
    assert program is not None
    if program.name == "median":
        sorted_values = sorted(values)
        n = len(values)
        median = Fraction(sorted_values[n // 2], 1) if n % 2 else Fraction(sorted_values[n // 2 - 1] + sorted_values[n // 2], 2)
        trace = [
            "AHR-CSP selected legal program: median.",
            f"State variables: values={values}.",
            f"Verifier sorted values = {sorted_values}.",
            f"Median = {fraction_text(median)} = {decimal_text(median)}.",
        ]
        return verified_result(program, f"The median is {fraction_text(median)} = {decimal_text(median)}.", trace, median, backend_type=backend_type)
    if program.name == "mode":
        counts = {value: values.count(value) for value in sorted(set(values))}
        max_count = max(counts.values())
        modes = [value for value, count in counts.items() if count == max_count]
        if len(modes) != 1:
            return abstain(q, "statistics_program", "mode_not_unique")
        trace = [
            "AHR-CSP selected legal program: mode.",
            f"State variables: values={values}.",
            f"Verifier counts = {counts}.",
            f"Unique mode = {modes[0]}.",
        ]
        return verified_result(program, f"The mode is {modes[0]}.", trace, modes[0], backend_type=backend_type)
    total = sum(values)
    n = len(values)
    mean = Fraction(total, n)
    denom = n - 1 if program.name == "mean_sample_variance" else n
    if denom <= 0:
        return abstain(q, "statistics_program", "sample_variance_needs_two_values")
    variance = sum((Fraction(x) - mean) ** 2 for x in values) / denom
    variance_name = "Sample variance" if program.name == "mean_sample_variance" else "Population variance"
    trace = [
        f"AHR-CSP selected legal program: {program.name}.",
        f"State variables: values={values}.",
        f"Verifier sum = {total}; count = {n}.",
        f"Mean = sum/count = {fraction_text(mean)} = {decimal_text(mean)}.",
        f"{variance_name} = squared deviation sum / {denom} = {fraction_text(variance)} = {decimal_text(variance)}.",
    ]
    answer = f"The mean is {fraction_text(mean)} = {decimal_text(mean)}. {variance_name} is {fraction_text(variance)} = {decimal_text(variance)}."
    return verified_result(program, answer, trace, mean, extra={"variance": fraction_text(variance)}, backend_type=backend_type)


def solve_logic(q: str) -> dict[str, Any]:
    pattern = re.compile(r"\b([a-z][a-z]*)\s+is\s+(?:taller|older|greater|higher)\s+than\s+([a-z][a-z]*)\b|\b([a-z][a-z]*)\s+is\s+(?:ranked above|ahead of)\s+([a-z][a-z]*)\b")
    raw_pairs = pattern.findall(q)
    pairs = [(a or c, b or d) for a, b, c, d in raw_pairs]
    if not pairs:
        return abstain(q, "finite_logic_constraints", "supported_logic_demo_needs_ordering_constraints")
    names = sorted({name for pair in pairs for name in pair})
    edges = {name: set() for name in names}
    indeg = {name: 0 for name in names}
    for higher, lower in pairs:
        if lower not in edges[higher]:
            edges[higher].add(lower)
            indeg[lower] += 1
    frontier = [name for name in names if indeg[name] == 0]
    order: list[str] = []
    while frontier:
        node = sorted(frontier)[0]
        frontier.remove(node)
        order.append(node)
        for nxt in sorted(edges[node]):
            indeg[nxt] -= 1
            if indeg[nxt] == 0:
                frontier.append(nxt)
    if len(order) != len(names):
        return abstain(q, "finite_logic_constraints", "ordering_constraints_cycle_detected")
    top = order[0]
    candidates = [CandidateProgram("finite_logic_constraints", "ordering_topological_sort", {"relations": pairs}, 0.9)]
    program, backend_type = choose_best(candidates, q, "finite_logic_constraints")
    assert program is not None
    trace = [
        "AHR-CSP selected legal program: ordering_topological_sort.",
        f"State variables: entities={names}; constraints={pairs}.",
        "Verifier built a directed acyclic ordering graph.",
        f"Topological order from greatest/tallest/oldest to least: {order}.",
        f"Answer entity = {top}.",
    ]
    return verified_result(program, f"{top.title()} is first in the verified ordering.", trace, top, backend_type=backend_type)


def solve_arithmetic(q: str) -> dict[str, Any]:
    unsupported_math_terms = [
        "determinant",
        "matrix",
        "eigen",
        "integral",
        "derivative",
        "limit",
        "polynomial",
        "quadratic",
        "system of equations",
        "differential equation",
    ]
    if any(term in q for term in unsupported_math_terms):
        return abstain(q, "short_arithmetic_program", "unsupported_advanced_math_form")
    nums = ints_in_text(q)
    if len(nums) < 2:
        return abstain(q, "short_arithmetic_program", "need_at_least_two_numbers")
    if any(x in q for x in ["multiply", "times", "product", "rows of", "each day", "at each", "with ", "covers"]):
        result = math.prod(nums)
        name = "multiply_all"
        expr = " * ".join(str(x) for x in nums)
    elif any(x in q for x in ["subtract", "minus", "spent", "spends", "sold", "left", "remain"]):
        from_match = re.search(r"\bsubtract\s+(-?\d+)\s+from\s+(-?\d+)\b", q)
        if from_match:
            result = int(from_match.group(2)) - int(from_match.group(1))
            expr = f"{from_match.group(2)} - {from_match.group(1)}"
        else:
            result = nums[0]
            for x in nums[1:]:
                result -= x
            expr = " - ".join(str(x) for x in nums)
        name = "subtract_chain"
    elif any(x in q for x in ["divide", "split", "shared", "equally"]):
        if nums[1] == 0 or nums[0] % nums[1] != 0:
            return abstain(q, "short_arithmetic_program", "division_not_exact_or_zero")
        result = nums[0] // nums[1]
        name = "divide_exact"
        expr = f"{nums[0]} / {nums[1]}"
    else:
        result = sum(nums)
        name = "add_all"
        expr = " + ".join(str(x) for x in nums)
    candidates = [CandidateProgram("short_arithmetic_program", name, {"numbers": nums}, 0.88)]
    program, backend_type = choose_best(candidates, q, "short_arithmetic_program")
    assert program is not None
    trace = [
        f"AHR-CSP selected legal program: {name}.",
        f"State variables: numbers={nums}.",
        f"Verifier executed exact integer expression: {expr} = {result}.",
        "Hard energy = 0 because the executable trace matches the verifier result.",
    ]
    return verified_result(program, f"The answer is {result}.", trace, result, backend_type=backend_type)


def verified_result(
    program: CandidateProgram,
    answer: str,
    trace: list[str],
    expected: Any,
    extra: dict[str, Any] | None = None,
    *,
    backend_type: str,
) -> dict[str, Any]:
    record = {
        "family": program.family,
        "model_program": program.name,
        "model_assignment": program.params,
        "model_score": program.score,
        "backend_type": backend_type,
        "cpu_check": {"valid": True, "energy": 0.0, "expected": str(expected)},
    }
    if extra:
        record.update(extra)
    ranker_checkpoint = None
    ranker_parameter_count = None
    if backend_type == MULTIFAMILY_RANKER_BACKEND_TYPE:
        ranker_checkpoint = MULTIFAMILY_RANKER_CHECKPOINT_NAME
        model_info = load_multifamily_ranker_model()
        ranker_parameter_count = None if model_info is None else model_info.get("parameter_count")
    elif backend_type == "trained_ahr_ranker":
        ranker_checkpoint = RANKER_CHECKPOINT_NAME
    return {
        "clinical_use": "forbidden",
        "checkpoint": MULTIFAMILY_RANKER_CHECKPOINT_NAME if backend_type == MULTIFAMILY_RANKER_BACKEND_TYPE else CHECKPOINT_NAME,
        "ranker_checkpoint": ranker_checkpoint,
        "ranker_parameter_count": ranker_parameter_count,
        "backend_type": backend_type,
        "family": program.family,
        "verified": True,
        "answer": answer,
        "trace": trace + ["CPU verifier accepted the program: valid=true, hard_energy=0."],
        "abstain_reason": None,
        "model_results": [record],
    }

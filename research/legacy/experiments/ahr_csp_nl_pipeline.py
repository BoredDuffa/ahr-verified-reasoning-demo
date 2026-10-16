#!/usr/bin/env python3
"""Natural-language translator and gate policy for AHR-CSP-only demos.

The translator is intentionally non-authoritative: it may only emit a typed
DSL. AHR-CSP execution plus CPU verification decides whether an answer exists.
"""

from __future__ import annotations

import json
import os
import re
import threading
from dataclasses import dataclass
from typing import Any, Callable


PRIMARY_TRANSLATOR = "Qwen/Qwen2.5-0.5B-Instruct"
FALLBACK_TRANSLATOR = "Qwen/Qwen3-0.6B"
TRAINED_TASK_FAMILIES = {"operation_synthesis_len2_factored", "operation_synthesis_len3_factored"}
DEMO_PROGRAM_FAMILIES = {
    "short_arithmetic_program",
    "probability_counting_program",
    "statistics_program",
    "finite_logic_constraints",
}
KNOWN_AHR_FAMILIES = {
    "operation_synthesis",
    "short_arithmetic_program",
    "probability_counting_program",
    "statistics_program",
    "finite_logic_constraints",
    "sequence_graph_reasoning",
    "unsupported",
}
UNTRAINED_FAMILIES = KNOWN_AHR_FAMILIES - {"operation_synthesis", "unsupported"} - DEMO_PROGRAM_FAMILIES
WORD_NUMBERS = {
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
}


@dataclass
class Translation:
    dsl: dict[str, Any]
    translator_model: str
    raw_output: str
    valid_json: bool
    source: str


def number_after(patterns: list[str], text: str) -> int | None:
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return int(match.group(1))
    for word, value in WORD_NUMBERS.items():
        if re.search(rf"\b{word}\b", text):
            return value
    return None


def normalize_question(question: str) -> str:
    return re.sub(r"\s+", " ", question.lower()).strip()


def strict_json_object(text: str) -> dict[str, Any] | None:
    text = text.strip()
    candidates = [text]
    first = text.find("{")
    last = text.rfind("}")
    if first >= 0 and last > first:
        candidates.append(text[first : last + 1])
    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


def rule_translate(question: str, model_name: str = "deterministic_schema_fallback") -> Translation:
    q = normalize_question(question)
    count = number_after([
        r"\b(\d+)\s+(?:operation\s+synthesis\s+)?examples?\b",
        r"\b(\d+)\s+length\s+[23]\s+operation",
        r"\b(\d+)\s+len[23]\s+operation",
        r"\bon\s+(\d+)\b",
        r"\b(\d+)\s+records?\b",
    ], q) or 1
    split = "val" if "validation" in q or " val " in f" {q} " else "train" if "train" in q else "test"
    if "operation synthesis" in q or "ahr-csp" in q or "ahr csp" in q or "run the model" in q:
        task_family = "operation_synthesis_len2_factored" if "len2" in q or "length 2" in q else "operation_synthesis_len3_factored"
        dsl = {
            "family": "operation_synthesis",
            "task_family": task_family,
            "request_kind": "sample",
            "count": max(1, min(count, 10)),
            "split": split,
            "assumptions": ["sample existing gated AHR-CSP dataset records"],
            "confidence": 0.95,
        }
    elif any(token in q for token in ["add", "plus", "subtract", "minus", "multiply", "product", "divide", "arithmetic", "split", "shared equally", "per kid", "per person", "spent", "spends", "sold", "left", "remains", "rows of", "each day", "at each"]):
        dsl = {"family": "short_arithmetic_program", "request_kind": "solve", "confidence": 0.8, "assumptions": []}
    elif any(token in q for token in ["coin", "dice", "die", "card", "deck", "bag", "urn", "balls", "probability", "bayes", "birthday", "chance", "odds"]):
        dsl = {"family": "probability_counting_program", "request_kind": "solve", "confidence": 0.8, "assumptions": []}
    elif any(token in q for token in ["mean", "average", "variance", "standard deviation", "median", "mode", "statistics", "stats"]):
        dsl = {"family": "statistics_program", "request_kind": "solve", "confidence": 0.8, "assumptions": []}
    elif any(token in q for token in ["sequence", "pattern", "next term"]) or ("graph" in q and "path" in q):
        dsl = {"family": "sequence_graph_reasoning", "request_kind": "solve", "confidence": 0.8, "assumptions": []}
    elif any(token in q for token in ["logic", "truth", "constraint", "sudoku", "proof", "taller than", "older than", "higher than", "greater than", "ranked above", "ahead of", "who is tallest", "who is oldest", "who is highest", "which is greatest"]):
        dsl = {"family": "finite_logic_constraints", "request_kind": "solve", "confidence": 0.8, "assumptions": []}
    else:
        dsl = {"family": "unsupported", "request_kind": "abstain", "confidence": 1.0, "assumptions": []}
    return Translation(dsl=dsl, translator_model=model_name, raw_output=json.dumps(dsl, sort_keys=True), valid_json=True, source="rules")


def normalize_translator_dsl(question: str, dsl: dict[str, Any]) -> dict[str, Any]:
    """Constrain Qwen output to the published AHR-CSP DSL contract.

    This function may correct routing/schema fields from the original question,
    but it never computes an answer.
    """

    q = normalize_question(question)
    out = dict(dsl)

    def mark_untrained_family(family: str) -> dict[str, Any]:
        return {
            "family": family,
            "request_kind": "solve",
            "count": 1,
            "split": "test",
            "assumptions": [],
            "confidence": max(float(out.get("confidence", 0.0) or 0.0), 0.8),
        }

    if "operation synthesis" in q or "ahr-csp" in q or "ahr csp" in q or "run the model" in q:
        fallback = rule_translate(question).dsl
        out.update(
            {
                "family": "operation_synthesis",
                "task_family": fallback["task_family"],
                "request_kind": "sample",
                "count": fallback["count"],
                "split": fallback["split"],
                "assumptions": fallback["assumptions"],
                "confidence": max(float(out.get("confidence", 0.0) or 0.0), 0.95),
            }
        )
        return out
    if any(token in q for token in ["weather", "poem", "story", "joke", "capital of", "who are you"]):
        return {"family": "unsupported", "request_kind": "abstain", "confidence": 1.0, "assumptions": ["not an AHR-CSP reasoning task"]}
    if any(token in q for token in ["logic grid", "sudoku", "satisfiable", "truth table", "taller than", "older than", "higher than", "greater than", "ranked above", "ahead of", "who is tallest", "who is oldest", "who is highest", "which is greatest"]):
        return mark_untrained_family("finite_logic_constraints")
    if any(token in q for token in ["add", "plus", "subtract", "minus", "multiply", "product", "divide", "arithmetic", "split", "shared equally", "per kid", "per person", "spent", "spends", "sold", "left", "remains", "rows of", "each day", "at each"]):
        return mark_untrained_family("short_arithmetic_program")
    if any(token in q for token in ["probability", "chance", "odds", "dice", "die", "coin", "card", "deck", "bag", "urn", "balls", "bayes", "birthday"]):
        return mark_untrained_family("probability_counting_program")
    if any(token in q for token in ["mean", "average", "variance", "standard deviation", "median", "mode", "statistics", "stats"]):
        return mark_untrained_family("statistics_program")
    if any(token in q for token in ["sequence", "pattern", "next term"]) or ("graph" in q and "path" in q):
        return mark_untrained_family("sequence_graph_reasoning")
    return out


class QwenJsonTranslator:
    """Lazy local Qwen translator with deterministic schema fallback."""

    def __init__(self, model_name: str = PRIMARY_TRANSLATOR, fallback_name: str = FALLBACK_TRANSLATOR, mode: str | None = None) -> None:
        self.model_name = model_name
        self.fallback_name = fallback_name
        self.mode = mode or os.environ.get("AHR_CSP_TRANSLATOR_MODE", "qwen")
        self._loaded_name: str | None = None
        self._tokenizer: Any = None
        self._model: Any = None
        self._load_error: str | None = None
        self._lock = threading.RLock()

    @property
    def status(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "primary": self.model_name,
            "fallback": self.fallback_name,
            "loaded": self._loaded_name,
            "load_error": self._load_error,
        }

    def _load(self) -> None:
        if self._model is not None or self._load_error is not None:
            return
        if self.mode == "rules":
            self._load_error = "translator_mode_rules"
            return
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            for name in [self.model_name, self.fallback_name]:
                try:
                    self._tokenizer = AutoTokenizer.from_pretrained(name)
                    self._model = AutoModelForCausalLM.from_pretrained(name, torch_dtype="auto", device_map="auto")
                    self._loaded_name = name
                    return
                except Exception as exc:  # pragma: no cover - depends on remote HF/cache state
                    self._load_error = f"{name}: {type(exc).__name__}: {exc}"
        except Exception as exc:  # pragma: no cover
            self._load_error = f"translator_import_error: {type(exc).__name__}: {exc}"

    def prompt(self, question: str) -> list[dict[str, str]]:
        return [
            {
                "role": "system",
                "content": (
                    "You are a JSON classifier for an AHR-CSP demo. "
                    "Output exactly one JSON object and nothing else. No prose. No final answer. "
                    "Allowed families: operation_synthesis, short_arithmetic_program, "
                    "probability_counting_program, statistics_program, finite_logic_constraints, "
                    "sequence_graph_reasoning, unsupported. "
                    "If the user asks to run AHR-CSP or operation synthesis, use operation_synthesis. "
                    "Required keys: family, task_family, request_kind, count, split, assumptions, confidence. "
                    "operation_synthesis uses the current neural checkpoint. "
                    "probability_counting_program, statistics_program, finite_logic_constraints, "
                    "and short_arithmetic_program use finite AHR-CSP program backends with exact verification. "
                    "sequence_graph_reasoning is parsed but not answered."
                ),
            },
            {
                "role": "user",
                "content": "Run the AHR-CSP model on 3 operation synthesis examples",
            },
            {
                "role": "assistant",
                "content": "{\"family\":\"operation_synthesis\",\"task_family\":\"operation_synthesis_len3_factored\",\"request_kind\":\"sample\",\"count\":3,\"split\":\"test\",\"assumptions\":[\"sample existing gated AHR-CSP dataset records\"],\"confidence\":0.95}",
            },
            {
                "role": "user",
                "content": "What is the probability of rolling a sum of seven with two dice?",
            },
            {
                "role": "assistant",
                "content": "{\"family\":\"probability_counting_program\",\"request_kind\":\"solve\",\"count\":1,\"split\":\"test\",\"assumptions\":[],\"confidence\":0.8}",
            },
            {
                "role": "user",
                "content": "Write me a poem about the model.",
            },
            {
                "role": "assistant",
                "content": "{\"family\":\"unsupported\",\"request_kind\":\"abstain\",\"count\":1,\"split\":\"test\",\"assumptions\":[\"not an AHR-CSP reasoning task\"],\"confidence\":1.0}",
            },
            {
                "role": "user",
                "content": "What is the weather tomorrow?",
            },
            {
                "role": "assistant",
                "content": "{\"family\":\"unsupported\",\"request_kind\":\"abstain\",\"count\":1,\"split\":\"test\",\"assumptions\":[\"not an AHR-CSP reasoning task\"],\"confidence\":1.0}",
            },
            {"role": "user", "content": question},
        ]

    def translate(self, question: str) -> Translation:
        with self._lock:
            self._load()
            if self._model is None or self._tokenizer is None:
                reason = self._load_error or "translator_unavailable"
                fallback = rule_translate(question, model_name=f"{self.model_name} unavailable; {reason}; deterministic fallback")
                fallback.source = "fallback"
                return fallback
            tokenizer = self._tokenizer
            model = self._model
            messages = self.prompt(question)
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer([text], return_tensors="pt").to(model.device)
            output_ids = model.generate(
                **inputs,
                max_new_tokens=180,
                do_sample=False,
                temperature=None,
                top_p=None,
            )
            generated = output_ids[0][inputs["input_ids"].shape[-1] :]
            raw = tokenizer.decode(generated, skip_special_tokens=True).strip()
        parsed = strict_json_object(raw)
        if parsed is None:
            fallback = rule_translate(
                question,
                model_name=f"{self._loaded_name or self.model_name} invalid JSON; deterministic schema fallback",
            )
            fallback.source = "qwen_invalid_json_fallback"
            fallback.dsl.setdefault("assumptions", [])
            fallback.dsl["assumptions"] = list(fallback.dsl["assumptions"]) + ["Qwen emitted non-object/invalid JSON; schema fallback used"]
            return fallback
        return Translation(dsl=normalize_translator_dsl(question, parsed), translator_model=self._loaded_name or self.model_name, raw_output=raw, valid_json=True, source="qwen")


def validate_dsl(dsl: dict[str, Any]) -> tuple[bool, str]:
    family = dsl.get("family")
    if family not in KNOWN_AHR_FAMILIES:
        return False, "unknown_family"
    confidence = dsl.get("confidence", 0)
    if not isinstance(confidence, (int, float)) or confidence < 0.75:
        return False, "low_confidence"
    if family == "unsupported":
        return True, "unsupported"
    if family in DEMO_PROGRAM_FAMILIES:
        if dsl.get("request_kind") not in {"solve", None}:
            return False, "invalid_demo_program_request_kind"
        return True, "demo_program_backend"
    if family != "operation_synthesis":
        return True, "known_but_untrained"
    task_family = dsl.get("task_family")
    if task_family not in TRAINED_TASK_FAMILIES:
        return False, "unsupported_operation_synthesis_task_family"
    if dsl.get("request_kind") != "sample":
        return False, "only_sample_requests_supported_for_current_checkpoint"
    count = dsl.get("count", 1)
    if not isinstance(count, int) or count < 1 or count > 10:
        return False, "count_out_of_range"
    split = dsl.get("split", "test")
    if split not in {"train", "val", "test"}:
        return False, "invalid_split"
    return True, "ok"


def response_contract(
    question: str,
    translation: Translation,
    *,
    checkpoint: str,
    verified: bool,
    answer: str | None,
    trace: list[str],
    abstain_reason: str | None,
    model_results: list[dict[str, Any]] | None = None,
    backend_type: str | None = None,
) -> dict[str, Any]:
    return {
        "clinical_use": "forbidden",
        "question": question,
        "translator_model": translation.translator_model,
        "translator_source": translation.source,
        "translator_valid_json": translation.valid_json,
        "translator_raw_output": translation.raw_output,
        "dsl": translation.dsl,
        "ahr_checkpoint": checkpoint,
        "verified": verified,
        "answer": answer,
        "trace": trace,
        "abstain_reason": abstain_reason,
        "model_results": model_results or [],
        "backend_type": backend_type or ("abstain" if abstain_reason else "unknown"),
    }


def paraphrase_rows() -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    templates = {
        "operation_synthesis": [
            "Run the AHR-CSP model on {n} operation synthesis examples",
            "Show {n} len3 operation synthesis records with the AHR CSP checkpoint",
            "Use the model for {n} operation synthesis test examples",
            "Run {n} length 2 operation synthesis examples",
        ],
        "probability_counting_program": [
            "What is the probability of exactly two heads in three coin flips?",
            "If I roll two dice, what is the chance the sum is seven?",
            "A disease test has sensitivity and specificity; compute the posterior probability.",
        ],
        "statistics_program": [
            "Find the mean and variance of these values.",
            "Compute the standard deviation for this small dataset.",
        ],
        "finite_logic_constraints": [
            "Solve this logic grid puzzle.",
            "Can this Sudoku board be solved?",
            "Check whether the graph constraints are satisfiable.",
        ],
        "sequence_graph_reasoning": [
            "What is the next term in the sequence?",
            "Find a path satisfying these graph constraints.",
        ],
        "short_arithmetic_program": [
            "Add 12 and 19, then multiply by 3.",
            "What arithmetic program maps 4 to 22?",
        ],
        "unsupported": [
            "What is the weather tomorrow?",
            "Write me a poem about the model.",
        ],
    }
    for family, qs in templates.items():
        for i in range(30):
            template = qs[i % len(qs)]
            rows.append((template.format(n=1 + (i % 10)), family))
    return rows

#!/usr/bin/env python3
"""Exact operation-synthesis utilities for the AHR MVP harness."""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

Seq = Tuple[int, ...]
Program = Tuple[str, ...]

OP_NAMES: Tuple[str, ...] = ("REV", "DELTA", "DOUBLE-EVENS", "ZIP-SELF")
MAX_OUTPUT_LEN = 64


@dataclass(frozen=True)
class ExecutionResult:
    output: Seq
    trace: Tuple[Seq, ...]
    truncated: bool


@dataclass(frozen=True)
class SynthesisResult:
    start: Seq
    target: Seq
    length: int
    checked: int
    solutions: Tuple[Program, ...]
    closest: Tuple[Tuple[int, Program, Seq, bool], ...]

    @property
    def passed(self) -> bool:
        return bool(self.solutions)

    @property
    def conclusion(self) -> str:
        return "sat" if self.solutions else "unsat_under_declared_operation_semantics"


def rev(seq: Seq) -> Seq:
    return tuple(reversed(seq))


def delta(seq: Seq) -> Seq:
    return tuple(seq[i] - seq[i - 1] for i in range(1, len(seq)))


def double_evens(seq: Seq) -> Seq:
    return tuple((2 * value if i % 2 == 0 else value) for i, value in enumerate(seq))


def zip_self(seq: Seq) -> Seq:
    if not seq:
        return ()
    out: List[int] = [seq[0]]
    for value in seq[1:]:
        out.extend([value, value])
    return tuple(out)


def apply_op(seq: Seq, op: str) -> Seq:
    if op == "REV":
        return rev(seq)
    if op == "DELTA":
        return delta(seq)
    if op == "DOUBLE-EVENS":
        return double_evens(seq)
    if op == "ZIP-SELF":
        return zip_self(seq)
    raise ValueError(f"unknown operation: {op}")


def execute(start: Sequence[int], program: Sequence[str], max_output_len: int = MAX_OUTPUT_LEN) -> ExecutionResult:
    seq: Seq = tuple(int(x) for x in start)
    trace: List[Seq] = [seq]
    truncated = False
    for op in program:
        seq = apply_op(seq, op)
        if len(seq) > max_output_len:
            truncated = True
            trace.append(seq[:max_output_len])
            break
        trace.append(seq)
    return ExecutionResult(output=seq, trace=tuple(trace), truncated=truncated)


def all_programs(length: int) -> Iterable[Program]:
    return itertools.product(OP_NAMES, repeat=length)


def distance(output: Seq, target: Seq) -> int:
    return abs(len(output) - len(target)) + sum(abs(a - b) for a, b in zip(output, target))


def synthesize(start: Sequence[int], target: Sequence[int], length: int, keep_closest: int = 10) -> SynthesisResult:
    start_seq = tuple(int(x) for x in start)
    target_seq = tuple(int(x) for x in target)
    checked = 0
    solutions: List[Program] = []
    closest: List[Tuple[int, Program, Seq, bool]] = []

    for program in all_programs(length):
        checked += 1
        result = execute(start_seq, program)
        if not result.truncated and result.output == target_seq:
            solutions.append(tuple(program))
        score = distance(result.output, target_seq)
        closest.append((score, tuple(program), result.output, result.truncated))

    closest = sorted(closest, key=lambda item: (item[0], item[1]))[:keep_closest]
    return SynthesisResult(
        start=start_seq,
        target=target_seq,
        length=length,
        checked=checked,
        solutions=tuple(solutions),
        closest=tuple(closest),
    )


def verify(start: Sequence[int], target: Sequence[int], program: Sequence[str]) -> dict:
    result = execute(start, program)
    target_seq = tuple(int(x) for x in target)
    return {
        "program": list(program),
        "output": list(result.output),
        "trace": [list(state) for state in result.trace],
        "truncated": result.truncated,
        "valid": (not result.truncated) and result.output == target_seq,
    }


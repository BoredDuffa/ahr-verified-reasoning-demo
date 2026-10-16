#!/usr/bin/env python3
"""Exact Tower-of-Hanoi search with mid-trajectory rule revision."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Iterable, Tuple

Peg = int
State = Tuple[Peg, ...]
Move = Tuple[int, str, str]

PEG_NAMES = ("A", "B", "C")
PEG_TO_ID = {name: i for i, name in enumerate(PEG_NAMES)}


@dataclass(frozen=True)
class HanoiResult:
    disks: int
    start: str
    target: str
    switch_after: int
    capped_peg: str
    cap: int
    checked_states: int
    moves: Tuple[Move, ...] | None
    final_state: State | None
    conclusion: str

    @property
    def passed(self) -> bool:
        return self.moves is not None


def initial_state(disks: int, start: str) -> State:
    return tuple([PEG_TO_ID[start]] * disks)


def target_state(disks: int, target: str) -> State:
    return tuple([PEG_TO_ID[target]] * disks)


def peg_counts(state: State) -> dict[str, int]:
    return {PEG_NAMES[peg]: sum(1 for value in state if value == peg) for peg in range(3)}


def top_disks(state: State) -> dict[int, int]:
    tops = {}
    for disk, peg in enumerate(state):
        tops.setdefault(peg, disk)
    return tops


def legal_moves(state: State) -> Iterable[tuple[int, Peg, Peg]]:
    tops = top_disks(state)
    for src, disk in tops.items():
        for dst in range(3):
            if dst == src:
                continue
            dst_top = tops.get(dst)
            if dst_top is None or disk < dst_top:
                yield disk, src, dst


def apply_move(state: State, disk: int, dst: Peg) -> State:
    values = list(state)
    values[disk] = dst
    return tuple(values)


def respects_cap(state: State, capped_peg: str, cap: int) -> bool:
    return peg_counts(state)[capped_peg] <= cap


def solve_rule_revision(
    disks: int,
    start: str = "A",
    target: str = "C",
    switch_after: int = 8,
    capped_peg: str = "B",
    cap: int = 2,
    max_depth: int | None = None,
) -> HanoiResult:
    if start not in PEG_TO_ID or target not in PEG_TO_ID or capped_peg not in PEG_TO_ID:
        raise ValueError("pegs must be A, B, or C")
    start_state = initial_state(disks, start)
    goal = target_state(disks, target)
    q = deque([(start_state, 0, tuple())])
    seen = {(start_state, 0)}
    checked = 0

    while q:
        state, depth, moves = q.popleft()
        checked += 1
        if state == goal and depth >= switch_after:
            return HanoiResult(disks, start, target, switch_after, capped_peg, cap, checked, moves, state, "sat")
        if max_depth is not None and depth >= max_depth:
            continue
        for disk, src, dst in legal_moves(state):
            next_state = apply_move(state, disk, dst)
            next_depth = depth + 1
            if next_depth > switch_after and not respects_cap(next_state, capped_peg, cap):
                continue
            key = (next_state, min(next_depth, switch_after + 1))
            if key in seen:
                continue
            seen.add(key)
            move = (disk + 1, PEG_NAMES[src], PEG_NAMES[dst])
            q.append((next_state, next_depth, moves + (move,)))

    return HanoiResult(disks, start, target, switch_after, capped_peg, cap, checked, None, None, "unsat_under_bounded_hanoi_rule_revision_search")


def verify_hanoi_solution(
    disks: int,
    moves: Iterable[Move],
    start: str = "A",
    target: str = "C",
    switch_after: int = 8,
    capped_peg: str = "B",
    cap: int = 2,
) -> dict:
    state = initial_state(disks, start)
    trace = [state]
    valid = True
    failure = None
    for index, (disk_label, src_name, dst_name) in enumerate(moves, start=1):
        disk = int(disk_label) - 1
        src = PEG_TO_ID[src_name]
        dst = PEG_TO_ID[dst_name]
        available = set(legal_moves(state))
        if (disk, src, dst) not in available:
            valid = False
            failure = {"move_index": index, "reason": "illegal_hanoi_move", "move": [disk_label, src_name, dst_name]}
            break
        state = apply_move(state, disk, dst)
        if index > switch_after and not respects_cap(state, capped_peg, cap):
            valid = False
            failure = {"move_index": index, "reason": "post_revision_cap_violation", "state": list(state)}
            break
        trace.append(state)
    if state != target_state(disks, target):
        valid = False
        failure = failure or {"reason": "target_not_reached", "state": list(state)}
    return {
        "valid": valid,
        "failure": failure,
        "final_state": list(state),
        "trace": [list(item) for item in trace],
        "peg_counts": peg_counts(state),
    }


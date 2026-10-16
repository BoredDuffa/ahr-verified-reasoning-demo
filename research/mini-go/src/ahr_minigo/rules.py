"""Small-board Go with positional superko, no suicide, and area scoring.

Black is 1, White is -1, and empty is 0. Board points are row-major integers;
``size * size`` denotes pass. Two consecutive passes finish the game. The rules
are intentionally explicit: there is no move-limit adjudication or dead-stone
removal. Players must capture dead stones before passing if it affects scoring.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Iterable, Mapping


class IllegalMove(ValueError):
    """The requested action violates the configured Go rules."""


@lru_cache(maxsize=16)
def _neighbors(size: int) -> tuple[tuple[int, ...], ...]:
    return tuple(
        tuple(
            rr * size + cc
            for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))
            if 0 <= rr < size and 0 <= cc < size
        )
        for r in range(size)
        for c in range(size)
    )


@dataclass(frozen=True, slots=True)
class GoState:
    size: int
    board: tuple[int, ...]
    to_play: int = 1
    komi: float = 0.5
    history: frozenset[tuple[int, ...]] = frozenset()
    consecutive_passes: int = 0
    move_number: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.size, int) or self.size < 2:
            raise ValueError("Board size must be an integer of at least 2.")
        board = tuple(self.board)
        if len(board) != self.size * self.size or any(x not in (-1, 0, 1) for x in board):
            raise ValueError("Board must contain size*size entries from {-1, 0, 1}.")
        if self.to_play not in (-1, 1):
            raise ValueError("to_play must be Black (1) or White (-1).")
        if self.consecutive_passes not in (0, 1, 2) or self.move_number < 0:
            raise ValueError("Invalid pass count or move number.")
        history = frozenset(tuple(item) for item in self.history)
        if any(len(item) != len(board) or any(x not in (-1, 0, 1) for x in item) for item in history):
            raise ValueError("History contains an invalid board.")
        object.__setattr__(self, "board", board)
        object.__setattr__(self, "history", history | {board})

    @property
    def terminal(self) -> bool:
        return self.consecutive_passes == 2

    @property
    def pass_action(self) -> int:
        return self.size * self.size

    def to_dict(self) -> dict[str, Any]:
        return to_dict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> GoState:
        return from_dict(value)


def initial_state(size: int = 3, komi: float = 0.5) -> GoState:
    return GoState(size=size, board=(0,) * (size * size), komi=komi)


def group_and_liberties(
    board: Iterable[int], size: int, point: int
) -> tuple[frozenset[int], frozenset[int]]:
    """Return the connected stone group and its empty neighboring points."""
    values = tuple(board)
    if not 0 <= point < size * size or not values[point]:
        raise ValueError("point must contain a stone.")
    color = values[point]
    group = {point}
    liberties: set[int] = set()
    pending = [point]
    adjacent = _neighbors(size)
    while pending:
        current = pending.pop()
        for neighbor in adjacent[current]:
            if values[neighbor] == 0:
                liberties.add(neighbor)
            elif values[neighbor] == color and neighbor not in group:
                group.add(neighbor)
                pending.append(neighbor)
    return frozenset(group), frozenset(liberties)


def groups_and_liberties(
    board: Iterable[int], size: int
) -> tuple[tuple[frozenset[int], frozenset[int]], ...]:
    """Return each stone group exactly once, in board-point order."""
    values = tuple(board)
    visited: set[int] = set()
    result = []
    for point, color in enumerate(values):
        if color and point not in visited:
            group, liberties = group_and_liberties(values, size, point)
            visited.update(group)
            result.append((group, liberties))
    return tuple(result)


def _trusted_successor(
    state: GoState,
    board: tuple[int, ...],
    history: frozenset[tuple[int, ...]],
    consecutive_passes: int,
) -> GoState:
    """Construct a successor after ``play`` has checked its rule invariants.

    Public construction and deserialization validate all supplied history.
    Internally, every earlier board is already validated and the new board is
    created by ``play``; rescanning the full history at every search node adds
    work proportional to the game's length without adding a new check.
    """
    child = object.__new__(GoState)
    for name, value in (
        ("size", state.size),
        ("board", board),
        ("to_play", -state.to_play),
        ("komi", state.komi),
        ("history", history),
        ("consecutive_passes", consecutive_passes),
        ("move_number", state.move_number + 1),
    ):
        object.__setattr__(child, name, value)
    return child


def play(state: GoState, action: int) -> GoState:
    if state.terminal:
        raise IllegalMove("The game has already ended after two passes.")
    if not isinstance(action, int) or not 0 <= action <= state.pass_action:
        raise IllegalMove("Action is outside the board and is not pass.")
    if action == state.pass_action:
        return _trusted_successor(
            state, state.board, state.history, state.consecutive_passes + 1,
        )
    if state.board[action] != 0:
        raise IllegalMove("The point is occupied.")
    board = list(state.board)
    board[action] = state.to_play
    checked: set[int] = set()
    for neighbor in _neighbors(state.size)[action]:
        if board[neighbor] == -state.to_play and neighbor not in checked:
            group, liberties = group_and_liberties(board, state.size, neighbor)
            checked.update(group)
            if not liberties:
                for stone in group:
                    board[stone] = 0
    _, own_liberties = group_and_liberties(board, state.size, action)
    if not own_liberties:
        raise IllegalMove("Suicide is not permitted.")
    new_board = tuple(board)
    if new_board in state.history:
        raise IllegalMove("Positional superko forbids repeating an earlier board.")
    return _trusted_successor(state, new_board, state.history | {new_board}, 0)


def legal_actions(state: GoState) -> tuple[int, ...]:
    if state.terminal:
        return ()
    legal = []
    for action in range(state.pass_action):
        if state.board[action]:
            continue
        try:
            play(state, action)
        except IllegalMove:
            continue
        legal.append(action)
    return tuple(legal) + (state.pass_action,)


def area_score(state: GoState) -> float:
    """Black area minus White area minus komi, also usable before game end.

    A nonterminal score is only a board evaluation, not a proven game result.
    Empty regions bordered by both colors, or neither color, count for neither.
    """
    score = float(sum(state.board)) - state.komi
    visited: set[int] = set()
    adjacent = _neighbors(state.size)
    for point, color in enumerate(state.board):
        if color or point in visited:
            continue
        region = {point}
        visited.add(point)
        pending = [point]
        borders: set[int] = set()
        while pending:
            current = pending.pop()
            for neighbor in adjacent[current]:
                if state.board[neighbor]:
                    borders.add(state.board[neighbor])
                elif neighbor not in visited:
                    visited.add(neighbor)
                    region.add(neighbor)
                    pending.append(neighbor)
        if len(borders) == 1:
            score += borders.pop() * len(region)
    return score


def is_terminal(state: GoState) -> bool:
    return state.terminal


def to_dict(state: GoState) -> dict[str, Any]:
    """JSON-compatible complete state, including superko history."""
    return {
        "size": state.size,
        "board": list(state.board),
        "to_play": state.to_play,
        "komi": state.komi,
        "history": [list(board) for board in sorted(state.history)],
        "consecutive_passes": state.consecutive_passes,
        "move_number": state.move_number,
    }


def from_dict(value: Mapping[str, Any]) -> GoState:
    return GoState(
        size=value["size"],
        board=tuple(value["board"]),
        to_play=value["to_play"],
        komi=value["komi"],
        history=frozenset(tuple(board) for board in value["history"]),
        consecutive_passes=value["consecutive_passes"],
        move_number=value["move_number"],
    )

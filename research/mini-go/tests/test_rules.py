"""Rule fixtures and differential checks against a separate coordinate oracle."""

import json
import random
import unittest

from ahr_minigo.rules import (
    GoState,
    IllegalMove,
    area_score,
    from_dict,
    group_and_liberties,
    groups_and_liberties,
    initial_state,
    legal_actions,
    play,
    to_dict,
)


def reference_move(state, action):
    """Independent coordinate-map implementation; return None if illegal."""
    n = state.size
    if state.consecutive_passes == 2 or action < 0 or action > n * n:
        return None
    if action == n * n:
        return state.board
    grid = {(r, c): state.board[r * n + c] for r in range(n) for c in range(n)}
    location = divmod(action, n)
    if grid[location]:
        return None
    grid[location] = state.to_play

    def adjacent(rc):
        r, c = rc
        return [(x, y) for x, y in ((r, c - 1), (r, c + 1), (r - 1, c), (r + 1, c)) if (x, y) in grid]

    def component(origin):
        color = grid[origin]
        found = {origin}
        frontier = [origin]
        liberty = False
        for point in frontier:
            for other in adjacent(point):
                if grid[other] == 0:
                    liberty = True
                if grid[other] == color and other not in found:
                    found.add(other)
                    frontier.append(other)
        return found, liberty

    # Determine all opposing dead groups before removing any, independently of
    # the production implementation's neighbor-by-neighbor removal order.
    captured = set()
    for other in adjacent(location):
        if grid[other] == -state.to_play:
            stones, has_liberty = component(other)
            if not has_liberty:
                captured |= stones
    for stone in captured:
        grid[stone] = 0
    _, has_liberty = component(location)
    if not has_liberty:
        return None
    result = tuple(grid[(r, c)] for r in range(n) for c in range(n))
    return None if result in state.history else result


def reference_score(state):
    """Independent per-empty-point reachability scoring (no shared helpers)."""
    n = state.size
    total = -state.komi
    for start, occupant in enumerate(state.board):
        if occupant:
            total += occupant
            continue
        discovered = {start}
        frontier = [start]
        sees_black = False
        sees_white = False
        for current in frontier:
            row, col = divmod(current, n)
            for rr, cc in ((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)):
                if not (0 <= rr < n and 0 <= cc < n):
                    continue
                other = rr * n + cc
                color = state.board[other]
                sees_black |= color == 1
                sees_white |= color == -1
                if color == 0 and other not in discovered:
                    discovered.add(other)
                    frontier.append(other)
        if sees_black and not sees_white:
            total += 1
        if sees_white and not sees_black:
            total -= 1
    return float(total)


class GoRulesTests(unittest.TestCase):
    def test_initial_and_occupied(self):
        state = initial_state()
        self.assertEqual(state.history, frozenset({(0,) * 9}))
        self.assertEqual(legal_actions(state), tuple(range(10)))
        child = play(state, 4)
        self.assertEqual(child.to_play, -1)
        self.assertEqual(child.board[4], 1)
        self.assertEqual(state.board, (0,) * 9)
        with self.assertRaises(IllegalMove):
            play(child, 4)
        for action in (-1, 10, 1.5):
            with self.assertRaises(IllegalMove):
                play(state, action)

    def test_single_capture(self):
        state = GoState(3, (0, -1, 1, -1, 1, 0, 0, 0, 0))
        result = play(state, 0)
        self.assertEqual(result.board, (1, 0, 1, -1, 1, 0, 0, 0, 0))

    def test_multiple_simultaneous_captures_rescue_apparent_suicide(self):
        state = GoState(3, (1, -1, 1, -1, 0, -1, 1, -1, 1))
        result = play(state, 4)
        self.assertEqual(result.board, (1, 0, 1, 0, 1, 0, 1, 0, 1))

    def test_suicide_and_connected_group_suicide(self):
        for board in (
            (0, -1, 0, -1, 0, -1, 0, -1, 0),
            (-1, 1, -1, -1, 0, -1, 0, -1, 0),
        ):
            with self.assertRaisesRegex(IllegalMove, "Suicide"):
                play(GoState(3, board), 4)

    def test_ko_repetition_and_history_roundtrip(self):
        board = [0] * 25
        for r, c in ((0, 1), (1, 0), (2, 1)):
            board[r * 5 + c] = 1
        for r, c in ((1, 1), (0, 2), (1, 3), (2, 2)):
            board[r * 5 + c] = -1
        state = GoState(5, tuple(board))
        child = play(state, 7)
        self.assertEqual(child.board[6], 0)
        with self.assertRaisesRegex(IllegalMove, "superko"):
            play(child, 6)
        restored = from_dict(json.loads(json.dumps(to_dict(child))))
        self.assertEqual(restored, child)
        with self.assertRaisesRegex(IllegalMove, "superko"):
            play(restored, 6)
        # The repetition check must inspect all earlier boards, not merely the
        # immediately previous board or the side to move.
        unrelated = (0,) * 25
        extended = GoState(5, child.board, to_play=-1, history=child.history | {unrelated})
        with self.assertRaisesRegex(IllegalMove, "superko"):
            play(extended, 6)

    def test_pass_exempt_from_superko_and_two_passes_finish(self):
        state = initial_state(5)
        once = play(state, 25)
        self.assertFalse(once.terminal)
        self.assertEqual(once.history, state.history)
        twice = play(once, 25)
        self.assertTrue(twice.terminal)
        self.assertEqual(legal_actions(twice), ())
        self.assertEqual(area_score(twice), -0.5)
        with self.assertRaises(IllegalMove):
            play(twice, 25)
        resumed = play(once, 0)
        self.assertEqual(resumed.consecutive_passes, 0)

    def test_area_stones_territory_neutral_and_komi(self):
        self.assertEqual(area_score(initial_state(3)), -0.5)
        self.assertEqual(area_score(GoState(3, (1, 1, 1, 1, 0, 1, 1, 1, 1))), 8.5)
        self.assertEqual(area_score(GoState(3, (-1, -1, -1, -1, 0, -1, -1, -1, -1))), -9.5)
        self.assertEqual(area_score(GoState(3, (1, 0, -1, 1, 0, -1, 1, 0, -1))), -0.5)
        self.assertEqual(area_score(GoState(3, (1, 0, -1, 1, 0, -1, 1, 0, -1), komi=2.5)), -2.5)

    def test_group_helpers(self):
        board = (1, 1, 0, 1, -1, 0, 0, 0, -1)
        group, liberties = group_and_liberties(board, 3, 0)
        self.assertEqual(group, frozenset({0, 1, 3}))
        self.assertEqual(liberties, frozenset({2, 6}))
        self.assertEqual(len(groups_and_liberties(board, 3)), 3)

    def test_random_legal_replay_against_independent_oracles(self):
        for size in (3, 5):
            for seed in range(8):
                rng = random.Random(100 * size + seed)
                state = initial_state(size)
                for _ in range(60):
                    expected = {
                        action: reference_move(state, action)
                        for action in range(size * size + 1)
                    }
                    expected = {action: board for action, board in expected.items() if board is not None}
                    with self.subTest(size=size, seed=seed, ply=state.move_number):
                        self.assertEqual(legal_actions(state), tuple(expected))
                        self.assertEqual(area_score(state), reference_score(state))
                        for action, expected_board in expected.items():
                            self.assertEqual(play(state, action).board, expected_board)
                    if state.terminal:
                        break
                    nonpasses = [action for action in expected if action != size * size]
                    action = rng.choice(nonpasses) if nonpasses and rng.random() > 0.1 else size * size
                    state = play(state, action)


if __name__ == "__main__":
    unittest.main()

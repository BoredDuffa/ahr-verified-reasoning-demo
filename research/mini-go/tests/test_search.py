"""Independent recursive oracles for bounded search and learned-backup wiring."""

import numpy as np
import pytest
import torch

from ahr_minigo.rules import GoState, area_score, initial_state, legal_actions, play
from ahr_minigo.search import build_tree, exact_values, neural_values


def recursive_oracle(state, depth, mode="adversarial"):
    """Traverse actual rules directly, without using the built-tree representation."""
    if depth == 0 or state.terminal:
        return area_score(state) / (state.size * state.size + abs(state.komi))
    children = [recursive_oracle(play(state, action), depth - 1, mode)
                for action in legal_actions(state)]
    if mode == "mean":
        return sum(children) / len(children)
    if mode == "cooperative" or state.to_play == 1:
        return max(children)
    return min(children)


class MeanBackup(torch.nn.Module):
    """A deliberately non-adversarial model detects any hidden exact backups."""

    def __init__(self):
        super().__init__()
        self.roles_seen = set()
        self.parents_seen = 0

    def forward(self, values, mask, roles):
        self.roles_seen.update(roles.tolist())
        self.parents_seen += len(roles)
        return values.masked_fill(~mask, 0).sum(1) / mask.sum(1)


def sample_states():
    opening = initial_state(3)
    white_to_play = play(opening, 4)
    midgame = play(play(white_to_play, 0), 1)
    one_pass = play(midgame, 9)
    terminal = play(one_pass, 9)
    capture = GoState(3, (1, -1, 1, -1, 0, -1, 1, -1, 1))
    return [opening, white_to_play, midgame, one_pass, terminal, capture]


@pytest.mark.parametrize("depth", [1, 2, 3])
def test_exact_tree_values_match_independent_recursive_oracle(depth):
    for state in sample_states():
        tree = build_tree(state, depth)
        actual = exact_values(tree)
        assert np.isfinite(actual).all()
        assert actual[0] == pytest.approx(recursive_oracle(state, depth), abs=1e-6)
        # Check the complete root-action vector, including White's minimization.
        assert tree.root_actions == legal_actions(state)
        expected = [recursive_oracle(play(state, action), depth - 1)
                    for action in tree.root_actions]
        np.testing.assert_allclose(actual[tree.children[0]], expected, atol=1e-6)


def test_complete_tree_contains_all_and_only_legal_successors():
    for state in sample_states():
        tree = build_tree(state, 3)
        flat = [index for level in tree.levels for index in level]
        assert sorted(flat) == list(range(len(tree.states)))
        assert len(flat) == len(set(flat))
        for depth, level in enumerate(tree.levels):
            for index in level:
                current = tree.states[index]
                children = tree.children[index]
                if depth == tree.depth or current.terminal:
                    assert children == []
                    assert np.isfinite(tree.leaf_values[index])
                else:
                    expected = [play(current, action) for action in legal_actions(current)]
                    assert [tree.states[child] for child in children] == expected
                    assert np.isnan(tree.leaf_values[index])
                    assert all(child in tree.levels[depth + 1] for child in children)


def test_neural_backups_use_the_model_at_every_internal_node():
    state = initial_state(3)
    tree = build_tree(state, 3)
    fake_model = MeanBackup()
    actual = neural_values(tree, fake_model)
    for depth, level in enumerate(tree.levels):
        for index in level:
            expected = recursive_oracle(tree.states[index], tree.depth - depth, "mean")
            assert actual[index] == pytest.approx(expected, abs=1e-6)
    assert fake_model.roles_seen == {-1.0, 1.0}
    assert fake_model.parents_seen == sum(bool(children) for children in tree.children)
    # Thus a learned path cannot secretly substitute exact minimax internally.
    assert not np.isclose(actual[0], exact_values(tree)[0], atol=1e-4)


def test_opponent_replies_cannot_be_treated_as_cooperating_choices():
    state = initial_state(3)
    adversarial = recursive_oracle(state, 2, "adversarial")
    cooperative = recursive_oracle(state, 2, "cooperative")
    assert adversarial == pytest.approx(-0.5 / 9.5)
    assert cooperative == pytest.approx(8.5 / 9.5)
    assert cooperative > adversarial + 0.9
    assert exact_values(build_tree(state, 2))[0] == pytest.approx(adversarial)


def test_terminal_nodes_need_no_model_and_invalid_depth_is_rejected():
    terminal = play(play(initial_state(3), 9), 9)
    model = MeanBackup()
    values = neural_values(build_tree(terminal, 3), model)
    assert model.parents_seen == 0
    np.testing.assert_allclose(values, [-0.5 / 9.5], atol=1e-7)
    with pytest.raises(ValueError, match="positive"):
        build_tree(initial_state(3), 0)

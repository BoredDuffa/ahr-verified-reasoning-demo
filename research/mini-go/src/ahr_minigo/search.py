"""Complete bounded Go trees, learned backups, and analytic control engines."""
from __future__ import annotations
from dataclasses import dataclass
from functools import lru_cache
import time
import numpy as np
import torch
from .rules import GoState, play, legal_actions, area_score


@dataclass
class Tree:
    states: list
    children: list
    levels: list
    root_actions: tuple
    depth: int
    leaf_values: np.ndarray


@lru_cache(maxsize=24)
def build_tree(state: GoState, depth: int) -> Tree:
    if depth < 1:
        raise ValueError("depth must be positive")
    states, children, levels = [state], [[]], [[0]]
    root_actions = legal_actions(state)
    for distance in range(depth):
        next_level = []
        for parent in levels[-1]:
            current = states[parent]
            for action in legal_actions(current):
                index = len(states)
                children[parent].append(index)
                states.append(play(current, action))
                children.append([])
                next_level.append(index)
        levels.append(next_level)
    scale = state.size ** 2 + abs(state.komi)
    leaves = np.array([area_score(s) / scale if not ch else np.nan
                       for s, ch in zip(states, children)], dtype=np.float32)
    return Tree(states, children, levels, root_actions, depth, leaves)


def exact_values(tree: Tree) -> np.ndarray:
    """Independent analytic reference for this finite-horizon objective only."""
    values = tree.leaf_values.copy()
    for indices in reversed(tree.levels):
        for parent in indices:
            if tree.children[parent]:
                descendants = values[tree.children[parent]]
                values[parent] = descendants.max() if tree.states[parent].to_play == 1 else descendants.min()
    return values


@torch.inference_mode()
def neural_values(tree: Tree, model) -> np.ndarray:
    """Every nonterminal internal backup is learned; root selection is separate."""
    values = tree.leaf_values.copy()
    for indices in reversed(tree.levels):
        parents = [p for p in indices if tree.children[p]]
        if not parents:
            continue
        for start in range(0, len(parents), 1024):
            batch = parents[start:start + 1024]
            width = max(len(tree.children[p]) for p in batch)
            child_values = np.zeros((len(batch), width), dtype=np.float32)
            mask = np.zeros_like(child_values, dtype=bool)
            roles = np.array([tree.states[p].to_play for p in batch], dtype=np.float32)
            for row, parent in enumerate(batch):
                count = len(tree.children[parent])
                child_values[row, :count] = values[tree.children[parent]]
                mask[row, :count] = True
            prediction = model(torch.from_numpy(child_values), torch.from_numpy(mask), torch.from_numpy(roles))
            if not torch.isfinite(prediction).all():
                raise RuntimeError("Nonfinite neural backup")
            values[batch] = prediction.detach().cpu().numpy()
    return values


def best_indices(scores, to_play, tolerance=1e-6):
    scores = np.asarray(scores)
    utility = to_play * scores
    return np.flatnonzero(utility >= utility.max() - tolerance)


def select_action(actions, scores, state, rng):
    best = best_indices(scores, state.to_play)
    # Shared, predeclared tie convention; neither trained nor analytic privileged.
    candidates = [actions[int(i)] for i in best]
    if state.pass_action in candidates:
        return state.pass_action
    return int(rng.choice(candidates))


class SearchPolicy:
    def __init__(self, name, model=None, depth3=3, depth5=2):
        self.name, self.model = name, model
        self.depth3, self.depth5 = depth3, depth5
        self._cache = {}

    def choose(self, state, rng, history=()):
        started = time.perf_counter()
        depth = self.depth3 if state.size == 3 else self.depth5
        key = (state, depth)
        cached = key in self._cache
        if not cached:
            tree = build_tree(state, depth)
            values = exact_values(tree) if self.model is None else neural_values(tree, self.model)
            self._cache[key] = (tree.root_actions, values[tree.children[0]].copy(), len(tree.states))
        actions, scores, nodes = self._cache[key]
        action = select_action(actions, scores, state, rng)
        return action, dict(nodes=0 if cached else nodes, nominal_tree_nodes=nodes,
                            cached=cached, elapsed_s=time.perf_counter() - started,
                            depth=depth)


class RandomPolicy:
    name = "random"

    def choose(self, state, rng, history=()):
        started = time.perf_counter()
        actions = legal_actions(state)
        return int(rng.choice(actions)), dict(nodes=len(actions), elapsed_s=time.perf_counter() - started)


class GreedyPolicy:
    name = "greedy"

    def choose(self, state, rng, history=()):
        started = time.perf_counter()
        actions = legal_actions(state)
        scores = [area_score(play(state, action)) for action in actions]
        return select_action(actions, scores, state, rng), dict(nodes=len(actions), elapsed_s=time.perf_counter() - started)


class GnuGoPolicy:
    name = "gnugo_level1"

    def __init__(self, binary, size, seed):
        from .gtp import GnuGo
        self.engine = GnuGo(binary, size=size, komi=.5, level=1, seed=seed)

    def choose(self, state, rng, history=()):
        started = time.perf_counter()
        self.engine.reset(history)
        action = self.engine.genmove(state.to_play)
        return action, dict(nodes=None, elapsed_s=time.perf_counter() - started,
                            depth=None, resets_and_replay_included=True)

    def close(self):
        self.engine.close()

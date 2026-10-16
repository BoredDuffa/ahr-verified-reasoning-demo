"""Held-out positions, paired mini-Go games, and auditable game replay."""
from __future__ import annotations
import json
import time
from pathlib import Path
import numpy as np
from .io import atomic_json, json_hash
from .rules import initial_state, play, legal_actions, area_score, to_dict, from_dict
from .search import build_tree, exact_values, neural_values, best_indices, select_action
from .search import SearchPolicy, RandomPolicy, GreedyPolicy, GnuGoPolicy
from .training import load_model, DEFAULT_CONFIG


def canonical_board(state):
    board = np.array(state.board).reshape(state.size, state.size)
    variants = []
    for k in range(4):
        b = np.rot90(board, k)
        variants.extend([tuple(b.ravel().tolist()), tuple(np.fliplr(b).ravel().tolist())])
    return (state.to_play, state.consecutive_passes, min(variants))


def generate_positions(n=192, seed=81047, size=3):
    rng = np.random.default_rng(seed)
    positions, seen = [], set()
    attempts = 0
    while len(positions) < n:
        attempts += 1
        if attempts > n * 500:
            raise RuntimeError("Insufficient distinct positions")
        state = initial_state(size)
        for _ in range(int(rng.integers(1, size * size * 2))):
            actions = legal_actions(state)
            if not actions:
                break
            action = int(rng.choice(actions))
            state = play(state, action)
        if state.terminal or len(legal_actions(state)) < 2:
            continue
        key = canonical_board(state)
        if key not in seen:
            seen.add(key)
            positions.append(to_dict(state))
    return positions


def evaluate_positions(output, config=None, n=192):
    config = dict(DEFAULT_CONFIG if config is None else config)
    output = Path(output)
    path = output / "data/heldout_positions.json"
    if path.exists():
        positions = json.loads(path.read_text())
        if len(positions) != n:
            raise ValueError("Heldout size changed")
    else:
        positions = generate_positions(n)
        atomic_json(path, positions)
    rows = []
    for seed in config["seeds"]:
        for mode in ("quantified", "existential", "untrained"):
            result_path = output / "results/positions" / f"{mode}_seed{seed}.json"
            if result_path.exists():
                rows.extend(json.loads(result_path.read_text()))
                continue
            model = load_model(output, mode, seed, config)
            local = []
            for i, raw in enumerate(positions):
                state = from_dict(raw)
                tree = build_tree(state, 3)
                began = time.perf_counter()
                oracle = exact_values(tree)
                exact_s = time.perf_counter() - began
                began = time.perf_counter()
                estimates = neural_values(tree, model)
                neural_s = time.perf_counter() - began
                exact_action_values = oracle[tree.children[0]]
                action_values = estimates[tree.children[0]]
                rng = np.random.default_rng(991000 + i)
                chosen = select_action(tree.root_actions, action_values, state, rng)
                action_index = tree.root_actions.index(chosen)
                best = best_indices(exact_action_values, state.to_play)
                utility = state.to_play * exact_action_values
                regret = float((utility.max() - utility[action_index]) * (state.size ** 2 + abs(state.komi)))
                local.append(dict(position=i, position_sha256=json_hash(raw), mode=mode, seed=seed,
                                  legal_actions=len(tree.root_actions), chosen=chosen,
                                  optimal=bool(action_index in best), informative=len(best) < len(tree.root_actions),
                                  regret_points=regret, nodes=len(tree.states),
                                  exact_backup_s=exact_s, neural_backup_s=neural_s,
                                  all_frontier_terminal=all(s.terminal for s, ch in zip(tree.states, tree.children) if not ch)))
            atomic_json(result_path, local)
            rows.extend(local)
            print(f"Positions {mode}/{seed}: {sum(r['optimal'] for r in local)}/{len(local)} horizon-optimal", flush=True)
    atomic_json(output / "results/positions_all.json", rows)
    return rows


def opening_moves(size, pair_id):
    if pair_id == 0:
        return []
    rng = np.random.default_rng(65177 + 101 * size + pair_id)
    state = initial_state(size)
    history = []
    for _ in range(2):
        actions = [a for a in legal_actions(state) if a != state.pass_action]
        action = int(rng.choice(actions))
        history.append((state.to_play, action))
        state = play(state, action)
    return history


def game_sgf(record):
    def escape(s):
        return str(s).replace("\\", "\\\\").replace("]", "\\]")
    size = record["size"]
    result = "Void" if record["status"] != "completed" else ("B+" if record["score_black"] > 0 else "W+") + f"{abs(record['score_black']):g}"
    parts = [f"(;GM[1]FF[4]SZ[{size}]KM[0.5]RU[Chinese positional superko; no suicide]PB[{escape(record['black'])}]PW[{escape(record['white'])}]RE[{result}]C[Bounded AHR pilot. {escape(record['status'])}]"]
    for move in record["moves"]:
        color = "B" if move["color"] == 1 else "W"
        action = move["action"]
        vertex = "" if action == size * size else chr(97 + action % size) + chr(97 + action // size)
        parts.append(f";{color}[{vertex}]")
    return "".join(parts) + ")\n"


def play_game(ahr, opponent, size, pair_id, color, seed, max_plies=100):
    state = initial_state(size)
    history = opening_moves(size, pair_id)
    moves = []
    for turn, action in history:
        state = play(state, action)
        moves.append(dict(color=turn, action=action, opening=True))
    rngs = {1: np.random.default_rng(seed * 100003 + pair_id * 101 + 1),
            -1: np.random.default_rng(seed * 100003 + pair_id * 101 + 2)}
    policies = {color: ahr, -color: opponent}
    name = {c: p.name for c, p in policies.items()}
    failure = None
    while not state.terminal and state.move_number < max_plies:
        actor = state.to_play
        try:
            action, diagnostic = policies[actor].choose(state, rngs[actor], history)
            if action not in legal_actions(state):
                raise ValueError(f"Engine returned illegal action {action}")
            state = play(state, action)
            history.append((actor, action))
            moves.append(dict(color=actor, action=action, opening=False, **diagnostic))
        except Exception as exc:
            failure = dict(actor=name[actor], error=type(exc).__name__, message=str(exc))
            break
    status = "engine_error" if failure else ("completed" if state.terminal else "ply_cutoff")
    score = area_score(state)
    return dict(size=size, opening_pair=pair_id, seed=seed, ahr_mode=ahr.name,
                opponent=opponent.name, ahr_color=color, black=name[1], white=name[-1],
                moves=moves, status=status, failure=failure, score_black=score,
                ahr_win=(bool(color * score > 0) if status == "completed" else None),
                final_state=to_dict(state), max_plies=max_plies)


def tournament(output, binary, config=None, opening_pairs=6, include_5x5=True):
    config = dict(DEFAULT_CONFIG if config is None else config)
    output = Path(output)
    records = []
    for size in (3, 5) if include_5x5 else (3,):
        seeds = config["seeds"] if size == 3 else config["seeds"][:1]
        opponents = ("random", "greedy", "minimax", "gnugo_level1") if size == 3 else ("gnugo_level1",)
        pairs = opening_pairs if size == 3 else 3
        for seed in seeds:
            for mode in ("quantified", "existential", "untrained"):
                ahr = SearchPolicy(mode, load_model(output, mode, seed, config))
                for opponent_name in opponents:
                    opponent = ({"random": lambda: RandomPolicy(),
                                 "greedy": lambda: GreedyPolicy(),
                                 "minimax": lambda: SearchPolicy("minimax"),
                                 "gnugo_level1": lambda: GnuGoPolicy(binary, size, seed)}[opponent_name])()
                    try:
                        for pair in range(pairs):
                            for color in (1, -1):
                                game_id = f"{size}x{size}_{mode}_s{seed}_{opponent_name}_p{pair}_{color}"
                                path = output / "games" / f"{game_id}.json"
                                if path.exists():
                                    record = json.loads(path.read_text())
                                else:
                                    record = play_game(ahr, opponent, size, pair, color, seed)
                                    record["game_id"] = game_id
                                    atomic_json(path, record)
                                    path.with_suffix(".sgf").write_text(game_sgf(record))
                                records.append(record)
                    finally:
                        if hasattr(opponent, "close"):
                            opponent.close()
                    selected = [r for r in records if r["size"] == size and r["seed"] == seed and r["ahr_mode"] == mode and r["opponent"] == opponent_name]
                    print(f"Games {size}x{size} {mode}/{seed} vs {opponent_name}: {sum(r['ahr_win'] is True for r in selected)}/{sum(r['status']=='completed' for r in selected)} wins; {sum(r['status']!='completed' for r in selected)} incomplete", flush=True)
    atomic_json(output / "results/games_all.json", records)
    return records


def audit_game_records(output):
    output = Path(output)
    files = sorted((output / "games").glob("*.json"))
    counts = dict(games=0, moves=0, completed=0, ply_cutoff=0, engine_error=0)
    for path in files:
        record = json.loads(path.read_text())
        state = initial_state(record["size"])
        for move in record["moves"]:
            if state.to_play != move["color"]:
                raise AssertionError("Turn mismatch")
            state = play(state, move["action"])
            counts["moves"] += 1
        if to_dict(state) != record["final_state"] or area_score(state) != record["score_black"]:
            raise AssertionError("Final state/score mismatch")
        if record["status"] == "completed":
            assert state.terminal and record["ahr_win"] == (record["ahr_color"] * area_score(state) > 0)
        counts["games"] += 1
        counts[record["status"]] += 1
    atomic_json(output / "results/game_replay_audit.json", counts)
    return counts

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_legacy() -> Any:
    here = Path(__file__).resolve().parent
    return load_module(here / "run_lean_ahr_v2_14_next5_cost_frontier.py", "lean_ahr_v214")


LEGACY = load_legacy()


def compact(text: str | None, limit: int = 700) -> str | None:
    return LEGACY.compact(text, limit)


def normalize_statement(name: str, statement: str) -> str:
    text = " ".join((statement or "").strip().split())
    if not text:
        raise ValueError("statement is empty")
    if "THEOREM_NAME" in text:
        return text
    return re.sub(r"^theorem\s+[A-Za-z_][A-Za-z0-9_']*", f"theorem THEOREM_NAME", text, count=1)


def spec_from_statement(name: str, statement: str, family: str | None = None) -> dict[str, Any]:
    return {
        "name": name,
        "split": "user",
        "family": family or "user_supplied",
        "statement": normalize_statement(name, statement),
    }


def parse_binders(statement: str) -> dict[str, list[str]]:
    binders: dict[str, list[str]] = defaultdict(list)
    for body in re.findall(r"\(([^()]*)\)", statement or ""):
        if ":" not in body:
            continue
        names, typ = body.split(":", 1)
        typ = typ.strip()
        for name in names.split():
            if re.match(r"^[A-Za-z_][A-Za-z0-9_']*$", name):
                binders[typ].append(name)
    return dict(binders)


def unique_proofs(proofs: list[list[str]]) -> list[list[str]]:
    out: list[list[str]] = []
    seen: set[tuple[str, ...]] = set()
    for proof in proofs:
        key = tuple(proof)
        if key not in seen:
            seen.add(key)
            out.append(proof)
    return out


def preferred_intro_count(statement: str, max_intros: int) -> int:
    head = (statement or "").rsplit(":= by", 1)[0]
    idx = head.rfind(") :")
    if idx >= 0:
        target = head[idx + 3 :]
    else:
        parts = head.split(":", 1)
        target = parts[1] if len(parts) == 2 else head
    count = 0
    depth = 0
    i = 0
    while i < len(target):
        ch = target[i]
        if ch == "(":
            depth += 1
            i += 1
            continue
        if ch == ")":
            depth = max(0, depth - 1)
            i += 1
            continue
        if depth == 0 and target.startswith("->", i):
            count += 1
            i += 2
            continue
        if depth == 0 and ch == "→":
            count += 1
        i += 1
    return max(0, min(max_intros, count))


def intro_prefix(n: int) -> list[str]:
    return [f"intro h{i}" for i in range(n)]


def hnames(n: int = 7) -> list[str]:
    return [f"h{i}" for i in range(n)]


def close_exprs(n: int, nat_vars: list[str]) -> list[str]:
    hs = hnames(n)
    exprs: list[str] = [
        "assumption",
        "contradiction",
        "rfl",
        "simp",
        "trivial",
    ]
    for h in hs:
        exprs.extend([
            f"exact {h}",
            f"exact {h}.left",
            f"exact {h}.right",
            f"exact False.elim {h}",
        ])
    for a in hs:
        for b in hs:
            if a != b:
                exprs.extend([
                    f"exact {a} {b}",
                    f"exact False.elim ({a} {b})",
                ])
    for a in hs:
        for b in hs:
            for c in hs:
                if len({a, b, c}) == 3:
                    exprs.extend([
                        f"exact {a} ({b} {c})",
                        f"exact {a} ({b}.left)",
                        f"exact {a} ({b}.right)",
                    ])
    for a in hs:
        for b in hs:
            for c in hs:
                for d in hs:
                    if len({a, b, c, d}) == 4:
                        exprs.extend([
                            f"exact {a} ({b} ({c} {d}))",
                            f"exact {a} ({b} ({c}.left))",
                            f"exact {a} ({b} ({c}.right))",
                        ])
    if len(nat_vars) >= 1:
        a = nat_vars[0]
        exprs.extend([
            f"exact Nat.le_refl {a}",
            f"exact Nat.lt_succ_self {a}",
            f"exact Nat.zero_le {a}",
            f"exact Nat.dvd_refl {a}",
            f"exact Nat.mul_one {a}",
            f"exact Nat.one_mul {a}",
            f"exact Exists.intro {a} rfl",
            f"exact Exists.intro ({a} + 0) rfl",
            f"exact Exists.intro (0 + {a}) rfl",
            f"exact Exists.intro ({a} + {a}) rfl",
        ])
    if len(nat_vars) >= 2:
        a, b = nat_vars[:2]
        exprs.extend([
            f"exact Nat.add_comm {a} {b}",
            f"exact Nat.mul_comm {a} {b}",
            f"exact Nat.le_add_right {a} {b}",
            f"exact Nat.le_add_left {a} {b}",
        ])
    if len(nat_vars) >= 3:
        a, b, c = nat_vars[:3]
        exprs.extend([
            f"exact Nat.add_assoc {a} {b} {c}",
            f"exact Nat.add_left_comm {a} {b} {c}",
            f"exact Nat.add_right_comm {a} {b} {c}",
            f"exact Nat.mul_assoc {a} {b} {c}",
            f"exact Nat.mul_left_comm {a} {b} {c}",
        ])
    for a in hs:
        for b in hs:
            exprs.extend([
                f"exact Nat.le_trans {a} {b}",
                f"exact Nat.dvd_add {a} {b}",
                f"exact Nat.dvd_trans {a} {b}",
            ])
    for a in hs:
        for b in hs:
            for c in hs:
                if len({a, b, c}) == 3:
                    exprs.extend([
                        f"exact Nat.le_trans {a} (Nat.le_trans {b} {c})",
                        f"exact Nat.dvd_trans {a} (Nat.dvd_trans {b} {c})",
                    ])
    return list(dict.fromkeys(exprs))


def generic_direct_proofs(statement: str, max_intros: int = 6) -> list[list[str]]:
    binders = parse_binders(statement)
    nat_vars = binders.get("Nat", [])
    priority: list[list[str]] = []
    if len(nat_vars) >= 1:
        a = nat_vars[0]
        priority.extend([
            [f"exact Nat.le_refl {a}"],
            [f"exact Nat.lt_succ_self {a}"],
            [f"exact Nat.zero_le {a}"],
            [f"exact Nat.dvd_refl {a}"],
            [f"exact Nat.mul_one {a}"],
            [f"exact Nat.one_mul {a}"],
            [f"exact Exists.intro {a} rfl"],
            [f"refine Exists.intro {a} ?_", "rfl"],
            ["simp"],
        ])
    if len(nat_vars) >= 2:
        a, b = nat_vars[:2]
        priority.extend([
            [f"exact Nat.add_comm {a} {b}"],
            [f"exact Nat.mul_comm {a} {b}"],
            [f"exact Nat.le_add_right {a} {b}"],
            [f"exact Nat.le_add_left {a} {b}"],
            ["rw [Nat.add_comm]"],
            ["rw [Nat.mul_comm]"],
        ])
    if len(nat_vars) >= 3:
        a, b, c = nat_vars[:3]
        priority.extend([
            [f"exact Nat.add_assoc {a} {b} {c}"],
            [f"exact Nat.add_left_comm {a} {b} {c}"],
            [f"exact Nat.add_right_comm {a} {b} {c}"],
            [f"exact Nat.mul_assoc {a} {b} {c}"],
            [f"exact Nat.mul_left_comm {a} {b} {c}"],
            ["rw [Nat.add_assoc]"],
            ["rw [Nat.mul_assoc]"],
        ])
    preferred_n = preferred_intro_count(statement, max_intros)
    intro_counts = [preferred_n] + [n for n in range(max_intros + 1) if n != preferred_n]
    for n in intro_counts:
        prefix = intro_prefix(n)
        if "∨" in statement or " Or " in f" {statement} ":
            priority.extend([
                prefix + ["cases h0 with | inl hp => simp_all | inr hq => simp_all"],
                prefix + ["exact Or.elim h0 h1 h2"],
                prefix + ["exact (Or.elim h0 h1 h2).left"],
                prefix + ["exact (Or.elim h0 h1 h2).right"],
            ])
        if "∧" in statement or " And " in f" {statement} ":
            priority.extend([
                prefix + ["exact h0.left"],
                prefix + ["exact h0.right"],
                prefix + ["exact h1 h0.left"],
                prefix + ["exact h1 h0.right"],
                prefix + ["constructor", "assumption", "assumption"],
                prefix + ["constructor", "exact h0", "exact h1"],
                prefix + ["constructor", "exact h1 h0", "exact h2 (h1 h0)"],
            ])
        if "Not " in statement or "¬" in statement:
            priority.extend([
                prefix + ["contradiction"],
                prefix + ["exact False.elim (h1 h0)"],
                prefix + ["exact False.elim (h0 h1)"],
                prefix + ["exact False.elim (h2 h0.right)"],
                prefix + ["exact False.elim (h1 h0.right)"],
            ])
        priority.extend(prefix + [close] for close in [
            "assumption",
            "contradiction",
            "rfl",
            "simp",
            "trivial",
            "exact h0 h1",
            "exact h1 h0",
            "exact h0 (h1 h2)",
            "exact h1 (h0 h2)",
            "exact h2 (h1 h0)",
            "exact h2 (h0 h1)",
            "exact h0 h1.left",
            "exact h0 h1.right",
            "exact h1 h0.left",
            "exact h1 h0.right",
            "exact False.elim (h0 h1)",
            "exact False.elim (h1 h0)",
            "exact False.elim (h0 h1.left)",
            "exact False.elim (h0 h1.right)",
            "exact False.elim (h1 h0.left)",
            "exact False.elim (h1 h0.right)",
            "exact h0 (h1 (h2 h3))",
            "exact h0 (h2 (h1 h3))",
            "exact h3 (h2 (h1 h0))",
            "exact h0 (h3 (h2 h1))",
            "exact Nat.le_trans h0 h1",
            "exact Nat.dvd_add h0 h1",
            "exact Nat.dvd_trans h0 h1",
            "exact Nat.le_trans h1 (Nat.le_trans h2 h3)",
            "exact Nat.dvd_trans h1 (Nat.dvd_trans h2 h3)",
        ])
        priority.extend([
            prefix + ["constructor", "assumption", "assumption"],
            prefix + ["constructor", "exact h1 h0", "exact h2 (h1 h0)"],
            prefix + ["cases h0 with | inl hp => simp_all | inr hq => simp_all"],
            prefix + ["exact (Or.elim h0 h1 h2).left"],
            prefix + ["exact (Or.elim h0 h1 h2).right"],
        ])
    if len(nat_vars) >= 1:
        a = nat_vars[0]
        priority.extend([
            [f"exact Nat.le_refl {a}"],
            [f"exact Nat.lt_succ_self {a}"],
            [f"exact Nat.zero_le {a}"],
            [f"exact Nat.dvd_refl {a}"],
            [f"exact Nat.mul_one {a}"],
            [f"exact Nat.one_mul {a}"],
            [f"exact Exists.intro {a} rfl"],
            [f"refine Exists.intro {a} ?_", "rfl"],
        ])
    if len(nat_vars) >= 2:
        a, b = nat_vars[:2]
        priority.extend([
            [f"exact Nat.add_comm {a} {b}"],
            [f"exact Nat.mul_comm {a} {b}"],
            [f"exact Nat.le_add_right {a} {b}"],
            [f"exact Nat.le_add_left {a} {b}"],
        ])
    if len(nat_vars) >= 3:
        a, b, c = nat_vars[:3]
        priority.extend([
            [f"exact Nat.add_assoc {a} {b} {c}"],
            [f"exact Nat.add_left_comm {a} {b} {c}"],
            [f"exact Nat.add_right_comm {a} {b} {c}"],
            [f"exact Nat.mul_assoc {a} {b} {c}"],
            [f"exact Nat.mul_left_comm {a} {b} {c}"],
        ])
    proofs: list[list[str]] = []
    for n in intro_counts:
        prefix = intro_prefix(n)
        hs = hnames(n)
        if n >= 1:
            proofs.extend(prefix + [close] for close in [
                "assumption",
                "contradiction",
                "rfl",
                "simp",
                "trivial",
            ])
        if n >= 2:
            proofs.extend(prefix + [close] for close in [
                "exact h0 h1",
                "exact h1 h0",
                "exact False.elim (h0 h1)",
                "exact False.elim (h1 h0)",
                "constructor",
            ])
        if n >= 3:
            proofs.extend(prefix + [close] for close in [
                "exact h0 (h1 h2)",
                "exact h1 (h0 h2)",
                "exact h2 (h1 h0)",
                "exact h2 (h0 h1)",
                "exact h0 h1.left",
                "exact h0 h1.right",
                "exact h1 h0.left",
                "exact h1 h0.right",
                "exact False.elim (h0 h1.left)",
                "exact False.elim (h0 h1.right)",
                "exact False.elim (h1 h0.left)",
                "exact False.elim (h1 h0.right)",
            ])
            proofs.extend([
                prefix + ["constructor", "exact h1 h0", "exact h2 (h1 h0)"],
                prefix + ["cases h0 with | inl hp => simp_all | inr hq => simp_all"],
                prefix + ["exact (Or.elim h0 h1 h2).left"],
                prefix + ["exact (Or.elim h0 h1 h2).right"],
            ])
        if n >= 4:
            proofs.extend(prefix + [close] for close in [
                "exact h0 (h1 (h2 h3))",
                "exact h0 (h2 (h1 h3))",
                "exact h3 (h2 (h1 h0))",
                "exact Nat.le_trans h1 (Nat.le_trans h2 h3)",
                "exact Nat.dvd_trans h1 (Nat.dvd_trans h2 h3)",
            ])
        if n >= 5:
            proofs.extend(prefix + [close] for close in [
                "exact h0 (h3 (h2 h1))",
                "cases h0 with | inl hp => simp_all | inr hq => simp_all",
            ])
        for close in close_exprs(n, nat_vars):
            proofs.append(prefix + [close])
        for a in hs:
            for b in hs:
                if a == b:
                    continue
                proofs.append(prefix + ["constructor", f"exact {a}", f"exact {b}"])
                proofs.append(prefix + ["constructor", f"exact {a} {b}", f"exact {a} {b}"])
        for a in hs:
            proofs.append(prefix + [f"cases {a} with | inl hp => simp_all | inr hq => simp_all"])
        if nat_vars:
            a = nat_vars[0]
            proofs.extend([
                prefix + [f"refine Exists.intro {a} ?_", "rfl"],
                prefix + [f"refine Exists.intro ({a} + 0) ?_", "rfl"],
                prefix + [f"refine Exists.intro (0 + {a}) ?_", "rfl"],
            ])
    return unique_proofs(priority + proofs)


def direct_by_family(spec: dict[str, Any]) -> tuple[str, list[str]] | None:
    candidates = [
        ("v28_direct_v16_deep_chain_conj", LEGACY.direct_v16_deep_chain_conj_proof),
        ("v29_direct_v17_contra_with_distractor", LEGACY.direct_v17_contra_with_distractor_proof),
        ("v210_direct_have_construct", LEGACY.direct_have_construct_proof),
        ("v211_direct_v17_deep_reordered_bridge", LEGACY.direct_v17_deep_reordered_bridge_proof),
        ("v212_direct_v16_project_contradiction", LEGACY.direct_v16_project_contradiction_proof),
        ("v213_direct_have_imp_chain", LEGACY.direct_have_imp_chain_proof),
        ("v214_direct_v17_reordered_chain", LEGACY.direct_v17_reordered_chain_proof),
        ("v214_direct_v17_distractor_projection", LEGACY.direct_v17_distractor_projection_proof),
        ("v214_direct_v16_or_to_local_conj", LEGACY.direct_v16_or_to_local_conj_proof),
        ("v214_direct_v24_nat_divisibility_nested_trans", LEGACY.direct_v24_nat_divisibility_nested_trans_proof),
        ("v214_direct_v24_nat_inequality_nested_trans", LEGACY.direct_v24_nat_inequality_nested_trans_proof),
    ]
    for strategy, fn in candidates:
        proof = fn(spec)
        if proof is not None:
            return strategy, proof
    return None


def final_check(spec: dict[str, Any], proof: list[str], out_dir: Path, lean_bin: str, timeout_s: int, suffix: str) -> dict[str, Any]:
    return LEGACY.final_check(spec, proof, out_dir, "user_direct", lean_bin, timeout_s, suffix)


def sanitize_check(res: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in res.items() if k not in {"stdout", "stderr"}}


def load_ranker(checkpoint: str, cpu: bool) -> tuple[Any, Any, dict[str, int], torch.device]:
    helper = load_module(Path("tools/train_lean_branch_cost_ranker_v0_9.py"), "lean_branch_cost_ranker_v09")
    ckpt = torch.load(checkpoint, map_location="cpu")
    vocab = ckpt["vocab"]
    hidden = int(ckpt.get("report", {}).get("model", {}).get("hidden_dim", 256))
    device = torch.device("cuda" if torch.cuda.is_available() and not cpu else "cpu")
    model = helper.BranchCostRanker(len(vocab), hidden).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return helper, model, vocab, device


def solve_user_spec(
    spec: dict[str, Any],
    out_dir: Path,
    lean_bin: str,
    timeout_s: int,
    max_direct: int,
    checkpoint: str,
    cpu: bool,
    max_k: int,
    max_depth: int,
    max_nodes: int,
    state_timeout_s: int,
    state_capture_retries: int,
    fallback: bool,
) -> dict[str, Any]:
    start = time.time()
    out_dir.mkdir(parents=True, exist_ok=True)
    lean_calls = 0
    first_error: dict[str, Any] | None = None

    family_direct = direct_by_family(spec)
    direct_candidates: list[tuple[str, list[str]]] = []
    if family_direct is not None:
        direct_candidates.append(family_direct)
    direct_candidates.extend(("v217_generic_direct", p) for p in generic_direct_proofs(spec["statement"]))

    for idx, (strategy, proof) in enumerate(direct_candidates[:max_direct], start=1):
        check = final_check(spec, proof, out_dir, lean_bin, timeout_s, f"direct_{idx:04d}")
        lean_calls += 1
        if check["accepted"]:
            return {
                "theorem": spec["name"],
                "family": spec.get("family", "user_supplied"),
                "solved": True,
                "verified": True,
                "proof_lines": proof,
                "proof_depth": len(proof),
                "lean_calls": lean_calls,
                "nodes_expanded": 0,
                "strategy": strategy,
                "trace": [{"strategy": strategy, "accepted": True}],
                "final_check": sanitize_check(check),
                "elapsed_s": round(time.time() - start, 3),
                "failure_reason": None,
                "clinical_use": "forbidden",
            }
        if first_error is None:
            first_error = sanitize_check(check)

    if fallback:
        helper, model, vocab, device = load_ranker(checkpoint, cpu)
        row = LEGACY.solve_spec(helper, model, vocab, device, spec, max_k, out_dir, lean_bin, timeout_s, state_timeout_s, state_capture_retries, max_depth, max_nodes)
        row["verified"] = bool(row.get("solved") and row.get("final_check", {}).get("accepted"))
        row["lean_calls"] = int(row.get("lean_calls", 0)) + lean_calls
        row["strategy"] = "learned_search_fallback" if row.get("solved") else "unsolved"
        row["elapsed_s"] = round(time.time() - start, 3)
        row["failure_reason"] = None if row.get("solved") else "no_verified_proof_found"
        return row

    return {
        "theorem": spec["name"],
        "family": spec.get("family", "user_supplied"),
        "solved": False,
        "verified": False,
        "proof_lines": [],
        "proof_depth": 0,
        "lean_calls": lean_calls,
        "nodes_expanded": 0,
        "strategy": "unsolved",
        "trace": [],
        "final_check": first_error,
        "elapsed_s": round(time.time() - start, 3),
        "failure_reason": "no_verified_proof_found",
        "clinical_use": "forbidden",
    }


def summarize_records(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def summarize_group(vals: list[dict[str, Any]]) -> dict[str, Any]:
        n = len(vals)
        solved = sum(1 for r in vals if r.get("solved"))
        return {
            "count": n,
            "solved_count": solved,
            "solve_rate": solved / n if n else 0.0,
            "mean_elapsed_s": sum(float(r.get("elapsed_s", 0)) for r in vals) / n if n else 0.0,
            "mean_lean_calls": sum(float(r.get("lean_calls", 0)) for r in vals) / n if n else 0.0,
            "mean_proof_depth": sum(float(r.get("proof_depth", 0)) for r in vals) / n if n else 0.0,
        }

    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_reason: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_family[str(row.get("family", "unknown"))].append(row)
        by_reason[str(row.get("failure_reason") or "solved")].append(row)
    return {
        "overall": summarize_group(rows),
        "by_family": {k: summarize_group(v) for k, v in sorted(by_family.items())},
        "by_failure_reason": {k: summarize_group(v) for k, v in sorted(by_reason.items())},
    }


def add_common_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--lean-bin", default="/home/benjamin/.elan/bin/lean")
    ap.add_argument("--checkpoint", default="checkpoints/lean_branch_cost_ranker_v0_9_repeats27_h256_e50.pt")
    ap.add_argument("--timeout-s", type=int, default=20)
    ap.add_argument("--state-timeout-s", type=int, default=20)
    ap.add_argument("--state-capture-retries", type=int, default=1)
    ap.add_argument("--max-k", type=int, default=8)
    ap.add_argument("--max-depth", type=int, default=12)
    ap.add_argument("--max-nodes", type=int, default=160)
    ap.add_argument("--max-direct", type=int, default=2500)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--no-fallback", action="store_true")

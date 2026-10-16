#!/usr/bin/env python3
"""
Experiment runner for the AHR-CSP 512 gate roadmap.

This file intentionally reuses the notebook implementation in
`AHR-CSP-v0-completed.ipynb`, but disables all notebook auto-run cells while
loading definitions. The runner adds the persisted experiment workflow that was
missing from the notebook source:

- Gate-C report verification.
- Original 512 dataset family splitting.
- Repaired factor-NLL / expected-violation loss patch.
- Family gate training.
- Uniqueness-controlled mixed dataset construction.
- Staged curriculum training.
- Phase-6 post-pass evaluation/export.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import itertools
import json
import math
import os
import random
import re
import sys
import time
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


EXPERIMENT_DIR = Path(__file__).resolve().parent
ARTIFACT_ROOT = EXPERIMENT_DIR.parent
PROJECT_ROOT = ARTIFACT_ROOT.parent
NOTEBOOK_PATH = PROJECT_ROOT / "AHR-CSP-v0-completed.ipynb"
DATA_DIR = ARTIFACT_ROOT / "data"
REPORT_DIR = ARTIFACT_ROOT / "reports"
CHECKPOINT_DIR = ARTIFACT_ROOT / "checkpoints"

# The notebook enables deterministic algorithms. Set this before torch is
# imported by the notebook loader to avoid repeated CuBLAS warnings.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

ORIGINAL_512_DATASET = DATA_DIR / "phase5p5_overfit_512_128_128.jsonl"
GATE_C_REPORT = REPORT_DIR / "phase5p6_gate_C_unique_64_a100_report.json"
GATE_C_DATASET = DATA_DIR / "phase5p6_unique_64.jsonl"

PASS_TARGETS = {
    "train_cpu_valid_rate": 0.95,
    "val_cpu_valid_rate": 0.90,
    "val_mean_hard_energy": 0.10,
}

FAMILY_GENERATORS = {
    "boolean_sat": "planted_boolean_ksat",
    "graph_coloring": "planted_graph_coloring",
    "alldifferent": "planted_alldifferent_permutation",
    "subset_sum": "planted_subset_sum",
}


def read_json(path: Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def summarize_records(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    from collections import Counter

    def counter(key: str) -> Dict[str, int]:
        return dict(Counter(str(r.get(key)) for r in rows))

    nums: Dict[str, Dict[str, float]] = {}
    for key in ("domain_product", "num_variables", "num_constraints"):
        vals = [float(r[key]) for r in rows if key in r and r[key] is not None]
        if vals:
            vals_sorted = sorted(vals)
            nums[key] = {
                "min": vals_sorted[0],
                "median": vals_sorted[len(vals_sorted) // 2],
                "mean": float(sum(vals_sorted) / len(vals_sorted)),
                "max": vals_sorted[-1],
            }

    return {
        "num_records": len(rows),
        "split_counts": counter("split"),
        "problem_type_counts": counter("problem_type"),
        "generator_counts": counter("generator"),
        "oracle_status_counts": counter("oracle_status"),
        "unique_solution_counts": counter("unique_solution"),
        **nums,
    }


def _disable_autorun_source(src: str, cell_index: int) -> str:
    """Disable notebook run flags and remove expensive unconditional calls."""
    replacements = {
        "RUN_PHASE5_SMOKE_TRAINING",
        "RUN_PHASE6_BUDGET_SWEEP_SMOKE",
        "RUN_PHASE6_ABLATION_SMOKE",
        "RUN_PHASE6_BASELINE_FAILURE_SMOKE",
        "RUN_PHASE6_SOLVE_WRAPPER_SMOKE",
        "RUN_PHASE6_EXPORT_SMOKE",
        "RUN_PHASE5P5_OVERFIT_TRAINING",
        "RUN_PHASE5P5_POST_OVERFIT_EVAL",
    }
    for name in replacements:
        src = re.sub(rf"^{name}\s*=\s*True\b", f"{name} = False", src, flags=re.M)

    if cell_index == 31:
        marker = "overfit_dataset_path, overfit_dataset_report = build_exact_overfit_dataset()"
        pos = src.find(marker)
        if pos >= 0:
            src = src[:pos] + "\nprint('Cell 30 definitions loaded; dataset auto-build skipped by runner.')\n"

    smoke_cut_markers = {
        12: "# Small smoke dataset.",
        13: "# DIMACS import smoke test.",
        14: "# Cell 13 smoke test.",
        15: "# Cell 14 smoke test.",
        21: "# Phase 4 smoke test.",
    }
    marker = smoke_cut_markers.get(cell_index)
    if marker is not None:
        pos = src.find(marker)
        if pos >= 0:
            src = src[:pos] + f"\nprint('Cell {cell_index} definitions loaded; smoke section skipped by runner.')\n"

    return src


def load_notebook_runtime() -> Dict[str, Any]:
    """
    Execute notebook definition cells into a namespace with auto-runs disabled.

    Cells 1-29 define schema/data/model/eval. Cell 30 is only a summary writer
    and is skipped. Cells 31-33 define overfit dataset/training/eval helpers.
    """
    try:
        import nbformat
    except ImportError as exc:
        raise RuntimeError("Use /opt/conda/bin/python; nbformat is not installed in /usr/bin/python3.") from exc

    module_name = "ahr512_notebook_runtime"
    ns: Dict[str, Any] = {
        "__name__": module_name,
        "__file__": str(NOTEBOOK_PATH),
    }
    runtime_module = ModuleType(module_name)
    runtime_module.__dict__.update(ns)
    sys.modules[module_name] = runtime_module
    ns = runtime_module.__dict__
    nb = nbformat.read(str(NOTEBOOK_PATH), as_version=4)
    wanted_cells = list(range(1, 30)) + [31, 32, 33]

    old_cwd = Path.cwd()
    quiet_load = os.environ.get("AHR512_VERBOSE_LOAD", "0") not in {"1", "true", "TRUE", "yes"}
    os.chdir(PROJECT_ROOT)
    try:
        with contextlib.ExitStack() as stack:
            if quiet_load:
                devnull = stack.enter_context(open(os.devnull, "w"))
                stack.enter_context(contextlib.redirect_stdout(devnull))
            for cell_index in wanted_cells:
                src = "".join(nb.cells[cell_index - 1].get("source", ""))
                src = _disable_autorun_source(src, cell_index)
                code = compile(src, f"{NOTEBOOK_PATH}:cell-{cell_index}", "exec")
                exec(code, ns)
    finally:
        os.chdir(old_cwd)

    return ns


def install_repaired_loss(ns: Dict[str, Any]) -> Any:
    """
    Install the repaired loss used by the A100 micro-gate reports.

    The notebook source already has true factor-table Worker support. The source
    does not persist the later factor-NLL loss, so we add it here explicitly.
    """
    torch = ns["torch"]
    F = ns["F"]

    @dataclasses.dataclass(frozen=True)
    class RepairedAHRLossConfig:
        assignment_weight: float = 0.5
        query_assignment_weight: float = 0.1
        fixed_weight: float = 1.0
        constraint_weight: float = 1.0
        expected_violation_weight: float = 1.0
        constraint_nll_weight: float = 2.0
        graph_conflict_weight: float = 0.0
        unsat_margin_weight: float = 0.0
        unsat_margin: float = 0.25
        entropy_weight: float = 0.0
        logit_l2_weight: float = 1.0e-7
        halt_weight: float = 0.0
        solve_energy_threshold: float = 0.5
        ignore_unsat_for_assignment: bool = True

    def compute_constraint_losses_repaired(output, batch, loss_config):
        expected_violation = output.expected_violation
        expected_energy = output.expected_energy
        sat_label = batch.tensors["sat_label"]
        constraint_mask = batch.tensors["constraint_mask"]

        sat_mask = sat_label == 1
        unsat_mask = sat_label == 0
        sat_constraint_mask = constraint_mask & sat_mask.unsqueeze(1)

        expected_satisfied = (1.0 - expected_violation).clamp(1.0e-8, 1.0)
        expected_violation_loss = ns["_safe_masked_mean"](
            expected_violation,
            sat_constraint_mask,
        )
        constraint_nll_loss = ns["_safe_masked_mean"](
            -torch.log(expected_satisfied),
            sat_constraint_mask,
        )

        constraint_count = constraint_mask.float().sum(dim=1).clamp_min(1.0)
        normalized_expected_energy = expected_energy / constraint_count

        if bool(unsat_mask.any().detach().cpu().item()):
            unsat_margin_values = F.relu(float(loss_config.unsat_margin) - normalized_expected_energy)
            unsat_margin_loss = ns["_safe_masked_mean"](unsat_margin_values, unsat_mask)
        else:
            unsat_margin_loss = ns["_zero_like_loss"](expected_energy)

        graph_conflict_terms = []
        probs = getattr(output, "probs", None)
        if probs is not None and hasattr(batch, "raw_problem_jsons"):
            for b, raw_problem in enumerate(batch.raw_problem_jsons):
                if str(raw_problem.get("problem_type")) != "graph_coloring":
                    continue
                name_to_index = {str(name): idx for idx, name in enumerate(batch.variable_names[b])}
                value_to_index = {
                    str(var_name): {value: idx for idx, value in enumerate(values)}
                    for var_name, values in zip(batch.variable_names[b], batch.candidate_values[b])
                }
                for constraint in raw_problem.get("constraints", []):
                    if constraint.get("op") != "notEquals":
                        continue
                    scope = constraint.get("scope") or constraint.get("args") or []
                    if len(scope) != 2:
                        continue
                    left, right = str(scope[0]), str(scope[1])
                    if left not in name_to_index or right not in name_to_index:
                        continue
                    left_values = value_to_index.get(left, {})
                    right_values = value_to_index.get(right, {})
                    common_values = [value for value in left_values if value in right_values]
                    if not common_values:
                        continue
                    left_indices = torch.tensor(
                        [left_values[value] for value in common_values],
                        device=probs.device,
                        dtype=torch.long,
                    )
                    right_indices = torch.tensor(
                        [right_values[value] for value in common_values],
                        device=probs.device,
                        dtype=torch.long,
                    )
                    left_probs = probs[b, name_to_index[left], :].gather(0, left_indices)
                    right_probs = probs[b, name_to_index[right], :].gather(0, right_indices)
                    graph_conflict_terms.append((left_probs * right_probs).sum())

        if graph_conflict_terms:
            graph_conflict_loss = torch.stack(graph_conflict_terms).mean()
        else:
            graph_conflict_loss = ns["_zero_like_loss"](expected_energy)

        # Keep the legacy key for old metric consumers.
        return {
            "constraint_loss": expected_violation_loss,
            "expected_violation_loss": expected_violation_loss,
            "constraint_nll_loss": constraint_nll_loss,
            "graph_conflict_loss": graph_conflict_loss,
            "unsat_margin_loss": unsat_margin_loss,
        }

    def compute_ahr_loss_repaired(output, batch, loss_config):
        assignment_losses = ns["compute_assignment_losses"](output, batch, loss_config)
        constraint_losses = compute_constraint_losses_repaired(output, batch, loss_config)
        regularization_losses = ns["compute_regularization_losses"](output, batch, loss_config)
        components = {
            **assignment_losses,
            **constraint_losses,
            **regularization_losses,
        }
        total = (
            float(loss_config.assignment_weight) * components["assignment_loss"]
            + float(loss_config.query_assignment_weight) * components["query_assignment_loss"]
            + float(loss_config.fixed_weight) * components["fixed_loss"]
            + float(loss_config.constraint_weight) * components["constraint_loss"]
            + float(loss_config.expected_violation_weight) * components["expected_violation_loss"]
            + float(loss_config.constraint_nll_weight) * components["constraint_nll_loss"]
            + float(getattr(loss_config, "graph_conflict_weight", 0.0)) * components["graph_conflict_loss"]
            + float(loss_config.unsat_margin_weight) * components["unsat_margin_loss"]
            + float(loss_config.entropy_weight) * components["entropy_loss"]
            + float(loss_config.logit_l2_weight) * components["logit_l2_loss"]
            + float(loss_config.halt_weight) * components["halt_loss"]
        )
        metrics = ns["compute_ahr_metrics"](output, batch)
        return ns["AHRLossResult"](total=total, components=components, metrics=metrics)

    ns["RepairedAHRLossConfig"] = RepairedAHRLossConfig
    ns["compute_constraint_losses"] = compute_constraint_losses_repaired
    ns["compute_ahr_loss"] = compute_ahr_loss_repaired
    return RepairedAHRLossConfig


def install_vectorized_factor_messages(ns: Dict[str, Any]) -> None:
    """
    Replace the notebook's per-tuple Python loop with a vectorized table pass.

    The notebook implementation loops over every allowed tuple and every slot
    in Python. This version still loops over batch items and constraints, but
    processes all allowed tuples for one factor with one gather/prod/scatter
    sequence on the selected device.
    """

    torch = ns["torch"]

    def compute_factor_messages_from_tables_vectorized(batch, probs, eps: float = 1.0e-8):
        value_mask = batch.tensors["value_mask"]
        constraint_mask = batch.tensors["constraint_mask"]
        constraint_scope = batch.tensors["constraint_scope"]
        constraint_scope_mask = batch.tensors["constraint_scope_mask"]
        factor_allowed_values = batch.tensors["factor_allowed_values"]
        factor_allowed_mask = batch.tensors["factor_allowed_mask"]
        factor_table_exact = batch.tensors["factor_table_exact"]

        B, N, D = probs.shape
        C = constraint_mask.shape[1]

        support = torch.zeros_like(probs)
        expected_satisfied = torch.zeros(B, C, device=probs.device, dtype=probs.dtype)

        for b in range(B):
            for c in range(C):
                if not bool(constraint_mask[b, c].item()):
                    continue
                if not bool(factor_table_exact[b, c].item()):
                    continue

                arity = int(constraint_scope_mask[b, c].sum().item())
                valid_rows = factor_allowed_mask[b, c]

                if arity == 0:
                    expected_satisfied[b, c] = 1.0 if bool(valid_rows.any().item()) else 0.0
                    continue

                if not bool(valid_rows.any().item()):
                    expected_satisfied[b, c] = 0.0
                    continue

                scope_vars = constraint_scope[b, c, :arity].long()
                allowed = factor_allowed_values[b, c, valid_rows, :arity].long()

                if not bool(((scope_vars >= 0) & (scope_vars < N)).all().item()):
                    expected_satisfied[b, c] = 0.0
                    continue

                valid_tuple_mask = torch.ones(allowed.shape[0], device=allowed.device, dtype=torch.bool)
                valid_tuple_mask = valid_tuple_mask & (allowed >= 0).all(dim=1) & (allowed < D).all(dim=1)
                if not bool(valid_tuple_mask.any().item()):
                    expected_satisfied[b, c] = 0.0
                    continue

                allowed = allowed[valid_tuple_mask]
                T = allowed.shape[0]

                var_probs = probs[b, scope_vars, :]  # [arity, D]
                expanded_probs = var_probs.unsqueeze(0).expand(T, arity, D)
                selected_probs = expanded_probs.gather(2, allowed.unsqueeze(-1)).squeeze(-1)
                tuple_mass = selected_probs.prod(dim=1)

                expected_satisfied[b, c] = tuple_mass.sum()

                for slot in range(arity):
                    v_idx = int(scope_vars[slot].item())
                    if v_idx < 0 or v_idx >= N:
                        continue
                    support[b, v_idx].scatter_add_(0, allowed[:, slot], tuple_mass)

        support = support.masked_fill(~value_mask, 0.0)
        expected_satisfied = expected_satisfied.clamp(0.0, 1.0)
        expected_violation = (1.0 - expected_satisfied).masked_fill(~constraint_mask, 0.0)
        support_z = ns["standardize_support"](support=support, value_mask=value_mask, eps=eps)
        mean_expected_violation = (
            expected_violation.sum(dim=1)
            / constraint_mask.float().sum(dim=1).clamp_min(1.0)
        )

        diagnostics = {
            "mean_expected_satisfied": float(expected_satisfied.detach().mean().cpu())
            if expected_satisfied.numel()
            else 1.0,
            "mean_expected_violation": float(mean_expected_violation.detach().mean().cpu()),
            "mean_support": float(support.detach().mean().cpu()) if support.numel() else 0.0,
            "implementation": "vectorized_factor_tables_v1",
        }

        return ns["WorkerStats"](
            support=support,
            support_z=support_z,
            expected_satisfied=expected_satisfied,
            expected_violation=expected_violation,
            mean_expected_violation=mean_expected_violation,
            diagnostics=diagnostics,
        )

    ns["compute_factor_messages_from_tables"] = compute_factor_messages_from_tables_vectorized


def make_debug_arch(ns: Dict[str, Any]) -> Any:
    return dataclasses.replace(
        ns["AHR_ARCH_CFG"],
        hidden_size=256,
        dropout=0.0,
        evaporation_rho=0.35,
        fixed_anchor_strength=0.2,
        worker_support_gain=0.5,
        scout_noise_scale=0.0,
        noise_during_eval=False,
    )


def make_repaired_train_config(
    ns: Dict[str, Any],
    name: str,
    epochs: int,
    batch_size: int,
    train_limit: Optional[int],
    val_limit: Optional[int],
    cpu_verify_items: Optional[int] = None,
    full_only: bool = False,
    graph_conflict_weight: float = 0.0,
) -> Any:
    RepairedLoss = ns["RepairedAHRLossConfig"]
    if full_only:
        deep_specs = (
            ns["DeepSupervisionSpec"]("full", 2, 2, 4, 1.0),
        )
    else:
        deep_specs = (
            ns["DeepSupervisionSpec"]("short", 1, 1, 1, 0.35),
            ns["DeepSupervisionSpec"]("full", 2, 2, 4, 1.0),
        )
    return dataclasses.replace(
        ns["DEFAULT_AHR_TRAINING_CONFIG"],
        batch_size=batch_size,
        epochs=epochs,
        max_train_batches=None,
        max_val_batches=None,
        lr=5.0e-4,
        weight_decay=1.0e-5,
        grad_clip_norm=1.0,
        train_noise_scale=0.0,
        eval_noise_scale=0.0,
        train_sat_only=True,
        val_sat_only=True,
        train_record_limit=train_limit,
        val_record_limit=val_limit,
        log_every=1,
        cpu_verify_items=batch_size if cpu_verify_items is None else int(cpu_verify_items),
        loss_config=RepairedLoss(graph_conflict_weight=float(graph_conflict_weight)),
        checkpoint_name=name,
        best_metric_name="cpu_valid_rate",
        greater_is_better=True,
        deep_supervision_specs=deep_specs,
    )


def verify_gate_c_report() -> Dict[str, Any]:
    report = read_json(GATE_C_REPORT)
    train = report.get("final", {}).get("train_eval", {})
    val = report.get("final", {}).get("val_eval", {})
    result = {
        "source": str(GATE_C_REPORT),
        "passed_in_report": bool(report.get("passed")),
        "train_cpu_valid_rate": float(train.get("cpu_valid_rate", 0.0)),
        "val_cpu_valid_rate": float(val.get("cpu_valid_rate", 0.0)),
        "val_mean_hard_energy": float(val.get("mean_hard_energy", 999.0)),
    }
    result["passes_targets"] = (
        result["train_cpu_valid_rate"] >= PASS_TARGETS["train_cpu_valid_rate"]
        and result["val_cpu_valid_rate"] >= PASS_TARGETS["val_cpu_valid_rate"]
        and result["val_mean_hard_energy"] <= PASS_TARGETS["val_mean_hard_energy"]
    )
    out = REPORT_DIR / "phase5p7_gate_c_verification_report.json"
    write_json(out, result)
    return result


def split_original_512_by_family() -> Dict[str, Any]:
    rows = read_jsonl(ORIGINAL_512_DATASET)
    family_dir = DATA_DIR / "phase5p7_family_gates"
    reports: Dict[str, Any] = {
        "source": str(ORIGINAL_512_DATASET),
        "source_summary": summarize_records(rows),
        "families": {},
    }

    families = sorted({str(r.get("problem_type")) for r in rows})
    for family in families:
        family_rows = [dict(r, family_gate_source=str(ORIGINAL_512_DATASET)) for r in rows if r.get("problem_type") == family]
        path = family_dir / f"phase5p7_512_family_{family}.jsonl"
        write_jsonl(path, family_rows)
        reports["families"][family] = {
            "dataset_path": str(path),
            "summary": summarize_records(family_rows),
        }

    out = REPORT_DIR / "phase5p7_family_split_report.json"
    write_json(out, reports)
    return reports


def _split_plan(train: int, val: int, test: int) -> List[str]:
    return ["train"] * train + ["val"] * val + ["test"] * test


def _balanced_family_for_index(index: int, families: Sequence[str]) -> str:
    return families[index % len(families)]


def _difficulty_for_family(family: str, rng: random.Random, domain_cap: int) -> Dict[str, Any]:
    # Keep close to the successful micro-gate range first.
    if family == "boolean_sat":
        n_vars = rng.randint(3, min(10, max(3, int(math.log2(max(domain_cap, 8))))))
        return {"n_vars": n_vars, "n_clauses": rng.randint(2 * n_vars, 6 * n_vars), "k": rng.choice([2, 3])}
    if family == "graph_coloring":
        n_nodes = rng.randint(4, 8)
        return {"n_nodes": n_nodes, "n_colors": rng.choice([3, 4]), "edge_density": rng.uniform(0.25, 0.65)}
    if family == "alldifferent":
        n = rng.randint(3, 6)
        return {"n": n, "n_givens": rng.randint(max(0, n - 3), max(0, n - 1))}
    if family == "subset_sum":
        n_items = rng.randint(3, 6)
        return {"n_items": n_items, "max_weight": rng.randint(8, 37), "force_nonempty": True}
    raise ValueError(f"Unknown family: {family}")


def build_unique_controlled_dataset(
    ns: Dict[str, Any],
    name: str,
    train: int,
    val: int,
    test: int,
    seed: int,
    domain_cap: int,
    require_unique: str,
    max_attempts_multiplier: int,
    families: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    dataset_path = DATA_DIR / f"{name}.jsonl"
    report_path = REPORT_DIR / f"{name}_report.json"
    if dataset_path.exists() and report_path.exists():
        print(f"[build-unique] reusing existing {dataset_path}", flush=True)
        return read_json(report_path)

    rng = random.Random(seed)
    families = tuple(families or FAMILY_GENERATORS.keys())
    for family in families:
        if family not in FAMILY_GENERATORS:
            raise ValueError(f"Unknown family: {family}")
    total = train + val + test
    split_values = _split_plan(train, val, test)
    accepted: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []
    seen_ids = set()
    attempts = 0
    max_attempts = max_attempts_multiplier * total

    unique_infeasible_families = {"graph_coloring"}
    policy = ns["CertifyPolicy"](
        prefer_planted=False,
        require_verified_sat=True,
        require_complete_unsat_proof=True,
        max_bruteforce_assignments=max(250_000, domain_cap),
        z3_timeout_ms=5_000,
    )

    start = time.time()
    print(
        f"[build-unique] name={name} target={total} families={list(families)} "
        f"domain_cap={domain_cap} require_unique={require_unique} max_attempts={max_attempts}",
        flush=True,
    )
    while len(accepted) < total and attempts < max_attempts:
        attempts += 1
        if attempts == 1 or attempts % 100 == 0:
            print(
                f"[build-unique] attempts={attempts} accepted={len(accepted)} rejected={len(rejected)}",
                flush=True,
            )
        family = _balanced_family_for_index(len(accepted), families)
        gen_name = FAMILY_GENERATORS[family]
        local_seed = rng.randrange(0, 2**31 - 1)
        local_rng = random.Random(local_seed)
        difficulty = _difficulty_for_family(family, local_rng, domain_cap)
        ctx = ns["GeneratorContext"](seed=local_seed, rng=local_rng, difficulty=difficulty, tag=name)

        try:
            candidate = ns["GENERATOR_REGISTRY"][gen_name].fn(ctx)
            certified = ns["certify_candidate"](candidate, policy=policy)
            if not certified.accepted or certified.problem is None:
                rejected.append({"family": family, "reason": certified.rejection_reason})
                continue
            problem = certified.problem
            pid = ns["problem_id"](problem)
            if pid in seen_ids:
                rejected.append({"family": family, "reason": "duplicate_problem_id"})
                continue
            seen_ids.add(pid)
            domain_product = ns["domain_product_size"](problem)
            if domain_product > domain_cap:
                rejected.append({"family": family, "reason": "domain_cap", "domain_product": domain_product})
                continue
            solution_count = getattr(problem.certificate, "solution_count", None)
            is_unique = solution_count == 1
            if require_unique == "all" and not is_unique:
                rejected.append({"family": family, "reason": "not_unique", "solution_count": solution_count})
                continue
            if require_unique == "feasible" and family not in unique_infeasible_families and not is_unique:
                rejected.append({"family": family, "reason": "not_unique_feasible_family", "solution_count": solution_count})
                continue

            split = split_values[len(accepted)]
            record = ns["certified_problem_to_record"](certified, stage_name=name)
            record.update(
                {
                    "split": split,
                    "stage": name,
                    "micro_dataset": name,
                    "unique_solution": is_unique,
                    "solution_count": solution_count,
                    "domain_product": domain_product,
                    "uniqueness_policy": require_unique,
                }
            )
            accepted.append(record)
            if len(accepted) == total or len(accepted) % 10 == 0:
                print(
                    f"[build-unique] accepted={len(accepted)}/{total} attempts={attempts} "
                    f"last_family={family} domain_product={domain_product} unique={is_unique}",
                    flush=True,
                )
        except Exception as exc:
            rejected.append({"family": family, "reason": repr(exc)})

    if len(accepted) < total:
        raise RuntimeError(f"Only built {len(accepted)} / {total} records after {attempts} attempts.")

    write_jsonl(dataset_path, accepted)
    report = {
        "dataset_path": str(dataset_path),
        "name": name,
        "elapsed_s": time.time() - start,
        "attempts": attempts,
        "accepted": len(accepted),
        "rejected": len(rejected),
        "config": {
            "train": train,
            "val": val,
            "test": test,
            "seed": seed,
            "domain_cap": domain_cap,
            "require_unique": require_unique,
            "families": list(families),
        },
        "summary": summarize_records(accepted),
        "rejection_sample": rejected[:50],
    }
    write_json(report_path, report)
    print(f"[build-unique] wrote {dataset_path} in {report['elapsed_s']:.1f}s", flush=True)
    return report


def _load_best_checkpoint_into_model(ns: Dict[str, Any], checkpoint_name: str, arch_config: Any, device: Any) -> Any:
    model = ns["AHRReasoner"](tensor_config=ns["TENSOR_CFG"], arch_config=arch_config).to(device)
    path = CHECKPOINT_DIR / f"{checkpoint_name}_best.pt"
    if path.exists():
        ns["load_ahr_checkpoint"](path, model=model, optimizer=None, map_location=device)
    else:
        raise FileNotFoundError(path)
    return model


def evaluate_splits(ns: Dict[str, Any], model: Any, dataset_path: Path, train_config: Any, batch_size: int, device: Any) -> Dict[str, Any]:
    split_reports: Dict[str, Any] = {}
    for split in ("train", "val", "test"):
        loader, data_report = ns["make_phase6_loader"](
            dataset_source=dataset_path,
            split=split,
            sat_only=True,
            batch_size=batch_size,
            limit=None,
        )
        eval_config = dataclasses.replace(
            train_config,
            max_val_batches=None,
            cpu_verify_items=batch_size,
        )
        metrics = ns["evaluate_ahr_model"](model=model, val_loader=loader, device=device, train_config=eval_config)
        split_reports[split] = {"data_report": data_report, "metrics": metrics}
    return split_reports


def gate_passed(split_reports: Dict[str, Any]) -> bool:
    train = split_reports.get("train", {}).get("metrics", {})
    val = split_reports.get("val", {}).get("metrics", {})
    return (
        float(train.get("cpu_valid_rate", 0.0)) >= PASS_TARGETS["train_cpu_valid_rate"]
        and float(val.get("cpu_valid_rate", 0.0)) >= PASS_TARGETS["val_cpu_valid_rate"]
        and float(val.get("mean_hard_energy", 999.0)) <= PASS_TARGETS["val_mean_hard_energy"]
    )


def train_gate(ns: Dict[str, Any], dataset_path: Path, name: str, epochs: int, batch_size: int) -> Dict[str, Any]:
    RepairedLoss = install_repaired_loss(ns)
    install_vectorized_factor_messages(ns)
    torch = ns["torch"]
    device_name = ns["get_device"]()
    device = torch.device("cpu" if device_name == "no-torch" else device_name)
    arch = make_debug_arch(ns)
    model = ns["AHRReasoner"](tensor_config=ns["TENSOR_CFG"], arch_config=arch).to(device)
    train_config = make_repaired_train_config(
        ns=ns,
        name=name,
        epochs=epochs,
        batch_size=batch_size,
        train_limit=None,
        val_limit=None,
    )

    model, history = ns["train_ahr_model"](
        model=model,
        dataset_source=dataset_path,
        train_config=train_config,
        device=device,
    )
    best_model = _load_best_checkpoint_into_model(ns, name, arch, device)
    split_reports = evaluate_splits(ns, best_model, dataset_path, train_config, batch_size, device)
    report = {
        "kind": "ahr_csp_512_gate_training",
        "name": name,
        "dataset_path": str(dataset_path),
        "created_at_unix": time.time(),
        "arch_config": ns["to_jsonable"](arch),
        "loss_config": dataclasses.asdict(RepairedLoss()),
        "train_config": ns["to_jsonable"](train_config),
        "history_tail": history[-5:],
        "split_reports": split_reports,
        "passed": gate_passed(split_reports),
        "targets": PASS_TARGETS,
    }
    write_json(REPORT_DIR / f"{name}_report.json", report)
    return report


def _maybe_load_checkpoint(
    ns: Dict[str, Any],
    model: Any,
    optimizer: Optional[Any],
    checkpoint_names: Sequence[str],
    device: Any,
) -> Optional[Dict[str, Any]]:
    for checkpoint_name in checkpoint_names:
        path = CHECKPOINT_DIR / f"{checkpoint_name}_best.pt"
        if path.exists():
            meta = ns["load_ahr_checkpoint"](
                path,
                model=model,
                optimizer=optimizer,
                map_location=device,
            )
            meta["path"] = str(path)
            return meta
    return None


def train_gate_fast(
    ns: Dict[str, Any],
    dataset_path: Path,
    name: str,
    epochs: int,
    batch_size: int,
    eval_every: int,
    patience: int,
    min_epochs: int,
    resume_names: Optional[Sequence[str]] = None,
    full_only: bool = False,
    graph_conflict_weight: float = 0.0,
) -> Dict[str, Any]:
    RepairedLoss = install_repaired_loss(ns)
    install_vectorized_factor_messages(ns)
    torch = ns["torch"]
    device_name = ns["get_device"]()
    device = torch.device("cpu" if device_name == "no-torch" else device_name)
    arch = make_debug_arch(ns)

    model = ns["AHRReasoner"](tensor_config=ns["TENSOR_CFG"], arch_config=arch).to(device)
    train_config = make_repaired_train_config(
        ns=ns,
        name=name,
        epochs=epochs,
        batch_size=batch_size,
        train_limit=None,
        val_limit=None,
        cpu_verify_items=0,
        full_only=full_only,
        graph_conflict_weight=graph_conflict_weight,
    )
    optimizer = ns["make_optimizer"](model=model, train_config=train_config)
    resumed_from = _maybe_load_checkpoint(
        ns=ns,
        model=model,
        optimizer=None,
        checkpoint_names=(name,) if resume_names is None else tuple(resume_names),
        device=device,
    )

    train_loader, val_loader, data_report = ns["make_phase5_loaders"](
        dataset_source=dataset_path,
        train_config=train_config,
    )

    last_ckpt_path = CHECKPOINT_DIR / f"{name}_last.pt"
    best_ckpt_path = CHECKPOINT_DIR / f"{name}_best.pt"
    history_path = REPORT_DIR / f"{name}_history.json"

    print("=== Fast Gate Training Setup ===")
    print("Device:", device)
    print("Trainable parameters:", ns["count_parameters"](model))
    print("Dataset source:", dataset_path)
    print("Data report:")
    ns["pprint"](data_report, sort_dicts=False)
    print("Resumed from:", resumed_from)
    print("Fast settings:", {
        "eval_every": eval_every,
        "patience": patience,
        "min_epochs": min_epochs,
        "cpu_verify_items_during_training": 0,
        "deep_supervision": "full_only" if full_only else "short_plus_full",
        "factor_messages": "vectorized_factor_tables_v1",
    })

    history: List[Dict[str, Any]] = []
    raw_global_step = resumed_from.get("global_step", 0) if resumed_from else 0
    global_step = int(raw_global_step or 0)
    best_metric = -math.inf
    best_energy = math.inf
    stale_evals = 0

    if resumed_from and isinstance(resumed_from.get("metrics"), dict):
        metrics = resumed_from["metrics"]
        best_metric = float(metrics.get("hard_valid_rate", metrics.get("cpu_valid_rate", -math.inf)))
        best_energy = float(metrics.get("mean_hard_energy", metrics.get("cpu_mean_energy", math.inf)))

    for epoch in range(1, int(epochs) + 1):
        epoch_start = time.time()
        train_metrics, global_step = ns["train_one_epoch"](
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            device=device,
            train_config=train_config,
            epoch=epoch,
            global_step=global_step,
        )

        should_eval = epoch == 1 or epoch % int(eval_every) == 0 or epoch == int(epochs)
        epoch_record: Dict[str, Any] = {
            "epoch": epoch,
            "global_step": global_step,
            "elapsed_s": time.time() - epoch_start,
            "train": train_metrics,
        }

        if should_eval:
            val_metrics = ns["evaluate_ahr_model"](
                model=model,
                val_loader=val_loader,
                device=device,
                train_config=train_config,
            )
            epoch_record["val"] = val_metrics
            metric = float(val_metrics.get("hard_valid_rate", 0.0))
            energy = float(val_metrics.get("mean_hard_energy", math.inf))

            ns["save_ahr_checkpoint"](
                path=last_ckpt_path,
                model=model,
                optimizer=optimizer,
                train_config=train_config,
                epoch=epoch,
                global_step=global_step,
                metrics=val_metrics,
            )

            improved = metric > best_metric + 1.0e-9 or (
                abs(metric - best_metric) <= 1.0e-9 and energy < best_energy
            )
            if improved:
                best_metric = metric
                best_energy = energy
                stale_evals = 0
                ns["save_ahr_checkpoint"](
                    path=best_ckpt_path,
                    model=model,
                    optimizer=optimizer,
                    train_config=train_config,
                    epoch=epoch,
                    global_step=global_step,
                    metrics=val_metrics,
                )
            else:
                stale_evals += 1

            print(
                f"[fast epoch {epoch}] elapsed={epoch_record['elapsed_s']:.1f}s "
                f"train_loss={train_metrics.get('loss_total', float('nan')):.4f} "
                f"val_valid={metric:.3f} val_energy={energy:.3f} "
                f"best_valid={best_metric:.3f} best_energy={best_energy:.3f} "
                f"stale_evals={stale_evals}"
            )

            target_hit = metric >= 0.90 and energy <= 0.10
            if epoch >= int(min_epochs) and target_hit:
                print("Fast gate target reached; stopping this stage early.")
                history.append(epoch_record)
                break
            if epoch >= int(min_epochs) and stale_evals >= int(patience):
                print("Fast gate plateau reached; stopping this stage.")
                history.append(epoch_record)
                break

        history.append(epoch_record)
        write_json(history_path, history)

    if not best_ckpt_path.exists():
        ns["save_ahr_checkpoint"](
            path=best_ckpt_path,
            model=model,
            optimizer=optimizer,
            train_config=train_config,
            epoch=history[-1]["epoch"] if history else 0,
            global_step=global_step,
            metrics=history[-1].get("val", {}) if history else {},
        )

    best_model = _load_best_checkpoint_into_model(ns, name, arch, device)
    final_eval_config = dataclasses.replace(train_config, cpu_verify_items=batch_size)
    split_reports = evaluate_splits(ns, best_model, dataset_path, final_eval_config, batch_size, device)
    report = {
        "kind": "ahr_csp_512_gate_fast_training",
        "name": name,
        "dataset_path": str(dataset_path),
        "created_at_unix": time.time(),
        "arch_config": ns["to_jsonable"](arch),
        "loss_config": dataclasses.asdict(RepairedLoss()),
        "train_config": ns["to_jsonable"](train_config),
        "fast_settings": {
            "eval_every": eval_every,
            "patience": patience,
            "min_epochs": min_epochs,
            "resume_names": list(resume_names or (name,)),
            "resumed_from": resumed_from,
            "factor_messages": "vectorized_factor_tables_v1",
            "deep_supervision": "full_only" if full_only else "short_plus_full",
            "graph_conflict_weight": graph_conflict_weight,
        },
        "history_tail": history[-10:],
        "split_reports": split_reports,
        "passed": gate_passed(split_reports),
        "targets": PASS_TARGETS,
    }
    write_json(REPORT_DIR / f"{name}_report.json", report)
    return report


def run_family_gates(ns: Dict[str, Any], epochs: int, batch_size: int, families: Optional[Sequence[str]]) -> Dict[str, Any]:
    split_report = split_original_512_by_family()
    selected = list(families or split_report["families"].keys())
    results: Dict[str, Any] = {"families": {}, "created_at_unix": time.time()}
    for family in selected:
        dataset_path = Path(split_report["families"][family]["dataset_path"])
        name = f"phase5p7_family_{family}_repaired"
        results["families"][family] = train_gate(ns, dataset_path, name, epochs, batch_size)
    write_json(REPORT_DIR / "phase5p7_family_gate_training_report.json", results)
    return results


def run_curriculum(ns: Dict[str, Any], stages: Sequence[str], epochs: int, batch_size: int) -> Dict[str, Any]:
    stage_reports: List[Dict[str, Any]] = []
    for stage in stages:
        if stage == "unique64":
            dataset_path = GATE_C_DATASET
        else:
            size = int(stage.replace("unique", ""))
            name = f"phase5p7_curriculum_unique_{size}"
            build_report = build_unique_controlled_dataset(
                ns=ns,
                name=name,
                train=size,
                val=max(16, size // 4),
                test=max(16, size // 4),
                seed=20260501 + size,
                domain_cap=46656,
                require_unique="feasible",
                max_attempts_multiplier=100,
            )
            dataset_path = Path(build_report["dataset_path"])

        gate_name = f"phase5p7_curriculum_{stage}_repaired"
        report = train_gate(ns, dataset_path, gate_name, epochs, batch_size)
        stage_reports.append(report)
        if not report["passed"]:
            break
    out = {"created_at_unix": time.time(), "stages": stage_reports}
    write_json(REPORT_DIR / "phase5p7_curriculum_report.json", out)
    return out


def run_curriculum_fast(
    ns: Dict[str, Any],
    stages: Sequence[str],
    epochs: int,
    batch_size: int,
    eval_every: int,
    patience: int,
    min_epochs: int,
    fresh: bool = False,
    full_only: bool = False,
) -> Dict[str, Any]:
    stage_reports: List[Dict[str, Any]] = []
    for stage in stages:
        if stage == "unique64":
            dataset_path = GATE_C_DATASET
        else:
            size = int(stage.replace("unique", ""))
            name = f"phase5p7_curriculum_fast_unique_{size}"
            build_report = build_unique_controlled_dataset(
                ns=ns,
                name=name,
                train=size,
                val=max(16, size // 4),
                test=max(16, size // 4),
                seed=20260501 + size,
                domain_cap=46656,
                require_unique="feasible",
                max_attempts_multiplier=100,
            )
            dataset_path = Path(build_report["dataset_path"])

        if fresh:
            prefix = "phase5p7_curriculum_fast_fresh_fullonly" if full_only else "phase5p7_curriculum_fast_fresh_shortfull"
            gate_name = f"{prefix}_{stage}_repaired"
            resume_names: Sequence[str] = ()
        else:
            prefix = "phase5p7_curriculum_fast_fullonly" if full_only else "phase5p7_curriculum_fast_shortfull"
            gate_name = f"{prefix}_{stage}_repaired"
            resume_names = (gate_name, f"phase5p7_curriculum_{stage}_repaired")
        report = train_gate_fast(
            ns=ns,
            dataset_path=dataset_path,
            name=gate_name,
            epochs=epochs,
            batch_size=batch_size,
            eval_every=eval_every,
            patience=patience,
            min_epochs=min_epochs,
            resume_names=resume_names,
            full_only=full_only,
        )
        stage_reports.append(report)
        if not report["passed"]:
            break
    out = {
        "created_at_unix": time.time(),
        "mode": "fast",
        "fresh": fresh,
        "deep_supervision": "full_only" if full_only else "short_plus_full",
        "stages": stage_reports,
    }
    if fresh:
        report_name = "phase5p7_curriculum_fast_fresh_fullonly_report.json" if full_only else "phase5p7_curriculum_fast_fresh_shortfull_report.json"
    else:
        report_name = "phase5p7_curriculum_fast_fullonly_report.json" if full_only else "phase5p7_curriculum_fast_shortfull_report.json"
    write_json(REPORT_DIR / report_name, out)
    return out


def run_curriculum_seeded(
    ns: Dict[str, Any],
    stages: Sequence[str],
    epochs: int,
    batch_size: int,
    eval_every: int,
    patience: int,
    min_epochs: int,
    full_only: bool = False,
) -> Dict[str, Any]:
    gate_c = verify_gate_c_report()
    stage_reports: List[Dict[str, Any]] = []
    previous_checkpoint = "phase5p6_gate_C_unique_64_a100"

    if "unique64" in stages:
        stage_reports.append(
            {
                "stage": "unique64",
                "mode": "verified_seed_checkpoint",
                "checkpoint_name": previous_checkpoint,
                "passed": bool(gate_c.get("passes_targets")),
                "gate_c": gate_c,
            }
        )
        if not gate_c.get("passes_targets"):
            out = {
                "created_at_unix": time.time(),
                "mode": "seeded",
                "deep_supervision": "full_only" if full_only else "short_plus_full",
                "stages": stage_reports,
            }
            write_json(REPORT_DIR / "phase5p7_curriculum_seeded_report.json", out)
            return out

    for stage in stages:
        if stage == "unique64":
            continue

        size = int(stage.replace("unique", ""))
        name = f"phase5p7_curriculum_seeded_unique_{size}"
        stage_domain_cap = 4096 if size <= 128 else 16384 if size <= 256 else 46656
        build_report = build_unique_controlled_dataset(
            ns=ns,
            name=name,
            train=size,
            val=max(16, size // 4),
            test=max(16, size // 4),
            seed=20260501 + size,
            domain_cap=stage_domain_cap,
            require_unique="feasible",
            max_attempts_multiplier=100,
        )
        dataset_path = Path(build_report["dataset_path"])

        gate_name = f"phase5p7_curriculum_seeded_{stage}_repaired"
        report = train_gate_fast(
            ns=ns,
            dataset_path=dataset_path,
            name=gate_name,
            epochs=epochs,
            batch_size=batch_size,
            eval_every=eval_every,
            patience=patience,
            min_epochs=min_epochs,
            resume_names=(previous_checkpoint,),
            full_only=full_only,
        )
        report["stage"] = stage
        report["build_report"] = build_report
        report["initialized_from"] = previous_checkpoint
        stage_reports.append(report)
        if not report["passed"]:
            break
        previous_checkpoint = gate_name

    out = {
        "created_at_unix": time.time(),
        "mode": "seeded",
        "deep_supervision": "full_only" if full_only else "short_plus_full",
        "stages": stage_reports,
    }
    write_json(REPORT_DIR / "phase5p7_curriculum_seeded_report.json", out)
    return out


def _json_domain_values(variable: Dict[str, Any]) -> List[Any]:
    domain = variable.get("domain", {})
    values = domain.get("values_")
    if values is not None:
        return list(values)
    if str(variable.get("sort")) == "Bool":
        return [False, True]
    if domain.get("kind") == "interval":
        lower = int(domain.get("lower", 0))
        upper = int(domain.get("upper", lower))
        return list(range(lower, upper + 1))
    raise ValueError(f"Unsupported variable domain for repair: {variable}")


def _problem_assignment_space(raw_problem_json: Dict[str, Any]) -> Tuple[List[str], Dict[str, List[Any]]]:
    names: List[str] = []
    domains: Dict[str, List[Any]] = {}
    for variable in raw_problem_json.get("variables", []):
        name = str(variable["name"])
        names.append(name)
        domains[name] = _json_domain_values(variable)
    return names, domains


def _best_energy(check: Dict[str, Any], default: float = 999.0) -> float:
    for key in ("total_energy", "energy", "hard_energy"):
        if key in check and check[key] is not None:
            return float(check[key])
    return default


def _iter_local_repairs(
    base_assignment: Dict[str, Any],
    names: Sequence[str],
    domains: Dict[str, Sequence[Any]],
    max_radius: int,
) -> Iterable[Tuple[int, List[str], Dict[str, Any]]]:
    for radius in range(1, int(max_radius) + 1):
        for combo in itertools.combinations(names, radius):
            alternatives: List[List[Any]] = []
            viable = True
            for name in combo:
                current = base_assignment.get(name)
                values = [value for value in domains.get(name, ()) if value != current]
                if not values:
                    viable = False
                    break
                alternatives.append(values)
            if not viable:
                continue
            for replacement_values in itertools.product(*alternatives):
                candidate = dict(base_assignment)
                for name, value in zip(combo, replacement_values):
                    candidate[name] = value
                yield radius, list(combo), candidate


def repair_assignment_with_verifier(
    ns: Dict[str, Any],
    raw_problem_json: Dict[str, Any],
    base_assignment: Dict[str, Any],
    max_radius: int,
    max_attempts: int,
) -> Dict[str, Any]:
    problem = ns["problem_from_jsonable"](raw_problem_json)
    raw_check = ns["verify_assignment"](problem, base_assignment)
    raw_valid = bool(raw_check.get("valid", False))
    raw_energy = _best_energy(raw_check, 999.0)
    if raw_valid:
        return {
            "raw_valid": True,
            "raw_energy": raw_energy,
            "repaired_valid": True,
            "repaired_energy": 0.0,
            "repair_radius": 0,
            "repair_variables": [],
            "attempts": 0,
            "attempt_limit_hit": False,
            "assignment": base_assignment,
        }

    names, domains = _problem_assignment_space(raw_problem_json)
    if not names:
        names = list(base_assignment.keys())
        domains = {name: [False, True] if isinstance(base_assignment[name], bool) else [base_assignment[name]] for name in names}

    attempts = 0
    best = {
        "energy": raw_energy,
        "radius": None,
        "variables": [],
        "assignment": base_assignment,
    }
    limit_hit = False
    for radius, changed_names, candidate in _iter_local_repairs(base_assignment, names, domains, max_radius):
        attempts += 1
        if attempts > int(max_attempts):
            limit_hit = True
            break
        check = ns["verify_assignment"](problem, candidate)
        energy = _best_energy(check, 999.0)
        if energy < float(best["energy"]):
            best = {
                "energy": energy,
                "radius": radius,
                "variables": changed_names,
                "assignment": candidate,
            }
        if bool(check.get("valid", False)):
            return {
                "raw_valid": False,
                "raw_energy": raw_energy,
                "repaired_valid": True,
                "repaired_energy": 0.0,
                "repair_radius": radius,
                "repair_variables": changed_names,
                "attempts": attempts,
                "attempt_limit_hit": False,
                "assignment": candidate,
            }

    return {
        "raw_valid": False,
        "raw_energy": raw_energy,
        "repaired_valid": False,
        "repaired_energy": float(best["energy"]),
        "repair_radius": best["radius"],
        "repair_variables": best["variables"],
        "attempts": attempts,
        "attempt_limit_hit": limit_hit,
        "assignment": best["assignment"],
    }


def summarize_repair_rows(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {"count": 0}
    count = len(rows)
    raw_valid = sum(1 for row in rows if row["raw_valid"])
    repaired_valid = sum(1 for row in rows if row["repaired_valid"])
    raw_energy = sum(float(row["raw_energy"]) for row in rows) / count
    repaired_energy = sum(float(row["repaired_energy"]) for row in rows) / count
    radius_counts: Dict[str, int] = {}
    limit_hits = 0
    for row in rows:
        radius_counts[str(row["repair_radius"])] = radius_counts.get(str(row["repair_radius"]), 0) + 1
        if row.get("attempt_limit_hit"):
            limit_hits += 1
    return {
        "count": count,
        "raw_valid": raw_valid,
        "raw_valid_rate": raw_valid / count,
        "raw_mean_energy": raw_energy,
        "repaired_valid": repaired_valid,
        "repaired_valid_rate": repaired_valid / count,
        "repaired_mean_energy": repaired_energy,
        "repair_lift": (repaired_valid - raw_valid) / count,
        "newly_repaired": sum(1 for row in rows if (not row["raw_valid"]) and row["repaired_valid"]),
        "attempt_limit_hits": limit_hits,
        "by_repair_radius": radius_counts,
    }


def repair_gate_passed(by_split: Dict[str, Any]) -> bool:
    train = by_split.get("train", {})
    val = by_split.get("val", {})
    return (
        float(train.get("repaired_valid_rate", 0.0)) >= PASS_TARGETS["train_cpu_valid_rate"]
        and float(val.get("repaired_valid_rate", 0.0)) >= PASS_TARGETS["val_cpu_valid_rate"]
        and float(val.get("repaired_mean_energy", 999.0)) <= PASS_TARGETS["val_mean_hard_energy"]
    )


def evaluate_repair_mode(
    ns: Dict[str, Any],
    dataset_path: Path,
    checkpoint_name: str,
    name: str,
    batch_size: int,
    max_radius: int,
    max_attempts: int,
) -> Dict[str, Any]:
    install_repaired_loss(ns)
    torch = ns["torch"]
    device_name = ns["get_device"]()
    device = torch.device("cpu" if device_name == "no-torch" else device_name)
    arch = make_debug_arch(ns)
    model = _load_best_checkpoint_into_model(ns, checkpoint_name, arch, device)
    model.eval()

    all_rows: List[Dict[str, Any]] = []
    by_split_rows: Dict[str, List[Dict[str, Any]]] = {}
    by_family_rows: Dict[str, List[Dict[str, Any]]] = {}
    examples: List[Dict[str, Any]] = []

    start = time.time()
    for split in ("train", "val", "test"):
        loader, data_report = ns["make_phase6_loader"](
            dataset_source=dataset_path,
            split=split,
            sat_only=True,
            batch_size=batch_size,
            limit=None,
        )
        rows: List[Dict[str, Any]] = []
        for batch0 in loader:
            batch = ns["batch_to_device"](batch0, device)
            with torch.no_grad():
                output = model(batch)
            hard_index = output.hard_index.detach().cpu()
            model_hard_energy = output.hard_energy.detach().cpu().tolist()
            for item_index, problem_id in enumerate(batch0.problem_ids):
                raw_problem_json = batch0.raw_problem_jsons[item_index]
                family = str(raw_problem_json.get("problem_type", "unknown"))
                base_assignment = ns["decode_batch_indices_to_assignment"](batch0, item_index, hard_index[item_index])
                repair = repair_assignment_with_verifier(
                    ns=ns,
                    raw_problem_json=raw_problem_json,
                    base_assignment=base_assignment,
                    max_radius=max_radius,
                    max_attempts=max_attempts,
                )
                row = {
                    "split": split,
                    "problem_id": problem_id,
                    "problem_type": family,
                    "n_vars": len(base_assignment),
                    "model_hard_energy": float(model_hard_energy[item_index]),
                    **{key: value for key, value in repair.items() if key != "assignment"},
                }
                rows.append(row)
                all_rows.append(row)
                by_family_rows.setdefault(family, []).append(row)
                if len(examples) < 25 and (not row["raw_valid"] or not row["repaired_valid"]):
                    examples.append(
                        {
                            **row,
                            "base_assignment": base_assignment,
                            "repaired_assignment": repair["assignment"],
                        }
                    )
        by_split_rows[split] = rows
        print(f"[repair-eval] split={split} {summarize_repair_rows(rows)}", flush=True)

    by_split = {split: summarize_repair_rows(rows) for split, rows in by_split_rows.items()}
    report = {
        "kind": "ahr_csp_verifier_guided_repair_eval",
        "created_at_unix": time.time(),
        "elapsed_s": time.time() - start,
        "name": name,
        "dataset_path": str(dataset_path),
        "checkpoint_name": checkpoint_name,
        "checkpoint_path": str(CHECKPOINT_DIR / f"{checkpoint_name}_best.pt"),
        "device": str(device),
        "batch_size": batch_size,
        "max_radius": max_radius,
        "max_attempts_per_problem": max_attempts,
        "targets": PASS_TARGETS,
        "passed": repair_gate_passed(by_split),
        "overall": summarize_repair_rows(all_rows),
        "by_split": by_split,
        "by_family": {family: summarize_repair_rows(rows) for family, rows in sorted(by_family_rows.items())},
        "failed_or_repaired_examples": examples,
    }
    write_json(REPORT_DIR / f"{name}_repair_eval_report.json", report)
    return report


def run_phase6_after_pass(ns: Dict[str, Any], dataset_path: Path, checkpoint_name: str, batch_size: int) -> Dict[str, Any]:
    install_repaired_loss(ns)
    torch = ns["torch"]
    device_name = ns["get_device"]()
    device = torch.device("cpu" if device_name == "no-torch" else device_name)
    arch = make_debug_arch(ns)
    train_config = make_repaired_train_config(ns, checkpoint_name, 1, batch_size, None, None)
    model = _load_best_checkpoint_into_model(ns, checkpoint_name, arch, device)
    split_reports = evaluate_splits(ns, model, dataset_path, train_config, batch_size, device)
    if not gate_passed(split_reports):
        raise RuntimeError("Refusing Phase 6: checkpoint does not pass the 512 gate.")

    budgets = (
        ns["AHRBudgetSpec"]("init_only", 0, 0, 0),
        ns["AHRBudgetSpec"]("short", 1, 1, 1),
        ns["AHRBudgetSpec"]("full", 2, 2, 4),
    )
    budget_sweep = ns["run_reasoning_budget_sweep"](
        model=model,
        dataset_source=dataset_path,
        split="val",
        budgets=budgets,
        colony_counts=(1, 4, 8, 16),
        base_config=ns["MultiColonyConfig"](
            n_colonies=4,
            noise_scale=0.05,
            stochastic=True,
            batch_size=batch_size,
            sat_only=True,
            cpu_verify_items_per_batch=batch_size,
        ),
        max_batches=None,
        record_limit=None,
        device=device,
    )
    ablations = ns["evaluate_ablation_suite"](
        model=model,
        dataset_source=dataset_path,
        split="val",
        ablations=(
            ns["AblationSpec"]("full_deterministic"),
            ns["AblationSpec"]("no_queen", skip_queen=True),
            ns["AblationSpec"]("no_worker", skip_worker=True),
            ns["AblationSpec"]("no_scout", skip_scout=True),
            ns["AblationSpec"]("init_only", budget=ns["AHRBudgetSpec"]("init_only", 0, 0, 0)),
        ),
        batch_size=batch_size,
        max_batches=None,
        record_limit=None,
        sat_only=True,
        device=device,
    )
    baselines = ns["evaluate_baseline_suite"](
        dataset_source=dataset_path,
        split="val",
        baselines=ns["DEFAULT_BASELINES"],
        batch_size=batch_size,
        max_batches=None,
        record_limit=None,
        sat_only=True,
    )
    failure_atlas = ns["collect_failure_atlas"](
        model=model,
        dataset_source=dataset_path,
        split="val",
        inference_config=ns["MultiColonyConfig"](
            n_colonies=16,
            budget=ns["AHRBudgetSpec"]("full", 2, 2, 4),
            noise_scale=0.05,
            stochastic=True,
            batch_size=batch_size,
            sat_only=True,
            cpu_verify_items_per_batch=batch_size,
        ),
        max_batches=None,
        max_failures=50,
        record_limit=None,
        device=device,
    )
    package = ns["export_ahr_package"](
        model=model,
        export_name=f"{checkpoint_name}_phase6_export",
        dataset_source=dataset_path,
        extra_reports={
            "split_reports": split_reports,
            "budget_sweep": budget_sweep,
            "ablations": ablations,
            "baselines": baselines,
            "failure_atlas": failure_atlas,
        },
    )
    report = {
        "created_at_unix": time.time(),
        "checkpoint_name": checkpoint_name,
        "dataset_path": str(dataset_path),
        "split_reports": split_reports,
        "budget_sweep": budget_sweep,
        "ablations": ablations,
        "baselines": baselines,
        "failure_atlas": failure_atlas,
        "export_dir": str(package),
    }
    write_json(REPORT_DIR / f"{checkpoint_name}_phase6_post_pass_report.json", report)
    return report


def cmd_status(_: argparse.Namespace) -> None:
    result = {
        "gate_c": verify_gate_c_report(),
        "original_512": summarize_records(read_jsonl(ORIGINAL_512_DATASET)),
    }
    write_json(REPORT_DIR / "phase5p7_roadmap_status_report.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="AHR-CSP 512 gate experiment runner")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("status", help="Verify Gate C report and summarize original 512 dataset.")
    sub.add_parser("split-family", help="Write family-specific JSONL datasets from original 512.")

    p_build = sub.add_parser("build-unique", help="Build a uniqueness-controlled mixed SAT dataset.")
    p_build.add_argument("--name", default="phase5p7_unique_controlled_512_128_128")
    p_build.add_argument("--train", type=int, default=512)
    p_build.add_argument("--val", type=int, default=128)
    p_build.add_argument("--test", type=int, default=128)
    p_build.add_argument("--seed", type=int, default=20260501)
    p_build.add_argument("--domain-cap", type=int, default=46656)
    p_build.add_argument("--require-unique", choices=("feasible", "all", "none"), default="feasible")
    p_build.add_argument("--max-attempts-multiplier", type=int, default=100)
    p_build.add_argument("--families", nargs="*", choices=tuple(FAMILY_GENERATORS.keys()))

    p_train = sub.add_parser("train", help="Train one repaired gate on a JSONL dataset.")
    p_train.add_argument("--dataset", required=True)
    p_train.add_argument("--name", required=True)
    p_train.add_argument("--epochs", type=int, default=120)
    p_train.add_argument("--batch-size", type=int, default=64)

    p_train_fast = sub.add_parser("train-fast", help="Train one repaired gate with vectorized factors and sparse validation.")
    p_train_fast.add_argument("--dataset", required=True)
    p_train_fast.add_argument("--name", required=True)
    p_train_fast.add_argument("--epochs", type=int, default=80)
    p_train_fast.add_argument("--batch-size", type=int, default=64)
    p_train_fast.add_argument("--eval-every", type=int, default=5)
    p_train_fast.add_argument("--patience", type=int, default=6)
    p_train_fast.add_argument("--min-epochs", type=int, default=10)
    p_train_fast.add_argument("--resume-from", nargs="*", default=None)
    p_train_fast.add_argument("--fresh", action="store_true", help="Do not load any previous checkpoint.")
    p_train_fast.add_argument("--full-only", action="store_true", help="Use only the full deep-supervision head.")
    p_train_fast.add_argument("--graph-conflict-weight", type=float, default=0.0, help="Dense expected same-color loss for graph-coloring notEquals edges.")

    p_family = sub.add_parser("train-family", help="Train repaired family-specific gates.")
    p_family.add_argument("--epochs", type=int, default=120)
    p_family.add_argument("--batch-size", type=int, default=64)
    p_family.add_argument("--families", nargs="*", choices=tuple(FAMILY_GENERATORS.keys()))

    p_curr = sub.add_parser("run-curriculum", help="Run staged curriculum until the first failed gate.")
    p_curr.add_argument("--stages", nargs="+", default=["unique64", "unique128", "unique256", "unique512"])
    p_curr.add_argument("--epochs", type=int, default=120)
    p_curr.add_argument("--batch-size", type=int, default=64)

    p_curr_fast = sub.add_parser("run-curriculum-fast", help="Run staged curriculum with vectorized factors, sparse validation, and plateau stop.")
    p_curr_fast.add_argument("--stages", nargs="+", default=["unique64", "unique128", "unique256", "unique512"])
    p_curr_fast.add_argument("--epochs", type=int, default=80)
    p_curr_fast.add_argument("--batch-size", type=int, default=64)
    p_curr_fast.add_argument("--eval-every", type=int, default=5)
    p_curr_fast.add_argument("--patience", type=int, default=6)
    p_curr_fast.add_argument("--min-epochs", type=int, default=10)
    p_curr_fast.add_argument("--fresh", action="store_true", help="Do not load previous curriculum checkpoints.")
    p_curr_fast.add_argument("--full-only", action="store_true", help="Use only the full deep-supervision head.")

    p_curr_seeded = sub.add_parser("run-curriculum-seeded", help="Run curriculum from the certified Gate C checkpoint.")
    p_curr_seeded.add_argument("--stages", nargs="+", default=["unique64", "unique128", "unique256", "unique512"])
    p_curr_seeded.add_argument("--epochs", type=int, default=150)
    p_curr_seeded.add_argument("--batch-size", type=int, default=64)
    p_curr_seeded.add_argument("--eval-every", type=int, default=1)
    p_curr_seeded.add_argument("--patience", type=int, default=12)
    p_curr_seeded.add_argument("--min-epochs", type=int, default=5)
    p_curr_seeded.add_argument("--full-only", action="store_true", help="Use only the full deep-supervision head.")

    p_phase6 = sub.add_parser("phase6-after-pass", help="Run Phase 6 only if checkpoint passes the gate.")
    p_phase6.add_argument("--dataset", required=True)
    p_phase6.add_argument("--checkpoint-name", required=True)
    p_phase6.add_argument("--batch-size", type=int, default=64)

    p_repair = sub.add_parser("repair-eval", help="Evaluate verifier-guided local repair on a checkpoint.")
    p_repair.add_argument("--dataset", required=True)
    p_repair.add_argument("--checkpoint-name", required=True)
    p_repair.add_argument("--name", required=True)
    p_repair.add_argument("--batch-size", type=int, default=64)
    p_repair.add_argument("--max-radius", type=int, default=2)
    p_repair.add_argument("--max-attempts", type=int, default=20000)

    args = parser.parse_args(argv)

    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    if args.cmd == "status":
        cmd_status(args)
        return 0
    if args.cmd == "split-family":
        print(json.dumps(split_original_512_by_family(), indent=2, sort_keys=True))
        return 0

    ns = load_notebook_runtime()

    if args.cmd == "build-unique":
        report = build_unique_controlled_dataset(
            ns=ns,
            name=args.name,
            train=args.train,
            val=args.val,
            test=args.test,
            seed=args.seed,
            domain_cap=args.domain_cap,
            require_unique=args.require_unique,
            max_attempts_multiplier=args.max_attempts_multiplier,
            families=args.families,
        )
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    if args.cmd == "train":
        report = train_gate(ns, Path(args.dataset), args.name, args.epochs, args.batch_size)
        print(json.dumps({"report": f"{REPORT_DIR / (args.name + '_report.json')}", "passed": report["passed"]}, indent=2))
        return 0
    if args.cmd == "train-fast":
        if args.fresh:
            resume_names = ()
        elif args.resume_from is not None:
            resume_names = args.resume_from
        else:
            resume_names = (args.name,)
        report = train_gate_fast(
            ns=ns,
            dataset_path=Path(args.dataset),
            name=args.name,
            epochs=args.epochs,
            batch_size=args.batch_size,
            eval_every=args.eval_every,
            patience=args.patience,
            min_epochs=args.min_epochs,
            resume_names=resume_names,
            full_only=args.full_only,
            graph_conflict_weight=args.graph_conflict_weight,
        )
        print(json.dumps({"report": f"{REPORT_DIR / (args.name + '_report.json')}", "passed": report["passed"]}, indent=2))
        return 0
    if args.cmd == "train-family":
        report = run_family_gates(ns, args.epochs, args.batch_size, args.families)
        print(json.dumps({"report": str(REPORT_DIR / "phase5p7_family_gate_training_report.json")}, indent=2))
        return 0
    if args.cmd == "run-curriculum":
        report = run_curriculum(ns, args.stages, args.epochs, args.batch_size)
        print(json.dumps({"report": str(REPORT_DIR / "phase5p7_curriculum_report.json"), "stages_run": len(report["stages"])}, indent=2))
        return 0
    if args.cmd == "run-curriculum-fast":
        report = run_curriculum_fast(
            ns=ns,
            stages=args.stages,
            epochs=args.epochs,
            batch_size=args.batch_size,
            eval_every=args.eval_every,
            patience=args.patience,
            min_epochs=args.min_epochs,
            fresh=args.fresh,
            full_only=args.full_only,
        )
        if args.fresh:
            report_name = "phase5p7_curriculum_fast_fresh_fullonly_report.json" if args.full_only else "phase5p7_curriculum_fast_fresh_shortfull_report.json"
        else:
            report_name = "phase5p7_curriculum_fast_fullonly_report.json" if args.full_only else "phase5p7_curriculum_fast_shortfull_report.json"
        print(json.dumps({"report": str(REPORT_DIR / report_name), "stages_run": len(report["stages"])}, indent=2))
        return 0
    if args.cmd == "run-curriculum-seeded":
        report = run_curriculum_seeded(
            ns=ns,
            stages=args.stages,
            epochs=args.epochs,
            batch_size=args.batch_size,
            eval_every=args.eval_every,
            patience=args.patience,
            min_epochs=args.min_epochs,
            full_only=args.full_only,
        )
        print(json.dumps({"report": str(REPORT_DIR / "phase5p7_curriculum_seeded_report.json"), "stages_run": len(report["stages"])}, indent=2))
        return 0
    if args.cmd == "phase6-after-pass":
        report = run_phase6_after_pass(ns, Path(args.dataset), args.checkpoint_name, args.batch_size)
        print(json.dumps({"report": str(REPORT_DIR / f"{args.checkpoint_name}_phase6_post_pass_report.json")}, indent=2))
        return 0
    if args.cmd == "repair-eval":
        report = evaluate_repair_mode(
            ns=ns,
            dataset_path=Path(args.dataset),
            checkpoint_name=args.checkpoint_name,
            name=args.name,
            batch_size=args.batch_size,
            max_radius=args.max_radius,
            max_attempts=args.max_attempts,
        )
        print(
            json.dumps(
                {
                    "report": str(REPORT_DIR / f"{args.name}_repair_eval_report.json"),
                    "passed": report["passed"],
                    "overall": report["overall"],
                    "by_split": report["by_split"],
                    "by_family": report["by_family"],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    raise AssertionError(args.cmd)


if __name__ == "__main__":
    raise SystemExit(main())

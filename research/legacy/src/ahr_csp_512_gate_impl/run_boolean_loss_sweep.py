#!/usr/bin/env python3
"""Small Boolean loss/cycle sweep for the n_vars<=5 gate."""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.ahr512_gate_runner import (  # noqa: E402
    CHECKPOINT_DIR,
    PASS_TARGETS,
    REPORT_DIR,
    evaluate_splits,
    gate_passed,
    install_repaired_loss,
    install_vectorized_factor_messages,
    load_notebook_runtime,
    make_debug_arch,
    make_repaired_train_config,
    write_json,
)


def compact_split_metrics(report: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {"passed": bool(report.get("passed"))}
    for split, obj in (report.get("split_reports") or {}).items():
        m = obj.get("metrics", obj)
        out[split] = {
            "cpu_valid_rate": m.get("cpu_valid_rate"),
            "cpu_mean_energy": m.get("cpu_mean_energy"),
            "assignment_acc": m.get("assignment_acc"),
            "exact_reference_match_rate": m.get("exact_reference_match_rate"),
        }
    return out


def make_case_arch(ns: Dict[str, Any], case: Dict[str, Any]) -> Any:
    arch = make_debug_arch(ns)
    updates = {}
    for key in ("n_queen_cycles", "n_worker_cycles", "n_scout_steps", "worker_support_gain", "decoder_delta_scale"):
        if key in case:
            updates[key] = case[key]
    return dataclasses.replace(arch, **updates) if updates else arch


def make_case_train_config(ns: Dict[str, Any], name: str, case: Dict[str, Any], epochs: int, batch_size: int) -> Any:
    cfg = make_repaired_train_config(
        ns=ns,
        name=name,
        epochs=epochs,
        batch_size=batch_size,
        train_limit=None,
        val_limit=None,
        cpu_verify_items=0,
        full_only=bool(case.get("full_only", False)),
    )
    loss_updates = {}
    for key in (
        "assignment_weight",
        "query_assignment_weight",
        "constraint_weight",
        "expected_violation_weight",
        "constraint_nll_weight",
        "logit_l2_weight",
    ):
        if key in case:
            loss_updates[key] = case[key]
    loss_cfg = dataclasses.replace(cfg.loss_config, **loss_updates) if loss_updates else cfg.loss_config
    cfg_updates = {"loss_config": loss_cfg}
    for key in ("lr", "weight_decay", "grad_clip_norm"):
        if key in case:
            cfg_updates[key] = case[key]
    return dataclasses.replace(cfg, **cfg_updates)


def train_case(
    ns: Dict[str, Any],
    dataset_path: Path,
    case: Dict[str, Any],
    epochs: int,
    batch_size: int,
    eval_every: int,
    patience: int,
    min_epochs: int,
) -> Dict[str, Any]:
    torch = ns["torch"]
    device_name = ns["get_device"]()
    device = torch.device("cpu" if device_name == "no-torch" else device_name)
    name = case["name"]
    arch = make_case_arch(ns, case)
    train_config = make_case_train_config(ns, name, case, epochs, batch_size)
    model = ns["AHRReasoner"](tensor_config=ns["TENSOR_CFG"], arch_config=arch).to(device)
    optimizer = ns["make_optimizer"](model=model, train_config=train_config)
    train_loader, val_loader, data_report = ns["make_phase5_loaders"](
        dataset_source=dataset_path,
        train_config=train_config,
    )

    last_ckpt_path = CHECKPOINT_DIR / f"{name}_last.pt"
    best_ckpt_path = CHECKPOINT_DIR / f"{name}_best.pt"
    history_path = REPORT_DIR / f"{name}_history.json"
    print(
        json.dumps(
            {
                "event": "case_setup",
                "name": name,
                "device": str(device),
                "dataset": str(dataset_path),
                "data_report": data_report,
                "arch": ns["to_jsonable"](arch),
                "train_config": ns["to_jsonable"](train_config),
            },
            sort_keys=True,
        ),
        flush=True,
    )

    history: List[Dict[str, Any]] = []
    best_metric = -math.inf
    best_energy = math.inf
    global_step = 0
    stale_evals = 0

    for epoch in range(1, int(epochs) + 1):
        t0 = time.time()
        train_metrics, global_step = ns["train_one_epoch"](
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            device=device,
            train_config=train_config,
            epoch=epoch,
            global_step=global_step,
        )
        record: Dict[str, Any] = {
            "epoch": epoch,
            "global_step": global_step,
            "elapsed_s": time.time() - t0,
            "train": train_metrics,
        }
        should_eval = epoch == 1 or epoch % eval_every == 0 or epoch == epochs
        if should_eval:
            val_metrics = ns["evaluate_ahr_model"](
                model=model,
                val_loader=val_loader,
                device=device,
                train_config=train_config,
            )
            record["val"] = val_metrics
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
            improved = metric > best_metric + 1e-9 or (abs(metric - best_metric) <= 1e-9 and energy < best_energy)
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
                f"[sweep {name} epoch {epoch}] train_valid={train_metrics.get('final/hard_valid_rate', train_metrics.get('hard_valid_rate', float('nan'))):.3f} "
                f"train_loss={train_metrics.get('loss_total', float('nan')):.4f} "
                f"val_valid={metric:.3f} val_energy={energy:.3f} best_valid={best_metric:.3f} best_energy={best_energy:.3f} stale={stale_evals}",
                flush=True,
            )
            target_hit = metric >= PASS_TARGETS["val_cpu_valid_rate"] and energy <= PASS_TARGETS["val_mean_hard_energy"]
            if epoch >= min_epochs and target_hit:
                history.append(record)
                break
            if epoch >= min_epochs and stale_evals >= patience:
                history.append(record)
                break
        history.append(record)
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

    best_model = ns["AHRReasoner"](tensor_config=ns["TENSOR_CFG"], arch_config=arch).to(device)
    ns["load_ahr_checkpoint"](best_ckpt_path, model=best_model, optimizer=None, map_location=device)
    final_eval_config = dataclasses.replace(train_config, cpu_verify_items=batch_size)
    split_reports = evaluate_splits(ns, best_model, dataset_path, final_eval_config, batch_size, device)
    report = {
        "kind": "boolean_loss_sweep_case",
        "name": name,
        "case": case,
        "dataset_path": str(dataset_path),
        "created_at_unix": time.time(),
        "arch_config": ns["to_jsonable"](arch),
        "train_config": ns["to_jsonable"](train_config),
        "history_tail": history[-10:],
        "best_training_metric": {"val_valid": best_metric, "val_energy": best_energy},
        "split_reports": split_reports,
        "passed": gate_passed(split_reports),
        "targets": PASS_TARGETS,
    }
    write_json(REPORT_DIR / f"{name}_report.json", report)
    print(json.dumps({"event": "case_done", "name": name, **compact_split_metrics(report)}, sort_keys=True), flush=True)
    return report


def default_cases() -> List[Dict[str, Any]]:
    return [
        {
            "name": "phase5p7_bool_le5_sweep_assign2_nll1_power35_b16",
            "assignment_weight": 2.0,
            "query_assignment_weight": 0.5,
            "constraint_nll_weight": 1.0,
        },
        {
            "name": "phase5p7_bool_le5_sweep_assign4_nll05_power35_b16",
            "assignment_weight": 4.0,
            "query_assignment_weight": 1.0,
            "constraint_nll_weight": 0.5,
            "constraint_weight": 0.5,
            "expected_violation_weight": 0.5,
        },
        {
            "name": "phase5p7_bool_le5_sweep_assign2_deeper_power35_b16",
            "assignment_weight": 2.0,
            "query_assignment_weight": 0.5,
            "constraint_nll_weight": 1.0,
            "n_queen_cycles": 3,
            "n_worker_cycles": 3,
            "n_scout_steps": 6,
        },
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=PROJECT_ROOT / "data" / "phase5p7_bool_nvars_le5.jsonl")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--patience", type=int, default=18)
    parser.add_argument("--min-epochs", type=int, default=35)
    parser.add_argument("--stop-on-pass", action="store_true")
    args = parser.parse_args()

    ns = load_notebook_runtime()
    install_repaired_loss(ns)
    install_vectorized_factor_messages(ns)

    results: List[Dict[str, Any]] = []
    start = time.time()
    for case in default_cases():
        report = train_case(
            ns=ns,
            dataset_path=args.dataset,
            case=case,
            epochs=args.epochs,
            batch_size=args.batch_size,
            eval_every=args.eval_every,
            patience=args.patience,
            min_epochs=args.min_epochs,
        )
        summary = {"name": case["name"], "case": case, **compact_split_metrics(report)}
        results.append(summary)
        if bool(report.get("passed")) and args.stop_on_pass:
            break
    out = {
        "created_at_unix": time.time(),
        "elapsed_s": time.time() - start,
        "dataset": str(args.dataset),
        "results": results,
    }
    report_path = REPORT_DIR / "phase5p7_bool_le5_loss_sweep_report.json"
    write_json(report_path, out)
    print(json.dumps({"report": str(report_path), **out}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

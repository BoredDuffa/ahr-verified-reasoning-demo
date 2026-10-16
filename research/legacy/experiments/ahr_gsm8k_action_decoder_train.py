#!/usr/bin/env python3
"""Constrained GSM8K arithmetic action decoder.

This experiment converts GSM8K equation traces into executable binary actions:

  add/sub/mul/div <operand_a> <operand_b> -> next result
  halt <operand> -> final answer

Operands are question numbers (N0...), previous computed results (R0...), or
literal constants (C:<value>).  At inference, a small policy predicts one action
at a time; every action is executed immediately by the exact Fraction evaluator.
The model can never emit free-form text.
"""

from __future__ import annotations

import argparse
import ast
import csv
import importlib.util
import json
import math
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "gsm8k_program_synthesis_v0"
REPORT_DIR = ROOT / "reports"
MODEL_PATH = ROOT / "checkpoints" / "gsm8k_action_decoder_v0.pt"
TOKEN_RE = re.compile(r"[a-z0-9]+")


def load_tools() -> Any:
    path = ROOT / "experiments" / "ahr_gsm8k_program_synthesis.py"
    spec = importlib.util.spec_from_file_location("ahr_gsm8k_program_synthesis", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


TOOLS = load_tools()


@dataclass(frozen=True)
class Action:
    op: str
    a: str
    b: str = ""

    def label(self) -> str:
        return f"{self.op}|{self.a}|{self.b}"


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def norm_num(value: str) -> str:
    return TOOLS.fraction_text(TOOLS.to_fraction(str(value)))


def value_of_source(src: str, numbers: Sequence[str], results: Sequence[Fraction]) -> Optional[Fraction]:
    try:
        if src.startswith("N"):
            return TOOLS.to_fraction(numbers[int(src[1:])])
        if src.startswith("R"):
            return results[int(src[1:])]
        if src.startswith("C:"):
            return TOOLS.to_fraction(src[2:])
    except Exception:
        return None
    return None


def source_for_leaf(raw: str, numbers: Sequence[str], results: Sequence[Fraction]) -> str:
    val = TOOLS.to_fraction(raw)
    for idx in range(len(results) - 1, -1, -1):
        if results[idx] == val:
            return f"R{idx}"
    for idx, number in enumerate(numbers):
        try:
            if TOOLS.to_fraction(number) == val:
                return f"N{idx}"
        except Exception:
            pass
    return f"C:{TOOLS.fraction_text(val)}"


def compile_expr(expr: str, numbers: Sequence[str], results: List[Fraction], actions: List[Action]) -> Optional[str]:
    expr = expr.replace(",", "").replace("$", "").replace("%", "/100")
    try:
        tree = ast.parse(expr, mode="eval")
    except Exception:
        return None

    def rec(node: ast.AST) -> Optional[str]:
        if isinstance(node, ast.Expression):
            return rec(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return source_for_leaf(str(node.value), numbers, results)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            inner = rec(node.operand)
            if inner is None:
                return None
            zero = "C:0"
            actions.append(Action("sub", zero, inner))
            val = execute_action(actions[-1], numbers, results)
            if val is None:
                return None
            results.append(val)
            return f"R{len(results) - 1}"
        if isinstance(node, ast.BinOp):
            left = rec(node.left)
            right = rec(node.right)
            if left is None or right is None:
                return None
            if isinstance(node.op, ast.Add):
                op = "add"
            elif isinstance(node.op, ast.Sub):
                op = "sub"
            elif isinstance(node.op, ast.Mult):
                op = "mul"
            elif isinstance(node.op, ast.Div):
                op = "div"
            else:
                return None
            actions.append(Action(op, left, right))
            val = execute_action(actions[-1], numbers, results)
            if val is None:
                return None
            results.append(val)
            return f"R{len(results) - 1}"
        return None

    return rec(tree)


def execute_action(action: Action, numbers: Sequence[str], results: Sequence[Fraction]) -> Optional[Fraction]:
    if action.op == "halt":
        return value_of_source(action.a, numbers, results)
    a = value_of_source(action.a, numbers, results)
    b = value_of_source(action.b, numbers, results)
    if a is None or b is None:
        return None
    try:
        if action.op == "add":
            return a + b
        if action.op == "sub":
            return a - b
        if action.op == "mul":
            return a * b
        if action.op == "div" and b != 0:
            return a / b
    except Exception:
        return None
    return None


def compile_row(row: Dict[str, Any]) -> Optional[List[Action]]:
    if not row.get("replay_valid"):
        return None
    numbers = list(row.get("numbers", []))
    results: List[Fraction] = []
    actions: List[Action] = []
    for eq in row.get("equations", []):
        before = len(results)
        src = compile_expr(str(eq["expr"]), numbers, results, actions)
        if src is None or len(results) == before:
            return None
        try:
            expected = TOOLS.to_fraction(eq["expected"])
        except Exception:
            return None
        if results[-1] != expected:
            return None
    if not results:
        return None
    actions.append(Action("halt", f"R{len(results) - 1}"))
    try:
        if results[-1] != TOOLS.to_fraction(row["final_answer"]):
            return None
    except Exception:
        return None
    return actions


def tokenize(question: str) -> List[str]:
    return TOKEN_RE.findall(question.lower())


class ActionDataset:
    def __init__(self, rows: Sequence[Dict[str, Any]], action_to_id: Dict[str, int], vocab: Dict[str, int], max_steps: int, start_action_id: int) -> None:
        self.items: List[Tuple[List[int], int, int, int, Dict[str, Any]]] = []
        for row in rows:
            actions = compile_row(row)
            if not actions or len(actions) > max_steps:
                continue
            ids = [vocab.get(tok, 1) for tok in tokenize(row["question"])]
            if not ids:
                continue
            for step, action in enumerate(actions):
                label = action.label()
                if label in action_to_id:
                    prev_id = start_action_id if step == 0 else action_to_id.get(actions[step - 1].label(), start_action_id)
                    self.items.append((ids, step, prev_id, action_to_id[label], row))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Tuple[List[int], int, int, int, Dict[str, Any]]:
        return self.items[idx]


def collate(batch: Sequence[Tuple[List[int], int, int, int, Dict[str, Any]]]) -> Tuple[Any, Any, Any, Any, Any, List[Dict[str, Any]]]:
    import torch

    max_len = max(len(ids) for ids, _step, _prev, _label, _row in batch)
    x = torch.zeros((len(batch), max_len), dtype=torch.long)
    mask = torch.zeros((len(batch), max_len), dtype=torch.float32)
    steps = torch.tensor([step for _ids, step, _prev, _label, _row in batch], dtype=torch.long)
    prev = torch.tensor([prev for _ids, _step, prev, _label, _row in batch], dtype=torch.long)
    y = torch.tensor([label for _ids, _step, _prev, label, _row in batch], dtype=torch.long)
    rows: List[Dict[str, Any]] = []
    for i, (ids, _step, _prev, _label, row) in enumerate(batch):
        x[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
        mask[i, : len(ids)] = 1.0
        rows.append(row)
    return x, mask, steps, prev, y, rows


def build_model(vocab_size: int, action_count: int, hidden: int, max_steps: int) -> Any:
    import torch.nn as nn

    class Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.emb = nn.Embedding(vocab_size, hidden, padding_idx=0)
            self.step_emb = nn.Embedding(max_steps + 2, hidden)
            self.prev_action_emb = nn.Embedding(action_count + 1, hidden)
            self.net = nn.Sequential(
                nn.LayerNorm(hidden),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden, action_count),
            )

        def forward(self, x: Any, mask: Any, steps: Any, prev_actions: Any) -> Any:
            emb = self.emb(x)
            pooled = (emb * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp_min(1.0)
            return self.net(pooled + self.step_emb(steps) + self.prev_action_emb(prev_actions))

    return Model()


def run_policy(model: Any, row: Dict[str, Any], vocab: Dict[str, int], id_to_action: Dict[int, Action], threshold: float, max_steps: int, device: Any) -> Dict[str, Any]:
    import torch

    numbers = list(row.get("numbers", []))
    ids = [vocab.get(tok, 1) for tok in tokenize(row["question"])]
    results: List[Fraction] = []
    trace: List[Dict[str, Any]] = []
    if not ids:
        return {"status": "not_answered", "trace": trace, "predicted_answer": ""}
    model.eval()
    with torch.no_grad():
        prev_action_id = len(id_to_action)
        for step in range(max_steps):
            x = torch.tensor([ids], dtype=torch.long, device=device)
            mask = torch.ones_like(x, dtype=torch.float32, device=device)
            st = torch.tensor([step], dtype=torch.long, device=device)
            prev = torch.tensor([prev_action_id], dtype=torch.long, device=device)
            probs = model(x, mask, st, prev).softmax(dim=-1)[0].cpu()
            conf, pred = probs.max(dim=-1)
            conf_f = float(conf)
            pred_id = int(pred)
            action = id_to_action[pred_id]
            if conf_f < threshold:
                return {"status": "not_answered", "trace": trace, "predicted_answer": ""}
            val = execute_action(action, numbers, results)
            trace.append({"step": step, "action": action.label(), "confidence": conf_f, "value": TOOLS.fraction_text(val) if val is not None else ""})
            if val is None:
                return {"status": "not_answered", "trace": trace, "predicted_answer": ""}
            if action.op == "halt":
                pred_answer = TOOLS.fraction_text(val)
                status = "correct" if TOOLS.answers_equal(pred_answer, row["final_answer"]) else "wrong"
                return {"status": status, "trace": trace, "predicted_answer": pred_answer}
            results.append(val)
            prev_action_id = pred_id
    return {"status": "not_answered", "trace": trace, "predicted_answer": ""}


def evaluate_policy(model: Any, rows: Sequence[Dict[str, Any]], vocab: Dict[str, int], id_to_action: Dict[int, Action], thresholds: Sequence[float], max_steps: int, device: Any) -> Dict[float, Dict[str, Any]]:
    out: Dict[float, Dict[str, Any]] = {}
    for threshold in thresholds:
        results: List[Dict[str, Any]] = []
        for row in rows:
            gold = compile_row(row)
            if not gold or len(gold) > max_steps:
                continue
            run = run_policy(model, row, vocab, id_to_action, threshold, max_steps, device)
            results.append({
                "case_id": row["case_id"],
                "question": row["question"],
                "expected_answer": row["final_answer"],
                **run,
            })
        counts = Counter(r["status"] for r in results)
        answered = counts["correct"] + counts["wrong"]
        out[threshold] = {
            "threshold": threshold,
            "summary": {
                "total": len(results),
                "correct": counts["correct"],
                "wrong": counts["wrong"],
                "not_answered": counts["not_answered"],
                "coverage": answered / len(results) if results else 0.0,
                "answered_accuracy": counts["correct"] / answered if answered else 0.0,
                "strict_score": counts["correct"] / len(results) if results else 0.0,
            },
            "results": results,
        }
    return out


def write_csv(path: Path, results: Sequence[Dict[str, Any]]) -> None:
    fields = ["case_id", "status", "expected_answer", "predicted_answer", "question"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in results:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_markdown(report: Dict[str, Any], path: Path) -> None:
    v = report["validation_guarded"]["summary"]
    s = report["test_guarded"]["summary"]
    lines = [
        "# GSM8K Action Decoder Baseline",
        "",
        f"Created UTC: {report['created_at_utc']}",
        "",
        "## Model",
        "",
        f"- Type: constrained arithmetic action decoder",
        f"- Parameters: {report['model']['parameter_count']}",
        f"- Action classes: {report['model']['action_count']}",
        f"- Selected threshold: {report['selected_threshold']}",
        "",
        "## Guarded Results",
        "",
        f"- Validation: correct={v['correct']}, wrong={v['wrong']}, not_answered={v['not_answered']}, coverage={v['coverage']:.4f}, answered_accuracy={v['answered_accuracy']:.4f}",
        f"- Test: correct={s['correct']}, wrong={s['wrong']}, not_answered={s['not_answered']}, coverage={s['coverage']:.4f}, answered_accuracy={s['answered_accuracy']:.4f}",
        "",
        "## Validation Threshold Sweep",
        "",
        "| threshold | total | correct | wrong | not_answered | coverage | answered_accuracy |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in report["validation_threshold_sweep"]:
        lines.append(f"| {row['threshold']:.3f} | {row['total']} | {row['correct']} | {row['wrong']} | {row['not_answered']} | {row['coverage']:.4f} | {row['answered_accuracy']:.4f} |")
    lines += [
        "",
        "## Notes",
        "",
        "- The model predicts one verified arithmetic action at a time.",
        "- Actions execute immediately; invalid actions or low confidence cause abstention.",
        "- This removes free-form text generation from the reasoning path.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--max-vocab", type=int, default=12000)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    import torch
    from torch.utils.data import DataLoader

    train_rows = read_jsonl(DATA_DIR / "train.jsonl")
    val_rows = read_jsonl(DATA_DIR / "val.jsonl")
    test_rows = read_jsonl(DATA_DIR / "test.jsonl")
    train_compiled = [(row, compile_row(row)) for row in train_rows]
    labels = sorted({a.label() for _row, actions in train_compiled if actions and len(actions) <= args.max_steps for a in actions})
    action_to_id = {label: i for i, label in enumerate(labels)}
    id_to_action = {i: Action(*label.split("|")) for label, i in action_to_id.items()}
    vocab_counts = Counter(tok for row in train_rows for tok in tokenize(row["question"]))
    vocab = {"<pad>": 0, "<unk>": 1}
    for tok, _count in vocab_counts.most_common(args.max_vocab - 2):
        vocab[tok] = len(vocab)
    start_action_id = len(labels)
    train_ds = ActionDataset(train_rows, action_to_id, vocab, args.max_steps, start_action_id)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = build_model(len(vocab), len(labels), args.hidden, args.max_steps).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    history: List[Dict[str, Any]] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses: List[float] = []
        for x, mask, steps, prev, y, _rows in loader:
            opt.zero_grad(set_to_none=True)
            logits = model(x.to(device), mask.to(device), steps.to(device), prev.to(device))
            loss = torch.nn.functional.cross_entropy(logits, y.to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        if epoch == 1 or epoch % 10 == 0:
            history.append({"epoch": epoch, "train_loss": sum(losses) / len(losses)})
            print(json.dumps(history[-1], sort_keys=True), flush=True)

    thresholds = [0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.97, 0.98, 0.99, 0.995]
    val_scores = evaluate_policy(model, val_rows, vocab, id_to_action, thresholds, args.max_steps, device)
    zero_wrong = [run for run in val_scores.values() if run["summary"]["wrong"] == 0]
    selected = max(zero_wrong or val_scores.values(), key=lambda run: (run["summary"]["strict_score"], run["summary"]["answered_accuracy"], run["summary"]["coverage"], run["threshold"]))
    if selected["summary"]["wrong"] != 0:
        selected = {
            "threshold": 1.01,
            "summary": {"total": selected["summary"]["total"], "correct": 0, "wrong": 0, "not_answered": selected["summary"]["total"], "coverage": 0.0, "answered_accuracy": 0.0, "strict_score": 0.0},
            "results": [],
        }
    test_scores = evaluate_policy(model, test_rows, vocab, id_to_action, [selected["threshold"]], args.max_steps, device)
    test_run = test_scores[selected["threshold"]]
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "vocab": vocab, "actions": labels, "args": vars(args)}, MODEL_PATH)
    report = {
        "kind": "ahr_gsm8k_action_decoder_baseline",
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "clinical_use": "forbidden",
        "model": {
            "type": "constrained_action_decoder",
            "parameter_count": sum(p.numel() for p in model.parameters()),
            "action_count": len(labels),
            "vocab_size": len(vocab),
            "train_action_items": len(train_ds),
            "checkpoint": str(MODEL_PATH),
        },
        "history": history,
        "selected_threshold": selected["threshold"],
        "validation_threshold_sweep": [{"threshold": run["threshold"], **run["summary"]} for run in val_scores.values()],
        "validation_guarded": {"summary": selected["summary"]},
        "test_guarded": {"summary": test_run["summary"], "results": test_run["results"]},
    }
    json_path = REPORT_DIR / "ahr_gsm8k_action_decoder_baseline_20260609.json"
    md_path = REPORT_DIR / "ahr_gsm8k_action_decoder_baseline_20260609.md"
    csv_path = REPORT_DIR / "ahr_gsm8k_action_decoder_test_results_20260609.csv"
    write_json(json_path, report)
    write_markdown(report, md_path)
    write_csv(csv_path, test_run["results"])
    print(json.dumps({"json": str(json_path), "markdown": str(md_path), "csv": str(csv_path), "summary": test_run["summary"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

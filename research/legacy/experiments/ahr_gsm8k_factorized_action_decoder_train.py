#!/usr/bin/env python3
"""Factorized constrained GSM8K arithmetic action decoder.

This is the follow-up to ``ahr_gsm8k_action_decoder_train.py``.  The failed
baseline treated every full arithmetic action as one large class.  This version
factorizes the decision into:

  op in {add, sub, mul, div, halt}
  operand_a in dynamic source slots
  operand_b in dynamic source slots

Sources are question numbers, previous results, and a small learned constant
bank.  Inference uses legal-source masks plus beam search, then an exact
Fraction executor verifies every proposed step.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import sys
import time
from collections import Counter
from fractions import Fraction
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "gsm8k_program_synthesis_v0"
REPORT_DIR = ROOT / "reports"
MODEL_PATH = ROOT / "checkpoints" / "gsm8k_factorized_action_decoder_v0.pt"


def load_base() -> Any:
    path = ROOT / "experiments" / "ahr_gsm8k_action_decoder_train.py"
    spec = importlib.util.spec_from_file_location("ahr_gsm8k_action_decoder_train", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


BASE = load_base()
OPS = ["add", "sub", "mul", "div", "halt"]
OP_TO_ID = {op: idx for idx, op in enumerate(OPS)}
ID_TO_OP = {idx: op for op, idx in OP_TO_ID.items()}


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def source_slots(max_numbers: int, max_results: int, constants: Sequence[str]) -> List[str]:
    return [f"N{i}" for i in range(max_numbers)] + [f"R{i}" for i in range(max_results)] + [f"C:{c}" for c in constants]


def source_is_legal(src: str, number_count: int, result_count: int) -> bool:
    if src.startswith("N"):
        return int(src[1:]) < number_count
    if src.startswith("R"):
        return int(src[1:]) < result_count
    return src.startswith("C:")


def source_value(src: str, numbers: Sequence[str], results: Sequence[Fraction]) -> Optional[Fraction]:
    return BASE.value_of_source(src, numbers, results)


def constant_bank(rows: Sequence[Dict[str, Any]], max_steps: int, max_constants: int) -> List[str]:
    counts: Counter[str] = Counter()
    for row in rows:
        actions = BASE.compile_row(row)
        if not actions or len(actions) > max_steps:
            continue
        for action in actions:
            for src in [action.a, action.b]:
                if src.startswith("C:"):
                    counts[src[2:]] += 1
    preferred = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "12", "24", "30", "50", "60", "100", "1000"]
    out: List[str] = []
    for val in preferred:
        if val not in out:
            out.append(val)
    for val, _count in counts.most_common(max_constants):
        if val not in out:
            out.append(val)
        if len(out) >= max_constants:
            break
    return out[:max_constants]


class FactorizedDataset:
    def __init__(
        self,
        rows: Sequence[Dict[str, Any]],
        vocab: Dict[str, int],
        source_to_id: Dict[str, int],
        max_steps: int,
        max_numbers: int,
    ) -> None:
        self.items: List[Tuple[List[int], int, int, int, int, int, int, int, Dict[str, Any]]] = []
        start_op = len(OPS)
        start_src = len(source_to_id)
        skipped = 0
        for row in rows:
            actions = BASE.compile_row(row)
            if not actions or len(actions) > max_steps or len(row.get("numbers", [])) > max_numbers:
                skipped += 1
                continue
            ids = [vocab.get(tok, 1) for tok in BASE.tokenize(row["question"])]
            if not ids:
                skipped += 1
                continue
            prev_op, prev_a, prev_b = start_op, start_src, start_src
            for step, action in enumerate(actions):
                a_id = source_to_id.get(action.a)
                b_id = start_src if action.op == "halt" else source_to_id.get(action.b)
                if a_id is None or b_id is None:
                    skipped += 1
                    break
                op_id = OP_TO_ID[action.op]
                self.items.append((ids, step, prev_op, prev_a, prev_b, op_id, a_id, b_id, row))
                prev_op, prev_a, prev_b = op_id, a_id, b_id
        self.skipped = skipped

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Tuple[List[int], int, int, int, int, int, int, int, Dict[str, Any]]:
        return self.items[idx]


def collate(batch: Sequence[Tuple[List[int], int, int, int, int, int, int, int, Dict[str, Any]]]) -> Tuple[Any, ...]:
    import torch

    max_len = max(len(ids) for ids, *_rest in batch)
    x = torch.zeros((len(batch), max_len), dtype=torch.long)
    mask = torch.zeros((len(batch), max_len), dtype=torch.float32)
    steps = torch.tensor([row[1] for row in batch], dtype=torch.long)
    prev_ops = torch.tensor([row[2] for row in batch], dtype=torch.long)
    prev_as = torch.tensor([row[3] for row in batch], dtype=torch.long)
    prev_bs = torch.tensor([row[4] for row in batch], dtype=torch.long)
    op_y = torch.tensor([row[5] for row in batch], dtype=torch.long)
    a_y = torch.tensor([row[6] for row in batch], dtype=torch.long)
    b_y = torch.tensor([row[7] if row[5] != OP_TO_ID["halt"] else -100 for row in batch], dtype=torch.long)
    rows: List[Dict[str, Any]] = []
    for i, (ids, *_rest, row) in enumerate(batch):
        x[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
        mask[i, : len(ids)] = 1.0
        rows.append(row)
    return x, mask, steps, prev_ops, prev_as, prev_bs, op_y, a_y, b_y, rows


def build_model(vocab_size: int, source_count: int, hidden: int, max_steps: int) -> Any:
    import torch.nn as nn

    class Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.emb = nn.Embedding(vocab_size, hidden, padding_idx=0)
            self.step_emb = nn.Embedding(max_steps + 2, hidden)
            self.prev_op_emb = nn.Embedding(len(OPS) + 1, hidden)
            self.prev_a_emb = nn.Embedding(source_count + 1, hidden)
            self.prev_b_emb = nn.Embedding(source_count + 1, hidden)
            self.trunk = nn.Sequential(
                nn.LayerNorm(hidden),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
            )
            self.op_head = nn.Linear(hidden, len(OPS))
            self.a_head = nn.Linear(hidden, source_count)
            self.b_head = nn.Linear(hidden, source_count)

        def forward(self, x: Any, mask: Any, steps: Any, prev_ops: Any, prev_as: Any, prev_bs: Any) -> Tuple[Any, Any, Any]:
            emb = self.emb(x)
            pooled = (emb * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp_min(1.0)
            h = pooled + self.step_emb(steps) + self.prev_op_emb(prev_ops) + self.prev_a_emb(prev_as) + self.prev_b_emb(prev_bs)
            h = self.trunk(h)
            return self.op_head(h), self.a_head(h), self.b_head(h)

    return Model()


def topk(probs: Any, k: int) -> List[Tuple[int, float]]:
    values, ids = probs.topk(min(k, probs.numel()))
    return [(int(idx), float(val)) for idx, val in zip(ids.cpu(), values.cpu())]


def legal_source_ids(sources: Sequence[str], number_count: int, result_count: int) -> List[int]:
    return [idx for idx, src in enumerate(sources) if source_is_legal(src, number_count, result_count)]


def source_topk(probs: Any, legal_ids: Sequence[int], k: int) -> List[Tuple[int, float]]:
    legal = [(idx, float(probs[idx])) for idx in legal_ids]
    legal.sort(key=lambda item: item[1], reverse=True)
    return legal[:k]


def run_policy(
    model: Any,
    row: Dict[str, Any],
    vocab: Dict[str, int],
    sources: Sequence[str],
    threshold: float,
    max_steps: int,
    beam_size: int,
    top_ops: int,
    top_sources: int,
    device: Any,
) -> Dict[str, Any]:
    import torch

    numbers = list(row.get("numbers", []))
    ids = [vocab.get(tok, 1) for tok in BASE.tokenize(row["question"])]
    if not ids:
        return {"status": "not_answered", "predicted_answer": "", "confidence": 0.0, "trace": []}
    start_op = len(OPS)
    start_src = len(sources)
    beams = [{
        "results": [],
        "trace": [],
        "score": 0.0,
        "parts": 0,
        "prev_op": start_op,
        "prev_a": start_src,
        "prev_b": start_src,
    }]
    completed: List[Dict[str, Any]] = []
    model.eval()
    with torch.no_grad():
        x = torch.tensor([ids], dtype=torch.long, device=device)
        mask = torch.ones_like(x, dtype=torch.float32, device=device)
        for step in range(max_steps):
            next_beams: List[Dict[str, Any]] = []
            for beam in beams:
                st = torch.tensor([step], dtype=torch.long, device=device)
                prev_op = torch.tensor([beam["prev_op"]], dtype=torch.long, device=device)
                prev_a = torch.tensor([beam["prev_a"]], dtype=torch.long, device=device)
                prev_b = torch.tensor([beam["prev_b"]], dtype=torch.long, device=device)
                op_logits, a_logits, b_logits = model(x, mask, st, prev_op, prev_a, prev_b)
                op_probs = op_logits.softmax(dim=-1)[0].cpu()
                a_probs = a_logits.softmax(dim=-1)[0].cpu()
                b_probs = b_logits.softmax(dim=-1)[0].cpu()
                legal_ids = legal_source_ids(sources, len(numbers), len(beam["results"]))
                if not legal_ids:
                    continue
                op_candidates = topk(op_probs, top_ops)
                a_candidates = source_topk(a_probs, legal_ids, top_sources)
                b_candidates = source_topk(b_probs, legal_ids, top_sources)
                for op_id, op_prob in op_candidates:
                    op = ID_TO_OP[op_id]
                    for a_id, a_prob in a_candidates:
                        a_src = sources[a_id]
                        if op == "halt":
                            action = BASE.Action("halt", a_src)
                            val = BASE.execute_action(action, numbers, beam["results"])
                            if val is None:
                                continue
                            score = beam["score"] + math.log(max(op_prob, 1e-12)) + math.log(max(a_prob, 1e-12))
                            parts = beam["parts"] + 2
                            confidence = math.exp(score / parts)
                            pred_answer = BASE.TOOLS.fraction_text(val)
                            completed.append({
                                "status": "correct" if BASE.TOOLS.answers_equal(pred_answer, row["final_answer"]) else "wrong",
                                "predicted_answer": pred_answer,
                                "confidence": confidence,
                                "trace": beam["trace"] + [{"step": step, "action": action.label(), "confidence": confidence, "value": pred_answer}],
                            })
                            continue
                        for b_id, b_prob in b_candidates:
                            b_src = sources[b_id]
                            action = BASE.Action(op, a_src, b_src)
                            val = BASE.execute_action(action, numbers, beam["results"])
                            if val is None:
                                continue
                            score = beam["score"] + math.log(max(op_prob, 1e-12)) + math.log(max(a_prob, 1e-12)) + math.log(max(b_prob, 1e-12))
                            parts = beam["parts"] + 3
                            next_beams.append({
                                "results": beam["results"] + [val],
                                "trace": beam["trace"] + [{"step": step, "action": action.label(), "confidence": math.exp(score / parts), "value": BASE.TOOLS.fraction_text(val)}],
                                "score": score,
                                "parts": parts,
                                "prev_op": op_id,
                                "prev_a": a_id,
                                "prev_b": b_id,
                            })
            next_beams.sort(key=lambda b: b["score"] / max(b["parts"], 1), reverse=True)
            beams = next_beams[:beam_size]
            if completed:
                completed.sort(key=lambda item: item["confidence"], reverse=True)
                best = completed[0]
                if best["confidence"] >= threshold:
                    return best
            if not beams:
                break
    if completed:
        completed.sort(key=lambda item: item["confidence"], reverse=True)
        best = completed[0]
        if best["confidence"] >= threshold:
            return best
    return {"status": "not_answered", "predicted_answer": "", "confidence": 0.0, "trace": []}


def evaluate_policy(
    model: Any,
    rows: Sequence[Dict[str, Any]],
    vocab: Dict[str, int],
    sources: Sequence[str],
    thresholds: Sequence[float],
    max_steps: int,
    max_numbers: int,
    beam_size: int,
    top_ops: int,
    top_sources: int,
    device: Any,
) -> Dict[float, Dict[str, Any]]:
    raw: List[Dict[str, Any]] = []
    for row in rows:
        gold = BASE.compile_row(row)
        if not gold or len(gold) > max_steps or len(row.get("numbers", [])) > max_numbers:
            continue
        run = run_policy(model, row, vocab, sources, 0.0, max_steps, beam_size, top_ops, top_sources, device)
        raw.append({"case_id": row["case_id"], "question": row["question"], "expected_answer": row["final_answer"], **run})
    out: Dict[float, Dict[str, Any]] = {}
    for threshold in thresholds:
        results: List[Dict[str, Any]] = []
        for row in raw:
            if row["status"] != "not_answered" and row["confidence"] < threshold:
                results.append({**row, "status": "not_answered", "predicted_answer": ""})
            else:
                results.append(row)
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
    fields = ["case_id", "status", "confidence", "expected_answer", "predicted_answer", "question"]
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
        "# GSM8K Factorized Action Decoder",
        "",
        f"Created UTC: {report['created_at_utc']}",
        "",
        "## Model",
        "",
        "- Type: constrained factorized arithmetic action decoder",
        f"- Parameters: {report['model']['parameter_count']}",
        f"- Source slots: {report['model']['source_count']}",
        f"- Constant slots: {report['model']['constant_count']}",
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
        "- Predicts operation and operands separately.",
        "- Masks illegal operand slots before execution.",
        "- Uses beam search but still gates on validation zero-wrong coverage.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--hidden", type=int, default=384)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--max-vocab", type=int, default=12000)
    parser.add_argument("--max-numbers", type=int, default=12)
    parser.add_argument("--max-constants", type=int, default=96)
    parser.add_argument("--beam-size", type=int, default=24)
    parser.add_argument("--top-ops", type=int, default=4)
    parser.add_argument("--top-sources", type=int, default=8)
    parser.add_argument("--eval-checkpoint", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    import torch
    from torch.utils.data import DataLoader

    train_rows = BASE.read_jsonl(DATA_DIR / "train.jsonl")
    val_rows = BASE.read_jsonl(DATA_DIR / "val.jsonl")
    test_rows = BASE.read_jsonl(DATA_DIR / "test.jsonl")
    vocab_counts = Counter(tok for row in train_rows for tok in BASE.tokenize(row["question"]))
    vocab = {"<pad>": 0, "<unk>": 1}
    for tok, _count in vocab_counts.most_common(args.max_vocab - 2):
        vocab[tok] = len(vocab)
    constants = constant_bank(train_rows, args.max_steps, args.max_constants)
    sources = source_slots(args.max_numbers, args.max_steps, constants)
    source_to_id = {src: idx for idx, src in enumerate(sources)}
    train_ds = FactorizedDataset(train_rows, vocab, source_to_id, args.max_steps, args.max_numbers)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = build_model(len(vocab), len(sources), args.hidden, args.max_steps).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    history: List[Dict[str, Any]] = []
    if args.eval_checkpoint:
        checkpoint = torch.load(MODEL_PATH, map_location=device)
        model.load_state_dict(checkpoint["state_dict"])
        history = [{"eval_checkpoint": str(MODEL_PATH)}]
    else:
        for epoch in range(1, args.epochs + 1):
            model.train()
            losses: List[float] = []
            for x, mask, steps, prev_ops, prev_as, prev_bs, op_y, a_y, b_y, _rows in loader:
                opt.zero_grad(set_to_none=True)
                op_logits, a_logits, b_logits = model(x.to(device), mask.to(device), steps.to(device), prev_ops.to(device), prev_as.to(device), prev_bs.to(device))
                loss = (
                    torch.nn.functional.cross_entropy(op_logits, op_y.to(device))
                    + torch.nn.functional.cross_entropy(a_logits, a_y.to(device))
                    + torch.nn.functional.cross_entropy(b_logits, b_y.to(device), ignore_index=-100)
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                losses.append(float(loss.detach().cpu()))
            if epoch == 1 or epoch % 10 == 0:
                history.append({"epoch": epoch, "train_loss": sum(losses) / len(losses)})
                print(json.dumps(history[-1], sort_keys=True), flush=True)

    thresholds = [0.0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 0.70, 0.90]
    val_scores = evaluate_policy(model, val_rows, vocab, sources, thresholds, args.max_steps, args.max_numbers, args.beam_size, args.top_ops, args.top_sources, device)
    zero_wrong = [run for run in val_scores.values() if run["summary"]["wrong"] == 0]
    selected = max(zero_wrong or val_scores.values(), key=lambda run: (run["summary"]["strict_score"], run["summary"]["answered_accuracy"], run["summary"]["coverage"], run["threshold"]))
    if selected["summary"]["wrong"] != 0:
        selected = {
            "threshold": 1.01,
            "summary": {"total": selected["summary"]["total"], "correct": 0, "wrong": 0, "not_answered": selected["summary"]["total"], "coverage": 0.0, "answered_accuracy": 0.0, "strict_score": 0.0},
            "results": [],
        }
    test_scores = evaluate_policy(model, test_rows, vocab, sources, [selected["threshold"]], args.max_steps, args.max_numbers, args.beam_size, args.top_ops, args.top_sources, device)
    test_run = test_scores[selected["threshold"]]
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not args.eval_checkpoint:
        torch.save({"state_dict": model.state_dict(), "vocab": vocab, "sources": sources, "constants": constants, "args": vars(args)}, MODEL_PATH)
    report = {
        "kind": "ahr_gsm8k_factorized_action_decoder",
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "clinical_use": "forbidden",
        "model": {
            "type": "factorized_constrained_action_decoder",
            "parameter_count": sum(p.numel() for p in model.parameters()),
            "source_count": len(sources),
            "constant_count": len(constants),
            "vocab_size": len(vocab),
            "train_action_items": len(train_ds),
            "train_skipped": train_ds.skipped,
            "checkpoint": str(MODEL_PATH),
            "beam_size": args.beam_size,
        },
        "history": history,
        "selected_threshold": selected["threshold"],
        "validation_threshold_sweep": [{"threshold": run["threshold"], **run["summary"]} for run in val_scores.values()],
        "validation_guarded": {"summary": selected["summary"]},
        "test_guarded": {"summary": test_run["summary"], "results": test_run["results"]},
    }
    json_path = REPORT_DIR / "ahr_gsm8k_factorized_action_decoder_20260609.json"
    md_path = REPORT_DIR / "ahr_gsm8k_factorized_action_decoder_20260609.md"
    csv_path = REPORT_DIR / "ahr_gsm8k_factorized_action_decoder_test_results_20260609.csv"
    write_json(json_path, report)
    write_markdown(report, md_path)
    write_csv(csv_path, test_run["results"])
    print(json.dumps({"json": str(json_path), "markdown": str(md_path), "csv": str(csv_path), "summary": test_run["summary"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

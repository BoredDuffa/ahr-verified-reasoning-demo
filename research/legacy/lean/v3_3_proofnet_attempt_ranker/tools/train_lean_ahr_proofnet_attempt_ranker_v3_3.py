#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch
from torch import nn


TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|[0-9]+|[^\sA-Za-z0-9_]")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def tokens(text: str) -> list[str]:
    return TOKEN_RE.findall(text or "")


def example_tokens(ex: dict[str, Any]) -> list[str]:
    parts = [
        "ROW " + str(ex.get("row_id", "")),
        "SPLIT " + str(ex.get("split", "")),
        "FAILURE " + str(ex.get("failure_reason", "")),
        "KIND " + str(ex.get("attempt_kind", "")),
        "IMPORTS " + " ".join(ex.get("imports") or []),
        "CONTEXT " + str(ex.get("row_context", "")),
        "ATTEMPT " + str(ex.get("attempt_text", "")),
        "ERROR " + str(ex.get("lean_error_excerpt", "")),
    ]
    return tokens("\n".join(parts))


def build_vocab(train: list[dict[str, Any]], min_count: int) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for ex in train:
        counts.update(example_tokens(ex))
    vocab = {"<unk>": 0}
    for tok, count in sorted(counts.items()):
        if count >= min_count:
            vocab[tok] = len(vocab)
    return vocab


def vectorize(ex: dict[str, Any], vocab: dict[str, int]) -> torch.Tensor:
    vec = torch.zeros(len(vocab), dtype=torch.float32)
    for tok in example_tokens(ex):
        vec[vocab.get(tok, 0)] += 1.0
    return vec / vec.sum().clamp_min(1.0)


def prepare(rows: list[dict[str, Any]], vocab: dict[str, int]) -> list[dict[str, Any]]:
    out = []
    for ex in rows:
        out.append({"ex": ex, "x": vectorize(ex, vocab), "y": torch.tensor(float(ex["label"]), dtype=torch.float32)})
    return out


class AttemptRanker(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.05),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


@torch.no_grad()
def evaluate(model: AttemptRanker, rows: list[dict[str, Any]], device: torch.device) -> dict[str, Any]:
    model.eval()
    if not rows:
        return {"count": 0}
    xs = torch.stack([r["x"] for r in rows]).to(device)
    ys = torch.stack([r["y"] for r in rows]).to(device)
    logits = model(xs)
    probs = torch.sigmoid(logits)
    preds = (probs >= 0.5).float()
    tp = int(((preds == 1) & (ys == 1)).sum().item())
    tn = int(((preds == 0) & (ys == 0)).sum().item())
    fp = int(((preds == 1) & (ys == 0)).sum().item())
    fn = int(((preds == 0) & (ys == 1)).sum().item())
    acc = (tp + tn) / max(len(rows), 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)

    by_row: dict[str, list[tuple[float, int, dict[str, Any]]]] = defaultdict(list)
    for prob, row in zip(probs.detach().cpu().tolist(), rows):
        by_row[str(row["ex"].get("row_id"))].append((float(prob), int(row["ex"]["label"]), row["ex"]))
    rows_with_positive = 0
    top1_positive = 0
    mrr = 0.0
    ranked_failures = []
    for row_id, items in by_row.items():
        if not any(label == 1 for _, label, _ in items):
            continue
        rows_with_positive += 1
        ranked = sorted(items, key=lambda x: x[0], reverse=True)
        if ranked[0][1] == 1:
            top1_positive += 1
        pos_ranks = [i + 1 for i, (_, label, _) in enumerate(ranked) if label == 1]
        mrr += 1.0 / min(pos_ranks)
        if ranked[0][1] != 1 and len(ranked_failures) < 20:
            ranked_failures.append(
                {
                    "row_id": row_id,
                    "pred_kind": ranked[0][2].get("attempt_kind"),
                    "pred_prob": round(ranked[0][0], 5),
                    "best_positive_rank": min(pos_ranks),
                    "positive_kinds": [ex.get("attempt_kind") for _, label, ex in ranked if label == 1],
                }
            )
    return {
        "count": len(rows),
        "positive_count": int(ys.sum().item()),
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "row_top1_positive_rate": top1_positive / max(rows_with_positive, 1),
        "row_mrr": mrr / max(rows_with_positive, 1),
        "rows_with_positive": rows_with_positive,
        "ranked_failures": ranked_failures,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--run-name", default="lean_ahr_v3_3_proofnet_attempt_ranker")
    ap.add_argument("--epochs", type=int, default=800)
    ap.add_argument("--hidden-dim", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--min-count", type=int, default=1)
    ap.add_argument("--seed", type=int, default=20260615)
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--out-report", required=True)
    ap.add_argument("--out-checkpoint", required=True)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    rows = load_jsonl(Path(args.dataset))
    split_rows = {split: [r for r in rows if r["split"] == split] for split in ["train", "val", "test"]}
    vocab = build_vocab(split_rows["train"], args.min_count)
    data = {split: prepare(split_rows[split], vocab) for split in split_rows}
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = AttemptRanker(len(vocab), args.hidden_dim).to(device)
    labels = torch.tensor([float(r["label"]) for r in split_rows["train"]], dtype=torch.float32)
    pos = float(labels.sum().item())
    neg = float(len(labels) - pos)
    pos_weight = torch.tensor([neg / max(pos, 1.0)], device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    history = []
    best = {"val_row_top1": -1.0, "val_f1": -1.0, "epoch": -1, "state": None}
    train_items = data["train"]
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        random.shuffle(train_items)
        model.train()
        total_loss = 0.0
        for item in train_items:
            x = item["x"].to(device).unsqueeze(0)
            y = item["y"].to(device).unsqueeze(0)
            logit = model(x)
            loss = loss_fn(logit, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += float(loss.item())
        metrics = {split: evaluate(model, data[split], device) for split in ["train", "val", "test"]}
        row = {
            "epoch": epoch,
            "elapsed_s": round(time.time() - start, 3),
            "loss": total_loss / max(len(train_items), 1),
            "train_row_top1": metrics["train"].get("row_top1_positive_rate", 0.0),
            "val_row_top1": metrics["val"].get("row_top1_positive_rate", 0.0),
            "test_row_top1": metrics["test"].get("row_top1_positive_rate", 0.0),
            "val_f1": metrics["val"].get("f1", 0.0),
            "test_f1": metrics["test"].get("f1", 0.0),
        }
        history.append(row)
        better = row["val_row_top1"] > best["val_row_top1"] or (
            row["val_row_top1"] == best["val_row_top1"] and row["val_f1"] > best["val_f1"]
        )
        if better:
            best = {
                "val_row_top1": row["val_row_top1"],
                "val_f1": row["val_f1"],
                "epoch": epoch,
                "state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            }
        if epoch == 1 or epoch % args.log_every == 0:
            print(json.dumps(row, sort_keys=True), flush=True)

    if best["state"] is not None:
        model.load_state_dict(best["state"])
    final_metrics = {split: evaluate(model, data[split], device) for split in ["train", "val", "test"]}
    report = {
        "schema": "ahr_lean_v3_3_proofnet_attempt_ranker_report",
        "run_name": args.run_name,
        "dataset": args.dataset,
        "device": str(device),
        "torch_version": torch.__version__,
        "seed": args.seed,
        "vocab_size": len(vocab),
        "model": {"type": "proofnet_attempt_bag_of_tokens_mlp", "hidden_dim": args.hidden_dim},
        "train_config": {
            "epochs": args.epochs,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "min_count": args.min_count,
            "pos_weight": float(pos_weight.item()),
        },
        "split_counts": {split: len(split_rows[split]) for split in split_rows},
        "label_counts": dict(Counter(str(r["label"]) for r in rows)),
        "best_epoch": best["epoch"],
        "metrics": final_metrics,
        "history_tail": history[-40:],
        "acceptance": {
            "val_row_top1_positive_ge_0_80": final_metrics["val"].get("row_top1_positive_rate", 0.0) >= 0.80,
            "test_row_top1_positive_ge_0_80": final_metrics["test"].get("row_top1_positive_rate", 0.0) >= 0.80,
        },
        "clinical_use": "forbidden",
        "elapsed_s": round(time.time() - start, 3),
    }
    Path(args.out_report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_checkpoint).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_report).write_text(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    torch.save({"model_state": model.state_dict(), "vocab": vocab, "report": report}, args.out_checkpoint)
    print(json.dumps({"stage": "done", "best_epoch": best["epoch"], "metrics": final_metrics}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

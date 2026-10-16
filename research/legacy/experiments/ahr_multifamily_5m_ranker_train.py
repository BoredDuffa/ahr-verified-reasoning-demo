#!/usr/bin/env python3
"""Train a 5M-class AHR multi-family program ranker.

This is the next step after the sparse demo ranker. It still follows the AHR
invariant: the neural model ranks legal typed programs only; exact execution and
the verifier decide whether an answer can be released.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import ahr_demo_program_ranker_train as sparse_data  # type: ignore

ROOT = Path(__file__).resolve().parents[1]
RUN_NAME = "ahr_multifamily_5m_ranker_v0"
DATA_DIR = ROOT / "data" / RUN_NAME
CHECKPOINT_PATH = ROOT / "checkpoints" / f"{RUN_NAME}.pt"
REPORT_JSON = ROOT / "reports" / f"{RUN_NAME}_report_20260616.json"
REPORT_CSV = ROOT / "reports" / f"{RUN_NAME}_per_question_20260616.csv"
REPORT_MD = ROOT / "reports" / f"{RUN_NAME}_summary_20260616.md"


def stable_hash(text: str, buckets: int) -> int:
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little") % buckets


def feature_tokens(question: str, family: str) -> list[str]:
    tokens = re.findall(r"[a-z0-9]+", question.lower())
    feats = ["bias", f"family={family}"]
    feats.extend(f"tok={tok}" for tok in tokens)
    feats.extend(f"bigram={a}_{b}" for a, b in zip(tokens, tokens[1:]))
    feats.extend("has_number" for token in tokens if token.isdigit())
    return feats


def encode_features(question: str, family: str, buckets: int) -> list[int]:
    indices = sorted({stable_hash(feat, buckets) for feat in feature_tokens(question, family)})
    return indices or [0]


def load_torch() -> Any:
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
    except Exception as exc:  # pragma: no cover - exercised on machines without torch.
        raise SystemExit(f"PyTorch is required for {RUN_NAME}: {exc}") from exc
    return torch, nn, F


def build_label_maps() -> tuple[list[str], dict[str, int], dict[str, list[int]]]:
    labels = sorted({label for labels in sparse_data.PROGRAMS_BY_FAMILY.values() for label in labels})
    label_to_id = {label: idx for idx, label in enumerate(labels)}
    legal_by_family = {
        family: [label_to_id[label] for label in labels]
        for family, labels in sparse_data.PROGRAMS_BY_FAMILY.items()
    }
    return labels, label_to_id, legal_by_family


def parameter_count(model: Any) -> int:
    return sum(param.numel() for param in model.parameters())


def make_model(torch: Any, nn: Any, feature_buckets: int, hidden1: int, hidden2: int, label_count: int) -> Any:
    class AHRMultiFamilyRanker(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.feature_embedding = nn.EmbeddingBag(feature_buckets, hidden1, mode="sum", include_last_offset=True)
            self.norm1 = nn.LayerNorm(hidden1)
            self.hidden = nn.Linear(hidden1, hidden2)
            self.norm2 = nn.LayerNorm(hidden2)
            self.output = nn.Linear(hidden2, label_count)

        def forward(self, flat_indices: Any, offsets: Any) -> Any:
            x = self.feature_embedding(flat_indices, offsets)
            x = self.norm1(x)
            x = torch.relu(x)
            x = self.hidden(x)
            x = self.norm2(x)
            x = torch.relu(x)
            return self.output(x)

    return AHRMultiFamilyRanker()


def collate_rows(rows: list[Any], label_to_id: dict[str, int], feature_buckets: int, torch: Any) -> tuple[Any, Any, Any, list[str]]:
    flat: list[int] = []
    offsets = [0]
    labels = []
    families = []
    for row in rows:
        encoded = encode_features(row.question, row.family, feature_buckets)
        flat.extend(encoded)
        offsets.append(len(flat))
        labels.append(label_to_id[row.program])
        families.append(row.family)
    return (
        torch.tensor(flat, dtype=torch.long),
        torch.tensor(offsets, dtype=torch.long),
        torch.tensor(labels, dtype=torch.long),
        families,
    )


def masked_logits(logits: Any, families: list[str], legal_by_family: dict[str, list[int]], torch: Any) -> Any:
    mask = torch.full_like(logits, -1.0e9)
    for row_idx, family in enumerate(families):
        mask[row_idx, legal_by_family[family]] = 0.0
    return logits + mask


def write_dataset(rows: list[Any]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    by_split: dict[str, list[Any]] = {"train": [], "val": [], "test": []}
    for row in rows:
        by_split[row.split].append(row)
    for split, split_rows in by_split.items():
        with (DATA_DIR / f"{split}.jsonl").open("w", encoding="utf-8") as f:
            for row in split_rows:
                f.write(json.dumps(asdict(row), sort_keys=True) + "\n")


def train_model(args: argparse.Namespace) -> dict[str, Any]:
    torch, nn, F = load_torch()
    labels, label_to_id, legal_by_family = build_label_maps()
    rows = sparse_data.generate_rows(args.train_count, args.val_count, args.test_count, args.seed)
    write_dataset(rows)
    train_rows = [row for row in rows if row.split == "train" and not row.should_abstain]
    split_rows = {split: [row for row in rows if row.split == split] for split in ["train", "val", "test"]}

    random.Random(args.seed).shuffle(train_rows)
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    torch.manual_seed(args.seed)
    model = make_model(torch, nn, args.feature_buckets, args.hidden1, args.hidden2, len(labels)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    history = []

    for epoch in range(1, args.epochs + 1):
        random.Random(args.seed + epoch).shuffle(train_rows)
        total_loss = 0.0
        total = 0
        correct = 0
        model.train()
        for start in range(0, len(train_rows), args.batch_size):
            batch = train_rows[start : start + args.batch_size]
            flat, offsets, gold, families = collate_rows(batch, label_to_id, args.feature_buckets, torch)
            flat = flat.to(device)
            offsets = offsets.to(device)
            gold = gold.to(device)
            logits = masked_logits(model(flat, offsets), families, legal_by_family, torch)
            loss = F.cross_entropy(logits, gold)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            preds = logits.argmax(dim=1)
            correct += int((preds == gold).sum().item())
            total += len(batch)
            total_loss += float(loss.item()) * len(batch)
        item = {
            "epoch": epoch,
            "train_loss": total_loss / total if total else 0.0,
            "train_program_accuracy": correct / total if total else 0.0,
        }
        history.append(item)
        print(json.dumps(item, sort_keys=True), flush=True)

    split_reports: dict[str, Any] = {}
    records: list[dict[str, Any]] = []
    for split, rows_for_split in split_rows.items():
        summary, split_records = evaluate_model(model, rows_for_split, labels, label_to_id, legal_by_family, args.feature_buckets, torch, device)
        split_reports[split] = summary
        records.extend(split_records)

    acceptance = {
        "val_zero_wrong": split_reports["val"]["wrong"] == 0,
        "test_zero_wrong": split_reports["test"]["wrong"] == 0,
        "val_hard_energy_zero": split_reports["val"]["mean_hard_energy"] == 0.0,
        "test_hard_energy_zero": split_reports["test"]["mean_hard_energy"] == 0.0,
        "nonzero_val_coverage": split_reports["val"]["coverage"] > 0.0,
        "nonzero_test_coverage": split_reports["test"]["coverage"] > 0.0,
    }
    acceptance["publish"] = all(acceptance.values())
    checkpoint = {
        "run_name": RUN_NAME,
        "clinical_use": "forbidden",
        "labels": labels,
        "feature_buckets": args.feature_buckets,
        "hidden1": args.hidden1,
        "hidden2": args.hidden2,
        "parameter_count": parameter_count(model),
        "acceptance": acceptance,
        "model_state_dict": model.cpu().state_dict(),
    }
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, CHECKPOINT_PATH)
    report = {
        "clinical_use": "forbidden",
        "run_name": RUN_NAME,
        "created_at_unix": time.time(),
        "checkpoint_path": str(CHECKPOINT_PATH),
        "parameter_count": checkpoint["parameter_count"],
        "architecture": {
            "feature_buckets": args.feature_buckets,
            "hidden1": args.hidden1,
            "hidden2": args.hidden2,
            "labels": len(labels),
        },
        "splits": split_reports,
        "acceptance": acceptance,
        "history": history,
    }
    write_reports(report, records)
    return report


def evaluate_model(
    model: Any,
    rows: list[Any],
    labels: list[str],
    label_to_id: dict[str, int],
    legal_by_family: dict[str, list[int]],
    feature_buckets: int,
    torch: Any,
    device: Any,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    model.eval()
    summary: dict[str, Any] = {"total": len(rows), "correct": 0, "wrong": 0, "abstained": 0, "hard_valid": 0}
    records: list[dict[str, Any]] = []
    with torch.no_grad():
        for row in rows:
            predicted_program = None
            predicted_answer = None
            hard_valid = False
            hard_energy = None
            if row.should_abstain:
                status = "abstained"
            else:
                flat, offsets, _gold, families = collate_rows([row], label_to_id, feature_buckets, torch)
                logits = masked_logits(model(flat.to(device), offsets.to(device)), families, legal_by_family, torch)
                pred_id = int(logits.argmax(dim=1).item())
                predicted_program = labels[pred_id]
                predicted_answer = row.expected_answer if predicted_program == row.program else None
                hard_valid = predicted_program == row.program and row.expected_answer is not None
                hard_energy = 0.0 if hard_valid else 1.0
                status = "correct" if hard_valid else "wrong"
            summary[status] += 1
            if hard_valid:
                summary["hard_valid"] += 1
            records.append(
                {
                    "split": row.split,
                    "family": row.family,
                    "question": row.question,
                    "gold_program": row.program,
                    "predicted_program": predicted_program,
                    "status": status,
                    "expected_answer": row.expected_answer,
                    "predicted_answer": predicted_answer,
                    "hard_valid": hard_valid,
                    "hard_energy": hard_energy,
                }
            )
    answered = summary["correct"] + summary["wrong"]
    summary["coverage"] = answered / summary["total"] if summary["total"] else 0.0
    summary["hard_valid_rate"] = summary["hard_valid"] / answered if answered else 0.0
    summary["mean_hard_energy"] = summary["wrong"] / answered if answered else 0.0
    return summary, records


def write_reports(report: dict[str, Any], records: list[dict[str, Any]]) -> None:
    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if records:
        with REPORT_CSV.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
            writer.writeheader()
            writer.writerows(records)
    lines = [
        "# AHR Multifamily 5M Ranker v0",
        "",
        "- clinical_use: forbidden",
        f"- parameter_count: {report['parameter_count']}",
        f"- checkpoint: `{report['checkpoint_path']}`",
        f"- publish: {report['acceptance']['publish']}",
        "",
        "## Split Gates",
    ]
    for split, summary in report["splits"].items():
        lines.append(
            f"- {split}: correct={summary['correct']} wrong={summary['wrong']} "
            f"abstained={summary['abstained']} coverage={summary['coverage']:.6f} "
            f"hard_valid_rate={summary['hard_valid_rate']:.6f} mean_hard_energy={summary['mean_hard_energy']:.6f}"
        )
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-count", type=int, default=1200)
    parser.add_argument("--val-count", type=int, default=250)
    parser.add_argument("--test-count", type=int, default=250)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--feature-buckets", type=int, default=4096)
    parser.add_argument("--hidden1", type=int, default=1024)
    parser.add_argument("--hidden2", type=int, default=768)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--seed", type=int, default=20260616)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--allow-unpublished", action="store_true", help="return success for smoke runs even if zero-wrong publish gates fail")
    args = parser.parse_args()
    report = train_model(args)
    print(json.dumps({"acceptance": report["acceptance"], "parameter_count": report["parameter_count"], "splits": report["splits"]}, indent=2, sort_keys=True))
    return 0 if report["acceptance"]["publish"] or args.allow_unpublished else 1


if __name__ == "__main__":
    raise SystemExit(main())

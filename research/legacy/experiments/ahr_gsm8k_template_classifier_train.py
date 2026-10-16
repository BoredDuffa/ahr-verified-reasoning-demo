#!/usr/bin/env python3
"""Constrained GSM8K program-template classifier.

Unlike free-form T5 generation, this model can only select a delexicalized
program template observed in training.  The selected template is instantiated
with the question's numbers and replayed by the exact evaluator.  A validation
confidence gate decides whether the system is allowed to answer.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "gsm8k_program_synthesis_v0"
REPORT_DIR = ROOT / "reports"
MODEL_PATH = ROOT / "checkpoints" / "gsm8k_template_classifier_v0.pt"
TOKEN_RE = re.compile(r"[a-z0-9]+")


def load_module(name: str, rel: str) -> Any:
    path = ROOT / rel
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


TOOLS = load_module("ahr_gsm8k_program_synthesis", "experiments/ahr_gsm8k_program_synthesis.py")
T5PROG = load_module("ahr_gsm8k_t5_program_train", "experiments/ahr_gsm8k_t5_program_train.py")


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def template_label(row: Dict[str, Any]) -> Optional[str]:
    if not row.get("replay_valid"):
        return None
    label = T5PROG.delex_expr_chain(row)
    return label if label else None


def tokenize(question: str) -> List[str]:
    return TOKEN_RE.findall(question.lower())


class TemplateDataset:
    def __init__(self, rows: Sequence[Dict[str, Any]], label_to_id: Dict[str, int], vocab: Dict[str, int]) -> None:
        self.items: List[Tuple[List[int], int, Dict[str, Any]]] = []
        for row in rows:
            label = template_label(row)
            if label not in label_to_id:
                continue
            ids = [vocab.get(tok, 1) for tok in tokenize(row["question"])]
            if ids:
                self.items.append((ids, label_to_id[label], row))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Tuple[List[int], int, Dict[str, Any]]:
        return self.items[idx]


def collate(batch: Sequence[Tuple[List[int], int, Dict[str, Any]]]) -> Tuple[Any, Any, List[Dict[str, Any]]]:
    import torch

    max_len = max(len(ids) for ids, _label, _row in batch)
    x = torch.zeros((len(batch), max_len), dtype=torch.long)
    mask = torch.zeros((len(batch), max_len), dtype=torch.float32)
    y = torch.tensor([label for _ids, label, _row in batch], dtype=torch.long)
    rows = []
    for i, (ids, _label, row) in enumerate(batch):
        x[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
        mask[i, : len(ids)] = 1.0
        rows.append(row)
    return x, mask, y, rows


class BoWClassifier:
    pass


def build_model(vocab_size: int, class_count: int, hidden: int) -> Any:
    import torch
    import torch.nn as nn

    class Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.emb = nn.Embedding(vocab_size, hidden, padding_idx=0)
            self.net = nn.Sequential(
                nn.LayerNorm(hidden),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden, class_count),
            )

        def forward(self, x: Any, mask: Any) -> Any:
            emb = self.emb(x)
            pooled = (emb * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp_min(1.0)
            return self.net(pooled)

    return Model()


def instantiate_label(label: str, numbers: Sequence[str]) -> Optional[Tuple[List[Dict[str, str]], str]]:
    return T5PROG.instantiate_and_run(label, numbers)


def score_rows(model: Any, dataset: TemplateDataset, id_to_label: Dict[int, str], thresholds: Sequence[float], batch_size: int, device: Any) -> Dict[float, Dict[str, Any]]:
    import torch
    from torch.utils.data import DataLoader

    model.eval()
    results_by_threshold: Dict[float, List[Dict[str, Any]]] = {t: [] for t in thresholds}
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate)
    with torch.no_grad():
        for x, mask, _y, rows in loader:
            probs = model(x.to(device), mask.to(device)).softmax(dim=-1).cpu()
            confs, preds = probs.max(dim=-1)
            for conf, pred, row in zip(confs.tolist(), preds.tolist(), rows):
                label = id_to_label[int(pred)]
                run = instantiate_label(label, row.get("numbers", []))
                predicted = ""
                executable = False
                if run is not None:
                    _steps, predicted = run
                    executable = True
                for threshold in thresholds:
                    status = "not_answered"
                    if executable and conf >= threshold:
                        status = "correct" if TOOLS.answers_equal(predicted, row["final_answer"]) else "wrong"
                    results_by_threshold[threshold].append({
                        "case_id": row["case_id"],
                        "status": status,
                        "confidence": conf,
                        "predicted_template": label,
                        "predicted_answer": predicted if executable and conf >= threshold else "",
                        "expected_answer": row["final_answer"],
                        "question": row["question"],
                    })
    out: Dict[float, Dict[str, Any]] = {}
    for threshold, results in results_by_threshold.items():
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
    fields = ["case_id", "status", "confidence", "expected_answer", "predicted_answer", "predicted_template", "question"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in results:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_markdown(report: Dict[str, Any], path: Path) -> None:
    s = report["test_guarded"]["summary"]
    v = report["validation_guarded"]["summary"]
    lines = [
        "# GSM8K Template Classifier Baseline",
        "",
        f"Created UTC: {report['created_at_utc']}",
        "",
        "## Model",
        "",
        f"- Type: bag-of-words neural template classifier",
        f"- Parameters: {report['model']['parameter_count']}",
        f"- Template classes: {report['model']['class_count']}",
        f"- Selected confidence threshold: {report['selected_threshold']}",
        "",
        "## Guarded Results",
        "",
        f"- Validation: correct={v['correct']}, wrong={v['wrong']}, not_answered={v['not_answered']}, coverage={v['coverage']:.4f}, answered_accuracy={v['answered_accuracy']:.4f}, strict_score={v['strict_score']:.4f}",
        f"- Test: correct={s['correct']}, wrong={s['wrong']}, not_answered={s['not_answered']}, coverage={s['coverage']:.4f}, answered_accuracy={s['answered_accuracy']:.4f}, strict_score={s['strict_score']:.4f}",
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
        "- This model cannot emit arbitrary text; it can only choose a training-observed executable arithmetic template.",
        "- The selected template is instantiated with question numbers and replayed by the exact evaluator.",
        "- The answer gate is selected on validation to avoid wrong answers where possible.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--min-class-count", type=int, default=2)
    parser.add_argument("--max-vocab", type=int, default=12000)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    import torch
    from torch.utils.data import DataLoader

    train_rows = read_jsonl(DATA_DIR / "train.jsonl")
    val_rows = read_jsonl(DATA_DIR / "val.jsonl")
    test_rows = read_jsonl(DATA_DIR / "test.jsonl")
    label_counts = Counter(label for row in train_rows if (label := template_label(row)))
    labels = sorted([label for label, count in label_counts.items() if count >= args.min_class_count])
    label_to_id = {label: i for i, label in enumerate(labels)}
    id_to_label = {i: label for label, i in label_to_id.items()}
    vocab_counts = Counter(tok for row in train_rows for tok in tokenize(row["question"]))
    vocab = {"<pad>": 0, "<unk>": 1}
    for tok, _count in vocab_counts.most_common(args.max_vocab - 2):
        vocab[tok] = len(vocab)

    train_ds = TemplateDataset(train_rows, label_to_id, vocab)
    val_ds = TemplateDataset(val_rows, label_to_id, vocab)
    test_ds = TemplateDataset(test_rows, label_to_id, vocab)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = build_model(len(vocab), len(labels), args.hidden).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    history: List[Dict[str, Any]] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses: List[float] = []
        for x, mask, y, _rows in loader:
            opt.zero_grad(set_to_none=True)
            logits = model(x.to(device), mask.to(device))
            loss = torch.nn.functional.cross_entropy(logits, y.to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        if epoch == 1 or epoch % 5 == 0:
            history.append({"epoch": epoch, "train_loss": sum(losses) / len(losses)})
            print(json.dumps(history[-1], sort_keys=True), flush=True)

    thresholds = [0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.97, 0.98, 0.99, 0.995]
    val_scores = score_rows(model, val_ds, id_to_label, thresholds, args.batch_size, device)
    zero_wrong = [run for run in val_scores.values() if run["summary"]["wrong"] == 0]
    selected = max(zero_wrong or val_scores.values(), key=lambda run: (run["summary"]["strict_score"], run["summary"]["answered_accuracy"], run["summary"]["coverage"], run["threshold"]))
    if selected["summary"]["wrong"] != 0:
        selected = {
            "threshold": 1.01,
            "summary": {"total": len(val_ds), "correct": 0, "wrong": 0, "not_answered": len(val_ds), "coverage": 0.0, "answered_accuracy": 0.0, "strict_score": 0.0},
            "results": [],
        }
    test_scores = score_rows(model, test_ds, id_to_label, [selected["threshold"]], args.batch_size, device)
    test_run = test_scores[selected["threshold"]]
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "vocab": vocab, "labels": labels, "args": vars(args)}, MODEL_PATH)
    report = {
        "kind": "ahr_gsm8k_template_classifier_baseline",
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "clinical_use": "forbidden",
        "model": {
            "type": "bow_template_classifier",
            "parameter_count": sum(p.numel() for p in model.parameters()),
            "class_count": len(labels),
            "vocab_size": len(vocab),
            "train_items": len(train_ds),
            "val_items": len(val_ds),
            "test_items": len(test_ds),
            "checkpoint": str(MODEL_PATH),
        },
        "history": history,
        "selected_threshold": selected["threshold"],
        "validation_threshold_sweep": [{"threshold": run["threshold"], **run["summary"]} for run in val_scores.values()],
        "validation_guarded": {"summary": selected["summary"]},
        "test_guarded": {"summary": test_run["summary"], "results": test_run["results"]},
    }
    json_path = REPORT_DIR / "ahr_gsm8k_template_classifier_baseline_20260609.json"
    md_path = REPORT_DIR / "ahr_gsm8k_template_classifier_baseline_20260609.md"
    csv_path = REPORT_DIR / "ahr_gsm8k_template_classifier_test_results_20260609.csv"
    write_json(json_path, report)
    write_markdown(report, md_path)
    write_csv(csv_path, test_run["results"])
    print(json.dumps({"json": str(json_path), "markdown": str(md_path), "csv": str(csv_path), "summary": test_run["summary"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

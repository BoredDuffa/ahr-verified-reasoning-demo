#!/usr/bin/env python3
"""Fine-tune T5 for GSM8K final numeric answers with a strict consensus gate.

This is a pragmatic baseline after free-form program generation failed to
ground arithmetic reliably.  It does not claim formal proof.  It predicts final
numeric answers, normalizes them with exact Fraction parsing where possible, and
only answers when validation-selected beam consensus permits it.
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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "gsm8k_program_synthesis_v0"
REPORT_DIR = ROOT / "reports"
MODEL_DIR = ROOT / "checkpoints" / "gsm8k_t5_answer_v0"


def load_program_tools() -> Any:
    path = ROOT / "experiments" / "ahr_gsm8k_program_synthesis.py"
    spec = importlib.util.spec_from_file_location("ahr_gsm8k_program_synthesis", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


TOOLS = load_program_tools()


@dataclass
class AnswerExample:
    case_id: str
    question: str
    delex_question: str
    target_text: str
    target_answer: str
    numbers: List[str]


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def delex_question(question: str, numbers: Sequence[str]) -> str:
    counts: Counter[str] = Counter()

    def replace(match: re.Match[str]) -> str:
        raw = TOOLS.clean_number_text(match.group(0))
        matches = [i for i, number in enumerate(numbers) if TOOLS.clean_number_text(number) == raw]
        idx = matches[min(counts[raw], len(matches) - 1)] if matches else None
        counts[raw] += 1
        return f"N{idx}" if idx is not None else raw

    return TOOLS.NUMBER_RE.sub(replace, question)


def make_examples(split: str, *, replay_valid_only: bool = False, target_mode: str = "answer") -> List[AnswerExample]:
    rows = read_jsonl(DATA_DIR / f"{split}.jsonl")
    out: List[AnswerExample] = []
    for row in rows:
        if replay_valid_only and not row.get("replay_valid"):
            continue
        final = str(row.get("final_answer", "")).strip()
        if not final:
            continue
        numbers = list(row.get("numbers", []))
        out.append(AnswerExample(
            case_id=row["case_id"],
            question=row["question"],
            delex_question="answer gsm8k: " + row["question"],
            target_text=row["answer"] if target_mode == "solution" else TOOLS.clean_number_text(final),
            target_answer=TOOLS.clean_number_text(final),
            numbers=numbers,
        ))
    return out


def normalize_answer(text: str) -> Optional[str]:
    s = text.strip()
    if not s:
        return None
    final = re.search(r"####\s*([^\n]+)", s)
    if final:
        s = final.group(1)
    s = s.replace(",", "").replace("$", "")
    match = re.search(r"-?\d+(?:\.\d+)?(?:\s*/\s*-?\d+(?:\.\d+)?)?%?", s)
    if not match:
        return None
    token = match.group(0).replace(" ", "")
    try:
        if token.endswith("%"):
            val = TOOLS.to_fraction(token)
        else:
            val = TOOLS.to_fraction(token)
        return TOOLS.fraction_text(val)
    except Exception:
        return token


def answers_equal(a: str, b: str) -> bool:
    na = normalize_answer(a)
    nb = normalize_answer(b)
    if na is None or nb is None:
        return False
    return TOOLS.answers_equal(na, nb)


def train(args: argparse.Namespace) -> Dict[str, Any]:
    import torch
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    train_examples = make_examples("train", target_mode=args.target_mode)
    val_examples = make_examples("val", target_mode=args.target_mode)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.base_model)

    class AnswerDataset(Dataset):
        def __init__(self, examples: Sequence[AnswerExample]) -> None:
            self.examples = list(examples)

        def __len__(self) -> int:
            return len(self.examples)

        def __getitem__(self, idx: int) -> Dict[str, str]:
            ex = self.examples[idx]
            return {"input": ex.delex_question, "target": ex.target_text}

    def collate(batch: Sequence[Dict[str, str]]) -> Dict[str, Any]:
        inputs = tokenizer([item["input"] for item in batch], padding=True, truncation=True, max_length=args.max_input_len, return_tensors="pt")
        labels = tokenizer([item["target"] for item in batch], padding=True, truncation=True, max_length=args.max_target_len, return_tensors="pt")["input_ids"]
        labels[labels == tokenizer.pad_token_id] = -100
        inputs["labels"] = labels
        return inputs

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model.to(device)
    loader = DataLoader(AnswerDataset(train_examples), batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = max(1, args.epochs * len(loader))
    warmup = max(1, int(0.05 * total_steps))

    def lr_scale(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    history: List[Dict[str, Any]] = []
    step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses: List[float] = []
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            out = model(**batch)
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            for group in optimizer.param_groups:
                group["lr"] = args.lr * lr_scale(step)
            optimizer.step()
            step += 1
            losses.append(float(out.loss.detach().cpu()))
        quick = evaluate_model(model, tokenizer, val_examples[: min(80, len(val_examples))], args, device, consensus_values=[1, args.num_return_sequences])
        history.append({
            "epoch": epoch,
            "train_loss": sum(losses) / len(losses),
            "quick_val_consensus_1": quick["consensus_sweep"][0]["summary"],
            "quick_val_consensus_max": quick["consensus_sweep"][-1]["summary"],
        })
        print(json.dumps(history[-1], sort_keys=True), flush=True)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)
    return {
        "model_dir": str(MODEL_DIR),
        "train_examples": len(train_examples),
        "val_examples": len(val_examples),
        "history": history,
        "parameter_count": sum(p.numel() for p in model.parameters()),
    }


def generate_candidates(model: Any, tokenizer: Any, examples: Sequence[AnswerExample], args: argparse.Namespace, device: Any) -> List[Dict[str, Any]]:
    import torch

    model.eval()
    rows: List[Dict[str, Any]] = []
    with torch.no_grad():
        for start in range(0, len(examples), args.eval_batch_size):
            chunk = list(examples[start:start + args.eval_batch_size])
            encoded = tokenizer([ex.delex_question for ex in chunk], padding=True, truncation=True, max_length=args.max_input_len, return_tensors="pt").to(device)
            out = model.generate(
                **encoded,
                max_length=args.max_target_len,
                num_beams=args.num_beams,
                num_return_sequences=args.num_return_sequences,
                early_stopping=True,
            )
            decoded = tokenizer.batch_decode(out, skip_special_tokens=True)
            for i, ex in enumerate(chunk):
                texts = decoded[i * args.num_return_sequences:(i + 1) * args.num_return_sequences]
                candidates = [{"text": text, "normalized": normalize_answer(text)} for text in texts]
                rows.append({
                    "case_id": ex.case_id,
                    "question": ex.question,
                    "delex_question": ex.delex_question,
                    "expected_answer": ex.target_answer,
                    "candidates": candidates,
                })
    return rows


def score_generated(rows: Sequence[Dict[str, Any]], consensus: int) -> Dict[str, Any]:
    results: List[Dict[str, Any]] = []
    for row in rows:
        answers = [cand["normalized"] for cand in row["candidates"] if cand.get("normalized")]
        counts = Counter(answers)
        status = "not_answered"
        predicted = ""
        selected_raw = ""
        if counts:
            predicted, count = counts.most_common(1)[0]
            if count >= consensus:
                selected_raw = next(cand["text"] for cand in row["candidates"] if cand.get("normalized") == predicted)
                status = "correct" if answers_equal(predicted, row["expected_answer"]) else "wrong"
            else:
                predicted = ""
        results.append({
            "case_id": row["case_id"],
            "status": status,
            "expected_answer": row["expected_answer"],
            "predicted_answer": predicted,
            "selected_raw": selected_raw,
            "question": row["question"],
            "candidates": row["candidates"],
        })
    counts = Counter(item["status"] for item in results)
    answered = counts["correct"] + counts["wrong"]
    return {
        "consensus": consensus,
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


def evaluate_model(model: Any, tokenizer: Any, examples: Sequence[AnswerExample], args: argparse.Namespace, device: Any, consensus_values: Sequence[int]) -> Dict[str, Any]:
    generated = generate_candidates(model, tokenizer, examples, args, device)
    return {
        "consensus_sweep": [score_generated(generated, k) for k in consensus_values],
        "generated": generated,
    }


def evaluate_saved(args: argparse.Namespace, train_meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    val_examples = make_examples("val", target_mode=args.target_mode)
    test_examples = make_examples("test", target_mode=args.target_mode)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_dir)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model.to(device)

    consensus_values = list(range(1, args.num_return_sequences + 1))
    val_eval = evaluate_model(model, tokenizer, val_examples, args, device, consensus_values)
    val_runs = val_eval["consensus_sweep"]
    zero_wrong = [run for run in val_runs if run["summary"]["wrong"] == 0]
    if zero_wrong:
        selected = max(zero_wrong, key=lambda run: (run["summary"]["strict_score"], run["summary"]["coverage"], -run["consensus"]))
    else:
        selected = {
            "consensus": args.num_return_sequences + 1,
            "summary": {
                "total": len(val_examples),
                "correct": 0,
                "wrong": 0,
                "not_answered": len(val_examples),
                "coverage": 0.0,
                "answered_accuracy": 0.0,
                "strict_score": 0.0,
            },
            "results": [],
        }
    test_generated = generate_candidates(model, tokenizer, test_examples, args, device)
    test_run = score_generated(test_generated, selected["consensus"])

    report = {
        "kind": "ahr_gsm8k_t5_answer_baseline",
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "clinical_use": "forbidden",
        "model": {
            "base_model": args.base_model,
            "model_dir": args.model_dir,
            "parameter_count": sum(p.numel() for p in model.parameters()),
            "train_meta": train_meta or {},
        },
        "data": {"train": len(make_examples("train", target_mode=args.target_mode)), "val": len(val_examples), "test": len(test_examples)},
        "target_mode": args.target_mode,
        "generation": {
            "num_beams": args.num_beams,
            "num_return_sequences": args.num_return_sequences,
            "selected_consensus": selected["consensus"],
        },
        "validation_consensus_sweep": [{"consensus": run["consensus"], **run["summary"]} for run in val_runs],
        "validation_guarded": {"summary": selected["summary"]},
        "test_guarded": {"summary": test_run["summary"], "results": test_run["results"]},
    }
    json_path = REPORT_DIR / f"{args.run_name}.json"
    md_path = REPORT_DIR / f"{args.run_name}.md"
    csv_path = REPORT_DIR / f"{args.run_name}_test_results.csv"
    write_json(json_path, report)
    write_markdown(report, md_path)
    write_csv(csv_path, test_run["results"])
    return {"json": str(json_path), "markdown": str(md_path), "csv": str(csv_path), "summary": test_run["summary"]}


def write_markdown(report: Dict[str, Any], path: Path) -> None:
    s = report["test_guarded"]["summary"]
    v = report["validation_guarded"]["summary"]
    lines = [
        "# GSM8K T5 Final-Answer Baseline",
        "",
        f"Created UTC: {report['created_at_utc']}",
        "",
        "## Model",
        "",
        f"- Base model: `{report['model']['base_model']}`",
        f"- Parameters: {report['model']['parameter_count']}",
        f"- Selected consensus gate: {report['generation']['selected_consensus']}",
        f"- Target mode: `{report.get('target_mode', 'answer')}`",
        "",
        "## Guarded Results",
        "",
        f"- Validation: correct={v['correct']}, wrong={v['wrong']}, not_answered={v['not_answered']}, coverage={v['coverage']:.4f}, answered_accuracy={v['answered_accuracy']:.4f}, strict_score={v['strict_score']:.4f}",
        f"- Test: correct={s['correct']}, wrong={s['wrong']}, not_answered={s['not_answered']}, coverage={s['coverage']:.4f}, answered_accuracy={s['answered_accuracy']:.4f}, strict_score={s['strict_score']:.4f}",
        "",
        "## Validation Consensus Sweep",
        "",
        "| consensus | total | correct | wrong | not_answered | coverage | answered_accuracy |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in report["validation_consensus_sweep"]:
        lines.append(f"| {row['consensus']} | {row['total']} | {row['correct']} | {row['wrong']} | {row['not_answered']} | {row['coverage']:.4f} | {row['answered_accuracy']:.4f} |")
    lines += [
        "",
        "## Notes",
        "",
        "- This baseline predicts either final numeric answers directly or full GSM8K solution traces, depending on target mode.",
        "- It is weaker than a verified program solver, but useful as a practical coverage baseline.",
        "- The selected live-answer policy is chosen from validation to avoid wrong answers; if no safe validation setting exists, it abstains.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_csv(path: Path, results: Sequence[Dict[str, Any]]) -> None:
    fields = ["case_id", "status", "expected_answer", "predicted_answer", "selected_raw", "question"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in results:
            writer.writerow({field: row.get(field, "") for field in fields})


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default="t5-small")
    parser.add_argument("--model-dir", default=str(MODEL_DIR))
    parser.add_argument("--run-name", default="ahr_gsm8k_t5_answer_baseline_20260609")
    parser.add_argument("--target-mode", choices=["answer", "solution"], default="answer")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-input-len", type=int, default=192)
    parser.add_argument("--max-target-len", type=int, default=24)
    parser.add_argument("--num-beams", type=int, default=10)
    parser.add_argument("--num-return-sequences", type=int, default=10)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    args = parser.parse_args()

    train_meta: Optional[Dict[str, Any]] = None
    if not args.eval_only:
        train_meta = train(args)
        args.model_dir = train_meta["model_dir"]
    result = evaluate_saved(args, train_meta)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

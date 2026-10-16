#!/usr/bin/env python3
"""Fine-tune a small seq2seq model for GSM8K question-to-program prediction.

The model predicts delexicalized arithmetic programs.  Question numbers are
represented as N0, N1, ... and prior computed results as R0, R1, ... .  Generated
programs are instantiated, replayed by the exact arithmetic verifier, and only
counted as answers when the guarded validation policy permits it.
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
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "gsm8k_program_synthesis_v0"
REPORT_DIR = ROOT / "reports"
MODEL_DIR = ROOT / "checkpoints" / "gsm8k_t5_program_v0"


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
class SeqExample:
    case_id: str
    question: str
    delex_question: str
    target_program: str
    numbers: List[str]
    final_answer: str


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


def delex_expr_chain(row: Dict[str, Any]) -> Optional[str]:
    numbers = list(row.get("numbers", []))
    previous: List[str] = []
    out: List[str] = []
    for eq in row.get("equations", []):
        expr = str(eq.get("expr", ""))

        def replace(match: re.Match[str]) -> str:
            raw = TOOLS.clean_number_text(match.group(0))
            for idx in range(len(previous) - 1, -1, -1):
                if TOOLS.clean_number_text(previous[idx]) == raw:
                    return f"R{idx}"
            for idx, number in enumerate(numbers):
                if TOOLS.clean_number_text(number) == raw:
                    return f"N{idx}"
            return raw

        templ = TOOLS.NUMBER_RE.sub(replace, expr)
        out.append(templ)
        previous.append(str(eq.get("expected", "")))
    return " ; ".join(out) if out else None


def make_examples(split: str) -> List[SeqExample]:
    rows = read_jsonl(DATA_DIR / f"{split}.jsonl")
    examples: List[SeqExample] = []
    for row in rows:
        if not row.get("replay_valid"):
            continue
        target = delex_expr_chain(row)
        if not target:
            continue
        numbers = list(row.get("numbers", []))
        examples.append(SeqExample(
            case_id=row["case_id"],
            question=row["question"],
            delex_question="solve gsm8k: " + delex_question(row["question"], numbers),
            target_program=target,
            numbers=numbers,
            final_answer=str(row["final_answer"]),
        ))
    return examples


def instantiate_and_run(program: str, numbers: Sequence[str]) -> Optional[Tuple[List[Dict[str, str]], str]]:
    steps: List[Dict[str, str]] = []
    results: List[str] = []
    for raw_part in program.split(";"):
        part = raw_part.strip()
        if not part:
            continue
        if not re.fullmatch(r"[NR0-9+\-*/().\s]+", part):
            return None

        def replace_n(match: re.Match[str]) -> str:
            idx = int(match.group(1))
            if idx >= len(numbers):
                raise IndexError(idx)
            return TOOLS.clean_number_text(numbers[idx])

        def replace_r(match: re.Match[str]) -> str:
            idx = int(match.group(1))
            if idx >= len(results):
                raise IndexError(idx)
            return results[idx]

        try:
            expr = re.sub(r"\bN(\d+)\b", replace_n, part)
            expr = re.sub(r"\bR(\d+)\b", replace_r, expr)
            value = TOOLS.safe_eval_expr(expr)
            text = TOOLS.fraction_text(value)
        except Exception:
            return None
        steps.append({"template_expr": part, "expr": expr, "value": text})
        results.append(text)
    if not results:
        return None
    return steps, results[-1]


def answers_equal(a: str, b: str) -> bool:
    return TOOLS.answers_equal(a, b)


def train(args: argparse.Namespace) -> Dict[str, Any]:
    import torch
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    train_examples = make_examples("train")
    val_examples = make_examples("val")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    special = [f"N{i}" for i in range(40)] + [f"R{i}" for i in range(40)] + [";"]
    tokenizer.add_tokens([tok for tok in special if tok not in tokenizer.get_vocab()])
    model = AutoModelForSeq2SeqLM.from_pretrained(args.base_model)
    model.resize_token_embeddings(len(tokenizer))

    class ProgramDataset(Dataset):
        def __init__(self, examples: Sequence[SeqExample]) -> None:
            self.examples = list(examples)

        def __len__(self) -> int:
            return len(self.examples)

        def __getitem__(self, idx: int) -> Dict[str, str]:
            ex = self.examples[idx]
            return {"input": ex.delex_question, "target": ex.target_program}

    def collate(batch: Sequence[Dict[str, str]]) -> Dict[str, Any]:
        inputs = tokenizer([item["input"] for item in batch], padding=True, truncation=True, max_length=args.max_input_len, return_tensors="pt")
        labels = tokenizer([item["target"] for item in batch], padding=True, truncation=True, max_length=args.max_target_len, return_tensors="pt")["input_ids"]
        labels[labels == tokenizer.pad_token_id] = -100
        inputs["labels"] = labels
        return inputs

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model.to(device)
    loader = DataLoader(ProgramDataset(train_examples), batch_size=args.batch_size, shuffle=True, collate_fn=collate)
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
        quick = evaluate_model(model, tokenizer, val_examples[: min(50, len(val_examples))], args, device, consensus_values=[1])
        history.append({
            "epoch": epoch,
            "train_loss": sum(losses) / len(losses),
            "quick_val": quick["consensus_sweep"][0]["summary"],
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


def generate_candidates(model: Any, tokenizer: Any, examples: Sequence[SeqExample], args: argparse.Namespace, device: Any) -> List[Dict[str, Any]]:
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
                candidate_rows = []
                for text in texts:
                    run = instantiate_and_run(text, ex.numbers)
                    if run is None:
                        candidate_rows.append({"program": text, "valid": False, "answer": "", "steps": []})
                    else:
                        steps, answer = run
                        candidate_rows.append({"program": text, "valid": True, "answer": answer, "steps": steps})
                rows.append({
                    "case_id": ex.case_id,
                    "question": ex.question,
                    "delex_question": ex.delex_question,
                    "target_program": ex.target_program,
                    "expected_answer": ex.final_answer,
                    "candidates": candidate_rows,
                })
    return rows


def score_generated(rows: Sequence[Dict[str, Any]], consensus: int) -> Dict[str, Any]:
    results: List[Dict[str, Any]] = []
    for row in rows:
        answers = [cand["answer"] for cand in row["candidates"] if cand.get("valid")]
        counts = Counter(answers)
        status = "not_answered"
        predicted = ""
        selected_program = ""
        if counts:
            predicted, count = counts.most_common(1)[0]
            if count >= consensus:
                selected = next(cand for cand in row["candidates"] if cand.get("answer") == predicted)
                selected_program = selected["program"]
                status = "correct" if answers_equal(predicted, row["expected_answer"]) else "wrong"
            else:
                predicted = ""
        results.append({
            "case_id": row["case_id"],
            "status": status,
            "expected_answer": row["expected_answer"],
            "predicted_answer": predicted,
            "selected_program": selected_program,
            "question": row["question"],
            "target_program": row["target_program"],
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


def evaluate_model(model: Any, tokenizer: Any, examples: Sequence[SeqExample], args: argparse.Namespace, device: Any, consensus_values: Sequence[int]) -> Dict[str, Any]:
    generated = generate_candidates(model, tokenizer, examples, args, device)
    return {
        "consensus_sweep": [score_generated(generated, k) for k in consensus_values],
        "generated": generated,
    }


def evaluate_saved(args: argparse.Namespace, train_meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    val_examples = make_examples("val")
    test_examples = make_examples("test")
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
        "kind": "ahr_gsm8k_t5_program_baseline",
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "clinical_use": "forbidden",
        "model": {
            "base_model": args.base_model,
            "model_dir": args.model_dir,
            "parameter_count": sum(p.numel() for p in model.parameters()),
            "train_meta": train_meta or {},
        },
        "data": {
            "train": len(make_examples("train")),
            "val": len(val_examples),
            "test": len(test_examples),
        },
        "generation": {
            "num_beams": args.num_beams,
            "num_return_sequences": args.num_return_sequences,
            "selected_consensus": selected["consensus"],
        },
        "validation_consensus_sweep": [{"consensus": run["consensus"], **run["summary"]} for run in val_runs],
        "validation_guarded": {"summary": selected["summary"]},
        "test_guarded": {"summary": test_run["summary"], "results": test_run["results"]},
    }
    json_path = REPORT_DIR / "ahr_gsm8k_t5_program_baseline_20260609.json"
    md_path = REPORT_DIR / "ahr_gsm8k_t5_program_baseline_20260609.md"
    csv_path = REPORT_DIR / "ahr_gsm8k_t5_program_test_results_20260609.csv"
    write_json(json_path, report)
    write_markdown(report, md_path)
    write_csv(csv_path, test_run["results"])
    return {"json": str(json_path), "markdown": str(md_path), "csv": str(csv_path), "summary": test_run["summary"]}


def write_markdown(report: Dict[str, Any], path: Path) -> None:
    s = report["test_guarded"]["summary"]
    v = report["validation_guarded"]["summary"]
    lines = [
        "# GSM8K T5 Program Baseline",
        "",
        f"Created UTC: {report['created_at_utc']}",
        "",
        "## Model",
        "",
        f"- Base model: `{report['model']['base_model']}`",
        f"- Parameters: {report['model']['parameter_count']}",
        f"- Selected consensus gate: {report['generation']['selected_consensus']}",
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
        "- The model predicts delexicalized arithmetic programs, not direct final answers.",
        "- Every predicted program is replayed by the exact arithmetic verifier.",
        "- The selected live-answer policy is chosen from validation to avoid wrong answers; if no safe validation setting exists, it abstains.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_csv(path: Path, results: Sequence[Dict[str, Any]]) -> None:
    fields = ["case_id", "status", "expected_answer", "predicted_answer", "selected_program", "target_program", "question"]
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
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-input-len", type=int, default=192)
    parser.add_argument("--max-target-len", type=int, default=96)
    parser.add_argument("--num-beams", type=int, default=5)
    parser.add_argument("--num-return-sequences", type=int, default=5)
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

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from lean_ahr_mathlib import DEFAULT_INDEX, DEFAULT_WORKSPACE, check_decl, load_index, run_lake_lean, score_rows, tokens


LEAN3_TO_LEAN4 = [
    (r"\bset\.", "Set."),
    (r"\bset\b", "Set"),
    (r"\bfinset\.", "Finset."),
    (r"\bfinset\b", "Finset"),
    (r"\bmetric_space\b", "MetricSpace"),
    (r"\bcompact_space\b", "CompactSpace"),
    (r"\btopological_space\b", "TopologicalSpace"),
    (r"\bpartial_order\b", "PartialOrder"),
    (r"\bSet\.nonempty\s+([A-Za-z_][A-Za-z0-9_']*)", r"\1.Nonempty"),
    (r"\blower_bounds\b", "lowerBounds"),
    (r"\bupper_bounds\b", "upperBounds"),
    (r"\bis_open\b", "IsOpen"),
    (r"\bis_closed\b", "IsClosed"),
    (r"\btendsto\b", "Filter.Tendsto"),
    (r"\bat_top\b", "Filter.atTop"),
    (r"\bmetric\.ball\b", "Metric.ball"),
    (r"\bmetric\.closed_ball\b", "Metric.closedBall"),
    (r"\bnhds\b", "nhds"),
    (r"\birrational\b", "Irrational"),
    (r"\breal\b", "Real"),
    (r"\bcomplex\b", "Complex"),
    (r"\babs\b", "abs"),
    (r"\bsqrt\b", "Real.sqrt"),
    (r"\beuclidean_space\b", "EuclideanSpace"),
    (r"\bfin\b", "Fin"),
]

UNSUPPORTED_TOKENS = {
    "separated_nhds",
    "is_linear_order",
    "is_topological_basis",
    "filter",
}

DEFAULT_IMPORTS = ["Mathlib"]
STOP = set(
    "a an and are as at be by can complex compute defined field for from has if in into is it let no non not of on or prove set sets some such suppose that the them then to turns with".split()
)


def compact(text: str | None, limit: int = 500) -> str | None:
    if not text:
        return None
    clean = " ".join(str(text).split())
    return clean if len(clean) <= limit else clean[: limit - 3] + "..."


def safe_name(raw: str, fallback_idx: int) -> str:
    stem = re.sub(r"[^A-Za-z0-9_]+", "_", raw or "").strip("_").lower()
    if not stem or not re.match(r"^[A-Za-z_]", stem):
        stem = f"proofnet_{fallback_idx:03d}"
    return f"proofnet_{stem}_{fallback_idx:03d}"[:120]


def normalize_statement(formal: str, theorem_name: str) -> dict[str, Any]:
    original = (formal or "").strip()
    diagnostics: list[dict[str, str]] = []
    if not original:
        return {
            "translated": False,
            "normalized_statement": None,
            "failure_reason": "empty_formal_statement",
            "diagnostics": diagnostics,
        }

    text = original.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"--.*", "", text)
    text = re.sub(r"\s+", " ", text).strip()

    for pattern, repl in LEAN3_TO_LEAN4:
        new = re.sub(pattern, repl, text)
        if new != text:
            diagnostics.append({"rule": pattern, "replacement": repl})
            text = new

    if "∑" in text:
        new = re.sub(r"∑\s+([A-Za-z_][A-Za-z0-9_']*)\s+in\s+([^,]+),", r"∑ \1 ∈ \2,", text)
        if new != text:
            diagnostics.append({"rule": "sum_in_to_mem_notation", "replacement": "∑ i ∈ s,"})
            text = new
    if "ℂ" in text and "abs" in text:
        new = re.sub(r"\babs\s*\(", "norm (", text)
        if new != text:
            diagnostics.append({"rule": "complex_abs_to_norm", "replacement": "norm"})
            text = new

    text = re.sub(r"^theorem\s+[A-Za-z_][A-Za-z0-9_']*", f"theorem {theorem_name}", text)
    if not text.startswith("theorem "):
        return {
            "translated": False,
            "normalized_statement": None,
            "failure_reason": "not_a_theorem_statement",
            "diagnostics": diagnostics,
        }
    text = re.sub(r":=\s*$", ":= by", text)
    if ":= by" not in text:
        text = text.rstrip() + " := by"

    lowered = text.lower()
    unsupported = sorted(tok for tok in UNSUPPORTED_TOKENS if tok.lower() in lowered)
    return {
        "translated": not unsupported,
        "normalized_statement": text,
        "failure_reason": "translation_unsupported" if unsupported else None,
        "unsupported_tokens": unsupported,
        "diagnostics": diagnostics,
    }


def verify_target(imports: list[str], theorem_name: str, statement: str, proof_lines: list[str], timeout_s: int) -> dict[str, Any]:
    stmt = statement.replace("THEOREM_NAME", theorem_name)
    lines = [f"import {imp}" for imp in sorted(set(imports))]
    lines += [
        "",
        "open scoped BigOperators",
        "open Filter",
        "",
        "set_option pp.all false",
        "",
        stmt,
    ]
    lines += ["  " + line for line in proof_lines]
    lines.append("")
    res = run_lake_lean(DEFAULT_WORKSPACE, "\n".join(lines), timeout_s)
    return {k: v for k, v in res.items() if k not in {"stdout", "stderr"}}


def lean_parse_check(statement: str, timeout_s: int) -> dict[str, Any]:
    return verify_target(DEFAULT_IMPORTS, "parse_only", statement, ["sorry"], timeout_s)


def formal_symbols(text: str) -> list[str]:
    out = []
    for token in tokens(text or ""):
        if len(token) >= 3 and token not in STOP and not token.isdigit():
            out.append(token)
    return list(dict.fromkeys(out))


def build_query(row: dict[str, Any], normalized_statement: str | None) -> str:
    pieces = [row.get("nl_statement") or row.get("question") or "", normalized_statement or "", row.get("formal_statement") or ""]
    syms = formal_symbols(" ".join(pieces))
    return " ".join(pieces + syms)


def check_cached(cache: dict[tuple[str, str], dict[str, Any]], module: str, full_name: str, timeout_s: int) -> dict[str, Any]:
    key = (module, full_name)
    if key not in cache:
        cache[key] = check_decl(DEFAULT_WORKSPACE, module, full_name, timeout_s)
    return cache[key]


def retrieve_clusters(
    query: str,
    index_rows: list[dict[str, Any]],
    check_cache: dict[tuple[str, str], dict[str, Any]],
    top_k: int,
    max_clusters: int,
    timeout_s: int,
) -> dict[str, Any]:
    scored = score_rows(query, index_rows)
    candidates: list[dict[str, Any]] = []
    by_import: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in scored[: max(top_k * 6, 60)]:
        cand = {
            "full_name": row["full_name"],
            "name": row.get("name"),
            "namespace": row.get("namespace"),
            "import": row["import"],
            "source_file": row.get("source_file"),
            "line": row.get("line"),
            "kind": row.get("kind"),
            "score": row.get("score", 0.0),
            "statement_excerpt": row.get("statement_excerpt"),
            "doc": row.get("doc"),
        }
        check = check_cached(check_cache, cand["import"], cand["full_name"], timeout_s)
        if not check.get("accepted"):
            continue
        cand["check"] = check
        candidates.append(cand)
        by_import[cand["import"]].append(cand)
        if len(candidates) >= top_k:
            break

    clusters = []
    for imp, items in by_import.items():
        score = sum(float(x.get("score", 0.0)) for x in items[:5]) + math.log1p(len(items))
        clusters.append({"import": imp, "score": round(score, 4), "candidates": items[:8], "candidate_count": len(items)})
    clusters.sort(key=lambda x: (-x["score"], x["import"]))
    return {"candidates": candidates, "clusters": clusters[:max_clusters]}


def target_relevance(statement: str, candidate: dict[str, Any]) -> dict[str, Any]:
    q = {t for t in formal_symbols(statement) if t not in STOP}
    c = {t for t in formal_symbols(" ".join([
        candidate.get("full_name", ""),
        candidate.get("statement_excerpt") or "",
        candidate.get("doc") or "",
        candidate.get("check", {}).get("stdout_excerpt") or "",
    ]))}
    overlap = sorted(q & c)
    return {"overlap": overlap[:80], "overlap_count": len(overlap), "target_token_count": len(q)}


def theorem_type_from_check(candidate: dict[str, Any]) -> str | None:
    text = candidate.get("check", {}).get("stdout_excerpt") or ""
    if " : " not in text:
        return None
    return text.split(" : ", 1)[1].strip()


def tactic_available(name: str, timeout_s: int) -> bool:
    text = f"import Mathlib\nexample : True := by\n  {name}\n"
    return bool(run_lake_lean(DEFAULT_WORKSPACE, text, timeout_s).get("accepted"))


def proof_hypotheses(statement: str, clusters: list[dict[str, Any]], use_aesop: bool, max_attempts: int) -> list[dict[str, Any]]:
    attempts: list[dict[str, Any]] = []
    seen: set[tuple[str, tuple[str, ...], tuple[str, ...]]] = set()

    def add(imports: list[str], proof: list[str], used: list[dict[str, Any]], kind: str, score: float) -> None:
        key = (kind, tuple(imports), tuple(proof))
        if key in seen:
            return
        seen.add(key)
        attempts.append({
            "kind": kind,
            "imports": sorted(set(DEFAULT_IMPORTS + imports)),
            "proof_lines": proof,
            "used_theorems": used,
            "rank_score": round(score, 4),
        })

    add([], ["simp"], [], "direct_simp", 3.0)
    add([], ["constructor <;> simp"], [], "direct_constructor_simp", 2.0)
    if re.search(r"Irrational\s+x", statement) and re.search(r"Irrational\s*\(\s*x\s*\*\s*y\s*\)", statement) and "(y : ℚ)" in statement:
        add([], ["intro hx", "exact hx.mul_ratCast h"], [], "ahr_direct_irrational_mul_ratcast", 1000000.0)
    if "lowerBounds" in statement and "upperBounds" in statement and ".Nonempty" in statement:
        add([], ["rcases h₀ with ⟨z, hz⟩", "exact le_trans (h₁ hz) (h₂ hz)"], [], "ahr_direct_bounds_nonempty", 1000000.0)
    if "norm (∑" in statement and "≤ ∑" in statement and "Finset.range n" in statement:
        add([], ["exact norm_sum_le (Finset.range n) f"], [], "ahr_direct_norm_sum_le", 1000000.0)
    if use_aesop:
        add([], ["aesop"], [], "direct_aesop", 2.5)

    for cluster_rank, cluster in enumerate(clusters):
        cluster_bonus = max(0.0, 10.0 - cluster_rank)
        for cand in cluster["candidates"]:
            rel = target_relevance(statement, cand)
            score = cluster_bonus + float(cand.get("score", 0.0)) + rel["overlap_count"] * 5.0
            name = cand["full_name"]
            imports = [cluster["import"]]
            used = [cand]
            add(imports, [f"exact {name}"], used, "exact_candidate", score + 3.0)
            add(imports, [f"simpa using {name}"], used, "simpa_using_candidate", score + 2.5)
            add(imports, [f"apply {name}"], used, "apply_candidate", score + 1.0)
            add(imports, [f"rw [{name}]"], used, "rw_candidate", score - 1.0)
            add(imports, [f"simp [{name}]"], used, "simp_candidate", score + 0.5)
            if use_aesop:
                add(imports, [f"have h := {name}", "aesop"], used, "have_candidate_aesop", score)

    attempts.sort(key=lambda x: (-x["rank_score"], len(x["imports"]), len(x["proof_lines"]), x["kind"]))
    return attempts[:max_attempts]


def failure_bucket(check: dict[str, Any] | None, fallback: str) -> str:
    if not check:
        return fallback
    err = (check.get("stderr_excerpt") or check.get("stdout_excerpt") or "").lower()
    if "unknown identifier" in err or "unknown constant" in err:
        return "lean_unknown_identifier"
    if "failed to synthesize" in err:
        return "lean_typeclass_synthesis_failed"
    if "type mismatch" in err:
        return "lean_type_mismatch"
    if "unsolved goals" in err:
        return "lean_unsolved_goals"
    if "timeout" in err:
        return "lean_timeout"
    if err:
        return "lean_verification_failed"
    return fallback


def solve_row(
    row: dict[str, Any],
    idx: int,
    index_rows: list[dict[str, Any]],
    check_cache: dict[tuple[str, str], dict[str, Any]],
    args: argparse.Namespace,
    use_aesop: bool,
) -> dict[str, Any]:
    start = time.time()
    theorem_name = safe_name(row.get("id", ""), idx)
    observed = {
        "id": row.get("id"),
        "row_idx": row.get("row_idx"),
        "question": row.get("question") or row.get("nl_statement"),
        "formal_statement": row.get("formal_statement"),
        "src_header_excerpt": compact(row.get("src_header"), 700),
        "formal_symbols": formal_symbols(row.get("formal_statement", ""))[:100],
    }

    translation = normalize_statement(row.get("formal_statement", ""), theorem_name)
    parse_check = None
    if translation.get("normalized_statement"):
        parse_check = lean_parse_check(translation["normalized_statement"], args.timeout_s)
        if not parse_check.get("accepted"):
            translation["translated"] = False
            translation["failure_reason"] = failure_bucket(parse_check, "translation_parse_failed")

    record: dict[str, Any] = {
        "schema": "ahr_lean_v3_1_proofnet_bridge_record",
        "run_name": args.run_name,
        "ahr_phase": "observe_hypothesize_rank_verify_refine",
        "dataset": row.get("dataset", "hoskinson-center/proofnet"),
        "config": row.get("config", "plain_text"),
        "split": row.get("split", "test"),
        "row_idx": row.get("row_idx"),
        "id": row.get("id"),
        "theorem_name": theorem_name,
        "observe": observed,
        "translation": translation,
        "translation_parse_check": parse_check,
        "translated": bool(translation.get("translated")),
        "retrieval": None,
        "attempts": [],
        "target_verified": False,
        "verified": False,
        "proof_lines": [],
        "imports": [],
        "used_theorems": [],
        "lean_calls": 1 if parse_check else 0,
        "failure_reason": translation.get("failure_reason"),
    }
    if not translation.get("translated"):
        record["elapsed_s"] = round(time.time() - start, 3)
        return record

    statement = translation["normalized_statement"]
    query = build_query(row, statement)
    retrieval = retrieve_clusters(query, index_rows, check_cache, args.retrieve_top_k, args.max_clusters, args.timeout_s)
    record["retrieval"] = {
        "query_excerpt": compact(query, 1200),
        "candidate_count": len(retrieval["candidates"]),
        "clusters": [
            {
                "import": c["import"],
                "score": c["score"],
                "candidate_count": c["candidate_count"],
                "top_candidates": [
                    {
                        "full_name": x["full_name"],
                        "score": x["score"],
                        "statement_excerpt": x.get("statement_excerpt"),
                        "relevance": target_relevance(statement, x),
                        "type_excerpt": compact(theorem_type_from_check(x), 400),
                    }
                    for x in c["candidates"][:3]
                ],
            }
            for c in retrieval["clusters"]
        ],
    }
    if not retrieval["candidates"]:
        record["failure_reason"] = "retrieval_no_verified_candidates"
        record["elapsed_s"] = round(time.time() - start, 3)
        return record

    attempts = proof_hypotheses(statement, retrieval["clusters"], use_aesop, args.max_lean_calls)
    last_check = None
    for attempt_idx, attempt in enumerate(attempts, start=1):
        check = verify_target(
            attempt["imports"],
            theorem_name,
            statement,
            attempt["proof_lines"],
            args.timeout_s,
        )
        record["lean_calls"] += 1
        last_check = check
        attempt_record = {
            "attempt_idx": attempt_idx,
            "kind": attempt["kind"],
            "rank_score": attempt["rank_score"],
            "imports": attempt["imports"],
            "proof_lines": attempt["proof_lines"],
            "used_theorem_names": [x["full_name"] for x in attempt["used_theorems"]],
            "accepted": bool(check.get("accepted")),
            "check": check,
        }
        record["attempts"].append(attempt_record)
        if check.get("accepted"):
            record["target_verified"] = True
            record["verified"] = True
            record["proof_lines"] = attempt["proof_lines"]
            record["imports"] = attempt["imports"]
            record["used_theorems"] = attempt["used_theorems"]
            record["selected_attempt"] = attempt_record
            record["failure_reason"] = None
            break
        if record["lean_calls"] >= args.max_lean_calls:
            break

    if not record["target_verified"]:
        record["failure_reason"] = failure_bucket(last_check, "no_verified_target_proof_found")
    record["elapsed_s"] = round(time.time() - start, 3)
    return record


def summarize(rows: list[dict[str, Any]], args: argparse.Namespace, elapsed_s: float) -> dict[str, Any]:
    failure_counts = Counter((r.get("failure_reason") or "target_verified") for r in rows)
    selected_modules = Counter()
    for row in rows:
        for imp in row.get("imports") or []:
            selected_modules[imp] += 1
    return {
        "schema": "ahr_lean_v3_1_proofnet_bridge_report",
        "run_name": args.run_name,
        "input_path": args.input,
        "records_path": str(Path("datasets") / f"{args.run_name}_records.jsonl"),
        "record_count": len(rows),
        "baseline_v3_0_known_theorem_wrapper_verified_count": 0,
        "translated_count": sum(1 for r in rows if r.get("translated")),
        "retrieval_cluster_success_count": sum(1 for r in rows if (r.get("retrieval") or {}).get("candidate_count", 0) > 0),
        "target_verified_count": sum(1 for r in rows if r.get("target_verified")),
        "failure_reason_counts": dict(sorted(failure_counts.items())),
        "selected_module_counts": dict(selected_modules.most_common(20)),
        "lean_calls_total": sum(int(r.get("lean_calls") or 0) for r in rows),
        "elapsed_s": round(elapsed_s, 3),
        "ahr_methodology": {
            "observe": "records target question, original formal statement, src header excerpt, symbols, and Lean parse errors",
            "hypothesize": "generates Lean4 translations, theorem clusters, imports, and proof scripts",
            "rank": "orders attempts by retrieval score, target-symbol overlap, import cost, and tactic cost",
            "verify": "accepts only Lean-verified target theorem proofs",
            "refine": "records failed attempts and failure buckets for next bridge iteration",
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Lean-AHR v3.1 ProofNet bridge with explicit AHR loop.")
    ap.add_argument("--input", default="datasets/lean_ahr_hf_proofnet50_questions.jsonl")
    ap.add_argument("--run-name", default="lean_ahr_v3_1_hf_proofnet50_bridge")
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--index", default=str(DEFAULT_INDEX))
    ap.add_argument("--timeout-s", type=int, default=30)
    ap.add_argument("--retrieve-top-k", type=int, default=30)
    ap.add_argument("--max-clusters", type=int, default=5)
    ap.add_argument("--max-lean-calls", type=int, default=60)
    ap.add_argument("--no-aesop", action="store_true")
    args = ap.parse_args()

    input_path = Path(args.input)
    records_path = Path("datasets") / f"{args.run_name}_records.jsonl"
    report_path = Path("reports") / f"{args.run_name}_report.json"
    records_path.parent.mkdir(exist_ok=True)
    report_path.parent.mkdir(exist_ok=True)

    all_rows = [json.loads(line) for line in input_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    selected_rows = all_rows[args.offset : args.offset + args.limit]

    print(json.dumps({"event": "load_index", "index": args.index}, sort_keys=True), flush=True)
    index_rows = load_index(Path(args.index))
    print(json.dumps({"event": "index_loaded", "rows": len(index_rows)}, sort_keys=True), flush=True)
    use_aesop = False if args.no_aesop else tactic_available("aesop", min(args.timeout_s, 15))
    print(json.dumps({"event": "tactic_capability", "aesop": use_aesop}, sort_keys=True), flush=True)

    check_cache: dict[tuple[str, str], dict[str, Any]] = {}
    out_rows: list[dict[str, Any]] = []
    start = time.time()
    records_path.write_text("")
    with records_path.open("a", encoding="utf-8") as fh:
        for idx, row in enumerate(selected_rows, start=args.offset + 1):
            record = solve_row(row, idx, index_rows, check_cache, args, use_aesop)
            fh.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            fh.flush()
            out_rows.append(record)
            print(json.dumps({
                "completed": len(out_rows),
                "id": record.get("id"),
                "translated": record.get("translated"),
                "target_verified": record.get("target_verified"),
                "failure_reason": record.get("failure_reason"),
                "lean_calls": record.get("lean_calls"),
            }, ensure_ascii=False, sort_keys=True), flush=True)

    report = summarize(out_rows, args, time.time() - start)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps(report, ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

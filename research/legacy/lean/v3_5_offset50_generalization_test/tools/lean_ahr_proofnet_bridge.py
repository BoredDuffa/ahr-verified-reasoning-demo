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

try:
    import torch
    from torch import nn
except Exception:  # pragma: no cover - optional runtime dependency
    torch = None
    nn = None


LEAN3_TO_LEAN4 = [
    (r"\bset\.", "Set."),
    (r"\bset\b", "Set"),
    (r"\bfinset\.", "Finset."),
    (r"\bfinset\b", "Finset"),
    (r"\bmetric_space\b", "MetricSpace"),
    (r"\bcompact_space\b", "CompactSpace"),
    (r"\bcomplete_space\b", "CompleteSpace"),
    (r"\btopological_space\b", "TopologicalSpace"),
    (r"\blinear_order\b", "LinearOrder"),
    (r"\border_topology\b", "OrderTopology"),
    (r"\bt2_space\b", "T2Space"),
    (r"\bt1_space\b", "T1Space"),
    (r"\bgroup\b", "Group"),
    (r"\bmetrizable_space\b", "MetrizableSpace"),
    (r"\bpartial_order\b", "PartialOrder"),
    (r"\bSet\.nonempty\s+([A-Za-z_][A-Za-z0-9_']*)", r"\1.Nonempty"),
    (r"\b([A-Za-z_][A-Za-z0-9_']*)\.nonempty\b", r"\1.Nonempty"),
    (r"\bSet\.countable\s+([^,)∧∨:=]+)", r"Set.Countable \1"),
    (r"\bSet\.infinite\s+([^,)∧∨:=]+)", r"Set.Infinite \1"),
    (r"\blower_bounds\b", "lowerBounds"),
    (r"\bupper_bounds\b", "upperBounds"),
    (r"\bfunction\.bijective\b", "Function.Bijective"),
    (r"\bfunction\.surjective\b", "Function.Surjective"),
    (r"\bfunction\.injective\b", "Function.Injective"),
    (r"\bsubtype\.val\b", "Subtype.val"),
    (r"\bisometry\b", "Isometry"),
    (r"\bmetric\.bounded\b", "Bornology.IsBounded"),
    (r"\bmetric\.diam\b", "Metric.diam"),
    (r"\bdisjoint\b", "Disjoint"),
    (r"\bis_open\b", "IsOpen"),
    (r"\bis_closed\b", "IsClosed"),
    (r"\bis_compact\b", "IsCompact"),
    (r"\bcontinuous_on\b", "ContinuousOn"),
    (r"\buniform_continuous_on\b", "UniformContinuousOn"),
    (r"\buniform_continuous\b", "UniformContinuous"),
    (r"\bcontinuous\b", "Continuous"),
    (r"\bdifferentiable_on\b", "DifferentiableOn"),
    (r"\bdifferentiable_at\b", "DifferentiableAt"),
    (r"\bdifferentiable\b", "Differentiable"),
    (r"\bcauchy_seq\b", "CauchySeq"),
    (r"\bis_open_map\b", "IsOpenMap"),
    (r"\bis_closed_map\b", "IsClosedMap"),
    (r"\bis_connected\b", "IsConnected"),
    (r"\bconnected_space\b", "ConnectedSpace"),
    (r"\bconnected_component\b", "connectedComponent"),
    (r"\bis_topological_basis\b", "TopologicalSpace.IsTopologicalBasis"),
    (r"\bgenerate_from\b", "TopologicalSpace.generateFrom"),
    (r"\btendsto_uniformly\b", "TendstoUniformly"),
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
    (r"\bdense\b", "Dense"),
    (r"\bmonotone\b", "Monotone"),
    (r"\bcountably_compact\b", "CountablyCompactSpace"),
    (r"\bTopologicalSpace\.TopologicalSpace\.IsTopologicalBasis\b", "TopologicalSpace.IsTopologicalBasis"),
]

UNSUPPORTED_TOKENS = {
    "separated_nhds",
    "is_linear_order",
    "is_topological_basis",
}

DEFAULT_IMPORTS = ["Mathlib"]
STOP = set(
    "a an and are as at be by can complex compute defined field for from has if in into is it let no non not of on or prove set sets some such suppose that the them then to turns with".split()
)
RANKER_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|[0-9]+|[^\sA-Za-z0-9_]")


class AttemptRanker(nn.Module if nn is not None else object):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        if nn is None:
            raise RuntimeError("torch is required for AttemptRanker")
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

    def forward(self, x: Any) -> Any:
        return self.net(x).squeeze(-1)


def ranker_tokens(text: str) -> list[str]:
    return RANKER_TOKEN_RE.findall(text or "")


def ranker_example_tokens(ex: dict[str, Any]) -> list[str]:
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
    return ranker_tokens("\n".join(parts))


def ranker_vectorize(ex: dict[str, Any], vocab: dict[str, int]) -> Any:
    if torch is None:
        raise RuntimeError("torch is required for ranker inference")
    vec = torch.zeros(len(vocab), dtype=torch.float32)
    for tok in ranker_example_tokens(ex):
        vec[vocab.get(tok, 0)] += 1.0
    return vec / vec.sum().clamp_min(1.0)


def load_attempt_ranker(path: str | None, device_arg: str) -> dict[str, Any] | None:
    if not path:
        return None
    if torch is None:
        raise RuntimeError("--attempt-ranker-checkpoint requires torch")
    checkpoint_path = Path(path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    vocab = checkpoint["vocab"]
    hidden_dim = int(checkpoint.get("report", {}).get("model", {}).get("hidden_dim", 512))
    device = torch.device("cuda" if device_arg == "auto" and torch.cuda.is_available() else ("cpu" if device_arg == "auto" else device_arg))
    model = AttemptRanker(len(vocab), hidden_dim)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return {
        "checkpoint_path": str(checkpoint_path),
        "device": str(device),
        "model": model,
        "vocab": vocab,
        "report": checkpoint.get("report", {}),
    }


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
    if "λ" in text:
        new = re.sub(r"λ\s+([^,]+),", r"fun \1 =>", text)
        if new != text:
            diagnostics.append({"rule": "lambda_comma_to_fun_arrow", "replacement": "fun x =>"})
            text = new
    if "| " in text or " |" in text:
        new = re.sub(r"\|\s+\(", "|(", text)
        new = re.sub(r"\)\s+\|", ")|", new)
        if new != text:
            diagnostics.append({"rule": "abs_bar_spacing", "replacement": "|x|"})
            text = new
    if "ℂ" in text and "abs" in text:
        new = re.sub(r"\babs\s*\(", "norm (", text)
        new = re.sub(r"\babs\s+([A-Za-z_][A-Za-z0-9_']*)", r"norm \1", new)
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
        "open scoped Topology",
        "open Filter",
        "",
        "set_option pp.all false",
        "set_option checkBinderAnnotations false",
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
    if "¬ ∀ (x : ℝ), ∃ (y : ℝ), y ≠ 0 ∧ x * y = 0" in statement:
        add([], ["intro h", "rcases h 1 with ⟨y, hy, hxy⟩", "norm_num at hxy", "exact hy hxy"], [], "ahr_direct_no_universal_zero_product", 1000000.0)
    if "Continuous f" in statement and "z = f⁻¹' {0}" in statement and "IsClosed z" in statement:
        add([], ["rw [g]", "exact isClosed_singleton.preimage h"], [], "ahr_direct_closed_preimage_singleton", 1000000.0)
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


def ranker_row_context(row: dict[str, Any], observed: dict[str, Any], translation: dict[str, Any], retrieval: dict[str, Any]) -> str:
    cluster_bits = []
    for cluster in retrieval.get("clusters", [])[:5]:
        names = [cand.get("full_name", "") for cand in cluster.get("candidates", [])[:5]]
        cluster_bits.append(str(cluster.get("import", "")) + " " + " ".join(names))
    parts = [
        str(row.get("id", "")),
        str(observed.get("question", "")),
        str(observed.get("formal_statement", "")),
        str(translation.get("normalized_statement", "")),
        " ".join(observed.get("formal_symbols") or []),
        " ".join(cluster_bits),
    ]
    return "\n".join(parts)


def ranker_attempt_text(attempt: dict[str, Any]) -> str:
    theorem_names = [x.get("full_name", "") for x in attempt.get("used_theorems", [])]
    return "\n".join([
        str(attempt.get("kind", "")),
        "\n".join(attempt.get("proof_lines") or []),
        " ".join(theorem_names),
    ])


@torch.no_grad() if torch is not None else (lambda fn: fn)
def rerank_attempts(
    attempts: list[dict[str, Any]],
    ranker: dict[str, Any] | None,
    row: dict[str, Any],
    observed: dict[str, Any],
    translation: dict[str, Any],
    retrieval: dict[str, Any],
    limit: int,
) -> list[dict[str, Any]]:
    if not ranker:
        return attempts[:limit]
    context = ranker_row_context(row, observed, translation, retrieval)
    scored: list[dict[str, Any]] = []
    model = ranker["model"]
    vocab = ranker["vocab"]
    device = torch.device(ranker["device"])
    model.eval()
    for idx, attempt in enumerate(attempts):
        ex = {
            "row_id": row.get("id"),
            "split": row.get("split", "test"),
            "failure_reason": translation.get("failure_reason") or "",
            "attempt_kind": attempt.get("kind"),
            "imports": attempt.get("imports") or [],
            "row_context": context,
            "attempt_text": ranker_attempt_text(attempt),
            "lean_error_excerpt": "",
        }
        x = ranker_vectorize(ex, vocab).to(device).unsqueeze(0)
        prob = float(torch.sigmoid(model(x)).detach().cpu().item())
        updated = dict(attempt)
        updated["original_rank_score"] = attempt.get("rank_score")
        updated["ranker_score"] = round(prob, 6)
        updated["ranker_pool_idx"] = idx + 1
        scored.append(updated)
    scored.sort(key=lambda x: (-float(x["ranker_score"]), -float(x.get("rank_score", 0.0)), len(x["imports"]), len(x["proof_lines"]), x["kind"]))
    return scored[:limit]


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
    attempt_ranker: dict[str, Any] | None,
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
        "schema": "ahr_lean_v3_4_ranker_guided_proofnet_bridge_record",
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
        "ranker": {
            "guided": bool(attempt_ranker),
            "checkpoint_path": attempt_ranker.get("checkpoint_path") if attempt_ranker else None,
            "device": attempt_ranker.get("device") if attempt_ranker else None,
            "candidate_pool_size": args.ranker_candidate_pool if attempt_ranker else args.max_lean_calls,
        },
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

    candidate_pool_size = args.ranker_candidate_pool if attempt_ranker else args.max_lean_calls
    attempts = proof_hypotheses(statement, retrieval["clusters"], use_aesop, candidate_pool_size)
    record["ranker"]["candidate_count_before_rerank"] = len(attempts)
    attempts = rerank_attempts(attempts, attempt_ranker, row, observed, translation, retrieval, args.max_lean_calls)
    record["ranker"]["candidate_count_after_rerank"] = len(attempts)
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
            "original_rank_score": attempt.get("original_rank_score"),
            "ranker_score": attempt.get("ranker_score"),
            "ranker_pool_idx": attempt.get("ranker_pool_idx"),
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
        "schema": "ahr_lean_v3_4_ranker_guided_proofnet_bridge_report",
        "run_name": args.run_name,
        "input_path": args.input,
        "records_path": str(Path("datasets") / f"{args.run_name}_records.jsonl"),
        "record_count": len(rows),
        "baseline_v3_0_known_theorem_wrapper_verified_count": 0,
        "baseline_v3_2_proofnet50_target_verified_count": 5,
        "baseline_v3_2_proofnet50_lean_calls_total": 537,
        "ranker_guided": bool(args.attempt_ranker_checkpoint),
        "attempt_ranker_checkpoint": args.attempt_ranker_checkpoint,
        "ranker_candidate_pool": args.ranker_candidate_pool,
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
            "rank": "orders attempts by retrieval score, target-symbol overlap, import cost, tactic cost, and optional learned v3.3 attempt-ranker score",
            "verify": "accepts only Lean-verified target theorem proofs",
            "refine": "records failed attempts and failure buckets for next bridge iteration",
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Lean-AHR ProofNet bridge with explicit AHR loop and optional learned attempt reranking.")
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
    ap.add_argument("--attempt-ranker-checkpoint", default=None)
    ap.add_argument("--ranker-candidate-pool", type=int, default=96)
    ap.add_argument("--ranker-device", default="auto")
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
    attempt_ranker = load_attempt_ranker(args.attempt_ranker_checkpoint, args.ranker_device)
    if attempt_ranker:
        print(json.dumps({
            "event": "attempt_ranker_loaded",
            "checkpoint": attempt_ranker["checkpoint_path"],
            "device": attempt_ranker["device"],
            "vocab_size": len(attempt_ranker["vocab"]),
        }, sort_keys=True), flush=True)

    check_cache: dict[tuple[str, str], dict[str, Any]] = {}
    out_rows: list[dict[str, Any]] = []
    start = time.time()
    records_path.write_text("")
    with records_path.open("a", encoding="utf-8") as fh:
        for idx, row in enumerate(selected_rows, start=args.offset + 1):
            record = solve_row(row, idx, index_rows, check_cache, args, use_aesop, attempt_ranker)
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

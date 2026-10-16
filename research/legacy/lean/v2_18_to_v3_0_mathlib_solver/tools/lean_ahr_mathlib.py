#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import re
import subprocess
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_WORKSPACE = Path("lean_ahr_mathlib_workspace")
DEFAULT_INDEX = Path("datasets/lean_ahr_mathlib_theorem_index.jsonl")


TOKEN_RE = re.compile(r"[A-Za-z0-9_']+")


def compact(text: str | None, limit: int = 900) -> str | None:
    if not text:
        return None
    clean = " ".join(str(text).strip().split())
    return clean if len(clean) <= limit else clean[: limit - 3] + "..."


def tokens(text: str) -> list[str]:
    base = [t.lower() for t in TOKEN_RE.findall(text or "")]
    extra: list[str] = []
    joined = " ".join(base)
    if "irrationality" in base:
        extra.append("irrational")
    if "square" in base and "root" in base:
        extra.extend(["sqrt", "sqrt_two"])
    if "pi" in base or "π" in (text or ""):
        extra.extend(["real", "pi", "irrational_pi"])
    if "sqrt" in base or "√" in (text or ""):
        extra.extend(["sqrt", "irrational_sqrt", "irrational_sqrt_two"])
    if "irrational" in base:
        extra.extend(["irrational", "not", "rat"])
    if "divides" in joined or "divisibility" in joined:
        extra.extend(["dvd", "nat", "divisibility"])
    if "less" in joined or "inequality" in joined:
        extra.extend(["le", "lt", "nat", "inequality"])
    return base + extra


def module_from_path(path: Path) -> str:
    parts = list(path.with_suffix("").parts)
    idx = parts.index("Mathlib") if "Mathlib" in parts else 0
    return ".".join(parts[idx:])


def run_lake_lean(workspace: Path, lean_text: str, timeout_s: int = 30) -> dict[str, Any]:
    start = time.time()
    with tempfile.NamedTemporaryFile("w", suffix=".lean", delete=False) as f:
        f.write(lean_text)
        tmp = Path(f.name)
    try:
        proc = subprocess.run(
            ["/home/benjamin/.elan/bin/lake", "env", "lean", str(tmp)],
            cwd=str(workspace),
            text=True,
            capture_output=True,
            timeout=timeout_s,
        )
        return {
            "accepted": proc.returncode == 0,
            "returncode": proc.returncode,
            "elapsed_s": round(time.time() - start, 4),
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "stdout_excerpt": compact(proc.stdout),
            "stderr_excerpt": None if proc.returncode == 0 else compact(proc.stderr or proc.stdout),
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "accepted": False,
            "returncode": -1,
            "elapsed_s": round(time.time() - start, 4),
            "stdout": exc.stdout if isinstance(exc.stdout, str) else "",
            "stderr": exc.stderr if isinstance(exc.stderr, str) else "",
            "stdout_excerpt": compact(exc.stdout if isinstance(exc.stdout, str) else ""),
            "stderr_excerpt": "timeout",
        }
    finally:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass


def check_decl(workspace: Path, module: str, full_name: str, timeout_s: int = 30) -> dict[str, Any]:
    res = run_lake_lean(workspace, f"import {module}\n#check {full_name}\n", timeout_s)
    return {k: v for k, v in res.items() if k not in {"stdout", "stderr"}}


def verify_theorem(
    workspace: Path,
    imports: list[str],
    theorem_name: str,
    statement: str,
    proof_lines: list[str],
    timeout_s: int = 30,
) -> dict[str, Any]:
    stmt = statement.replace("THEOREM_NAME", theorem_name)
    lines = [f"import {imp}" for imp in imports]
    lines += ["", "set_option pp.all false", "", stmt]
    lines += ["  " + line for line in proof_lines]
    lines.append("")
    res = run_lake_lean(workspace, "\n".join(lines), timeout_s)
    return {k: v for k, v in res.items() if k not in {"stdout", "stderr"}}


def load_index(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def score_rows(query: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    q = Counter(tokens(query))
    scored: list[dict[str, Any]] = []
    for row in rows:
        rtoks = row.get("tokens") or tokens(" ".join([
            row.get("full_name", ""),
            row.get("statement_excerpt", ""),
            row.get("doc", "") or "",
        ]))
        r = Counter(rtoks)
        overlap = sum((1.0 + math.log1p(r[t])) * w for t, w in q.items() if t in r)
        name = row.get("full_name", "").lower()
        bonus = 0.0
        for t in q:
            if t and t in name:
                bonus += 2.5
        if "pi" in q and "irrational" in q and "irrational_pi" in name:
            bonus += 50.0
        if "sqrt" in q and "irrational" in q and "two" in q and "irrational_sqrt_two" in name:
            bonus += 50.0
        qwords = set(q)
        phrase_bonuses = [
            ({"nat", "le", "trans"}, "nat.le_trans", 45.0),
            ({"nat", "dvd", "trans"}, "nat.dvd_trans", 45.0),
            ({"nat", "dvd", "add"}, "nat.dvd_add", 45.0),
            ({"nat", "add", "comm"}, "nat.add_comm", 45.0),
            ({"nat", "mul", "comm"}, "nat.mul_comm", 45.0),
            ({"nat", "le", "refl"}, "nat.le_refl", 45.0),
            ({"zero", "less", "one"}, "zero_lt_one", 35.0),
        ]
        for words, target_name, value in phrase_bonuses:
            if words <= qwords and target_name in name:
                bonus += value
        if overlap + bonus > 0:
            item = dict(row)
            item["score"] = round(overlap + bonus, 4)
            scored.append(item)
    scored.sort(key=lambda x: (-float(x["score"]), x.get("full_name", "")))
    return scored


def retrieve(
    query: str,
    top_k: int,
    index_path: Path = DEFAULT_INDEX,
    workspace: Path = DEFAULT_WORKSPACE,
    verify: bool = True,
    timeout_s: int = 30,
) -> list[dict[str, Any]]:
    rows = score_rows(query, load_index(index_path))
    out: list[dict[str, Any]] = []
    for row in rows[: max(top_k * 5, top_k)]:
        item = {
            "full_name": row["full_name"],
            "name": row["name"],
            "namespace": row.get("namespace"),
            "import": row["import"],
            "source_file": row["source_file"],
            "line": row["line"],
            "kind": row["kind"],
            "score": row["score"],
            "statement_excerpt": row.get("statement_excerpt"),
            "doc": row.get("doc"),
        }
        if verify:
            item["check"] = check_decl(workspace, row["import"], row["full_name"], timeout_s)
            if not item["check"].get("accepted"):
                continue
        out.append(item)
        if len(out) >= top_k:
            break
    return out

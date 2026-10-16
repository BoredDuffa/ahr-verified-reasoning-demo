#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from lean_ahr_mathlib import DEFAULT_INDEX, DEFAULT_WORKSPACE, module_from_path, tokens


DECL_RE = re.compile(r"^\s*(?:@\[[^\]]+\]\s*)?(?:(protected)\s+)?(theorem|lemma)\s+([A-Za-z_][A-Za-z0-9_'.]*)\b(.*)$")


BUILTIN_DECLS = [
    ("Nat.le_trans", "theorem Nat.le_trans : a ≤ b -> b ≤ c -> a ≤ c"),
    ("Nat.dvd_trans", "theorem Nat.dvd_trans : a ∣ b -> b ∣ c -> a ∣ c"),
    ("Nat.dvd_add", "theorem Nat.dvd_add : a ∣ b -> a ∣ c -> a ∣ b + c"),
    ("Nat.add_comm", "theorem Nat.add_comm : a + b = b + a"),
    ("Nat.mul_comm", "theorem Nat.mul_comm : a * b = b * a"),
    ("Nat.le_refl", "theorem Nat.le_refl : a ≤ a"),
    ("zero_lt_one", "theorem zero_lt_one : 0 < 1"),
]


def consume_statement(lines: list[str], i: int) -> str:
    parts = [lines[i].strip()]
    j = i + 1
    while j < len(lines) and len(" ".join(parts)) < 1200:
        text = lines[j].strip()
        if not text:
            break
        if re.match(r"^\s*(theorem|lemma|def|class|instance|namespace|section|end)\b", text):
            break
        parts.append(text)
        if " := " in text or text.endswith(" := by") or text == ":= by" or text.startswith(":= by"):
            break
        j += 1
    return " ".join(parts)


def nearby_doc(lines: list[str], i: int) -> str | None:
    docs: list[str] = []
    j = i - 1
    while j >= 0 and len(docs) < 8:
        text = lines[j].strip()
        if not text:
            if docs:
                break
            j -= 1
            continue
        if text.startswith("/--") or text.startswith("--"):
            docs.append(text.strip("-/ "))
            j -= 1
            continue
        break
    if not docs:
        return None
    return " ".join(reversed(docs))[:900]


def build_index(workspace: Path, output: Path, limit: int | None = None) -> dict[str, int]:
    mathlib_root = workspace / ".lake/packages/mathlib/Mathlib"
    if not mathlib_root.exists():
        raise RuntimeError(f"mathlib source root not found: {mathlib_root}")
    count = 0
    files = 0
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as out:
        for name, statement in BUILTIN_DECLS:
            row = {
                "name": name.split(".")[-1],
                "full_name": name,
                "namespace": ".".join(name.split(".")[:-1]) or None,
                "kind": "theorem",
                "import": "Mathlib",
                "source_file": "builtin",
                "line": 0,
                "statement_excerpt": statement,
                "doc": "Lean/Mathlib builtin seed declaration for retrieval.",
            }
            row["tokens"] = tokens(" ".join([name, statement, row["doc"]]))
            out.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
        for path in sorted(mathlib_root.rglob("*.lean")):
            files += 1
            module = module_from_path(path)
            rel = str(path.relative_to(workspace))
            lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
            namespace_stack: list[str] = []
            for i, line in enumerate(lines):
                ns = re.match(r"^\s*namespace\s+([A-Za-z_][A-Za-z0-9_'.]*)\b", line)
                if ns:
                    namespace_stack.extend(ns.group(1).split("."))
                    continue
                end = re.match(r"^\s*end(?:\s+([A-Za-z_][A-Za-z0-9_'.]*))?\b", line)
                if end and namespace_stack:
                    name = end.group(1)
                    if name and namespace_stack[-1] == name.split(".")[-1]:
                        namespace_stack.pop()
                    elif not name:
                        namespace_stack.pop()
                    continue
                m = DECL_RE.match(line)
                if not m:
                    continue
                kind = m.group(2)
                name = m.group(3)
                namespace = ".".join(namespace_stack) if namespace_stack else None
                full_name = name if "." in name or not namespace else f"{namespace}.{name}"
                statement = consume_statement(lines, i)
                doc = nearby_doc(lines, i)
                row = {
                    "name": name,
                    "full_name": full_name,
                    "namespace": namespace,
                    "kind": kind,
                    "import": module,
                    "source_file": rel,
                    "line": i + 1,
                    "statement_excerpt": statement[:1000],
                    "doc": doc,
                }
                row["tokens"] = tokens(" ".join([full_name, module, statement, doc or ""]))
                out.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                count += 1
                if limit and count >= limit:
                    return {"files_seen": files, "declarations": count}
    return {"files_seen": files, "declarations": count}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workspace", default=str(DEFAULT_WORKSPACE))
    ap.add_argument("--output", default=str(DEFAULT_INDEX))
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    report = build_index(Path(args.workspace), Path(args.output), args.limit)
    report["workspace"] = args.workspace
    report["output"] = args.output
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()

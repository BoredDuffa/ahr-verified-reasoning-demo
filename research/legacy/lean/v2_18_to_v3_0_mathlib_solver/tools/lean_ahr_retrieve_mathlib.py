#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from lean_ahr_mathlib import DEFAULT_INDEX, DEFAULT_WORKSPACE, retrieve


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", required=True)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--index", default=str(DEFAULT_INDEX))
    ap.add_argument("--workspace", default=str(DEFAULT_WORKSPACE))
    ap.add_argument("--timeout-s", type=int, default=30)
    ap.add_argument("--no-verify", action="store_true")
    args = ap.parse_args()
    rows = retrieve(
        args.query,
        args.top_k,
        Path(args.index),
        Path(args.workspace),
        verify=not args.no_verify,
        timeout_s=args.timeout_s,
    )
    print(json.dumps({
        "schema": "ahr_lean_mathlib_retrieval_v2_18",
        "query": args.query,
        "top_k": args.top_k,
        "count": len(rows),
        "candidates": rows,
    }, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()

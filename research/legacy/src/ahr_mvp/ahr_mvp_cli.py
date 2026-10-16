#!/usr/bin/env python3
"""Interactive MVP CLI for verified AHR-CSP reasoning artifacts.

This is intentionally narrow. It exposes the stable operation-synthesis
frontier and deterministic verifier-backed utilities before a broader natural
language interface is ready.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from answer import format_verified_answer
from graph_csp import search_graph, verify_graph
from hanoi import solve_rule_revision, verify_hanoi_solution
from operation_synthesis import OP_NAMES, synthesize, verify
from router import route_prompt, route_prompt_file

ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = ROOT / "reports" / "reasoning_progress_since_last_report_20260509"
EXAMPLE_DIR = Path(__file__).resolve().parent / "examples"

FRONTIER_REPORTS = {
    "v16k": REPORT_DIR / "phase5r16k_operation_synthesis_len3_hint_bridge_fullgpu_b64_report.json",
    "v16l": REPORT_DIR / "phase5r16l_operation_synthesis_len3_dehint_bridge_fullgpu_b64_report.json",
    "v16m": REPORT_DIR / "phase5r16m_operation_synthesis_len3_dehint_bridge_fullgpu_b64_report.json",
}

EXAMPLE_PROMPTS = {
    "graph_two_triangles_unsat": EXAMPLE_DIR / "graph_two_triangles_unsat.txt",
    "hanoi_5disk_after8_bcap2": EXAMPLE_DIR / "hanoi_5disk_after8_bcap2.txt",
    "operation_len5_unsat": EXAMPLE_DIR / "operation_len5_unsat.txt",
}


def parse_int_list(text: str) -> list[int]:
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(f"expected JSON integer list: {exc}") from exc
    if not isinstance(value, list) or not all(isinstance(x, int) for x in value):
        raise argparse.ArgumentTypeError("expected JSON integer list, e.g. '[1, 4, 9]'")
    return value


def parse_edge_list(text: str) -> list[list[int]]:
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(f"expected JSON edge list: {exc}") from exc
    if not isinstance(value, list) or not all(isinstance(edge, list) and len(edge) == 2 for edge in value):
        raise argparse.ArgumentTypeError("expected JSON edge list, e.g. '[[0, 1], [1, 2]]'")
    return value


def load_report(path: Path) -> dict:
    if not path.exists():
        return {"available": False, "path": str(path)}
    data = json.loads(path.read_text())
    split_reports = data.get("split_reports", {})
    metrics = {}
    for split in ("train", "val", "test"):
        split_metrics = split_reports.get(split, {}).get("metrics", {})
        metrics[split] = {
            "cpu_validity": split_metrics.get("cpu_valid_rate"),
            "mean_hard_energy": split_metrics.get("mean_hard_energy"),
        }
    return {
        "available": True,
        "path": str(path),
        "passed": data.get("passed"),
        "dataset_path": data.get("dataset_path"),
        "metrics": metrics,
    }


def cmd_status(_args: argparse.Namespace) -> int:
    result = {
        "clinical_use": "forbidden",
        "mvp_scope": "operation_synthesis_graph_csp_hanoi_exact_verifiers_and_frontier_status",
        "supported_operations": list(OP_NAMES),
        "supported_modules": [
            "operation_synthesis_exact_search",
            "small_graph_exact_search",
            "hanoi_mid_rule_revision_bfs",
        ],
        "frontier": {name: load_report(path) for name, path in FRONTIER_REPORTS.items()},
        "notes": [
            "v16l is the current passable de-hint frontier.",
            "v16m failed at 30% unhinted length-3; do not treat it as a stable checkpoint.",
        ],
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def cmd_synthesize(args: argparse.Namespace) -> int:
    result = synthesize(args.start, args.target, args.length)
    payload = {
        "clinical_use": "forbidden",
        "kind": "operation_synthesis_exact_search",
        "start": list(result.start),
        "target": list(result.target),
        "length": result.length,
        "checked": result.checked,
        "conclusion": result.conclusion,
        "solutions": [list(program) for program in result.solutions],
        "closest": [
            {
                "score": score,
                "program": list(program),
                "output": list(output),
                "truncated": truncated,
            }
            for score, program, output, truncated in result.closest
        ],
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if result.passed else 2


def cmd_verify(args: argparse.Namespace) -> int:
    payload = {
        "clinical_use": "forbidden",
        "kind": "operation_synthesis_program_verification",
        "start": args.start,
        "target": args.target,
        **verify(args.start, args.target, args.program),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["valid"] else 2


def cmd_graph_search(args: argparse.Namespace) -> int:
    result = search_graph(
        vertices=args.vertices,
        diameter=args.diameter,
        edge_critical_connected=args.edge_critical_connected,
        triangles=args.triangles,
    )
    payload = {
        "clinical_use": "forbidden",
        "kind": "small_graph_exact_search",
        "vertices": result.vertices,
        "checked": result.checked,
        "conclusion": result.conclusion,
        "solution": [list(edge) for edge in result.solution] if result.solution is not None else None,
        "verification": result.verification,
        "constraints": {
            "connected": True,
            "diameter": args.diameter,
            "edge_critical_connected": args.edge_critical_connected,
            "triangles": args.triangles,
        },
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if result.passed else 2


def cmd_verify_graph(args: argparse.Namespace) -> int:
    payload = {
        "clinical_use": "forbidden",
        "kind": "small_graph_verification",
        **verify_graph(
            vertices=args.vertices,
            edges=args.edges,
            diameter=args.diameter,
            edge_critical_connected=args.edge_critical_connected,
            triangles=args.triangles,
        ),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["valid"] else 2


def cmd_hanoi_revision(args: argparse.Namespace) -> int:
    result = solve_rule_revision(
        disks=args.disks,
        start=args.start,
        target=args.target,
        switch_after=args.switch_after,
        capped_peg=args.capped_peg,
        cap=args.cap,
        max_depth=args.max_depth,
    )
    verification = None
    if result.moves is not None:
        verification = verify_hanoi_solution(
            disks=args.disks,
            moves=result.moves,
            start=args.start,
            target=args.target,
            switch_after=args.switch_after,
            capped_peg=args.capped_peg,
            cap=args.cap,
        )
    payload = {
        "clinical_use": "forbidden",
        "kind": "hanoi_mid_rule_revision_bfs",
        "disks": result.disks,
        "start": result.start,
        "target": result.target,
        "switch_after": result.switch_after,
        "capped_peg": result.capped_peg,
        "cap": result.cap,
        "checked_states": result.checked_states,
        "conclusion": result.conclusion,
        "move_count": len(result.moves) if result.moves is not None else None,
        "moves": [
            {"disk": disk, "from": src, "to": dst}
            for disk, src, dst in result.moves
        ]
        if result.moves is not None
        else None,
        "verification": verification,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if result.passed else 2


def cmd_route_prompt(args: argparse.Namespace) -> int:
    if args.prompt_file:
        payload = route_prompt_file(args.prompt_file)
    else:
        payload = route_prompt(args.prompt)
    print(json.dumps(payload, indent=2, sort_keys=True))
    conclusion = payload.get("result", {}).get("conclusion")
    return 0 if conclusion == "sat" else 2


def cmd_ask(args: argparse.Namespace) -> int:
    if args.prompt_file:
        payload = route_prompt_file(args.prompt_file)
    else:
        payload = route_prompt(args.prompt)
    print(format_verified_answer(payload))
    if args.show_json:
        print("\nRaw evidence:")
        print(json.dumps(payload, indent=2, sort_keys=True))
    return 2 if payload.get("module") == "unsupported" else 0


def cmd_demo(args: argparse.Namespace) -> int:
    payloads = []
    unsupported = False
    for name, path in EXAMPLE_PROMPTS.items():
        payload = route_prompt_file(path)
        payloads.append((name, payload))
        unsupported = unsupported or payload.get("module") == "unsupported"
        print(f"## {name}")
        print(format_verified_answer(payload))
        print()
    if args.json:
        print("Raw evidence:")
        print(json.dumps({name: payload for name, payload in payloads}, indent=2, sort_keys=True))
    return 2 if unsupported else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_status = sub.add_parser("status", help="Show the current archived AHR operation-synthesis frontier.")
    p_status.set_defaults(func=cmd_status)

    p_syn = sub.add_parser("synthesize-operation", help="Exact search for a fixed-length operation program.")
    p_syn.add_argument("--start", required=True, type=parse_int_list)
    p_syn.add_argument("--target", required=True, type=parse_int_list)
    p_syn.add_argument("--length", required=True, type=int)
    p_syn.set_defaults(func=cmd_synthesize)

    p_verify = sub.add_parser("verify-operation", help="Verify a proposed operation program.")
    p_verify.add_argument("--start", required=True, type=parse_int_list)
    p_verify.add_argument("--target", required=True, type=parse_int_list)
    p_verify.add_argument("--program", nargs="+", required=True, choices=OP_NAMES)
    p_verify.set_defaults(func=cmd_verify)

    p_graph = sub.add_parser("graph-search", help="Exact search for a small simple graph satisfying constraints.")
    p_graph.add_argument("--vertices", required=True, type=int)
    p_graph.add_argument("--diameter", type=int)
    p_graph.add_argument("--triangles", type=int)
    p_graph.add_argument("--edge-critical-connected", action="store_true")
    p_graph.set_defaults(func=cmd_graph_search)

    p_verify_graph = sub.add_parser("verify-graph", help="Verify a proposed small graph against constraints.")
    p_verify_graph.add_argument("--vertices", required=True, type=int)
    p_verify_graph.add_argument("--edges", required=True, type=parse_edge_list)
    p_verify_graph.add_argument("--diameter", type=int)
    p_verify_graph.add_argument("--triangles", type=int)
    p_verify_graph.add_argument("--edge-critical-connected", action="store_true")
    p_verify_graph.set_defaults(func=cmd_verify_graph)

    p_hanoi = sub.add_parser("hanoi-revision", help="BFS solve Tower of Hanoi with a mid-trajectory peg cap.")
    p_hanoi.add_argument("--disks", required=True, type=int)
    p_hanoi.add_argument("--start", default="A", choices=("A", "B", "C"))
    p_hanoi.add_argument("--target", default="C", choices=("A", "B", "C"))
    p_hanoi.add_argument("--switch-after", type=int, default=8)
    p_hanoi.add_argument("--capped-peg", default="B", choices=("A", "B", "C"))
    p_hanoi.add_argument("--cap", type=int, default=2)
    p_hanoi.add_argument("--max-depth", type=int)
    p_hanoi.set_defaults(func=cmd_hanoi_revision)

    p_route = sub.add_parser("route-prompt", help="Route a known prompt shape into the narrow MVP solvers.")
    route_source = p_route.add_mutually_exclusive_group(required=True)
    route_source.add_argument("--prompt")
    route_source.add_argument("--prompt-file", type=Path)
    p_route.set_defaults(func=cmd_route_prompt)

    p_ask = sub.add_parser("ask", help="Answer a known prompt shape with concise verified evidence.")
    ask_source = p_ask.add_mutually_exclusive_group(required=True)
    ask_source.add_argument("--prompt")
    ask_source.add_argument("--prompt-file", type=Path)
    p_ask.add_argument("--show-json", action="store_true", help="Append the raw routed evidence JSON.")
    p_ask.set_defaults(func=cmd_ask)

    p_demo = sub.add_parser("demo", help="Run all saved MVP example prompts and print verified answers.")
    p_demo.add_argument("--json", action="store_true", help="Append raw routed evidence for every example.")
    p_demo.set_defaults(func=cmd_demo)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

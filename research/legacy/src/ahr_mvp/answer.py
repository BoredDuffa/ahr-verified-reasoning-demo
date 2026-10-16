#!/usr/bin/env python3
"""Human-readable answer formatting for routed AHR MVP results."""

from __future__ import annotations

from typing import Any


def _constraint_summary(parsed: dict[str, Any]) -> str:
    parts = ["connected"]
    if parsed.get("diameter") is not None:
        parts.append(f"diameter exactly {parsed['diameter']}")
    if parsed.get("edge_critical_connected"):
        parts.append("disconnects after any single edge removal")
    if parsed.get("triangles") is not None:
        parts.append(f"exactly {parsed['triangles']} triangles")
    return ", ".join(parts)


def _format_adjacency(verification: dict[str, Any]) -> str:
    adjacency = verification.get("adjacency", {})
    rows = []
    for node in sorted(adjacency, key=lambda item: int(item)):
        rows.append(f"{node}: {adjacency[node]}")
    return "\n".join(rows)


def _format_graph(payload: dict[str, Any]) -> str:
    parsed = payload["parsed"]
    result = payload["result"]
    if result["conclusion"] != "sat":
        return (
            "No such graph exists under the MVP's exhaustive simple-graph search.\n"
            f"Evidence: checked {result['checked']} graphs on {parsed['vertices']} vertices; "
            f"none satisfied {_constraint_summary(parsed)}."
        )
    verification = result["verification"] or {}
    checks = verification.get("checks", {})
    return (
        "A graph exists.\n"
        "Adjacency list:\n"
        f"{_format_adjacency(verification)}\n"
        "Verification: "
        f"connected={checks.get('connected')}, "
        f"diameter={checks.get('diameter')}, "
        f"edge_critical_connected={checks.get('edge_critical_connected')}, "
        f"triangle_count={checks.get('triangle_count')}."
    )


def _format_hanoi(payload: dict[str, Any]) -> str:
    parsed = payload["parsed"]
    result = payload["result"]
    if result["conclusion"] != "sat":
        return (
            "No legal completion exists under the MVP's bounded Hanoi rule-revision search.\n"
            f"Evidence: checked {result['checked_states']} states for {parsed['disks']} disks "
            f"from peg {parsed['start']} to peg {parsed['target']}; after move {parsed['switch_after']}, "
            f"peg {parsed['capped_peg']} may hold at most {parsed['cap']} disks."
        )
    moves = result["moves"] or []
    rendered_moves = "\n".join(
        f"{index}. move disk {move['disk']} from {move['from']} to {move['to']}"
        for index, move in enumerate(moves, start=1)
    )
    return (
        f"A legal minimum completion exists in {result['move_count']} moves.\n"
        f"{rendered_moves}\n"
        f"Verification: {result['verification']}."
    )


def _format_operation(payload: dict[str, Any]) -> str:
    parsed = payload["parsed"]
    result = payload["result"]
    if result["conclusion"] == "sat":
        programs = [" -> ".join(program) for program in result["solutions"]]
        return (
            "A composition exists.\n"
            f"Input: {parsed['start']}\n"
            f"Target: {parsed['target']}\n"
            f"Length: {parsed['length']}\n"
            f"Programs: {programs}"
        )

    closest = result.get("closest", [])[:3]
    closest_lines = "\n".join(
        f"- {' -> '.join(item['program'])}: output {item['output']} with score {item['score']}"
        for item in closest
    )
    return (
        "No composition exists under the declared operation semantics.\n"
        f"Evidence: checked all {result['checked']} programs of length {parsed['length']} from "
        f"{parsed['start']} to {parsed['target']}.\n"
        f"Closest candidates:\n{closest_lines}"
    )


def format_verified_answer(payload: dict[str, Any]) -> str:
    """Convert one routed payload into a concise answer with explicit evidence."""
    module = payload.get("module")
    if module == "graph":
        return _format_graph(payload)
    if module == "hanoi":
        return _format_hanoi(payload)
    if module == "operation_synthesis":
        return _format_operation(payload)
    supported = payload.get("result", {}).get("supported_modules", [])
    return (
        "Unsupported by the current MVP router.\n"
        f"Supported modules: {supported}.\n"
        "Use route-prompt for the raw parser output."
    )

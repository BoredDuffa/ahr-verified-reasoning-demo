#!/usr/bin/env python3
"""Conservative prompt router for the narrow AHR MVP harness."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from graph_csp import search_graph
from hanoi import solve_rule_revision, verify_hanoi_solution
from operation_synthesis import synthesize

NUMBER_WORDS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
}


def _number_value(text: str | None) -> int | None:
    if text is None:
        return None
    value = text.lower()
    if value.isdigit():
        return int(value)
    return NUMBER_WORDS.get(value)


def _extract_number(patterns: list[str], text: str) -> int | None:
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            value = _number_value(match.group(1))
            if value is not None:
                return value
    return None


def _extract_lists(text: str) -> list[list[int]]:
    lists: list[list[int]] = []
    for raw in re.findall(r"\[[^\]]*\]", text):
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(value, list) and all(isinstance(item, int) for item in value):
            lists.append(value)
    return lists


def _route_graph(prompt: str) -> dict[str, Any] | None:
    lowered = prompt.lower()
    graph_terms = ("graph" in lowered) or ("vertices" in lowered) or ("vertex" in lowered)
    if not graph_terms:
        return None
    vertices = _extract_number([r"\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten)[-\s]?vertex"], prompt)
    diameter = _extract_number([r"\bdiameter\s+(?:exactly\s+)?(\d+|one|two|three|four|five|six|seven|eight|nine|ten)"], prompt)
    triangles = _extract_number([r"\bexactly\s+(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+triangles?"], prompt)
    edge_critical = (
        ("any single edge" in lowered or "single edge" in lowered or "edge is removed" in lowered)
        and ("disconnected" in lowered or "removed" in lowered)
    )
    if vertices is None:
        return None

    result = search_graph(
        vertices=vertices,
        diameter=diameter,
        edge_critical_connected=edge_critical,
        triangles=triangles,
    )
    return {
        "clinical_use": "forbidden",
        "kind": "ahr_mvp_routed_prompt",
        "module": "graph",
        "router_confidence": 0.9,
        "parsed": {
            "vertices": vertices,
            "diameter": diameter,
            "edge_critical_connected": edge_critical,
            "triangles": triangles,
        },
        "result": {
            "kind": "small_graph_exact_search",
            "vertices": result.vertices,
            "checked": result.checked,
            "conclusion": result.conclusion,
            "solution": [list(edge) for edge in result.solution] if result.solution is not None else None,
            "verification": result.verification,
        },
    }


def _route_hanoi(prompt: str) -> dict[str, Any] | None:
    lowered = prompt.lower()
    if "hanoi" not in lowered and "disk" not in lowered:
        return None
    disks = _extract_number([r"\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+disks?"], prompt)
    switch_after = _extract_number([r"\bafter\s+move\s+(\d+|one|two|three|four|five|six|seven|eight|nine|ten)"], prompt)
    cap = _extract_number([r"\bat\s+most\s+(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+disks?"], prompt)
    start_target = re.search(r"\bfrom\s+peg\s+([ABC])\s+to\s+peg\s+([ABC])", prompt, flags=re.IGNORECASE)
    capped_peg = re.search(r"\bpeg\s+([ABC])\s+may\s+hold\s+at\s+most", prompt, flags=re.IGNORECASE)
    if disks is None:
        return None

    start = start_target.group(1).upper() if start_target else "A"
    target = start_target.group(2).upper() if start_target else "C"
    result = solve_rule_revision(
        disks=disks,
        start=start,
        target=target,
        switch_after=switch_after if switch_after is not None else 8,
        capped_peg=capped_peg.group(1).upper() if capped_peg else "B",
        cap=cap if cap is not None else 2,
    )
    verification = None
    if result.moves is not None:
        verification = verify_hanoi_solution(
            disks=result.disks,
            moves=result.moves,
            start=result.start,
            target=result.target,
            switch_after=result.switch_after,
            capped_peg=result.capped_peg,
            cap=result.cap,
        )
    return {
        "clinical_use": "forbidden",
        "kind": "ahr_mvp_routed_prompt",
        "module": "hanoi",
        "router_confidence": 0.9,
        "parsed": {
            "disks": result.disks,
            "start": result.start,
            "target": result.target,
            "switch_after": result.switch_after,
            "capped_peg": result.capped_peg,
            "cap": result.cap,
        },
        "result": {
            "kind": "hanoi_mid_rule_revision_bfs",
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
        },
    }


def _route_operation(prompt: str) -> dict[str, Any] | None:
    lowered = prompt.lower()
    has_operation_terms = any(name.lower() in lowered for name in ("rev", "delta", "double-evens", "zip-self"))
    lists = _extract_lists(prompt)
    if not has_operation_terms or len(lists) < 2:
        return None
    length = _extract_number(
        [
            r"\bexactly\s+(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+of\s+these\s+operations",
            r"\bcomposition\s+of\s+exactly\s+(\d+|one|two|three|four|five|six|seven|eight|nine|ten)",
            r"\blength\s+(\d+|one|two|three|four|five|six|seven|eight|nine|ten)",
        ],
        prompt,
    )
    if length is None:
        return None

    result = synthesize(lists[0], lists[1], length)
    return {
        "clinical_use": "forbidden",
        "kind": "ahr_mvp_routed_prompt",
        "module": "operation_synthesis",
        "router_confidence": 0.85,
        "parsed": {
            "start": lists[0],
            "target": lists[1],
            "length": length,
        },
        "result": {
            "kind": "operation_synthesis_exact_search",
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
        },
    }


def route_prompt(prompt: str) -> dict[str, Any]:
    """Route one prompt into one of the narrow exact MVP modules.

    This is intentionally a conservative parser for known prompt families, not
    a broad natural-language interface.
    """
    for router in (_route_graph, _route_hanoi, _route_operation):
        routed = router(prompt)
        if routed is not None:
            return routed
    return {
        "clinical_use": "forbidden",
        "kind": "ahr_mvp_routed_prompt",
        "module": "unsupported",
        "router_confidence": 0.0,
        "parsed": {},
        "result": {
            "conclusion": "unsupported_prompt_for_current_mvp",
            "supported_modules": ["graph", "hanoi", "operation_synthesis"],
        },
    }


def route_prompt_file(path: Path) -> dict[str, Any]:
    return route_prompt(path.read_text())

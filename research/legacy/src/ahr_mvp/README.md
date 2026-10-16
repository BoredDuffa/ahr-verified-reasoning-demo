# AHR MVP Harness

This is the first narrow interactive harness around the current verified AHR-CSP frontier.

It is not a general natural-language reasoner yet. It provides:

- archived frontier status for v16k/v16l/v16m operation-synthesis runs,
- a conservative prompt router for the three supported MVP prompt families,
- exact operation-synthesis search over `REV`, `DELTA`, `DOUBLE-EVENS`, and `ZIP-SELF`,
- verifier output for proposed operation programs,
- exact bounded search for small simple graph constraints,
- exact Tower-of-Hanoi BFS with a mid-trajectory peg-cap rule,
- explicit `clinical_use=forbidden` metadata.

## Examples

Show the current frontier:

```bash
python3 src/ahr_mvp/ahr_mvp_cli.py status
```

Run the full saved-prompt demo:

```bash
python3 src/ahr_mvp/ahr_mvp_cli.py demo
```

Expected result: three readable verified answers, covering the graph impossibility proof, the Hanoi post-revision impossibility proof, and the length-5 operation-synthesis impossibility proof.

Verify a program:

```bash
python3 src/ahr_mvp/ahr_mvp_cli.py verify-operation \
  --start '[1, 2, 3]' \
  --target '[3, 2, 1]' \
  --program REV
```

Search for a fixed-length program:

```bash
python3 src/ahr_mvp/ahr_mvp_cli.py synthesize-operation \
  --start '[1, 2, 3]' \
  --target '[3, 2, 1]' \
  --length 1
```

Route one of the saved user-style prompts:

```bash
python3 src/ahr_mvp/ahr_mvp_cli.py ask \
  --prompt-file src/ahr_mvp/examples/graph_two_triangles_unsat.txt
```

Expected answer: the MVP states that no such graph exists and cites the exhaustive check over all `32768` simple graphs on 6 vertices.

```bash
python3 src/ahr_mvp/ahr_mvp_cli.py route-prompt \
  --prompt-file src/ahr_mvp/examples/graph_two_triangles_unsat.txt
```

Expected conclusion: `unsat_under_exhaustive_simple_graph_search` after checking all `32768` simple graphs on 6 vertices.

```bash
python3 src/ahr_mvp/ahr_mvp_cli.py route-prompt \
  --prompt-file src/ahr_mvp/examples/hanoi_5disk_after8_bcap2.txt
```

Expected conclusion: `unsat_under_bounded_hanoi_rule_revision_search` for the 5-disk, after-move-8, peg-B-cap-2 rule revision.

```bash
python3 src/ahr_mvp/ahr_mvp_cli.py route-prompt \
  --prompt-file src/ahr_mvp/examples/operation_len5_unsat.txt
```

Expected conclusion: `unsat_under_declared_operation_semantics` after checking all `1024` length-5 programs.

Use `ask` for a readable answer and `route-prompt` for raw JSON evidence. Add `--show-json` to `ask` when both are useful.

Search the 6-vertex graph example:

```bash
python3 src/ahr_mvp/ahr_mvp_cli.py graph-search \
  --vertices 6 \
  --diameter 3 \
  --edge-critical-connected \
  --triangles 2
```

Solve the 5-disk Hanoi mid-rule example:

```bash
python3 src/ahr_mvp/ahr_mvp_cli.py hanoi-revision \
  --disks 5 \
  --start A \
  --target C \
  --switch-after 8 \
  --capped-peg B \
  --cap 2
```

The MVP returns exit code `0` for verified success and `2` for verified no-solution/invalid-program cases.
The `ask` command returns exit code `0` when a prompt is supported and answered, including verified no-solution answers; it returns `2` only when the prompt is outside the current MVP router.

## Router Scope

The router is intentionally narrow and deterministic. It extracts explicit numbers, pegs, arrays, and operation names from known prompt shapes, then delegates to exact standard-library solvers. Unsupported prompts return `unsupported_prompt_for_current_mvp` rather than a guessed answer.

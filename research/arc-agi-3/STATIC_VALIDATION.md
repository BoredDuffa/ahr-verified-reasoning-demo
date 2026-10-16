# AHR v2 XL draft validation

Status: **authored and statically checked; not executed**.

The delivery contains the original 16 notebooks, numbered 00–15. `DELIVERY_MANIFEST.json` records each file's SHA256, code-source hash, cell counts, dependency order and exported definitions.

## Checks performed

- Notebook format 4.5 validated with `nbformat`.
- Every code cell parsed using Python 3.12 grammar and compiled without execution.
- Undefined-name checks performed on each notebook together with its explicit dependency exports.
- Dependency graph checked for complete 00–15 coverage and cycles.
- Tagged library cells checked for definitions, imports and constants; upstream execution cells are excluded by the loader.
- Every code cell has a null execution count and empty outputs.
- Interfaces reviewed across environment actions, observed replay, demonstration targets, full model outputs, checkpoints, held-out evaluation and the visual dashboard.

## Still to verify when running individual notebooks

Kernel installation and current GPU behavior; official toolkit access and local assets; tensor shapes and numerical behavior under the selected precision; actual full-model parameter count and memory use; exact checkpoint continuation; training progress; benchmark results; notebook widget rendering.

The checkpoints and continuation tests are implemented but have not passed at authoring time. Training cells check for real matching recovery reports before starting. No pass report, trained checkpoint, data collection, model output or benchmark result was generated during this task.

## Scope of the first drafts

The full Queen, Worker and Scout architecture is present. XL dimensions remain provisional until profiling. Training uses one selected GPU per run; independent ablation jobs can use the other GPUs. The initial joint objective trains from recorded single transitions, so recurrent multi-step credit assignment remains a refinement. The ARC adapter uses the official public toolkit; exact Kaggle submission packaging is outside this unverified first draft.

Game families and genuine demonstration data must be selected or recorded before supervised training. Random exploration is explicitly not expert supervision. Offline public held-out metrics are not represented as hidden competition scores.

Remote delivery is verified by reading notebook files and comparing SHA256 hashes with the local manifest. This reads file contents only; it does not run notebook cells.

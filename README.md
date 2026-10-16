# Ant Hill Reasoning (AHR)

**Benjamin Falkenburg · research source and evidence · updated 16 October 2026**

AHR studies **Queen–Worker–Scout hierarchical search over reasoning trajectories**, with learned guidance and independent, domain-specific checks. The Queen allocates goals and computation, Workers construct and refine candidates, and Scouts explore alternatives. Acceptance belongs to a specified verifier; a learned score alone does not establish correctness.

This repository brings the June browser demonstration together with the subsequent mathematical theory, Lean experiments, Go development, economic-credit experiment and ARC-style research notebooks. Different implementations test different parts of the architecture. They are **not one released general-purpose model**.

**Start with the [research overview](docs/RESEARCH_OVERVIEW.md), [results and limitations](docs/RESULTS.md), or [source notebook index](research/NOTEBOOK_INDEX.md).**

## Try and read

- [Research page](https://boredduffa.github.io/ahr-verified-reasoning-demo/research.html)
- [Interactive exact-reasoning demo](https://boredduffa.github.io/ahr-verified-reasoning-demo/)
- [Unified mathematical theory, v1.2 — PDF](papers/AHR_Unified_Theory.pdf) and [LaTeX source](papers/AHR_Unified_Theory.tex)
- [Economics, reward design and the proposed AI-safety study](docs/ECONOMIC_REWARDS.md)
- [Reproduction instructions and current gaps](docs/REPRODUCING.md)

The browser demo uses hand-written exact backends and proof templates for supported question families. It does **not** load the trained rankers, Go models or ARC model. Unsupported requests abstain. Its behaviour is separate from the recorded learned-model experiments below.

## Selected evidence

| Work | Recorded result | Interpretation |
|---|---|---|
| Lean-AHR v2.0 | 270/270 top-1 completions | Bounded suite; learned ranking plus hand-coded priors and Lean checking. Historical report, not a fresh recompile. |
| Multi-family program ranker | 4,998,161 parameters; 340/360 test questions answered correctly, 20 abstentions, zero recorded wrong answers | 94.4% coverage on the specified generated suite; exact execution and verification contribute to correctness. |
| Lean-AHR ProofNet bridge | v3.6/v3.7: 2/50 target proofs on offset-50; 33/50 translation/retrieval coverage | Small development slices; translation coverage is not proof success. The ranker did not establish a general proof-solving gain. |
| AHR-JEPA Go v3 | 294 wins, 473 losses, one unknown out of 768 planned games in C versus B | Three-seed controlled comparison; benefit and promotion criterion not established. |
| Economic-credit Go study | Economics 4/40 wins, control 2/40, base 5/40 on the same development screen | Descriptive only. The **base** was selected; its fresh KataGo one-visit confirmation was 3/100, so the incumbent was retained. |
| Latest Go follow-up | Diagnosis completed; 16 October collection stopped below the 1,024-row minimum | The optimizer was refused. This is a failed collection stage, not a successful new training result. |
| AHR v2 XL / ARC-style workflow | 16 source notebooks | Local static draft snapshot; no trained checkpoint or official ARC score established by this release. |

Every row is expanded with its evidence link and limitations in [RESULTS.md](docs/RESULTS.md). Related FMM-JEPA vision work is indexed separately; its engineering or representation results do not establish AHR reasoning performance.

## Repository map

| Location | Contents |
|---|---|
| `ahr_core.js`, `app.js`, `index.html` | Lightweight exact browser demo |
| [`papers/`](papers/) | Existing mathematical monograph and historical Lean report |
| [`research/legacy/`](research/legacy/) | Earlier finite-search, program-ranking, Sudoku and Lean source and reports |
| [`research/mini-go/`](research/mini-go/) | Independent bounded quantified-backup probe, rules and existing tests |
| [`research/go/`](research/go/) | Live-source snapshot: 14 foundation notebooks, Go v2, JEPA v3, native v4/v5 and result records |
| [`research/arc-agi-3/`](research/arc-agi-3/) | 16-notebook Queen–Worker–Scout / Lorentz-workspace draft |
| [`research/related/fmm-jepa/`](research/related/fmm-jepa/) | Related vision research, source modules and selected reports |
| [`research/SOURCE_MANIFEST.json`](research/SOURCE_MANIFEST.json) | Original and published hashes, source provenance and notebook transformations |
| [`research/RESULTS_LEDGER.json`](research/RESULTS_LEDGER.json) | Machine-readable checked headline claims |

## Publication status

This is a source-and-evidence update, not a newly executed benchmark or a portable trained-model release. Original notebook code cells are preserved; bulky saved displays and execution counters are cleared. Report files retain the recorded measurements. Checkpoints, large generated datasets, private company/clinical implementations, credentials, unrelated projects and third-party paper collections are excluded. Source hashes in historical reports refer to the original versions, which may differ from the current source snapshot.

See the [publication audit](docs/PUBLICATION_AUDIT.md) for coverage, omissions, validation and the exact boundary between historical records and current execution status.

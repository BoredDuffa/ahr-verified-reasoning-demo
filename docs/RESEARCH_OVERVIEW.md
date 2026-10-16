# AHR research overview

Snapshot: **16 October 2026**. Author: Benjamin Falkenburg.

## The mathematical contract

The canonical theoretical object is Queen/Worker/Scout hierarchical energy minimisation over verified reasoning trajectories. For a problem `x`, let `C_B(x)` be the candidates actually explored within a finite budget `B`, and let `V(x, trajectory, answer)` be the declared verifier. The release rule is:

```text
verified = {(trajectory, answer) in C_B(x) : V(x, trajectory, answer) = 1}
if verified is empty: return unresolved / abstain
otherwise: select a verified candidate using the specified energy and tie rule
```

The **Queen** manages strategy, compatibility, budgets and stopping. A **Worker** constructs or repairs a local candidate. A **Scout** proposes alternatives or explores new branches. Shared memory records useful evidence and attempts. These roles are mathematical primitives; a particular implementation may share parameters or approximate their interaction.

Hard constraints exclude inadmissible candidates, conventionally through infinite energy. A finite penalty can be traded off against another objective and is not equivalent to this exclusion.

Two soundness statements must be separated:

1. The release rule accepts only candidates that pass `V`.
2. Accepted candidates solve the intended problem **if** `V` is sound for that domain and the formalised problem matches the intended one.

The first does not establish the second by itself. Exhaustive coverage and safe pruning are additional assumptions for completeness. Learned energies, finite search budgets and geometric representations do not supply those assumptions automatically. For Go, legal moves and bounded tactical proofs are checkable; a predicted win probability is not a proof of full-game optimality.

The [93-page monograph](../papers/AHR_Unified_Theory.pdf) gives the spaces, transitions, role-compatibility relation, objective, finite modus-ponens model, and conditional existence, soundness, pruning and coverage results. It is the existing July v1.2 artifact, not a new peer-reviewed publication or a claim that every notebook implements every theorem.

## Development since the June demonstration

| Period / branch | Development | What is established |
|---|---|---|
| June: finite reasoning and program ranking | Typed program/action ranking, exact backends, abstention, translator gates and a roughly 5M-parameter ranker | Recorded bounded-suite performance; no unrestricted mathematics or broad natural-language correctness guarantee |
| June: Lean-AHR v3.0–v3.8 | Mathlib retrieval, Lean3-to-Lean4 translation repairs, attempt ranking and ProofNet bridge diagnostics | Small verified proof counts, translation improvements and clear generalisation failures; known-theorem wrappers remain distinct from new proof synthesis |
| July: unified theory | Formal Queen–Worker–Scout compatibility, hard/soft energies, safe-pruning and coverage conditions | A conditional mathematical framework with a finite worked model; empirical architectural advantage remains a testable question |
| September: foundation Go notebooks | Independent rules, CSP/QCSP encoding, grouped data, native model, search, transfer, adaptation, self-play, GTP, arena, play interface and export | Executed engineering and experiment records; no perfect Go or competitive rank claim |
| September–October: Go v3/v4/v5 | Predictive/JEPA comparisons, native-role search, tactical curricula, learning gates and recovery | Mixed and negative outcomes are retained; failed readiness gates do not become final evaluations |
| October: Go v2 teacher/search work | Batched PUCT, tactical Worker reading, offline KataGo labels, teacher-trained Scout/Worker, strength/speed protocols and later diagnosis | Complete source snapshot and recorded controlled comparisons; soft move-time overruns are reported |
| October: economic credit | Frozen-buffer control/economics training, opportunity-cost weighting, auxiliary credit prediction, screen and promotion gates | Implemented pilot, but no confirmed improvement from economics |
| October: AHR v2 XL | Encoder/world model, Lorentz workspace, memory, Worker, Scout, Queen, integration and ARC-style training/evaluation notebooks | Static local draft delivery, not an official ARC result |

## Research strands that should remain separate

**Proof search:** Lean can check a supplied formal proof. It does not certify that a language interface translated the original question correctly, or that retrieval discovered a novel proof.

**Go:** the public source includes model proposals, search, independent rules and bounded tactical checks. The earlier `AHRBackupNet` mini-Go probe only learns quantified tree-backup operations; it is not the whole AHR theory or a full constraint solver.

**Geometry and JEPA:** the Lorentz workspace and latent prediction mechanisms are representation/search hypotheses. They do not establish causality or correctness. Related FMM-JEPA vision work is provided under `research/related/` and must be interpreted using its own endpoints and controls.

**Economic incentives:** the implemented Go credit pilot and proposed AI-agent oversight experiment have different objectives, data and evidence status. [Economic rewards note](ECONOMIC_REWARDS.md).

**Verification in applied systems:** a proposal/ranking component can sit behind typed inputs, evidence provenance, deterministic rule checks and abstention. Private healthcare integration material is excluded here. This repository establishes no clinical efficacy, approved medical rule set or safety certification.

# Results and limitations

Audited from saved artifacts on **16 October 2026**. These are historical experiment records; the publication did not rerun model training or recompile every Lean proof. A file that once said `RUNNING` is not evidence that its process is running now.

## Bounded arithmetic and program ranking

**Lean-AHR v2.0:** the [31 May report](../papers/Lean_AHR_v2_0_Report.pdf) records 270/270 top-1 completions, with mean 16.20 Lean calls and 6.46 seconds per top-1 record. The search combines learned branch-cost/ranking with hand-coded priors and Lean verification. This is a bounded synthetic arithmetic/propositional suite, not a general proof benchmark. The PDF and associated source are historical evidence; proof objects were not all freshly recompiled for this update.

**Multi-family ranker:** the [16 June primary report](../research/legacy/reports/ahr_multifamily_5m_ranker_v0_report_20260616.json) records 4,998,161 parameters and an eight-epoch run. Test and validation each contain 360 questions: 340 correct accepted answers, 20 abstentions, zero recorded wrong answers, hard-valid rate 1.0 among answered questions and mean hard energy zero. Training contains 1,440 questions, 1,360 accepted and 80 abstentions. Coverage is `340/360 = 94.44%`, not 100% of all questions. Candidate families, generated data, exact execution and acceptance logic constrain the meaning of this result. It was not wired into this repository's JavaScript demo.

## Lean-AHR beyond the arithmetic suite

The [Lean archive](../research/legacy/lean/) preserves milestone reports and implementation snapshots. Counts refer to target-verified proofs, not merely successful translation or theorem retrieval.

| Version | Slice | Translation / retrieval | Target proofs | Important qualification |
|---|---|---:|---:|---|
| v3.1 | First 50 ProofNet test rows | 6/50 | 3/50 | Small bridge experiment |
| v3.2 | Same first 50 | 26/50 | 5/50 | 537 Lean calls |
| v3.4 | Same first 50, attempt ranker | 26/50 | 5/50 | No net solve or call-count gain over v3.2 |
| v3.5 | Same first 50, translation repair | 33/50 | 5/50 | More parsing did not increase verified proofs |
| v3.5 held-out attempt | Rows 50–99 | 13/50 | 0/50 | Failure to generalise to this next slice |
| v3.6 | Rows 50–99, repaired after inspection | 33/50 | 2/50 | 769 calls; now development evidence |
| v3.7 | Same inspected slice | 33/50 | 2/50 | 768 calls; no proof-count gain |

The v3.8 refresh smoke checked six selected rows and verified two, matching its v3.7 smoke baseline with 54 versus 105 Lean calls. This is **not a completed new 50-row generalisation result**. Its ranker report has only two test rows containing positives and zero threshold F1 on validation/test; a perfect conditional rank metric on those rows is not broad calibrated correctness.

The v3.0 50/50 known-theorem wrapper result used ten unique prompt families and existing Mathlib theorems. It must not be described as inventing fifty new proofs. Public ProofNet test slices used repeatedly for repairs are development material after that reuse.

## Go foundations and early learning

The [starter validation](../research/go/reports/03_dataset_validation.json) records 232 examples across six families, 218 known labels and 14 unresolved examples, with zero canonical-group overlap across train/development/test. This validates that small dataset and its recovery, not a trained Go player.

The [CSP pretraining record](../research/go/reports/05_training_results.json) records 12 epochs / 936 optimizer steps. Selected development top-1 fraction is 0.51, also 0.51 at initialization; that endpoint does not show improved ranking. The [bounded solving comparison](../research/go/reports/05_constraint_solving_results.json) must be read with the exact search/verifier budgets and development-data scope. Model-guided ordering does not itself certify a learned improvement over exact checking.

The independent [mini-Go protocol](../research/mini-go/docs/PILOT_PROTOCOL.md) tests quantified backup transfer on complete depth-limited trees. Terminal scores are exact under its rules; horizon scores are heuristics. Its saved training files and tests are retained without turning training completion into a full transfer/tournament result.

## AHR-JEPA Go v3

The [three-seed family report](../research/go/reports/ahr_jepa_v3/family_summary.json) compares C versus B on 768 planned games: **294 wins, 473 losses, one `UNKNOWN_DEADLINE`**; 767 games completed. B/C share initial weights, games, updates, sampler seed and curriculum. JEPA adds computation, so exact FLOP parity is not claimed. The recorded primary protocol uses a one-second inference budget and pairs both colours of each opening.

The reported complete-pair mean score is 0.383812, with a paired Hoeffding interval `[0.314416, 0.453208]`, conditional on these trained models and opening distribution. One seed won 130/256 while the other two won 62/255 and 102/256: selecting only the favourable seed would change the conclusion. The family is incomplete, its promotion criterion failed, and `jepa_benefit_established` is false. These counts are a comparison against B, **not 294 wins against KataGo**.

## Native Go v4/v5

The [v4 pipeline snapshot](../research/go/reports/ahr_native_v4/pipeline_status.json) includes development evaluations with zero wins, 29 losses and three deadline-unknown games out of 32 planned. Its old `RUNNING` field is a historical status, not a live-process assertion.

The [v5 completion record](../research/go/reports/ahr_native_v5/completion.json) is `BUDGET_COMPLETE_FINAL_NOT_READY`, with `final_readiness = FAIL`, no confirmation and no final result. It records 102,118 optimizer steps, but zero completed self-play games in that training stage: it remained in the bounded curriculum/learning-gate process. Development arena failures and unknown outcomes remain visible. Optimizer steps do not establish successful Go learning or passing the gate.

## Economic-credit Go study, 11–12 October

The [main result record](../research/go/reports/ahr_v2/ahr-v2-katago-econ-20261011-v1/results.json) records two 20,000-step supervised runs and matched 1,000-step guided-control and guided-economics runs sharing a frozen buffer.

On the common 40-game KataGo one-visit development screen:

| Candidate | Wins | Losses | Unknown |
|---|---:|---:|---:|
| Base | 5 | 35 | 0 |
| Standard teacher-guided control | 2 | 38 | 0 |
| Economics variant | 4 | 36 | 0 |

The economics arm won two more games than the control, but its mean paired score margin was 1.25 points worse, and the base won more games than either. This is descriptive screen evidence; no confirmed economic benefit is established. The **base** was selected for confirmation.

The [fresh confirmation extract](../research/go/reports/katago_confirmation_summary.json) records **3 wins, 97 losses, no unknowns** in 100 games of 9×9, komi 7.5, against the pinned KataGo profile at **one visit**. This is a deliberately limited opponent configuration. The original 17,455,360-byte report is identified by SHA-256 in the extract; per-game traces stay in research storage.

The [evaluation recovery result](../research/go/reports/ahr_v2/ahr-v2-katago-econ-20261011-v1-eval-recovery-v1/results.json) records 16/20 wins against GNU Go for the selected candidate versus 17/20 for the incumbent. The promotion decision was **KEEP_CHAMPION**: 3/100 was below the required 55/100 KataGo wins. Higher KataGo gates were skipped.

The nominal two-second move budget was soft: the KataGo confirmation records 3,615 overruns among 3,832 AHR moves, with mean 2.0213 seconds and maximum 2.2194 seconds. The report's independent-Bernoulli Wilson intervals are labelled descriptive because openings are paired; they are not used here as independent-game inferential certification. No Elo, amateur rank or engine-strength equivalence is inferred.

## Latest Go follow-up, 16 October

The [latest controller status](../research/go/reports/ahr_v2/ahr-v2-k21-20261016-gpu0-v1/status.json) records diagnosis completed, then `ERROR` at `collect_0`, updated 16:56:56 UTC. Its error states that fresh collection produced fewer than the required 1,024 rows and optimizer training was refused. The [saved diagnosis](../research/go/reports/ahr_v2/ahr-v2-k21-20261016-gpu0-v1/results.json) contains 80/80 sampled positions, including 17 marked contestable. Teacher-estimated regret is a diagnostic, not an exact full-game value. Engineering smoke and recovery checks are separate from this failed production collection.

## AHR v2 XL / ARC-style notebooks

The [local delivery README](../research/arc-agi-3/README.md) and [static manifest](../research/arc-agi-3/DELIVERY_MANIFEST.json) describe 16 draft notebooks. They include the Queen, Worker, Scout, encoder, world model, literal Lorentz coordinates, memory, integration, training and evaluation paths. The snapshot is **unexecuted first-draft code**. A separate hardware/setup report does not establish trained weights, successful full Run All, hidden-game generalisation or an official ARC-AGI-3 score. No such score is claimed here.

## Related FMM-JEPA vision research

The separate [source and reports](../research/related/fmm-jepa/) provide context for latent prediction, geometric representations and careful negative-result reporting. They do not validate AHR reasoning or Go strength.

The [compute-gate-002 result](../research/related/fmm-jepa/artifacts/notebook_19/compute-gate-002/result.json) is a complete **two-seed development pilot**, with three variants and initial, matched-update and final evaluations. Its final mean accuracy differences versus ViT are +0.3465 percentage points for `learned_optimized` and +0.2228 for `learned_global_prior`, but effects vary in sign by seed, kNN differences are negative, and both flags are `NO_COMPLETE_FAVORABLE_SIGNAL`. No confidence interval or significance claim is made from two pairs. The record says zero official-test pixels read for this pilot. Engineering profiling and prediction quality are different endpoints.

The [56-pair follow-up status](../research/related/fmm-jepa/artifacts/notebook_20/multiday-confirmation-001/study_status.json) is a snapshot marked `RUNNING`, phase `56_pairs_training`; its [result placeholder](../research/related/fmm-jepa/artifacts/notebook_20/multiday-confirmation-001/result.json) is not a completed confirmation. It is excluded from completed-results claims.

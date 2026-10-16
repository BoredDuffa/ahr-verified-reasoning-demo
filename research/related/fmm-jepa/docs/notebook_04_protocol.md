# Bounded semantic follow-up: registered decisions

Run ID: `semantic-followup-001`. The persistent budget starts at 2026-09-26 23:25:06 UTC and ends no later than 2026-09-27 02:25:06 UTC. Preparation, diagnostics, calibration, recovery checks, training, and evaluation count toward this limit. Target completion is two hours; no deadline extension is automatic.

## Diagnostic selection, before outcomes

Use the saved 500-update ViT and semantic checkpoints from `semantic-local-pilot-002`. Compare online-context and EMA-target frozen features with identical training-only normalization, classifier initialization/order, and a fixed final probe epoch 20. Use the audited 10,100 training / 2,020 development images. Do not access official test images. Report paired development-image intervals as conditional exploratory evidence.

If either architecture's online-minus-teacher development top-1 point estimate is at least 0.5 percentage points, select the EMA factor: cosine EMA starting at 0.996 versus 0.99, both ending at 1.0. Otherwise select the global-feature factor: semantic feature count 32 versus 64, with ViT as a reference and EMA starting at 0.996 for all arms. This rule selects a follow-up hypothesis; it is not a significance test.

## Engineering selection

Screen physical microbatches 16, 32, 64, 128 with accumulation 8, 4, 2, 1 respectively, keeping effective batch 128. Compare real FP32 deterministic JEPA steps using identical recorded data/augmentation/mask identities. Discard all calibration weights. Select a common feasible microbatch by the sum of ViT and semantic median complete-step times, independent of accuracy; certify accumulated-gradient agreement within declared numerical tolerances. Report peaks and timing scopes, including preprocessing. At most 70 GiB measured tensor allocation is permitted per case.

## Paired training

Use three independent training seeds: 20260926, 20260927, 20260928. Share initialization, samples, augmentation and mask streams within each seed across compared arms. Hold probe initialization constant. The EMA experiment has four arms per seed; the feature experiment has three. Choose the common update budget from remaining wall time and measured speed before new training outcomes, round down to complete 100-update blocks, and reserve time for all probes and reporting. Preserve original sources and artifacts.

Use full-state checkpointing and exact uninterrupted-versus-resumed smoke certificates at the chosen actual batch and schedule. Train in detached tmux with one writer per run, retaining the latest two valid checkpoints and best recovery checkpoint. Primary comparison uses the latest common completed update count, never minimum JEPA loss. Any unequal cutoff is incomplete evidence and cannot support an architecture ranking.

## Evaluation and interpretation

Evaluate frozen EMA features at fixed probe epoch 20, including matching untrained controls. Reconstruct predictions from saved feature caches and final classifier weights. Report per-seed learning gains, architecture differences, and the selected mathematical factor's effect. Summarize paired seed differences without treating evaluation images as independent training runs. Three seeds are a screen, not a guarantee of statistical resolution. Low effective rank is advisory; rank alone is not a quality target.

Report actual complete-step time, peak allocation, equal-update quality and image exposures. Synthetic scale measurements, if performed, are separate from learned quality. A speed gain against an unoptimized predecessor does not establish superiority to an equally configured ViT. No held-out test use, external publication, or new unbounded run is authorized by this notebook.

All new executable logic lives in tagged notebook cells under `experiments/04_followup/` and the Notebook04 consumer. Existing `fmm_jepa/*.ipynb` files remain unchanged so prior experiment source identities remain valid.

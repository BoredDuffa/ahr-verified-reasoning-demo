# Completed bounded follow-up

Run `semantic-followup-001` started 2026-09-26 at 23:25:06 UTC, with an absolute deadline of 2026-09-27 at 02:25:06 UTC. [Notebook04](../notebooks/04_Bounded_Semantic_Followup.ipynb) performs the full workflow; [usage and recovery instructions](notebook_04_usage.md) describe exact resume.

**Completed 2026-09-27 at 01:44:50 UTC in 139.73 minutes**, including preparation, diagnostics, calibration, recovery checks, training and evaluation. All 12 arms reached 300 updates with identical within-seed data traces and zero failed training attempts; all 18 fixed-epoch probes completed. The independent CPU audit passed. The worker and deadline watcher have exited.

The diagnostic selected EMA starts 0.996 versus 0.99. The comparison uses ViT and semantic-local encoders, both EMA settings, and three paired seeds. Engineering selected physical batch128 with accumulation1. Classifier evaluation uses fixed final epoch20. The official test remains reserved. The short scaling companion also completed within the original deadline.

The faster-EMA semantic model averaged 5.31% development top-1 versus 4.27% ViT and led in every seed. Its learning gain over initialization remains uncertain. Actual semantic training was about 15% slower and used 3.2 times the peak allocated memory of ViT. The measured scaling screen through 4,096 tokens also found no efficiency advantage. Read the [combined interpretation](../artifacts/notebook_04/semantic-followup-001/bounded_summary.md), [full results](../artifacts/notebook_04/semantic-followup-001/factorial_results.md) and [independent audit](../artifacts/notebook_04/semantic-followup-001/independent_review.json).

- Historical worker, now exited: `fmm-followup-semantic-followup-001`.
- Historical deadline watcher, now exited: `fmm-deadline-semantic-followup-001`.
- [Current status](../artifacts/notebook_04/semantic-followup-001/pilot_status.json).
- [Registered plan](../artifacts/notebook_04/semantic-followup-001/plan.json).
- [Decision rules](notebook_04_protocol.md).

## Previous completed pilot: semantic-local attention

Run `semantic-local-pilot-002` compares fresh ViT, original FMM and the revised semantic-local mixer on the audited Food-101 training/development split. Each arm has a registered budget of 500 updates, effective batch 128, followed by matched initial/trained frozen-feature probes. The official test split is reserved. The persisted run deadline is 2026-09-25 18:26:01 UTC; completing early does not authorize another run.

**Completed 2026-09-25 at 16:33:16 UTC in 67.25 minutes.** All three models reached 500 updates with matched traces and no failed attempts. Both pilot tmux sessions have exited. The independent results review passed. [Results](../artifacts/notebook_03/semantic-local-pilot-002/semantic_results.md) and [audit evidence](../artifacts/notebook_03/semantic-local-pilot-002/independent_review.json) are saved.

The candidate scored 6.83% development top-1 versus 5.94% ViT and 5.20% original FMM. Its conditional paired-image interval is positive against FMM, while improvement over initialization and over ViT remain unresolved. Actual training uses 65.2% less peak allocated memory than FMM, but steps are only 2.0% faster; ViT remains faster and smaller. This is partial promise from one short paired seed, not a robust architecture ranking.

- Main notebook: [`03_Semantic_Local_Attention_Pilot.ipynb`](../notebooks/03_Semantic_Local_Attention_Pilot.ipynb). Run All prepares, launches or reconnects, monitors and displays results.
- Historical worker session (now exited): `fmm-semantic-semantic-local-pilot-002`. While live, attach with `tmux attach -t fmm-semantic-semantic-local-pilot-002`.
- Detach: **Ctrl-b**, then **d**.
- Status: [`pilot_status.json`](../artifacts/notebook_03/semantic-local-pilot-002/pilot_status.json).
- Worker log: [`tmux_driver.log`](../artifacts/notebook_03/semantic-local-pilot-002/tmux_driver.log).
- Checkpoints: `checkpoints/semantic-local-pilot-002/{vit,fmm,semantic}/`.
- Independent completed-run review: [`03_Semantic_Local_Results_Review.ipynb`](../notebooks/03_Semantic_Local_Results_Review.ipynb).
- Protocol and interpretation limits: [`semantic_pilot_protocol.md`](semantic_pilot_protocol.md).
- Exact resume and checkpoint instructions: [`notebook_03_usage.md`](notebook_03_usage.md).

The current status file is authoritative. Use the existing run ID with unchanged sources/configuration to reconnect or resume; do not launch a second writer. Preparation run `semantic-local-pilot-001` failed on an environment-metadata field before production training, and its artifacts are retained separately.

## Historical Tiny ImageNet pilot

Run `20260923T214538Z-9663c15f`; paired5000updates/model, effectivebatch128,20k/2k data, four30epoch probes. Held-out disabled.

- Attach: `tmux attach -t fmm-jepa-pilot-20260923T214538Z-9663c15f`
- Detach: **Ctrl-b**, then **d**
- Status: `/home/benjamin/workspace/ahr_go/fmm_jeppa/artifacts/notebook_01/20260923T214538Z-9663c15f/pilot_status.json`
- Execution log: `/home/benjamin/workspace/ahr_go/fmm_jeppa/artifacts/notebook_01/20260923T214538Z-9663c15f/pilot_execution.log`
- Driver log: `/home/benjamin/workspace/ahr_go/fmm_jeppa/artifacts/notebook_01/20260923T214538Z-9663c15f/tmux_driver.log`
- Driver notebook: `/home/benjamin/workspace/ahr_go/fmm_jeppa/artifacts/notebook_01/20260923T214538Z-9663c15f/pilot_driver.ipynb`
- Actual-batch/resume certificate: `/home/benjamin/workspace/ahr_go/fmm_jeppa/artifacts/notebook_01/pilot-preflight-20260923T214240Z-493ef4df/preflight_certificate.json`

Monitor checkpoint manifests under `checkpoints/20260923T214538Z-9663c15f-{vit,fmm}/manifest.json`; CSVs finalize when each model completes. Exact resume uses the same driver/config/source after confirming no prior writer remains. Validated latest checkpoints are loaded; invalid state is an error.

Independent results audit is queued in the same session, window `postrun-review`. It waits for completed training/probes, then writes `pilot_results_review.md`, `interpretation_review.json`, review figures and `pilot_review_status.json`. Inspect it with `tmux select-window -t fmm-jepa-pilot-20260923T214538Z-9663c15f:postrun-review` after attaching.

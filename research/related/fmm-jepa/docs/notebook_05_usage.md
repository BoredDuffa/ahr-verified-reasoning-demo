# Notebook 05 usage and exact resume

Open [05_Learning_and_Resolution_Scaling.ipynb](../notebooks/05_Learning_and_Resolution_Scaling.ipynb) with the existing `ahr-go` kernel (`/home/benjamin/workspace/ahr_go/.venv/bin/python`). This is the repaired study with fresh paired models. The failed original run remains preserved under `artifacts/notebook_05/learning-resolution-001/`, including a verified source snapshot. The user has authorized the repaired long run. Run All launches it or reconnects to the same run; the launch requires the passing numerical-repair certificate described below.

## Choose the scope before the first run

The configuration cell defaults to:

```python
RUN_ID = "learning-resolution-stable-001"
PROFILE = "staged"
```

| Profile | Requested work | Planning estimate | Hard maximum |
|---|---|---:|---:|
| `staged` (default) | Learning, then resolution and data-volume stages if the learning gate passes |12–24 hours|24 hours|
| `learning_only` | The longer three-seed learning comparison |3–4 hours|4 hours|

These historical estimates predate the numerical repair and are not promises. Actual calibration replaces them before each stage. The 128-resolution history suggests about 2.6 hours of training alone; calibration, full-state recovery tests, data preparation, feature extraction and 18 probes add time. The 256-resolution cost has not yet been measured with this study's actual batch and schedule. Run All shows the planning estimate, then measures real runtime before stage training. If the unchanged 1,000-update budget cannot fit the remaining original clock, it records an incomplete-budget outcome instead of shortening the experiment.

The default `staged` profile can end after learning. Its preregistered gate requires the semantic model's final-minus-own-initialization three-seed 95% t interval to have a positive lower bound, with finite and nonconstant final representations. If this is not established, resolution and data-volume stages remain `NOT_RUN`; this is not evidence that learning is impossible.

## Start

1. Set the desired `PROFILE` and a fresh `RUN_ID` before launching. Leave them unchanged for that run thereafter.
2. Restart the project kernel and click **Run All**.
3. The launch cell `launch-staged-study` creates or reconnects to a detached tmux worker. The following cells monitor progress and show the saved reports when it finishes.

Run All performs the requested experiment; there is no preview-only switch. The first launch persists the original deadline and source/configuration identity. All enabled stages, calibration, preflight and evaluation share that one clock. Closing the notebook, browser or SSH connection does not stop the detached worker.

For the default run, attach from a terminal with:

```bash
tmux attach -t fmm-scaling-learning-resolution-stable-001
```

Detach with **Ctrl-b, then d**. An independent deadline watcher runs as `fmm-scaling-deadline-learning-resolution-stable-001`. Use the actual run ID in both names if you chose another one. An existing worker is reused rather than duplicated.

A terminal invocation of the main notebook is also possible after you have set its configuration cell:

```bash
cd /home/benjamin/workspace/ahr_go/fmm_jeppa
/home/benjamin/workspace/ahr_go/.venv/bin/python -m jupyter nbconvert \
  --execute --to notebook --inplace \
  --ExecutePreprocessor.kernel_name=ahr-go \
  --ExecutePreprocessor.timeout=-1 \
  notebooks/05_Learning_and_Resolution_Scaling.ipynb
```

Experiment logic remains in notebooks. The main notebook launches its own detached worker; this terminal process primarily monitors it.

## What is held fixed

There are three paired seeds, two architectures and 1,000 successful updates per model per enabled stage. The 300-update checkpoint is a milestone on the same 1,000-update learning-rate/EMA schedule, not a separately scheduled short run. Both architectures receive the same examples, augmentation and masks within each seed, with effective batch 128 and FP32 arithmetic.

The learning stage uses 128×128 images and 10,100 pretraining images. Resolution changes only image size to 256×256. Data volume uses 128×128 images and the full 65,600-image audited pretraining pool. Every probe still fits on the original 10,100 labeled identities and evaluates on the same 2,020 development identities. The official test remains reserved.

Calibration tries microbatch 128, falling back to 64, 32 or 16 only when required by OOM or the 70 GiB allocation limit. It measures three warmup plus five complete real-data updates per architecture, certifies accumulation numerics and discards its weights. Main training starts from fresh registered initialization and first passes exact checkpoint/restart verification. The launcher verifies `artifacts/notebook_05/repair_validation/stability_certificate.json`: failure reproduction, operator regressions, bounded late-checkpoint stress and fresh full-state exact resume must pass for the exact stable-attention export hash. Every semantic model, including the evaluation teacher and initial controls, uses `semantic-stable-v1`.

## Numerical failure policy

The first nonfinite numerical attempt stops the run before any further sample consumption. The previous clean full-state checkpoint remains valid; restarting replays from that checkpoint rather than skipping the failed examples. Saved states containing failed attempts, mismatched sample exposure or missing successful traces are rejected. This policy protects paired sample/augmentation/mask streams. A stable attention repair does not itself establish improved accuracy, efficiency or scalability.

The monitor reads each arm's durable checkpoint counters, showing successful updates and saved failed attempts. Counts can lag live computation by up to 50 updates. Its remaining-training forecast uses measured calibration times and excludes outstanding evaluation, preflight and checkpoint I/O.

## Outputs

For the default run, evidence is rooted at `artifacts/notebook_05/learning-resolution-stable-001/`:

- `study_binding.json`, `budget_clock.json` and `registered_preview.json`: original configuration, frozen sources, passing repair-certificate hash and persisted clock.
- `study_status.json`, `worker.log` and `deadline_watchdog_result.json`: progress, errors and deadline enforcement.
- `study_results.md` and `combined_results.json`: final combined readout, including skipped or incomplete stages.
- `stages/<stage_id>/`: immutable data manifests, verified image membership, calibration results, runtime estimates, registered plans, recovery certificates, training/evaluation evidence, `stage_results.md`, `stage_quality.png` and `stage_compute.png`.

Stage IDs are `learning`, `resolution` and `data_volume`. Calibration states live under `checkpoints/<run_id>/stages/<stage_id>/calibration/`. Main arm states live under `checkpoints/<run_id>/stages/<stage_id>/arms/<arm_id>/`, including a resumable `manifest.json` and immutable milestone snapshots under `milestones/`. Probe states have their own checkpoint directories referenced by evaluation evidence.

The model comparisons separate learning gain, architecture quality and compute cost. Three paired seeds improve the evidence but do not guarantee statistical resolution or establish asymptotic scaling.

## Resume exactly

1. Keep the same `RUN_ID`, `PROFILE`, configuration values, notebook code and tagged exported definitions. Keep the existing artifact and checkpoint folders.
2. Restart the project kernel and run the main notebook top-to-bottom again. The `launch-staged-study` cell reconnects to an existing worker or reconstructs the run from valid saved state.
3. The original `budget_clock.json` remains authoritative. Completed evidence is checked and reused; remaining work resumes while that original deadline permits it.

Do not delete manifests or clocks to force a restart. Corrupt, missing or mismatched state is an explicit error. A completed run displays existing results without additional training. An expired deadline does not extend automatically; its evidence stays available. A fresh experiment with changed settings requires a different run ID chosen before that new launch.

The full-state checkpoint includes model and teacher, optimizer, scheduler, EMA position, RNG streams, sampler position, configuration, metrics and trace identities. Main training saves at epoch/stage boundaries and at most every 50 successful updates or 15 minutes. Atomic writes retain the latest two valid resumable states and the best; one writer owns each run. Final and milestone evaluation use their explicitly registered snapshots, not whichever checkpoint had the smallest JEPA loss.

Read the [scientific protocol](notebook_05_protocol.md) for data integrity, calibration, budget assumptions and interpretation limits. The workspace is on the persistent boot disk; before deleting or recycling the instance, copy required final/best checkpoints off-instance and verify their recorded hashes.

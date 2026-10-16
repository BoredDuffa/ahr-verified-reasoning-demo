# Notebook04: bounded semantic follow-up

Open `notebooks/04_Bounded_Semantic_Followup.ipynb` in the `Python (ahr-go)` kernel and click Run All. The notebook launches or reconnects to a detached worker, monitors it, and displays the saved results. It performs the saved online-versus-teacher diagnostic, physical-batch measurements, actual-configuration recovery checks, three-seed paired training, and fixed-epoch frozen-feature evaluations.

Current run: `semantic-followup-001`. Its original clock began 2026-09-26 at 23:25:06 UTC, and its maximum deadline is 2026-09-27 at 02:25:06 UTC. That clock includes preparation and never resets on reconnection. The mathematical factor is selected by the rule in `notebook_04_protocol.md` before new training outcomes. The common update count is frozen from measured speed and remaining time.

Worker tmux session: `fmm-followup-semantic-followup-001`. Attach with `tmux attach -t fmm-followup-semantic-followup-001`; detach with Ctrl-b, then d. Closing the notebook monitor does not terminate the worker. The initial controller session is `fmm-followup-control-001`.

An independent watcher, `fmm-deadline-semantic-followup-001`, enforces the original absolute deadline outside the model process. It terminates only this run's worker process tree if still active at the limit and records an incomplete result. A disposable-process smoke test verified this cutoff before training. This protects the cap even when nested model timers restore their own alarms.

Status and evidence: `artifacts/notebook_04/semantic-followup-001/`. Worker log: `worker.log`. Checkpoints: `checkpoints/semantic-followup-001/`, with separate diagnostics, calibration, preflight, arm and probe subdirectories. Each training arm retains full optimizer/scheduler/EMA/RNG/sampler/config/metrics state, latest two valid checkpoints and best recovery checkpoint. Writes are atomic and each run has one writer.

To cold-resume the current registered run, open `notebooks/04_Followup_Exact_Resume.ipynb`, retain the same run ID, sources, configuration and clock, restart the project kernel and Run All. The resume cell is `launch-exact-followup-resume`. A live worker is reconnected without starting another writer; a completed run shows saved evidence. Invalid manifests, changed identities and an expired incomplete run are errors; they never silently start over. The original Notebook04 remains the immutable experiment entry point, but use the exact-resume notebook if the worker has stopped.

The recovery adapter repairs one serialization issue in metadata: the original engineering `weighted_losses` dictionaries had integer microbatch keys, while JSON deserialization returns string keys. Sorting changes their canonical hash order. The adapter restores only those keys, proves that all JSON-observable values are identical, and requires the resulting hash to match the original registration exactly. It patches only reads of this exact run plan in the frozen workflow and restores the reader afterward. It changes no model or training values, source definitions, checkpoint payloads or deadline. Validation evidence is `exact_resume_identity_certificate.json`; a cold worker uses `exact_resume_worker.ipynb` and `exact_resume_worker.log` under the existing run artifact directory, the same tmux session name, and the original independent deadline watchdog.

The recovery notebook has passed a fresh-kernel CPU dry check of original source hashes, every arm protocol, the restored registration hash and reader restoration. Its launch/monitor cells were intentionally not executed while the original worker was training.

Core experiment definitions are tagged notebook cells under `experiments/04_followup/`; the narrowly scoped recovery adapter is tagged in `notebooks/04_Followup_Exact_Resume.ipynb`. Previous `fmm_jepa` modules remain unchanged. Only the audited Food101 training/development membership is used; the official test remains reserved. Workspace storage remains the persistent boot disk; instance deletion policy is still unverified. No instance recycling is part of this experiment.

The seed-level statistical summaries require SciPy. The active environment uses SciPy 1.18.1; it was installed before evaluation without changing NumPy, PyTorch or frozen notebook definitions. The dependency is recorded in `pyproject.toml`.

Operational interruptions, exact recovery and the bounded fresh-status wait are documented in [Notebook04 recovery notes](notebook_04_recovery_incidents.md).

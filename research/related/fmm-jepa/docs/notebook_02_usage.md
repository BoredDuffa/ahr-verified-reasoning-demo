# Notebook 02 — Run All

Open `notebooks/02_Food101_Main.ipynb` with the `ahr-go` kernel and click **Run All**. No mode changes or download toggles are needed.

Run All performs the compute benchmark, downloads Food-101 if missing, audits and prepares splits, verifies exact checkpoint recovery, launches the full paired pilot in detached tmux, waits while displaying progress, then displays all four evaluations and the report/figures. A failed required check stops with an error.

## Configuration

`configs/food101_pilot.yaml` defines 128×128 images, patch8,256tokens,width192,depth4,5,000updates for each model and effective batch128. User settings are in the main notebook's `runtime-controls` cell. Keep RUN_ID to resume; choose a new ID for a different experiment. Official test predictions remain disabled.

## Resume and reconnect

Open the notebook with the same RUN_ID and unchanged config/source, then click **Run All** again. A validated preflight allows preparation stages to be reused. Active workers are reconnected without competing for their writer lock. Interrupted training resumes from the latest valid checkpoint. Completed runs show saved results. Invalid source/checkpoint identities stop clearly rather than silently restarting.

Food support modules are refreshed at notebook setup, including when the earlier preview version was loaded in the same kernel. Reload the notebook from disk after receiving an update so Run All uses the current cells.

## Long training and progress

The worker runs independently in `fmm-food101-<RUN_ID>`. Attach with `tmux attach -t fmm-food101-<RUN_ID>`; detach Ctrl-b then d. Status/logs are in `artifacts/notebook_02/<RUN_ID>/pilot_status.json` and `tmux_driver.log`. The notebook refreshes stage, saved updates and probe epochs every10seconds and displays results after completion. A failed or vanished worker raises with its log details.

Interrupting the progress monitor leaves detached training running. Run All again to reconnect. Checkpoints are under `checkpoints/<RUN_ID>/{vit,fmm,probes,preflight}/`; latest two valid states and best are retained. Saves occur every100updates, epoch boundaries and at most15minutes between completed-update saves. Full state recovery is verified before long training.

## Validation boundary

The assistant changed and statically checked the notebook without executing its cells, downloading data or launching an experiment. Notebook format, Python syntax, local tagged imports, automatic stage ordering and historical source preservation were checked. Actual runtime/numerical validation occurs when you run it; no new PASS certificate or Food-101 result has been claimed.

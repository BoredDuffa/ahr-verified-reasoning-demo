# Notebook03: semantic-local attention promise test

Open `notebooks/03_Semantic_Local_Attention_Pilot.ipynb` in the **Python (ahr-go)** kernel and click **Run All**. The notebook prepares, calibrates, certifies recovery, launches detached training, monitors it and displays the saved report. It does not default to a preview.

The three arms are fresh ViT, original FMM and normalized semantic summaries with exact nearby exponential interactions. All use the audited Food101 training/development membership from `food101-pilot-001`, identical augmentation/mask streams, a shared nonmixer initialization and predictor, and the same successful-update/probe budget. Semantic qkv/output projection initialization also matches ViT. Counts and compute are reported rather than treated as matched. Official test prediction remains NOT_RUN.

Calibration discards its weights and freezes an equal budget <=1500 updates using measured timing only. Requested microbatch16 × accumulation8 = effective128. Training advances in equal 100-update rounds. Target runtime is about two hours; calibration, benchmark, exact-resume checks, training and evaluation share a persisted maximum three-hour wall-clock deadline with twenty minutes reserved for evaluation. Early common-round termination is labeled; unequal/incomplete runs do not support architecture conclusions. One paired seed cannot establish robust superiority.

Artifacts: `artifacts/notebook_03/<RUN_ID>/`. Full-state checkpoints: `checkpoints/<RUN_ID>/{vit,fmm,semantic}/`. Each training checkpoint includes model/EMA teacher, optimizer, scheduler, precision/scaler state, all RNG state, consumed data cursor, global step, config/protocol and metrics; latest two plus best recovery checkpoint are retained with hashes. Atomic writes and advisory locks enforce one writer. Checkpoints occur every epoch boundary, 50 updates or fifteen minutes, whichever comes first.

For the completed run, open `notebooks/03_Semantic_Local_Results_Review.ipynb` and click Run All. It independently reconstructs the saved probe predictions, verifies source/data/checkpoint identities and the time limit, refreshes the report's compute figures and actual training-cost table, and embeds both figures. It does not launch training. The current default run `semantic-local-pilot-002` completed all three 500-update arms and six probes in 67.25 minutes; its independent audit passed.

To resume exactly, keep source files and configuration unchanged, restart the same kernel, preserve RUN_ID, and click Run All. The resume cell ID is `launch-detached-semantic-pilot`. A live tmux worker is reconnected; completed stages are read; saved checkpoint hashes and state counters are checked. An invalid manifest or expired time cap never silently restarts or extends a run.

Tmux session: `fmm-semantic-<RUN_ID>`. Attach: `tmux attach -t fmm-semantic-<RUN_ID>`. Detach: Ctrl-b then d. Interrupting the notebook monitor does not stop training. Worker log: `artifacts/notebook_03/<RUN_ID>/tmux_driver.log`.

Workspace storage is on the existing persistent boot disk. Instance deletion policy remains unverified. No instance recycle/delete is part of this pilot; copy final checkpoints off-instance and verify hashes before any later recycle.

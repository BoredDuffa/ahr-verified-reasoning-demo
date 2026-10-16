# Overnight full JEPA comparison

The authorized run manifold-long-001 launched at 01:02:14 Europe/Dublin on 7 October 2026. Its original 12-hour deadline is 13:02:14 the same day. Training, evaluation, visualization and recovery all share that immutable clock.

Calibration selected 1,800 successful updates for every model: six fresh paired seeds (20261401–20261406), three encoders (ViT, fixed spatial kernel, learned coordinate kernel), physical/effective batch 128. Each arm receives 230,400 image exposures; all18 total4,147,200. Initial,350-update andfinalEMAencoders each get fixed20epochlinearprobes:54evaluations. The primary comparison is final learned-kernel minus ViT development accuracy across six paired seeds. More seeds do not guarantee significance.

Open notebooks/11_Overnight_Manifold_JEPA.ipynb with the ahr-go kernel. Run All reconnects to the actual detached worker or resumes valid checkpoints under the unchanged original binding. It does not restart completed training. The worker passed actual four-versus-two-plus-two bitwise recovery for every architecture and real teacher/probe integration before long training.

The kernel sums are exact dense sums; this is a representation-learning study, not an FMM acceleration claim. Both online and EMA target encoders use each architecture. Predictor, masking, loss, optimizer/scheduler, initialization controls and labeled evaluation protocol are shared. Active parameter counts differ.

All architectures use shared source-bound execution improvements. Vectorized validity/overlap guards preserved all outputs, losses, metrics and parameter gradients bitwise and cut measured CUDA compute by4–6x. This also accelerates ViT. Ordered6threadinput processing preserved original pixels/masks/metadata/RNG and sampler position; it did not demonstrate a standalone preprocessing speedup. Full wall-time and memory are measured in the study.

Worker: fmm-manifold-long-manifold-long-001
Attach: tmux attach -t fmm-manifold-long-manifold-long-001
Detach: Ctrl-b, then d
Deadline watcher: fmm-manifold-long-deadline-manifold-long-001
Hourly agent supervisor: fmm-manifold-long-heartbeat-manifold-long-001

GPU0 is used. The unrelated AHR job on GPU1 is preserved. The worker and supervisors continue without the user's browser connection.

The supervisor records progress every30seconds and requests a Codex CLI review everyhour (firstscheduledreview02:02Dublin). It also requests a review if the worker stops or makes no observed progress for20minutes. Authenticated shell access was tested before launch. An agent may implement a validated notebook adapter for an operational fault and resume the same study. It cannot edit frozen registered exports, model/loss/optimizer/seeds/data/quality endpoints, skip batches, reset checkpoints or extend the clock. Automatic unmodified checkpoint restarts are limited tothree. True numerical trajectory failures or invalid evidence must remain blocked/incomplete. Recovery from every possible failure is not guaranteed.

Logs:
- artifacts/notebook_11/manifold-long-001/worker.log
- heartbeat/latest_heartbeat.json
- heartbeat/agent_reviews/<time>/answer.json and events.jsonl
- heartbeat/supervisor.log

Checkpoints: checkpoints/manifold-long-001/. Full model/teacher, optimizer/scheduler/scaler, all RNGs, consumed sampler/augmentation/mask position, global update, config and metrics are saved at every50updates or15minutes and epoch/stage boundary. Atomic writes and manifests verify the latest two valid states and best. There is one writer per run.

The workspace is on the persistent boot disk; instance-deletion policy is unverified. Copy final/best checkpoints off-instance and verify hashes before recycling/deleting the instance.

Visual review after completion:
- notebooks/11_Overnight_Results_Review.ipynb (Run All is read-only; never starts training)
- artifacts/notebook_11/manifold-long-001/stages/learning/visual_report/report.html
- stage_results.md plus PNGs, CSVs and hashed visual evidence

The visual review includes development accuracy for every seed atinitial/350/final, paired effects/uncertainty, own-initialization gains, trainingloss curves, complete-step time and memory, per-encoderPCA and realimage nearest-neighbor grids. Feature views are not pixel reconstruction. Missing scores on cutoff remain unknown; a partial run cannot establish a full architectural winner. Official test images remain reserved.

Primary paired-mean t inference and exact sign-direction inference have different nulls. The sign test is a robustness check, not an alternative endpoint to cherry-pick. All conclusions are conditional on the reused development membership; secondary contrasts and visual examples are exploratory.

Codex CLI scheduled-job reference: https://learn.chatgpt.com/docs/non-interactive-mode

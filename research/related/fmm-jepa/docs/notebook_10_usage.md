# Notebook 10: learned-coordinate kernel JEPA pilot

Open notebooks/10_Manifold_Kernel_JEPA_Pilot.ipynb with the ahr-go kernel. Run All launches the actual full JEPA experiment in detached tmux, or reconnects to the same existing run. The online and EMA target encoders are replaced; the JEPA predictor, masking, target normalization and self-supervised loss remain unchanged.

The three encoders are ViT, fixed-position spatial kernel, and a learned-coordinate kernel. All have width 192, four blocks, three heads, shared stem/MLP/norm/predictor initialization and independent state. The kernel controls additionally share value/output projections and initial kernel parameters. They use exact dense normalized positive Gaussian-mixture sums. This pilot does not implement or claim FMM acceleration or a proved manifold.

Two fresh model seeds (20261301/20261302), 350 successful updates per arm and batch 128 are fixed. Each initial/final frozen EMA encoder gets the same fixed 20 epoch linear probe on 10,100 training and 2,020 development images. Official test remains reserved. These are exploratory results on reused development membership; two seeds cannot establish a general winner.

Expected runtime: approximately 1–2 hours if actual calibration permits, hard cap three hours from first launch, including preflight, training, evaluation and any recovery. The original clock never resets. Timing calibration uses discarded weights and cannot alter quality endpoints, variants, seeds or update budget. If the full planned comparison cannot fit, it stops as INCOMPLETE.

Before long training, the worker must create real checkpoints and verify bitwise four-update versus interrupted two-plus-two recovery for all three encoders. It then checks fresh feature extraction and fixed-probe recovery. Only a fully completed, paired six-arm training result can enter the 12-probe comparison.

Session: fmm-manifold-kernel-manifold-kernel-pilot-001
Attach: tmux attach -t fmm-manifold-kernel-manifold-kernel-pilot-001
Detach: Ctrl-b, then d
GPU: cuda:0. The unrelated AHR worker on GPU 1 is preserved.

Artifacts: artifacts/notebook_10/manifold-kernel-pilot-001/
Status: study_status.json
Worker log: worker.log
Final report: stages/learning/stage_results.md
Checkpoint root: checkpoints/manifold-kernel-pilot-001/

Checkpoint every epoch/stage boundary, 50 updates or 15 minutes, whichever comes first. Atomic full-state saves include model/teacher, optimizer, scheduler, precision/scaler, all RNGs, stream/sampler positions, config, metrics and update counts; latest two valid resumable checkpoints and best are retained. Manifests fail on corruption. There is one writer per run.

Exact resume: Run All unchanged with the same RUN_ID before the original deadline, using a fresh ahr-go kernel. It reconnects to an active worker or resumes saved states. After completion it shows the existing report. After expiry it refuses to extend the old run.

The workspace is on the persistent boot disk. Instance-deletion policy remains unverified. Copy final/best checkpoints off-instance and verify hashes before recycling/deleting the instance.

Read the learned-minus-spatial and learned-minus-ViT seed differences, own-initialization learning gains, measured step time and peak memory. Geometry diagnostics describe a fixed small image sample and selected far blocks; they do not prove global compressibility. A promising representation result would justify a separate dense-versus-FMM approximation/scaling test with identical weights.

Evaluation device recovery (7 October 2026): all six encoder runs completed cleanly. The descriptive geometry check received CUDA images after feature extraction had restored the teacher to CPU. The validated adapter moves only diagnostic inputs to the teacher's actual device. For exact evaluation recovery, use recoveries/10_Evaluation_Recovery.ipynb and Run All unchanged before the original 00:59:58 Europe/Dublin deadline. It reuses the same session, original evaluation identity, completed scores and cached features; a guard forbids encoder training. Original failure records are preserved in evaluation_recovery_v1/. New caches may be added for unfinished evaluations, with the same registered feature computation and identity. Frozen training/model/probe sources remain unchanged.

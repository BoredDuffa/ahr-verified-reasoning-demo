# Shared experiment protocol: registered future work

This document specifies future Notebooks 01–04. None of its real-data experiments is executed by Notebook 00. Concrete dataset identities, revisions, access/license conditions, and numerical budgets are unresolved until the relevant adapter is built and verified.

## Dataset adapter contract

Each sample must expose:

| Field | Contract |
| --- | --- |
| `image` | Decoded RGB image, transformed to a float tensor `[3,H,W]` using a versioned transform specification. A batch is `[B,3,H,W]`. |
| `sample_id` | Stable identifier scoped by dataset repository and immutable revision; retained through derived splits and evaluation. |
| `split` | Explicit project role: SSL train, probe train, development, or final/lockbox evaluation. Native source split is separately recorded. |
| `source_dimensions` | Original decoded `(height,width)` before resizing/cropping; preserve them in resolution studies. |
| `labels` / `annotations` | Optional task-specific metadata. Excluded from JEPA pretraining loss and available only to declared evaluation or diagnostic code. |

Before download, pin the Hugging Face repository, immutable revision, configuration/subset, schema, license/access conditions, image and annotation fields, sample-ID construction, source split mapping, and expected counts. Do not infer that a repository exists from a dataset's common name. Save verification URLs, timestamps, schema checks, counts, file hashes where feasible, and the deterministic split manifest.

Deterministic project split assignment uses stable sample IDs and a recorded seed. Check duplicate or near-duplicate examples and overlapping IDs across train/development/final roles; do not assign image derivatives to separate roles. Lockbox data remains outside training and tuning. Tune only on development data, freeze the protocol, then evaluate the final split. If a standard held-out set is repurposed for development, it cannot simultaneously be the untouched final claim.

## Register comparisons before running

Record the hypothesis, model configs/hashes, primary quality metric, resource metrics, seeds, training examples/steps, effective batch size, precision, mask distribution, optimizer/schedule, augmentation policy, stop conditions, evaluation task, checkpoint-selection rule, and selected ablations before execution. Include both approximately parameter-matched and unchanged-width variants only if explicitly registered; report actual parameter differences in either case.

Keep two comparison questions separate:

- **Equal data/steps:** matched example exposures, batch accounting, masks, and stopping steps; elapsed compute may differ.
- **Equal compute/time:** fixed measured resource or wall-clock budget; completed steps/examples may differ and must be reported.

The main architectural comparison changes the encoder mixer. Use the same predictor recipe, loss, EMA policy, input transforms, label isolation, and evaluation procedure. Register near-only, order changes, and receiver-gating changes as separate ablations, selecting a bounded subset rather than a full combinatorial sweep. A collapse-prevention penalty introduced only for FMM changes the scientific comparison; revise the protocol or make it an explicit ablation.

By default, each dataset experiment and each independent comparison seed starts with fresh initialization, including the teacher copied from its fresh context encoder. Neither synthetic smoke weights nor a previous dataset checkpoint are silently inherited. A transfer/warm-start arm records the exact source run, checkpoint SHA-256, training data exposure, and changed optimizer state policy.

## Learning and evaluation gates

Self-supervised optimization uses only images/masks. Freeze the chosen encoder checkpoint before label-based probe fitting; record whether features come from the student or teacher, which layer and pooling rule are used, and how invalid slots are excluded. Fit the probe on the registered probe-training split, tune its limited hyperparameters on development, and evaluate once on the held-out final role. Probe labels never update the pretrained encoder in a “frozen” evaluation. Fine-tuning is a different protocol.

Tiny ImageNet and Food-101 can use frozen single-label linear classification with registered top-1 accuracy as a primary metric. COCO can begin with image-level multi-label targets derived from training annotations, using per-class average precision and a declared macro mean; declare absent categories, crowd handling, annotation completeness, and threshold policy. Detection requires a separate head, matching/IoU metrics, and training protocol.

For COCO spatial evidence, pre-register a bounded diagnostic such as frozen-feature prediction of relative object-centroid displacement bins for object pairs, stratified by spatial separation and object size. Derive train/development/evaluation examples only from their image split, exclude ambiguous/missing annotations, and compare against a coordinate-only/label-prior baseline. Report pair counts, distance strata, and confidence intervals. Such a probe checks recoverable spatial information; it does not isolate causal long-distance communication or prove semantic preservation. A separately controlled image perturbation diagnostic may be useful, but must define what remains fixed and its distribution shift.

At checkpoints, inspect feature dispersion across images/tokens, feature norms, covariance/effective rank when feasible, near/far branch norms, and loss/gradient distributions. Prediction loss and teacher EMA alone are not anti-collapse proofs. Use multiple independent seeds for important conclusions when affordable; publish intended, completed, failed, and censored seed counts rather than hiding incomplete runs.

## Dataset-specific decision points

| Notebook | Registered decision before execution |
| --- | --- |
| 01 — Tiny ImageNet | Verify adapter and class/split mapping; candidate 64×64 images, patch size 8; small fresh FMM/ViT pilots and frozen probe. Genuine learning is the continuation gate. |
| 02 — Food-101 | Record native dimensions; choose one main representation resolution and a small ablation subset. Label upsampled-resolution sweeps as compute stress tests. |
| 03 — COCO | Fix image-level versus detection scope, annotation-derived labels, spatial diagnostic, and metric definitions. Keep comparison pretraining exposures explicit. |
| 04 — ImageNet-1K | Verify version/access/permitted use and persistent/off-instance storage; estimate time and per-device memory from a successful pilot. Choose a feasible budget before download. |

## Resource measurement contract

Benchmarks distinguish mixer, encoder block/full encoder, and full JEPA training step. Label forward-only and forward-plus-backward separately. Save raw observations, warm-up count, repeats, median and dispersion, batch, visible and full-grid tokens, target tokens, hierarchy order/cell counts, feature width/depth, dtype, device, library/backend, thread count, and workload seed/hash. Equivalent comparisons use identical masks and token definitions. A full-image teacher must appear in whole-JEPA accounting.

Synchronize CUDA before/after timed intervals. Report peak allocated and/or reserved CUDA bytes with explicit reset boundaries and whether persistent model/optimizer tensors are included. Use sequential isolated model lifetimes or fresh measurement processes where possible. CPU process high-water RSS is not tensor allocation and cannot be treated as a per-model incremental peak; if only RSS is available, label it as a process observation. Python allocation tracing does not measure all native PyTorch tensor memory.

Optimized PyTorch scaled-dot-product attention is the performance baseline; record the selected backend through profiler evidence or mark backend resolution unavailable. A naive dense implementation may validate mathematics but cannot be the sole speed comparator. Report slower FMM results as observed. Finite log–log slopes do not prove asymptotic scaling, and upsampling does not add source image information.

## Compute, checkpoint, and resume rules

No GPU count is guaranteed. CPU correctness and an optional single-GPU pilot are supported first. One GPU or selected 2–4 A100-class GPUs may later be practical only after profiling and budgeting. Effective global batch is `per_device_batch × world_size × gradient_accumulation_steps`; record the actual final partial batch and accumulation behavior. More GPUs do not make an oversized single example fit on one device.

Notebook 00 does not implement a distributed launcher. Future distributed work must register sampler sharding, per-rank RNG, synchronization, optimizer/scaler state, and notebook-compatible launch/resume instructions. The brief's future script-launch suggestion does not override the user's notebook-only source rule; any new source-file policy requires an explicit user change. Do not depend on notebook-local multiprocessing history.

Any run expected to exceed 15 minutes requires a short interruption/resume validation first. Checkpoint every stage/epoch boundary and at least every 15 minutes or 250 optimizer steps, whichever is earlier. Use a single writer, temporary file plus atomic rename, checksums, and a manifest under `checkpoints/<run_id>/`; retain the latest two valid resumable checkpoints and the best. Save model/student/predictor/teacher, optimizer, scheduler/scaler or explicit null, all RNG/generator states, sample/sampler position, step, config, metrics, and applicable replay/distributed state. Resume validates schema/config/hash and verifies restored step/metrics; invalid state is an error, never a silent restart.

The workspace is described as persistent boot-disk storage, but instance-deletion retention must be checked separately. Before recycling/destroying an instance, copy best/final checkpoints off-instance and verify hashes. Unmounted local NVMe is not an acceptable persistent copy. Notebook 00's synthetic checkpoint does not authorize instance lifecycle changes.

## Synthesis and reproducibility

Notebook 05 reads completed manifests and metrics; it neither fills missing results nor retrains implicitly. Show quality next to throughput, peak-memory definition, compute estimates and wall-clock cost. State the number of completed independent seeds, estimate uncertainty appropriate to the sampling unit, and preserve failed/partial runs. Keep synthetic correctness, small pilots, and full experiments separate in every aggregate table.

All later notebooks load the Notebook 00 handoff, check source and schema versions, and rerun a compact core test set. They preserve exact artifact paths and hash references. A changed implementation or environment may invalidate an old PASS and requires affected certificates to rerun.

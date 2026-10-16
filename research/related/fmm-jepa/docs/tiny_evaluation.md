# Frozen representation evaluation

All implementation lives in tagged `ahr-export` cells in `fmm_jepa/tiny_evaluation.ipynb`. Import it through the project's existing tagged notebook loader. `tests/test_frozen_probe.ipynb` contains independent CPU certificates; running that notebook requires no dataset download.

The primary image representation is the complete-image EMA target encoder. `VisionEncoder` applies its learned final LayerNorm to every patch, after which evaluation averages the valid patch vectors. At the registered native 64 × 64 resolution and patch size 8, all 64 output patches are valid. There is no predictor, CLS token, or extra nonaffine normalization from the JEPA loss in this representation.

## Interfaces

- `encoder_state_hash(encoder)` hashes named tensors, shapes, dtypes and values independently of mutable checkpoint container files.
- `extract_features(encoder, dataset, cache_dir, identity, batch_size=32, device=None)` consumes a dataset whose items contain `image`, `label`, and `sample_id`. The Tiny ImageNet module provides `EvaluationDataset(adapter, sample_ids, config)`. Features and labels are retained on CPU.
- Cache `identity` must include `dataset_revision`, ordered `sample_ids`, `checkpoint_hash`, `encoder_kind`, `preprocessing`, `pooling`, and `normalization`. Pooling is `mean_final_layernorm_patches`. Values should be JSON serializable. The checkpoint hash must equal `encoder_state_hash(encoder)`.
- `fit_linear_probe(train_cache, dev_cache, output_dir, config, model_id, encoder=encoder)` returns `summary`, per-epoch `metrics`, per-image `predictions`, `performance`, and an inference-only `selected_probe`. Use `vit_untrained`, `vit_trained`, `fmm_untrained`, and `fmm_trained` as the four model IDs.
- `compute_diagnostics(encoder, dataset_or_images, config)` accepts the same deterministic dataset interface or an image tensor. It preserves encoder parameters, device, individual module modes, gradient flags, and Python/NumPy/PyTorch random streams. Evaluation clears the shared geometry cache on entry and exit so inference-mode geometric constants cannot contaminate later autograd training. A native 8 × 8 grid FMM diagnostic-then-backward regression verifies this lifecycle. Select the fixed one-image-per-class development subset before any comparison outcomes.
- `compare_representations(results, config)` verifies equal probe settings and initialization, aligns predictions by stable ID, and calculates trained-minus-initial deltas separately from the trained FMM-minus-trained ViT comparison.

## Probe protocol

The classifier is exactly `Linear(D, 200)` optimized with cross-entropy and AdamW. Defaults are learning rate 0.01, weight decay 0.0001, batch size 256, and a shared probe initialization seed and minibatch permutation seed. The smoke configuration fits two epochs; pilot fits 30. Every representation receives the entire fixed budget. Highest development top-1 selects the epoch; ties select the earliest. Standardization is fitted on probe-training features only, with population standard deviation clamped to `1e-6`; those unchanged statistics are applied to development and optional held-out features. Pretraining and probe training intentionally reuse the selected source-training pool. Development is excluded from encoder pretraining.

The public probe function freezes a supplied encoder, uses evaluation mode, and verifies its tensor-state hash before and after. The optimizer receives only the classifier's weight and bias. Cached feature tensors have no connection to an encoder autograd graph. Each epoch records training and development loss, top-1, top-5, example/class counts, checkpoint identity, seed, and the minibatch permutation hash. Predictions contain the true label, actual top-five class indices, correctness flags, sample identity and model/probe checkpoint identities.

## Checkpoints and exact resume

Pass a canonical project checkpoint folder, for example `checkpoints/<run_id>-probes`, as `output_dir`. Each model owns its subdirectory and a nonblocking filesystem writer lock. Resumable snapshots contain classifier and optimizer states, all RNG states, the explicit current permutation and consumed offset, constant learning-rate scheduler position, disabled-scaler declaration, configuration/cache identities, standardization, completed epochs, successful updates, and metrics. Save boundaries are every epoch, 250 probe optimizer updates, or 15 minutes. The atomically replaced `probe_manifest.json` identifies the latest valid state and best development epoch. At least the latest two valid snapshots and the best snapshot are retained. A malformed manifest, hash mismatch, orphan checkpoint, or changed configuration raises instead of restarting silently.

Resume by calling `fit_linear_probe` again with the identical arguments. `stop_after_updates=1` is a bounded certificate hook: it saves the exact partially consumed epoch and returns `INTERRUPTED`. Tests compare a fresh-state continuation with uninterrupted training and require identical selected state hashes and predictions. The compact `selected_probe.pt` is an inference export; the manifest points to the separate full resumable training snapshots.

Feature extraction writes progress after 32 batches or 15 minutes, and a complete cache atomically. Rerunning with identical arguments resumes at the first unprocessed sample. Valid complete caches are reused; stale identities, changed feature contents, sample-order mismatches, or missing commit metadata raise. Reuse produces an explicit `NOT_RUN` extraction timing rather than a fabricated measurement.

## Diagnostics and interpretation

Dispersion is measured before optional feature standardization. It includes feature standard deviations, embedding norms, between-image cosines, covariance eigenvalues and effective rank, and token variation across images at matching patch positions. With covariance eigenvalues `lambda`, effective rank is `exp(-sum(q * log(q)))` for `q = lambda / sum(lambda)`. Exactly zero variance gives rank zero. The maximum possible covariance rank is `min(D, n - 1)`.

Registered trace `<= 1e-12` and effective rank `< 2` flags are advisory diagnostics, not universal collapse theorems. Position-dependent tokens repeated identically across images have zero matching-position dispersion, even if positions differ strongly within one image. Nonzero FMM near/far message norms establish activity only. Training-time branch gradient norms are collected by the training runner.

Paired bootstrap intervals resample evaluation images while retaining the pairing between model predictions. They describe evaluation-sample uncertainty conditional on those fixed models and exclude variability from training seeds. Development was used for probe epoch selection, so these intervals are exploratory and do not remove selection bias. A one-seed smoke run always remains integration-only, with learning inconclusive and Notebook 02 readiness false regardless of scores.

Held-out evaluation rejects smoke mode. It requires explicit `run_mode=pilot`, `evaluation.heldout_enabled=true`, a selected probe with choices fixed, matching representation identity, and sample disjointness from both training and development. No held-out fitting or statistics estimation occurs.

## Timing scope

Feature timing includes lazy decoding, deterministic preprocessing, transfers, encoder forward, CPU feature transfer, and periodic progress writes; complete-cache final publication and initial model transfer are outside its timer. Probe timing includes classifier updates, per-epoch train/development scoring, and checkpoint writes. It excludes encoder extraction. Batch medians and median absolute deviations retain all measured batches and exclude no warmup batch. CUDA measurements synchronize around these dedicated timing regions and record allocated-memory baseline/peak, which include already live tensors and exclude reserved allocator memory. CPU tensor peak memory is explicitly unmeasured. Caller orchestration must load and evaluate encoders sequentially to avoid cross-model live allocations.

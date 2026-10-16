# Notebook 05 learning and scaling protocol

The user has authorized fixing the numerical failure and starting a fresh long run. The repaired study uses `learning-resolution-stable-001`; the failed `learning-resolution-001` evidence and its verified source snapshot remain preserved. Reusable preparation imports still load definitions only. The launch requires independent repair checks, then actual-configuration calibration and exact-resume preflight before long training.

## Stability repair and failure policy

The repair certificate is `artifacts/notebook_05/repair_validation/stability_certificate.json` with schema `fmm-jepa-semantic-stability/v1`, status `PASS`, numerical backend `semantic-stable-v1`, and the SHA-256 of the newline-joined tagged stable-attention definitions. Its `checks` must report `PASS` for `reproduced_failure`, `operator_regressions`, `late_checkpoint_stress` and `exact_resume`. The study binds the certificate file hash at its first launch and verifies it on resume. Diagnostic checkpoint continuations are isolated and never become scientific-study initial weights.

A first numerical failure aborts immediately (`max_consecutive_failures=1`), leaving the last clean checkpoint for exact replay. The runner refuses any saved failed-attempt state or exposure/trace mismatch before resuming. Thus an error cannot silently consume a different sample stream for one architecture. Fresh scientific arms start from matched initialization under the new run ID; failed-run checkpoints are used only as repair diagnostics.

The replacement method applies consistently to every semantic model, including calibration, paired initialization, checkpoint reconstruction, context encoders, EMA teachers, initial-feature controls and final evaluation. The replacement's source is included in each stage's immutable source hashes. Neither passing numerical checks nor completing the experiment establishes a positive scientific result without the registered learning comparisons.

## Fixed scientific scope

The study compares ViT with the semantic-local operator using the `semantic-stable-v1` numerical implementation. Its repair is loaded from `experiments/05_scaling/stable_attention.ipynb` and installed in memory; frozen `fmm_jepa` definitions are not edited. Both use the same scaffold, fixed FP32 arithmetic, effective batch 128, data/augmentation/mask streams within each seed, and 1,000 successful pretraining updates. There are three paired initialization seeds. The learning schedule and its1,000-update horizon remain unchanged during short runtime calibration. Root configuration fixes the same EMA schedule for both architectures; the current proposed start is 0.99.

| Stage | Image size | Patch8 grid tokens | Pretraining identities | Fixed probe-fit identities | Development identities |
|---|---:|---:|---:|---:|---:|
| Learning |128×128|256|10,100|10,100|2,020|
| Resolution |256×256|1,024|10,100|10,100|2,020|
| Data volume |128×128|256|65,600|10,100|2,020|

The resolution and data-volume stages proceed only if the controller's preregistered learning gate passes after stage1. The default gate requires a positive lower confidence bound for learning gain; the full contrast, confidence interval and seed aggregation are defined in the frozen evaluation protocol. A failed or unresolved gate remains a recorded result and does not trigger an automatic architecture or hyperparameter search.

Each stage accounts for 18 fixed-final-epoch probe jobs: six fresh initialization controls and 12 trained controls at the 300- and 1,000-update milestones. Probe fitting uses the same10,100 identities and final 20-epoch budget. No expanded pretraining labels are added to the data-volume stage's probe fit. Initialization features may be reused only when their complete encoder/data/preprocessing identity matches.

## Immutable source and data membership

The source is the existing pinned Food101 cache and `artifacts/notebook_02/food101-pilot-001/` manifests. Notebook 05 does not rewrite that cache, old notebooks, the old split, labels, source configuration or duplicate audit.

The original audited training pool contains 65,600 identities. The full development reserve contains 10,094 identities, of which the same 2,020 are used for this study. All training identities remain disjoint from the entire development reserve and all 25,246 retained official-test identities. The exact original decoded-RGB duplicate exclusions are preserved; this does not rule out near duplicates or semantic similarities.

For every selected pretraining, probe-fit and development image, preparation verifies the original encoded byte count and SHA-256 against the original decoded-image audit record. Verification therefore covers 12,120 distinct images in the learning/resolution stages and 67,620 in the data-volume stage. Test image bytes are never read by Notebook 05. Existing test identity metadata is used only to preserve its reservation.

New immutable stage evidence is written under `artifacts/notebook_05/<run_id>/stages/<stage_id>/`: `dataset_manifest.json`, a derived `split_manifest.json`, `stage_data_manifest.json`, and `image_verification.json`. The source dataset fingerprint is retained; each derived stage split gets a new identity. The returned bundle exposes `train_ids` for pretraining and a separate `probe_train_ids` list for supervised probe fitting.

## Calibration after the user starts the notebook

Calibration measures complete real-data JEPA updates for both architectures: three warmup updates followed by five measured updates. The measured step includes preprocessing/transfers, forward/backward, clipping, optimizer, scheduler and EMA; checkpoint writing is excluded from per-step timing. Fresh calibration weights are discarded and never become learning-stage checkpoints.

It first tries microbatch 128 with one accumulation step. If either architecture runs out of memory or exceeds the 70 GiB allocation limit, both are considered at 64×2, then 32×4, then 16×8. The selected setting is the largest common feasible microbatch. Effective batch remains128 and the sample/occurrence/augmentation/mask traces must agree across completed cases. Numerical failures other than OOM are errors, not reasons to silently change the experiment.

The actual first 128-image batch also verifies weighted JEPA loss and all trainable gradients against 16×8 accumulation. The per-sample loss weights and natural packed padding are retained. This certificate uses the unchanged Notebook 04 helper in an isolated stage folder and binds its evidence to the full Notebook 05 source/configuration/data identity. Gradient tolerance is `rtol=3e-4, atol=5e-6`; weighted-loss absolute tolerance is `3e-6`. No teacher gradients are permitted.

Calibration checkpoints are under `checkpoints/<run_id>/stages/<stage_id>/calibration/<case>/`. The existing full-state checkpoint implementation retains the latest two valid resumable states and the best, uses atomic writes and one writer, and records model/optimizer/scheduler/EMA/RNG/sampler/configuration/metrics. Calibration saves at 0, 4 and 8 successful updates. Main training has its own required preflight and checkpoint cadence.

## Runtime and hard deadlines

Historical128-resolution measurements were approximately 1.455 seconds per ViT update and 1.676 seconds per semantic update at effective batch 128. At three seeds and 1,000 updates each, that implies 9,393 seconds (2.61 hours) of training before uncertainty and other work. A planning estimate applies a 1.2 training-time margin, adds a combined 1,800-second evaluation/preflight/report reserve covering18 probe jobs, and adds 300 seconds for preparation/calibration. This totals about 3.71 hours; it is a forecast, not a completion guarantee.

The higher-resolution stage has no measured Notebook 05 estimate before calibration. No reliable exact full-study runtime is claimed. A rough full-study planning range is12–24 hours, including higher-resolution work. Default maximum budgets are 24 hours for all enabled stages or 4 hours for learning alone, selected in the main configuration before launch. The actual clock begins at the authorized repaired-study launch and is persisted once; calibration, preflight, training and evaluation share that original absolute deadline. No stage or restart resets it.

Before a stage begins actual learning, the controller recomputes the estimate using that stage's measured calibration and the remaining original time. If the registered1,000-update, three-seed stage is forecast not to fit, it stops with `INCOMPLETE_BUDGET`, preserves earlier stage results, and explains the unchanged requested budget. It never silently lowers the update count, drops seeds, shortens probe training, or extends the deadline. Starting a differently configured run is a user decision made before that new run begins. Unknown later-stage cost does not veto a feasible learning stage before its result exists.

The learning and data-volume combined reserve is 1,800 seconds; the resolution stage uses 3,600 seconds for evaluation/preflight/report work. These explicit reserves may still be exceeded, especially by higher-resolution feature extraction. The estimator reads `config.budget.evaluation_reserve_seconds` and refuses values below the stage minimum. The persistent external watchdog enforces the hard maximum even if a numerical library call or nested helper alarm does not return promptly. Time-limited results remain incomplete, not negative learning evidence.

## APIs and resumption

`load05_module(root, name)` uses only `ahr-export` cells with the canonical module name `fmm05_<name>`. Imports do not configure CUDA or construct models.

`prepare_stage_bundle(root, artifact_dir, stage, config, verify_images=True)` returns the source adapter, immutable stage manifest, verified pretraining/probe/development IDs and verification evidence. `stage` is one of the exact registered stage dictionaries; `config.image_size` must match it.

`calibrate_stage(root, artifact_dir, stage, config, bundle, deadline_unix)` runs only at user execution. Its stable result includes `selected_microbatch`, `selected_accumulation_steps`, `selected_model_step_seconds`, `equivalence_status`, `budget_estimate`, `fits_budget`, and inspectable per-case evidence. `estimate_stage_budget(...)` is CPU-only; `require_stage_budget(...)` records the estimate and refuses to begin an over-budget stage.

To resume, run `notebooks/05_Learning_and_Resolution_Scaling.ipynb` top-to-bottom (resume cell `launch-staged-study`) with the same run ID, stage configuration, frozen exported sources and original clock. A completed calibration is verified and reused. An interrupted calibration resumes from its latest valid complete optimizer boundary; corrupt or mismatched evidence causes an explicit error. Original completed results remain immutable. All long training is launched by the main notebook into its detached tmux session, independently of the browser connection.

## Interpretation boundaries

Three paired seeds improve on a one-seed screen but do not guarantee a conclusive ranking. Development-image uncertainty and training-seed uncertainty must remain distinct. A positive representation-learning result, an architecture comparison and a compute tradeoff are separate claims. Fixed-resolution throughput and memory measurements do not establish an asymptotic scaling law. The official Food101 test remains `NOT_RUN` throughout this study.

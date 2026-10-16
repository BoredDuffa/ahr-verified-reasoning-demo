# Notebook 01 executed evidence

Run `20260923T214538Z-9663c15f`; predecessor `20260923T163118Z-57c8d609`. Profile **pilot**.

Engineering ready: **True**. Learning evidence: **positive**. Ready for Notebook02: **True**.

All engineering gates passed; registered pilot learning criterion met. Review raw diagnostics and limitations before a separately authorized next stage.

## Data and protocol

Intended repository: `thethinkmachine/tiny-imagenet`; source `9f4331d7160a7f6e90c12e3533d658e9c8c7ae3c`; converted Parquet `5318f3a410192b1480adce2ea61e8e2c139adb53`.
Selected memberships: {'training': 20000, 'development': 2000, 'heldout': 10000}. Duplicate-cleaned pools: {'training': 89975, 'development': 9998, 'heldout': 10000}. 27 exact duplicate copies excluded before selection; no relabeling. All 110,000 source images decoded and class mapping verified.
Native 64×64, patch8,64 tokens. Same-scaffold dim192,depth4, FMMp4,leaf2; not parameter matched. Devicecuda:0,FP32. Equal effectivebatch128.

## Actual model budgets

|Model|Trainable parameters|Teacher bytes|Successful updates|Attempts|Image exposures|
|---|---:|---:|---:|---:|---:|
|vit|2,263,296|7,267,584|5000|5000|640000|
|fmm|2,606,088|8,638,752|5000|5000|640000|

## Frozen representations

Complete-image EMA target, final affine LayerNorm patch outputs averaged over64positions; predictor absent. Training-only feature standardization. Linear200-class AdamW, lr0.01,weightdecay0.0001,batch256,30epochs for every control. Best development top1, earliest tie. Initialization and data order matched.

|Representation|Dev top1|Dev top5|Examples|
|---|---:|---:|---:|
|vit_untrained|5.00%|13.70%|2000|
|vit_trained|8.85%|22.30%|2000|
|fmm_untrained|4.60%|14.70%|2000|
|fmm_trained|9.85%|24.30%|2000|

|Paired top1 contrast|Delta percentage points|95% interval|
|---|---:|---|
|delta_vit|+3.85|[2.4, 5.2]|
|delta_fmm|+5.25|[3.85, 6.550000000000001]|
|delta_architecture|+1.00|[-0.45124999999999993, 2.4012499999999988]|

Intervals resample development images conditional on these fixed models. They do not measure variation across training seeds. Uniform0.5% chance is only a sanity reference.
Held-out predictive evaluation: **NOT_RUN**. Dataset integrity auditing is separate from predictive evaluation.

## Raw representation diagnostics

|Encoder|Step|Mean feature std|Mean cosine|Effective rank|Rank ceiling|Far norm|
|---|---:|---:|---:|---:|---:|---:|
|vit|0|0.2105|0.8123|2.540|192|None|
|vit|2500|0.5036|0.3665|13.874|192|None|
|vit|5000|0.5105|0.3168|17.340|192|None|
|fmm|0|0.1877|0.7699|2.526|192|18.3722288949149|
|fmm|2500|0.2391|0.6828|6.091|192|38.29637895311628|
|fmm|5000|0.2467|0.6855|7.560|192|41.080932344709126|

Diagnostics use200fixed development images before probe standardization. Rank<2 and variance≤1e−12 are advisory registered warnings. Nonzero far output/gradient is branch activity, not useful long-range semantics. Full covariance spectra and matching-position across-image dispersion are saved in representation_diagnostics.json.

## Measured performance

Full per-phase timings, quartiles, sample counts, peak allocations, backend events and workload definitions are in performance_metrics.json and its CSV. All dedicated encoder measurements use the same complete64-token real-image batch. Training step timing includes data/preprocessing and optimizer/EMA; excludes checkpoint/diagnostic overhead, which is reported in total wall time.

|Model|Data/update ms|JEPA update ms|Encoder forward ms|Encoder peak MiB|Training invocation peak MiB|Training wall s|
|---|---:|---:|---:|---:|---:|---:|
|vit|237.562|1356.484|4.300|73.566|171.069|7074.428|
|fmm|232.320|2179.002|22.913|99.053|492.842|11192.926|

Training invocation peak includes diagnostic allocations and backend workspace; encoder peaks use a separate full-grid measurement. CUDA allocator peaks exclude reserved cache and context memory. Probe fitting runs on CPU; CPU tensor peak memory was not measured.

|Representation|Feature extraction wall s (train+dev)|Probe wall s|
|---|---:|---:|
|vit_untrained|19.151|7.868|
|vit_trained|19.028|7.823|
|fmm_untrained|31.317|7.814|
|fmm_trained|31.203|7.910|

## Certificates and execution

|Gate|Status|Evidence|
|---|---|---|
|E1|PASS|predecessor_verification.json|
|E2|PASS|verification_suite.json|
|E3|PASS|verification_suite.json|
|E4|PASS|verification_suite.json|
|E5|PASS|verification_suite.json|
|E6|PASS|verification_suite.json|
|E8|PASS|verification_suite.json|
|E7|PASS|training_metrics.csv|
|E9|PASS|verification_suite.json|
|E10|PASS|resume_certificate.json|
|E11|PASS|comparison_summary.json|
|E12|PASS|comparison_summary.json|
|E13|PASS|performance_metrics.json|
|E14|PASS|clean_execution.json|

Resume: controlled CPU FP32, single thread, deterministic algorithms, K4/J2 on real data; exact model/optimizer/scheduler/RNG/consumed-order/augmentation/mask comparison for both models. An earlier two-thread FMM test had identical losses but parameter differences up to3.28e−7, retained under development_verification/threaded_resume_observation.json. This does not support cross-hardware/software bitwise equivalence.

## Answers supported by this run

1. Does the real-data pipeline work? Yes, all14 engineering gates passed for the executed profile.
2. Did FMM learn useful features? This pilot supports improvement over FMM initialization under the registered criterion; the conclusion remains conditional on one training seed and development-based probe selection.
3. Advantage over ViT? The paired interval does not resolve a development top-1 advantage. The FMM-minus-ViT difference is +1.00 percentage points, paired 95% interval [-0.45124999999999993, 2.4012499999999988]. Interpret quality alongside the measured compute; one seed does not establish a robust architectural advantage.
4. Ready for Food-101? The recorded readiness criteria are met. Review the raw diagnostics and limitations, then separately authorize and preregister a fresh-initialization Food-101 comparison.

## Resume and next stage

All executable project source remains in notebooks. The import hook loads only tagged `ahr-export` definition cells and enables ordinary `from fmm_jepa...` imports. The main notebook calls the same reusable runner in both profiles.

The registered pilot has been executed. Review its learning contrasts, representation diagnostics and compute before choosing any new experiment; this report does not request an automatic rerun. Long-running continuation belongs in the recorded detached tmux session.

Exact resume: set `FMM_JEPA_01_RESUME=20260923T214538Z-9663c15f` and `FMM_JEPA_01_PROFILE=pilot`, then rerun the notebook from its bootstrap cell through the detached executor. It validates protocol/source/config/dataset identities and loads checkpoint manifests; invalid states raise rather than restart. Training/probe checkpoints retain latest two valid plus best under checkpoints/<run_id>-*. Initial encoder snapshots are immutable controls, not resumable checkpoints.

## Limitations and next stage

- One paired training seed; development selects probe epochs, so the paired bootstrap is exploratory and excludes training-seed variability.
- Initial integration run20260923T173233Z-82b71a99 failed when diagnostic inference geometry entered training; preserved evidence and a diagnostic-to-backward regression accompany the cache-boundary fix.
- Source valid held out from predictive evaluation unless explicit pilot flag.
- Exact resume certificate scoped to CPU FP32 one thread and pinned software; no cross-device bitwise claim.
- 64-token timings cannot establish asymptotic scaling; p4FMM has16nodes/cell and greater parameter count.
- No mixed precision/multiworker/distributed resume validated.
- Persistent boot disk; instance deletion policy unverified; verify off-instance best/final hashes before recycle.

Review the completed Tiny ImageNet pilot, its own-initialization controls, raw representation diagnostics and measured compute. If the recorded criteria are met and the next stage is separately authorized, preregister one bounded same-scaffold Food-101 comparison from fresh initialization with all four controls. Otherwise investigate the observed limitation before choosing a new experiment; no automatic pilot rerun or dataset transition.

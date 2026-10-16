# Semantic-local bounded pilot: screening criteria

Registered before training outcomes for a single-seed three-arm exploratory Food-101 screen. The candidate uses normalized positive semantic features plus exact local exponential weights. The original FMM and ViT are contemporaneous equal-update controls. All three use the same data membership, augmentations, masks, scaffold, JEPA predictor and training schedule; initial/trained target encoders receive matching frozen-feature probes.

The run must finish within the user's 2–3 hour bound, including compute measurements, preflight, training and evaluation. Calibrate the common update budget before the production protocol is frozen. A deadline cutoff is incomplete evidence; it never authorizes extending the limit automatically. Official test remains untouched.

Assess three separate questions, without a universal success score:

1. **Learning:** report semantic trained-minus-initial development top-1 and the paired 95% development-image bootstrap interval. Label learning promising only when its interval lower bound exceeds zero and the registered raw-embedding degeneracy advisories are absent. Inspect raw rank/variance even when flags are absent.
2. **Quality comparison:** report semantic-minus-ViT and semantic-minus-original-FMM top-1 deltas and paired intervals. Intervals crossing zero mean unresolved; they do not establish equivalence or noninferiority. One training seed cannot estimate architecture seed uncertainty. Development-selected probe epochs and exploratory multiple contrasts limit interpretation.
3. **Cost repair:** a predeclared promising threshold is at least 20% lower median complete optimizer-step time than the original FMM, at the same accumulated effective batch. Report ViT ratios, encoder-forward timings, full-step timings and both total/incremental peak CUDA allocations separately. If actual accumulated-step timing is unavailable, identify the synthetic one-microbatch scope explicitly. A cost win alone is insufficient evidence of useful representation learning.

Compute uses isolated live GPU models, the actual scaffold/dtype, identical synthetic fixtures/masks/batch, at least three warmup and ten measured calls per case. The default scaling grid is 256 and 1,024 full-grid tokens. Synchronized timing excludes data preprocessing/transfers/checkpoints/profiler and includes finite checks. Full JEPA includes loss/backward/clipping/optimizer/scheduler/EMA. Reserved allocator/context memory is outside CUDA tensor allocation peaks. Saved source/environment hashes and raw timings permit audit; two grid sizes do not establish an asymptotic exponent or crossover beyond measured sizes.

The finite random feature realization has no proved task-specific softmax approximation guarantee. Bounded operator certificates cover algebra, masks, derivatives and recovery, not learned semantic quality.

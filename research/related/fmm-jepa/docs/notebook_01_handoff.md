# Notebook 01 handoff

# Notebook 00 engineering report

Current milestone: reusable core and verification.

Notebook: `notebooks/00_FMM_JEPA_Core_Theory_and_Verification.ipynb`.

Clean execution: this run reached its final cell; the external `clean_execution.json` records completed clean-kernel validation.

Device and precision: cpu/float32 neural and recovery, cpu/float64 classical identities and gradchecks, cuda:0/float32 small optimizer/EMA smoke and mixer profiling.

Modules: bootstrap, geometry, operators, models, data, training, evaluation and reporting, all tagged notebook modules.

| Model | Context | Predictor | Total trainable | Frozen teacher |
|---|---:|---:|---:|---:|
| FMM | 130,020 | 16,992 | 147,012 | 130,020 |
| VIT | 112,448 | 16,992 | 129,440 | 112,448 |

Required gates: **{'PASS': 20}**. Gradient isolation and EMA: PASS; exact CPU checkpoint/resume: PASS.

| Classical grid | Order p | Absolute L2 error | Stabilized relative L2 error |
|---|---:|---:|---:|
| 8 × 8 | 2 | 1.15596 | 0.00892736 |
| 8 × 8 | 4 | 0.0338871 | 0.000261706 |
| 8 × 8 | 6 | 0.000592741 | 4.57767e-06 |
| 7 × 9 | 2 | 1.95093 | 0.00523608 |
| 7 × 9 | 4 | 0.087732 | 0.000235464 |
| 7 × 9 | 6 | 0.00178678 | 4.79554e-06 |

Synthetic optimization: 12 steps; first loss 0.558172, last loss 0.498536. This is a software smoke test.

Largest measured grid: **28 × 28**. Timing below is forward + backward; MAD is repeat dispersion. Full raw rows include all sizes and forward-only results.

| Device / scope | Grid tokens | Model | Median ms | MAD ms | Peak allocated MiB |
|---|---:|---|---:|---:|---:|
| cpu / mixer | 192 | FMM | 8.899 | 0.053 | not measured |
| cpu / mixer | 192 | VIT | 1.427 | 0.037 | not measured |
| cpu / encoder_block | 192 | FMM | 9.739 | 0.027 | not measured |
| cpu / encoder_block | 192 | VIT | 2.286 | 0.045 | not measured |
| cpu / full_jepa | 192 | FMM | 23.252 | 0.047 | not measured |
| cpu / full_jepa | 192 | VIT | 11.949 | 0.012 | not measured |
| cuda:0 / mixer | 784 | FMM | 14.215 | 0.040 | 89.44 |
| cuda:0 / mixer | 784 | VIT | 4.342 | 0.006 | 65.92 |

FMM is slower in 22/22 measured comparisons. Observed baseline backends: cpu: torch SDPA CPU flash; cuda:0: torch SDPA memory-efficient. This is an implementation measurement, not a quality–compute comparison or asymptotic proof.

Known bottlenecks and limitations:
- Useful representation learning, real-dataset scores, generalization, novelty and a quality-compute advantage are not established.
- Prototype uses Python-level hierarchy/edge operations and synchronous diagnostic scalar reads; no custom fused FMM kernels.
- Allocated empty square-tree cells count toward cost; very thin rectangular grids can have substantial padding overhead.
- FMM and ViT widths match in this smoke, but actual parameter counts differ; future comparisons must register budgets.
- CPU peak tensor memory was not measured. CUDA memory is live allocator accounting, not total device/process memory.
- Mixed precision, distributed execution and CUDA bitwise resume were not tested; CPU float32 resume was tested.
- Bounded classical interpolation results do not transfer to learned semantic translations.
- Conventional Transformer predictor remains an attention cost; no whole-JEPA linear-time claim.
- Boot disk instance-deletion policy is unverified; checkpoints have not been copied off-instance.

Artifact directory: `/home/benjamin/workspace/ahr_go/fmm_jeppa/artifacts/notebook_00/20260923T163118Z-57c8d609`.

Ready for Notebook 01: **YES — all required gates passed**.

Next task: Notebook 01 — Tiny ImageNet learning pilot.

**Foundation status: verified.**

**Research hypothesis status: not yet established**; learning and comparative evidence belong to later notebooks.

Machine contract: [handoff_00_to_01.json](../handoff_00_to_01.json).

```json
{
  "entry_checks": [
    "reject schema/config/source incompatibilities",
    "rerun fmm_tests.test_geometry.run_geometry_tests and fmm_tests.test_models.run_model_tests",
    "verify checkpoint resume on chosen backend before long runs",
    "do not load synthetic smoke weights for default comparison"
  ],
  "adapter_requirements": [
    "pin and verify Hugging Face repository and immutable revision",
    "verify schema, license/access conditions and source dimensions",
    "register deterministic train/development/evaluation split IDs; prevent image duplication leakage",
    "keep final/lockbox evaluation separate from development",
    "check all class IDs and label mapping; labels excluded from SSL"
  ],
  "pilot": {
    "source_resolution": [
      64,
      64
    ],
    "patch_size": 8,
    "initialization": "fresh independently for FMM and ViT",
    "task": "frozen-encoder single-label linear probe with registered accuracy metric",
    "seeds": "register before launch and report actual completed count",
    "budget": "small matched pilot only; choose steps/batch after measured memory and learning checks",
    "checkpoint": "verify exact resume before any >15-minute run"
  }
}
```

Presentation verification: all seven rendered figures inspected; readable HTML exported. Full HTML browser/MathJax layout inspection was not run because no browser engine is installed. Open `Notebook_00_readable.html` in a browser to inspect that layout.

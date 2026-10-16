# Reading and reproducing the research

## Browser demonstration

Serve the repository root with a static server, for example `python3 -m http.server 8000`, then open `http://localhost:8000/`. No API key, hosted model, network inference or trained weights are required for the demo. Run its existing route regression suite with:

```bash
node tests/demo_routes.js
```

This checks bounded exact examples and abstention. It does not reproduce the learned-model research.

## Lightweight mathematical and Go checks

The earlier exact-search code uses standard Python and can be checked independently:

```bash
python3 -m unittest discover -s research/legacy/tests -p 'test_*.py'
```

For the mini-Go probe, use Python 3.12 and the dependencies in `research/mini-go/pyproject.toml`, plus pytest. From its directory:

```bash
cd research/mini-go
python -m pytest -q
```

Its tests compare legal moves/scoring and bounded tree values to independent oracles, check that neural backups really call the model, and exercise recovery. Passing them does not reproduce training or tournament strength.

## Scientific notebooks

The [notebook index](../research/NOTEBOOK_INDEX.md) lists the source snapshot and original execution evidence. `research/SOURCE_MANIFEST.json` contains original-file, published-file and code-cell hashes. Output clearing changes the notebook file hash while preserving source cells. Source pins in old results may refer to earlier revisions, not the current notebook source.

These are **research workspace snapshots**, not a self-contained release of every model and dataset. No clean-environment end-to-end execution is claimed for this publication. Saved reports are inspectable without rerunning the notebooks. Reproducing a particular experiment requires its matching source revision, pinned dataset, rules, dependencies, configuration and checkpoint. Checkpoint metadata and hashes are not the checkpoint weights themselves.

### Go workspace layout

`research/go/` preserves `notebooks/`, `experiments/ahr_v2/`, `experiments/ahr_jepa_v3/`, `experiments/ahr_native_v4/`, `experiments/ahr_native_v5/`, `config/` and selected `reports/`. Set the project-root controls in the chosen notebook to a prepared workspace with that layout. Some historical cells retain original absolute paths; review and configure them before execution. Do not run every variant sequentially as if it were one pipeline.

Start with `notebooks/00_instance_setup_and_recovery.ipynb` and the relevant branch's `00` contract/shared notebook. Import only cells tagged `ahr-export`; importing definitions must not run another notebook's training. Configure the declared data and weights, independently check the rules, then use that branch's recovery validation and explicit run cell. One-visit KataGo and GNU Go level 1 are particular baseline configurations, not generic engine strength.

External assets such as KataGo, GNU Go, datasets and trained checkpoints are not bundled. The recorded source/rules/model/data identities govern which assets are compatible. Dataset grouping and consumed development/evaluation families must be preserved.

### ARC-style draft

Read [`research/arc-agi-3/README.md`](../research/arc-agi-3/README.md), `NOTEBOOK_CONTRACT.md` and `STATIC_VALIDATION.md`. Work through notebooks 00–15 separately. The local delivery manifest records unexecuted drafts. Hardware setup is separate from model integration, training and evaluation. The exact competition submission contract and an official score are not established by this source release.

### Related FMM-JEPA

The related workspace preserves `notebooks/`, notebook modules in `fmm_jepa/`, experiment modules in `experiments/`, `configs/`, `docs/` and selected `artifacts/`. Set `FMM_JEPA_ROOT` to its prepared project directory and follow its tagged-cell import contract. Datasets, weights and some large/raw artifacts are external. Notebook 20 is an ongoing study snapshot, not a completed result.

## Long runs and recovery

The remote-instance agreement requires all authored executable project logic to remain in `.ipynb` notebooks. Long runs must first pass an interruption/resume smoke test. Use interruption-safe checkpoints under `checkpoints/<run_id>/`, keep latest two plus best, and save model/adapters, optimizer, scheduler/scaler, RNG, sampler/data position, configuration, step and metrics at stage/epoch boundaries and at least every 15 minutes or 250 optimizer steps. Verify persistent storage and off-instance backup before instance deletion. No training run, long evaluation, paid provisioning or remote modification was performed for this repository update.

## What must still be verified

- Clean-environment execution of each scientific notebook with its exact external dependencies.
- Restoration and hash verification of the matching trained checkpoints and datasets.
- Fresh Lean compilation of the historical proof objects if independent proof reproduction is required.
- Repair and revalidation of the latest Go collection failure before any further training claim.
- Final receipts and frozen analysis for unfinished studies; readiness, launch, smoke tests and training loss are separate evidence states.

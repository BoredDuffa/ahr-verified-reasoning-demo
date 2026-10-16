# AHR v2 XL — ARC-AGI-3 notebook workspace

**Working first drafts. No notebook has been executed. No model is trained.**

The user confirmed the original **16 notebooks, numbered 00–15**, rather than combining setup and recovery into 15. These files implement the full Queen–Worker–Scout research architecture and its surrounding data, training, evaluation and demonstration workflow. They are intended for review and Run All **one notebook at a time**.

Remote project: `/workspace/arc-agi-3-ahrv2`  
SSH alias: `arc-agi-3-ahrv2`  
Notebook directory: `/workspace/arc-agi-3-ahrv2/notebooks`

## Start here

1. Open `00_setup_and_hardware.ipynb` in VS Code on the remote workspace. Review its settings, then Run All when ready. It creates the project environment and performs short hardware checks.
2. Select the **AHR v2 XL (Python 3.12)** kernel for notebooks 01–15.
3. Run `01_runtime_and_recovery.ipynb`. It must actually pass before long data collection or training.
4. Notebook 02 prepares missing public game assets through the official SDK when `ACQUIRE_PUBLIC_ASSETS=True`, then lists the local games. Select disjoint game families in 03. Dataset paths, game choices and demonstration provenance must be explicit. Empty game lists are required user configuration, not a fabricated dataset.
5. Review model components 04–08; run the full model integration and recovery validation in 09 before training.
6. Review and run training, comparisons, evaluation, bundling and the visual demo separately. No notebook should automatically execute another notebook's training cells.

## Notebook map

| Notebook | What it does |
|---|---|
| [00_setup_and_hardware](notebooks/00_setup_and_hardware.ipynb) | Persistent project setup, pinned dependencies, kernel installation, hardware and short CUDA checks |
| [01_runtime_and_recovery](notebooks/01_runtime_and_recovery.ipynb) | Deterministic sampler, complete atomic checkpoints, run locks and exact continuation diagnostic |
| [02_arc_environment](notebooks/02_arc_environment.ipynb) | Public asset preparation, official ARC observation/action interface, legal click coordinates and environment recovery limits |
| [03_datasets_and_replay](notebooks/03_datasets_and_replay.ipynb) | Game-family splits, observation-only collection, demonstration provenance, validated replay and batching |
| [04_encoder_and_world_model](notebooks/04_encoder_and_world_model.ipynb) | Grid encoder, latent dynamics, predicted rewards, termination and uncertainty |
| [05_lorentz_workspace_and_memory](notebooks/05_lorentz_workspace_and_memory.ipynb) | Literal 1+3 Minkowski events, Lorentz boosts, bounded evidence memory and branch provenance |
| [06_worker](notebooks/06_worker.ipynb) | Conditional energy ranking, proposal heads and bounded local refinement |
| [07_scout](notebooks/07_scout.ipynb) | Diverse exploratory proposals, energy refinement and novelty estimates |
| [08_queen](notebooks/08_queen.ipynb) | Transformer controller, subgoals, role allocation, halting, action/click policy and value |
| [09_full_ahr_integration](notebooks/09_full_ahr_integration.ipynb) | Complete bounded planning loop, action gates, traces, model sizing and full-model recovery validation |
| [10_supervised_training](notebooks/10_supervised_training.ipynb) | Demonstration targets plus observed dynamics and energy objectives; resumable optimization |
| [11_joint_training](notebooks/11_joint_training.ipynb) | Joint offline updates from recorded experience, with explicit supervised warm start |
| [12_baselines_and_ablations](notebooks/12_baselines_and_ablations.ipynb) | Independent runs and meaningful geometry/controller/Scout comparisons |
| [13_benchmark_and_generalization](notebooks/13_benchmark_and_generalization.ipynb) | Frozen-model evaluation through public interfaces and honest local versus official score reporting |
| [14_inference_and_model_bundle](notebooks/14_inference_and_model_bundle.ipynb) | Load trained checkpoints; package weights, configuration and notebook sources |
| [15_play_and_watch](notebooks/15_play_and_watch.ipynb) | Visual replay and fresh play, with the real board and per-role reasoning traces |

## Research boundaries

- The architecture is complete in scope; the initial XL width/depth and training schedule are **provisional and unprofiled**. The first drafts do not establish a parameter-count, speed or ARC score claim.
- The shared semantic representation is separate from literal Minkowski event coordinates. Lorentz geometry is an experimental representation hypothesis. It does not establish truth, physical causation or correct game rules.
- Energy models rank candidates. The official environment determines what happened. Imagined world-model trajectories remain labelled predictions.
- The first training path is a single explicitly selected GPU per run. Independent ablation/seed jobs can be assigned to other GPUs. Eight-GPU distributed training speed and scaling are not claimed.
- Joint training initially uses advantage-weighted updates on recorded transitions. It is not labelled on-policy PPO or an autonomous online reinforcement-learning campaign.
- Random exploration is not expert supervision. Notebook 10 requires explicit demonstration labels with teacher provenance.
- Development-game evaluation is not proof of generalization to the competition's hidden games. No private game state, hidden game source or oracle simulator is used by the policy.
- The public ARC toolkit adapter is implemented. The exact Kaggle submission runner contract was not verified from the accessible data page; no competition submission package or score is claimed, and no submission is made.

## Execution and recovery

All user-authored executable project logic stays in `.ipynb` files. Library cells carry the `ahr-export` tag; the loader imports only those cells and their declared dependencies. It never executes upstream run cells.

Checkpoints live in `checkpoints/<run_id>/`. The runtime caps cadence at 15 minutes/250 optimizer steps and defaults to 10 minutes/200 steps, with stage and epoch saves. Hash-verified checkpoints retain the latest two plus best, and include optimizer/scheduler/scaler, RNG, data/sampler position, source/config identity and stage state. A changed dataset, configuration or source requires an explicit new run or migration.

Exact optimization resume is separate from external game-session recovery. The public service's hidden state cannot be reconstructed from a notebook checkpoint; interrupted online episodes are explicitly incomplete and must not be silently treated as exact continuations.

Your existing hardware inspection is in [REMOTE_INSTANCE_AUDIT.md](REMOTE_INSTANCE_AUDIT.md). In particular, the initial unexplained utilization on GPUs 0–2, 450 W limits and 512 GiB persistent-volume capacity remain relevant to the first hardware review. No software installation or GPU experiment was performed as part of authoring these notebooks.

## Validation at delivery

The [delivery manifest](DELIVERY_MANIFEST.json) records notebook hashes, schema checks, Python syntax checks, export dependency coverage and unexecuted status. [Validation notes](STATIC_VALIDATION.md) distinguish completed checks from the later runtime review. These checks do **not** establish that Run All succeeds. Runtime, mathematical behavior, learning quality, kernel selection and visual outputs must be checked during the requested notebook-by-notebook review.

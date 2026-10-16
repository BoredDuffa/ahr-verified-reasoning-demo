# AHR v2 XL first-draft notebook contract

User confirmed 16 WORKING FIRST DRAFTS, original 00–15 layout. Author only; do not run notebooks, install remote dependencies, collect data, train, evaluate, or submit. Static JSON/schema/AST/interface validation is allowed. Final status must state unexecuted first drafts.

Remote delivery: `/workspace/arc-agi-3-ahrv2/`; local mirror: this directory. All authored executable project logic resides in `notebooks/*.ipynb`; no exported .py/.sh helpers. Third-party installed packages are dependencies, not authored project code. Source remains notebook-only even for detached jobs: launch Python with a short `-c` expression that loads tagged notebook exports; log/config files are non-code.

## Files and ownership

00_setup_and_hardware.ipynb — root
01_runtime_and_recovery.ipynb — root
02_arc_environment.ipynb — environment agent
03_datasets_and_replay.ipynb — environment agent
04_encoder_and_world_model.ipynb — model agent
05_lorentz_workspace_and_memory.ipynb — model agent
06_worker.ipynb — model agent
07_scout.ipynb — model agent
08_queen.ipynb — model agent
09_full_ahr_integration.ipynb — model agent
10_supervised_training.ipynb — training agent
11_joint_training.ipynb — training agent
12_baselines_and_ablations.ipynb — training agent
13_benchmark_and_generalization.ipynb — training agent
14_inference_and_model_bundle.ipynb — training agent
15_play_and_watch.ipynb — environment agent (coordinate model APIs)

## Notebook format

Use nbformat v4.5. Every executable cell is plain Python (no magics/shell escape). Empty outputs and execution_count=null. Metadata kernel name `ahr-v2`, display_name `AHR v2 XL (Python 3.12)`. Tagged library cells use `ahr-export`; only imports, definitions and harmless constants, no installation, filesystem writes, downloads, model allocation or execution at import. Each notebook has opening markdown purpose / inputs / outputs / expected runtime (estimate, profile required) / checkpoint directory / exact resume instruction / status: UNEXECUTED FIRST DRAFT. Markdown precedes focused code cells. All expensive cells explicitly mark what Run All will do.

## Loading definitions (no hidden kernel state)

Every notebook except 00 begins with this bootstrap (untagged):

```python
from pathlib import Path
import json
ROOT = next((p for p in (Path.cwd(), *Path.cwd().parents) if (p / 'notebooks' / '00_setup_and_hardware.ipynb').exists()), None)
if ROOT is None:
    raise RuntimeError('Open a kernel in /workspace/arc-agi-3-ahrv2 or its notebooks directory.')
_nb = json.loads((ROOT / 'notebooks' / '00_setup_and_hardware.ipynb').read_text())
for _cell in _nb['cells']:
    if _cell['cell_type'] == 'code' and 'ahr-export' in _cell.get('metadata', {}).get('tags', []):
        exec(compile(''.join(_cell['source']), '00_setup_and_hardware.ipynb', 'exec'), globals())
load_exports([DEPENDENCY_IDS], globals())
```

00 exports `load_exports(ids, namespace)`, `default_config()`, `atomic_json(path,data)`, `canonical_hash(value)`, `sha256_file(path)`, `source_manifest(root=ROOT)`, `ensure_layout(root=ROOT)`, `load_config(root=ROOT)`.
Loader recursively resolves dependencies: 01→00;02→01;03→02;04→01;05→04;06→05;07→06;08→07;09→02,03,08;10→09;11→10;12→11;13→12;14→13;15→14. Exports share explicit namespace, loader tracks already loaded IDs per namespace. No execution cells from dependencies are run.

## Shared config (dictionary)

`cfg=load_config()` merges saved `config/ahr_v2_xl.json` onto full defaults; root creates file when notebook00 is run. Modules accept this config dictionary. It has:

```
model: d_model=1024,n_heads=16,encoder_layers=8,world_layers=6,queen_layers=8,
       energy_hidden=2048,energy_layers=4,patch_size=4,grid_size=64,n_colors=16,
       action_dim=8,event_spatial_dim=3,dropout=0.0
reasoning: worker_candidates=16,scout_candidates=16,refine_steps=4,max_depth=3,
           max_model_calls=128,memory_capacity=256,history_length=32,geometry='lorentz',
           queen_mode='learned',scout_enabled=True,latent_clip=8.0
train: seed=42,device='cuda:0',batch_size=8,lr=0.0001,weight_decay=0.01,
       max_steps=2000,grad_clip=1.0,checkpoint_seconds=600,checkpoint_steps=200,
       max_wall_seconds=3600,precision='bf16',stage='supervised'
data: train_games=[],dev_games=[],heldout_games=[],split_manifest='data/splits.json',
      replay_path='data/replay/train.jsonl',expert_path='data/demonstrations/train.jsonl'
environment: operation_mode='OFFLINE',game_id=None,max_actions=256,seed=42
```

These are full architecture dimensions, provisional until profiling; do not quietly substitute tiny training models. Tiny deterministic fixtures permitted only for recovery/unit checks, explicitly not model training or evidence.

## Core tensor/API contract

Observations: `np.ndarray` or torch long `[64,64]`, palette 0..15; batches `[B,64,64]`. Encoder `GridEncoder(cfg)(grids)->(tokens[B,T,D], pooled[B,D])`.
Actions: integer 0=RESET,1..7=ACTION1..ACTION7. `ActionSpec(action:int,x:Optional[int]=None,y:Optional[int]=None)`, `.to_dict()`. ACTION6 requires x,y in 0..63; no fixed-coordinate substitution. Masks `[B,8]`, available actions from public observation only. Click coordinates learned/proposed explicitly.
World model `LatentWorldModel(cfg).forward(latent[B,D], action[B], xy[B,2])->dict(next_latent[B,D], reward[B], done_logit[B], uncertainty[B])`. Predictive ensemble/disagreement or learned calibrated uncertainty; distinguish prediction from observed fact.
Worker `WorkerEBM(cfg).propose(context[B,D], goal[B,D], mask[B,8], count:int)->dict(latents[B,K,D], actions[B,K], xy[B,K,2], energies[B,K])`; `.energy(context,candidate,goal)` broadcasting supported. Scout analogous `ScoutEBM(cfg)`; explicit proposal heads + bounded energy refinement, not energy alone generating.
Queen `QueenController(cfg).forward(history[B,H,D], memory[B,M,D], mask[B,8])->dict(goal[B,D], worker_weight[B], scout_weight[B], halt_logit[B], action_logits[B,8], value[B], click_logits[B,2,64])`; padding optional but avoid all-masked attention.
`AHRModel(cfg)` contains encoder,world,worker,scout,queen. `.forward(grids,actions=None,xy=None)->dict` supplies differentiable `action_logits[B,8],click_logits[B,2,64],value[B],latent[B,D]` plus world output when action supplied. Own `.cfg`.
`AHRModel.act(observation,available_actions,episode_state=None, budget=None)->(ActionSpec,episode_state,trace)`; full bounded Queen/Worker/Scout planning, memory+event-DAG, per-role energies, proposals, observed vs imagined labels, no hidden game oracle. `episode_state` portable serializable dict of CPU values/tensors. `.reset_episode()` if internal buffers; prefer passed state.

Literal Minkowski event coordinates and Lorentz transforms, finite bounded boosts and FP32 geometry. Semantic latents separate from event coordinates; frame transformation is a representational hypothesis, not proof of truth/causality. Verified observed transitions have explicit provenance. Hard gates validate legal actions and budgets, not unknown game rules. Energy ranks candidates; it cannot verify imagined state truth. Event-DAG branches cannot merge mutually exclusive evidence.

## Environment/replay contracts

`ARCSession(cfg, operation_mode=None)`: public official toolkit adapter. `.list_games()->list[str]`; `.reset(game_id)->obs_dict`; `.step(ActionSpec)->obs_dict`; `.close()->scorecard_or_None`; public `state_dict()` and `resume` must not claim exact hidden remote-state restoration if unsupported. `obs_dict` includes `grid`, `available_actions`, `levels_completed`, `state`, `game_id`, `guid` when available. No game source/private state access; no benchmark games in train split. Competition API and scorecard rules explicit.

JSONL transition schema: `episode_id,step,game_id,split,observation[64x64],action{action,x,y},next_observation[64x64],reward,done,available_actions,next_available_actions,source,teacher_action(optional),episode_start(optional)` plus provenance. `load_transitions(path,split=None)->list[dict]` validates; `transition_batch(rows,device)->dict(grids,actions,xy,next_grids,rewards,dones,masks,teacher_mask)` torch tensors. Teacher labels only explicit demonstrations; random rollouts must never become expert targets. Click `xy` normalized to [-1,1] for model inputs; click targets integer64 bins derived original records.
`ReplayBuffer(capacity,seed)` add/sample/state_dict/load_state_dict. `SplitRegistry` or equivalent must reject overlapping GAME FAMILIES (prefix game id before '-') between train/dev/heldout. Public developer game split is not official hidden generalization evidence.

## Runtime/recovery contract

01 exports `seed_everything(seed)`, `capture_rng()`, `restore_rng(state)`, `StatefulSampler(size,seed)` with `.next_indices(batch_size)` and state methods.
`CheckpointManager(root,run_id,config,source_hashes=None)` `.save(model,optimizer=None,scheduler=None,scaler=None,step=0,metrics=None,extra=None,is_best=False)`; `.load_latest(model,optimizer=None,scheduler=None,scaler=None,map_location='cpu')->dict|None`. Full RNG restored on load; payload contains step/metrics/extra/config. Atomic hashed files, manifest, latest2+best, source/config mismatch rejection; fallback older valid checkpoint. `.due(step)` 600 seconds/200 steps; always save boundaries and interruption. Checkpoints rooted `checkpoints/<run_id>/`.
`validate_recovery(root=ROOT)->dict` runs tiny deterministic optimizer continuation check and writes `reports/recovery_gate.json` with source hash. `require_recovery_gate(root=ROOT)` demands passed matching source; no fixture result pre-saved. Full AHR continuation validation in notebook09 required before training10/11, saved `reports/full_model_recovery_gate.json`; implement `require_full_model_gate(cfg,root=ROOT)` in integration module. Training gates do not manufacture success.
`RunLock(root,run_id)` context manager avoids concurrent writer; `append_metrics(path,row)`; `runtime_fingerprint()` Python/torch/CUDA/version identity; actual GPU anomaly probe in00. Training records source/data/config/checkpoint hashes.

## Training / evaluation / demo

Supervised stage uses explicit teacher masks only, alongside world dynamics, reward/done and contrastive energy objectives on valid observed transitions. Joint stage uses real experience, advantage actor-critic + dynamics + energy losses; replay supports exact optimization continuation. Do not claim exact online environment continuation if toolkit does not support it; checkpoint at episode boundaries or mark interrupted episodes discarded/replayed separately without duplicating optimization.
Training exports `train_supervised(cfg,run_id,root=ROOT)` and `train_joint(cfg,run_id,root=ROOT)`; their implementation may share `train_stage`. Notebook execution cells call actual stage after gates, allowing user Run All later. No auto-execution while authoring. Full source checkpointing includes data/sampler/replay/episode state. Warm start next stage from prior best explicit path; no silent random restart.
Ablations must change real behavior (Euclidean geometry, fixed Queen scheduling, no Scout), independent run IDs, matched data/budget, no fabricated result tables. Holdout benchmark freezes model hash; official scorecard score when available, local measures labeled proxies. Online competition cannot be exact-resumed after arbitrary process crash without public toolkit support.
Bundle exports include `build_model_from_checkpoint(checkpoint_path,cfg=None,device='cpu')` and `export_bundle(...)` with notebook source manifest + config+weights, no .py. Demo is ipywidgets/Matplotlib grid playback with per-role decisions, energy traces and observed vs imagined display. No Gradio external service necessary. Model predicts ACTION6 coords; a clickable canvas option can be deferred but full replay and fresh-run function must exist. A random/untrained model must be clearly labelled and never silently substituted for missing checkpoints.

No API keys in source/outputs. No submissions or acceptance of competition rules. Exact Kaggle runner integration only if verified from official current docs, otherwise explicitly separate generic official toolkit adapter from submission packaging.

# Model and data contracts (schema 1)

The reusable package is stored in notebooks, as required by the workspace agreement. The bootstrap installs imports by executing only explicitly tagged `ahr-export` definition cells. No training or test execution is triggered by an import. `fmm_jepa.models`, `fmm_jepa.data`, and `fmm_tests.test_models` are public imports after installing the bootstrap. See the main README for bootstrap instructions.

## Tensors and geometry

| Object | Contract |
|---|---|
| Images | Floating `[B,3,H,W]`; H and W positive multiples of patch size |
| Original patch grid | `(H // patch_size, W // patch_size)`, arbitrary positive rectangle |
| Original index | Row-major integer `row * grid_width + column` |
| Patch coordinates | `(x,y)=((column+0.5)/grid_width,(row+0.5)/grid_height)` |
| Mask indices / validity | Matching `long [B,V]` and `bool [B,V]`, on the image device; each sample has at least one valid unique in-range index |
| Invalid entries | Ignored; the sampler uses index `-1`. Invalid outputs are exactly zero. |
| Encoder output | `[B,V,D]` in the exact supplied original-index order, including padded slots |
| Complete encoder input | Omitting indices and validity produces `[B,N_grid,D]` in full row-major order |
| Targets | Tensor `[B,T]` or list of such tensors; corresponding validity tensor/list; each target block has at least one valid unique position per sample |
| Predictor output | List of `[B,T_k,D]`, each in the supplied target block's order; padded outputs zero |

`VisionEncoder(kind='fmm'|'vit', patch_size=8, dim=64, depth=2, heads=4, order=2, leaf_size=2, receiver_gate=True)` implements `forward(images, indices=None, valid=None)`. Both architectures use the same raw patch extraction, linear patch embedding, fixed Fourier positions, pre-normalized residual scaffolding, 4D-width GELU MLP and final LayerNorm. The FMM path contains no attention. For ViT, embedding width must be divisible by its head count.

Raw pixels are unfolded on the complete grid, then visible patches are gathered **before** any learned embedding or mixing. Thus the context model computes no hidden patch features, although raw extraction still allocates full-grid storage. The leakage gate perturbs every hidden image patch and requires unchanged visible context output. Positional features preserve the original geometry after masking; compressed sequence positions are not substituted for image coordinates. Fixed Fourier positions support rectangular grids without learned positional-table resizing. The frequencies are a simple geometric encoding choice and do not constitute a semantic guarantee.

`EncoderBlock(kind, dim, heads=4, order=2, leaf_size=2, receiver_gate=True)` and `AttentionMixer(dim, heads=4)` accept `forward(x, indices, valid, grid)` with `[B,V,D]` features. The latter uses `torch.nn.functional.scaled_dot_product_attention` with valid-key masking and no dropout. `observed_attention_backend(device='cpu', dtype=torch.float32, dim=32, heads=4, tokens=16)` reports actual profiler dispatcher events for its stated workload; backend selection can change by dtype, shape, hardware, mask, or PyTorch version. A recorded CPU backend must not be reused as a CUDA backend claim.

## Synthetic data and masks

`synthetic_images(batch_size, height, width, generator)` returns CPU float32 `[B,3,H,W]` values in `[0,1]` generated from colored Gaussian blobs, sinusoidal stripes and noise. The explicit generator must be a CPU `torch.Generator`. Images have no class labels and support software smoke tests only.

`sample_masks(grid, batch_size, generator, context_area=(0.65,0.90), target_area=(0.08,0.18), aspect_ratio=(0.5,2.0), target_count=2, max_retries=32)` returns:

- `context_indices`, `context_valid`: padded CPU tensors.
- `target_indices`, `target_valid`: corresponding lists of padded CPU tensors, one per target block.
- `metadata`: grid, intentional overlap policy, retry cap and observed fallback counts.

Target rectangles are sampled independently, so different target blocks may intentionally overlap. A random per-image anchor is forbidden in every target proposal, leaving at least one possible context patch. After bounded retries, a target becomes a singleton distinct from that anchor. A context rectangle is sampled and the complete union of target positions is removed. If this remains empty after the retry cap, context becomes the anchor. Area and aspect ranges describe proposals: rounding, clipping and target removal change the realized shape and area. A grid must contain at least two patches. This implementation preserves variable counts with masks, rather than truncating a batch to its smallest count. It differs from the official I-JEPA sampling procedure; this is an explicit synthetic foundation choice that later protocols must review.

Masks can be moved to an accelerator after drawing. Restore the CPU generator state from the checkpoint before sampling the next batch. The sampler does not use global RNG state.

## JEPA and loss

`JEPA(config)` takes the model configuration dictionary with the encoder keys above and `predictor_dim=32`, `predictor_depth=1`, optional `predictor_heads` (default encoder heads). Other configuration entries are preserved in `model.config` but do not silently alter the architecture. Predictor width must be divisible by predictor heads.

`model(images, context_indices, context_valid, target_indices, target_valid)` returns a dictionary containing:

- `loss`: differentiable scalar tensor.
- `prediction`: target-block list of `[B,T_k,D]` predicted representations.
- `target`: equally shaped stopped-gradient normalized teacher features.
- `context_features`: `[B,V,D]`.
- `metrics`: detached Python floats for loss, per-token representation norms, featurewise dispersion and available FMM near/far norms.

Context indices must exclude the union of all targets; overlap is checked and rejected. The target encoder intentionally processes the full image. Its output receives featurewise, nonaffine LayerNorm before target indexing. The predictor receives context features and the original context/target coordinates. A learned mask token plus target Fourier coordinates stands in for each target; true target features never enter the predictor. Each target block is predicted separately using a small ordinary Transformer over context and target-position tokens. Target tokens can interact with other target positional tokens within that block. The attention predictor remains a potential whole-system bottleneck.

The objective is SmoothL1 with beta 1 between predictions and normalized, detached teacher features. For each block and each batch sample, element losses are summed over valid positions and features, divided by that sample's valid target count times D, and then averaged over samples and blocks. This gives each block and sample equal weight despite different padded lengths. Overlapping targets contribute once for each block containing them. A hand-computed uneven-padding fixture verifies this normalization. There is no reconstruction objective, negatives, contrastive loss, or collapse-prevention regularizer.

Teacher parameters initialize as an exact independent deep copy of context parameters, are frozen and stay in evaluation mode even when `model.train()` is called. `model.update_teacher(momentum)` must be called only after a successful student optimizer step: it applies `target = momentum * target + (1 - momentum) * context`. Floating/complex buffers receive EMA; integer buffers are copied. Current encoders have no train/eval-dependent normalization or dropout, but the explicit policy protects future changes. The official pinned implementation does not explicitly call teacher eval and updates parameters only; this documented policy is intentional. Construct the optimizer from parameters whose `requires_grad` is true.

## Future dataset contract

Each future adapter must return image, stable sample identifier, split, source width/height, and optional labels or annotations. It must pin and verify repository ID, revision, schema, license/access conditions and split mapping. Keep train/development/lockbox identifiers in separate manifests. Labels and annotations do not enter JEPA loss; only a registered frozen-representation evaluation may consume them. No real dataset adapter or download is included now.

## Executable verification

`from fmm_tests.test_models import run_model_tests` returns JSON-safe PASS/FAIL gate records. Gates check rectangular coordinates and shapes, input-order equivariance, bounded multiblock masking, exact hidden-pixel invariance, padded NaN isolation in both mixers, predictor/teacher alignment, a hand-calculated loss reduction, finite student/predictor gradients and parameter updates, absent target gradients, bitwise initialization copying, numeric EMA including buffers, deterministic CPU repeats, and observed SDPA dispatch. The notebook orchestrator saves the returned records in its certificate ledger. Tests are bounded numerical/software evidence, not evidence of useful representations.

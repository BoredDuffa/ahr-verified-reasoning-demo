# Tiny ImageNet provenance and reproducible input contract

Notebook 01 uses [thethinkmachine/tiny-imagenet](https://huggingface.co/datasets/thethinkmachine/tiny-imagenet). Verified on 2026-09-23 UTC:

- Source revision: `9f4331d7160a7f6e90c12e3533d658e9c8c7ae3c`.
- Actual converted Parquet revision: `5318f3a410192b1480adce2ea61e8e2c139adb53`.
- Dataset fingerprint: `e1b8185f419b8154124105d18b76b90e83eb70dbcccd06b91d01b4237e6dfec3`.
- Actual splits: `train` = 100,000 labeled images (500/class); `valid` = 10,000 labeled images (50/class), with 200 matching WordNet class IDs across both split schemas. No unlabeled test split is loaded.
- Downloaded compressed source: 161,067,911 bytes. This includes the complete source for metadata/integrity auditing. Smoke selects 2,000 training and 1,000 development examples; downloaded bytes are separate from selected example counts and training exposures.

Converted Parquet lacks embedded Hugging Face ClassLabel metadata. Its Arrow schema is `image: struct<bytes: binary, path: string>` and `label: int64`. We bind the semantic mapping from immutable `dataset_infos.json`, cross-check both live Viewer split schemas, and verify each downloaded shard's SHA-256 equals the immutable source file's LFS SHA-256. The pinned metadata's legacy key is `Maysee--tiny-imagenet`, and the card's copied example names another mirror. Loading always uses the requested repository; no mirror migration or remote code occurs.

The API declares `private=false`, `gated=false`. The card declares `license=[]` and contains ImageNet access terms describing non-commercial research/education. Public access is not interpreted as unrestricted reuse permission. Exact declarations are retained in the dataset manifest.

## Full-source integrity and duplicate audit

All 110,000 images decoded successfully at native 64×64. Original modes were 98,179 training RGB + 1,821 training grayscale, and 9,832 validation RGB + 168 validation grayscale. Every model input is converted to RGB. The adapter retains compressed bytes in Arrow and decodes selected images on demand; a separate first-use audit decodes the complete source once and caches its evidence.

SHA-256 over decoded RGB dimensions and pixels found 27 exact duplicate pairs. All 27 pairs have conflicting source class labels. The predeclared rule reserves the held-out partition first, then development, then training; retains the lowest original row within the winning partition; preserves that member's source label; and excludes other members before selecting experimental subsets. No relabeling is attempted. Source-label ambiguity and undetected near/semantic duplicates remain limitations.

With split seed 314159, exclusions were: 12 training duplicates retained in training, 8 training duplicates retained in development, 5 training duplicates retained in held-out, and 2 development duplicates retained in held-out. Thus 15 cross-partition and 12 within-training duplicate copies were removed. The held-out pool remains intact. Decoding held-out images solely for integrity/duplicate exclusion is separate from predictive evaluation, model selection and preprocessing-statistic fitting.

## Exact separation and subsets

For each class, a SHA-256-derived local seed permutes all 500 source-training rows. The first 50 form development and the remaining 450 form the training/probe pool. Duplicate exclusions leave 89,975 training-pool and 9,998 development images; held-out remains 10,000. Labels are used for stratification and frozen probes, never passed to JEPA's loss. Probe fitting intentionally reuses the selected pretraining pool.

Subset seed 1702 is independent of data order, augmentation, masks, model initialization and probes. Selection uses a local per-class permutation of the already-separated pools:

| Profile | Training | Development | Reserved held-out |
|---|---:|---:|---:|
| Smoke | 2,000 (10/class) | 1,000 (5/class) | 10,000 |
| Pilot | 20,000 (100/class) | 2,000 (10/class) | 10,000 |
| Explicit full pool | 89,975 | 9,998 | 10,000 |

Exact pool/subset IDs, exclusions, class counts, seeds and hashes are under `data/tiny_imagenet/5318f3a410192b1480adce2ea61e8e2c139adb53/split_*.json`. `data/tiny_imagenet/profile_membership_verification.json` verifies all three data configurations without launching their training. IDs are `repository@converted_revision:original_split:original_row` and do not depend on selected order.

## Input API and randomness

Reusable definitions are tagged `ahr-export` cells in `fmm_jepa/tiny_imagenet.ipynb`. After standard bootstrap, ordinary imports provide:

- `prepare_dataset(project_root, config)`: dictionary containing `dataset`, `dataset_manifest`, `split_manifest`, `audit`, `train_ids`, `dev_ids`, and `heldout_ids`.
- `dataset.get(sample_id)`: PIL RGB image, source label, ID, original dimensions/mode/split.
- `make_batch(dataset, sample_ids, occurrences, config, training=True)`: CPU `images` (B×3×64×64), `labels`, `sample_ids`, `occurrences`, `augmentation_ids`, `mask_ids`, detailed metadata and a packed `masks` dictionary. Padding indices are −1 with separate Boolean valid masks.
- `EvaluationDataset(dataset, sample_ids, config)`: deterministic `image`, `label`, `sample_id` items. `evaluation_batch` supplies deterministic batch tensors.
- `preprocessing_descriptor(config)`: canonical preprocessing identity for feature caches.

A stream seed is derived from its base seed, stream name, sample ID and sample occurrence. Augmentation and mask streams are distinct. Global torch/NumPy RNG consumption and rebatching cannot change an example's image or mask. Evaluation draws no training randomness. Data ordering belongs to the runner's separate stream.

Training uses one moderate random resized crop (area 0.6–1.0, aspect 0.75–4/3), PIL bilinear resize to 64, and horizontal flip probability 0.5. Fixed channel mean/std are both (0.5,0.5,0.5); no split is used to estimate normalization. Teacher and student share the augmented image and coordinates. Evaluation uses full native RGB pixels with the same normalization.

The verified core sampler proposes four target blocks with area 0.15–0.20/aspect 0.75–1.5 and a context block with area 0.85–1.0. Target unions are removed from context. Complete masks retry at most 16 times until at least four visible tokens remain; core rectangle retries are capped at 32. Realized integer shapes/areas and deviations from continuous requested ranges are recorded per example. Targets may overlap each other; context cannot overlap any target. The final fallback repeats a 3×3 target four times and uses the 55-token complement as context. Its use is logged, and a forced fixture tests it.

## Reproduction and recovery

From the project root, run:

```text
/home/benjamin/workspace/ahr_go/.venv/bin/jupyter nbconvert --to notebook --execute --inplace tests/test_tiny_imagenet_pipeline.ipynb --ExecutePreprocessor.timeout=600
```

The notebook writes `data/tiny_imagenet/pipeline_certificate.json`. It tests current source/schema, exact split reproducibility/disjointness, decoded duplicate exclusion, all-class balancing, stateless pairing, batching independence, deterministic evaluation, context/target isolation and bounded fallback.

Downloads and JSON writes use temporary files followed by atomic replacement. Source files are SHA-256-verified at every preparation. Invalid manifests, audits or caches raise an error rather than silently restarting or substituting data. Resume an interrupted initial download/audit by calling the same preparation function: completed files are reused and partial files rewritten. Audits are reused only when their version, fingerprint and payload hash match. No model checkpoints are owned by this module.

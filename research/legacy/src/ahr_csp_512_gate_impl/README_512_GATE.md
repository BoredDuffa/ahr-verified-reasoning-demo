# AHR-CSP 512 Gate Implementation

This directory contains the executable implementation of the 512-gate roadmap.
The runner is designed to live on the remote AHR-CSP machine under:

`/home/benjamin/.1AHR_maths&words/ahr_csp_v0/experiments/ahr512_gate_runner.py`

Use the conda Python on the remote host:

```bash
/opt/conda/bin/python /home/benjamin/.1AHR_maths\&words/ahr_csp_v0/experiments/ahr512_gate_runner.py status
```

## What It Implements

- Verifies the existing Gate C unique-64 report against the 512-gate acceptance criteria.
- Splits the original failed 512/128/128 dataset into family-specific JSONL datasets.
- Loads the notebook definitions with all auto-run cells disabled.
- Installs the repaired loss that was present in the A100 reports but not persisted in notebook source:
  - expected violation loss
  - factor negative log likelihood
  - stronger fixed/query/constraint weighting
- Trains repaired family gates.
- Builds uniqueness-controlled mixed SAT datasets with balanced families.
- Runs staged curriculum gates.
- Runs Phase 6 evaluation/export only after a checkpoint passes the gate.

## Commands

Status and dataset summary:

```bash
/opt/conda/bin/python /home/benjamin/.1AHR_maths\&words/ahr_csp_v0/experiments/ahr512_gate_runner.py status
```

Create family split datasets:

```bash
/opt/conda/bin/python /home/benjamin/.1AHR_maths\&words/ahr_csp_v0/experiments/ahr512_gate_runner.py split-family
```

Train one family first, for a short smoke run:

```bash
/opt/conda/bin/python /home/benjamin/.1AHR_maths\&words/ahr_csp_v0/experiments/ahr512_gate_runner.py train-family --families subset_sum --epochs 2 --batch-size 16
```

Full family gates:

```bash
/opt/conda/bin/python /home/benjamin/.1AHR_maths\&words/ahr_csp_v0/experiments/ahr512_gate_runner.py train-family --epochs 120 --batch-size 64
```

Build the uniqueness-controlled 512/128/128 dataset:

```bash
/opt/conda/bin/python /home/benjamin/.1AHR_maths\&words/ahr_csp_v0/experiments/ahr512_gate_runner.py build-unique \
  --name phase5p7_unique_controlled_512_128_128 \
  --train 512 --val 128 --test 128 \
  --require-unique feasible \
  --domain-cap 46656
```

Train the repaired gate on that dataset:

```bash
/opt/conda/bin/python /home/benjamin/.1AHR_maths\&words/ahr_csp_v0/experiments/ahr512_gate_runner.py train \
  --dataset /home/benjamin/.1AHR_maths\&words/ahr_csp_v0/data/phase5p7_unique_controlled_512_128_128.jsonl \
  --name phase5p7_unique_controlled_512_repaired \
  --epochs 120 --batch-size 64
```

Run staged curriculum:

```bash
/opt/conda/bin/python /home/benjamin/.1AHR_maths\&words/ahr_csp_v0/experiments/ahr512_gate_runner.py run-curriculum \
  --stages unique64 unique128 unique256 unique512 \
  --epochs 120 --batch-size 64
```

Run Phase 6 only after a gate-passing checkpoint:

```bash
/opt/conda/bin/python /home/benjamin/.1AHR_maths\&words/ahr_csp_v0/experiments/ahr512_gate_runner.py phase6-after-pass \
  --dataset /path/to/gate_dataset.jsonl \
  --checkpoint-name phase5p7_unique_controlled_512_repaired \
  --batch-size 64
```

## Acceptance

A gate passes only when:

- train CPU-validity `>= 0.95`
- validation CPU-validity `>= 0.90`
- validation mean hard energy `<= 0.10`

All reports are written under:

`/home/benjamin/.1AHR_maths&words/ahr_csp_v0/reports`

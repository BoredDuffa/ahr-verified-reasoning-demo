# Notebook04 recovery notes

Run: `semantic-followup-001`. These notes describe operational interruptions and their recovery; they do not alter results or the registered comparison.

## Plan metadata serialization

The registered in-memory plan contained integer microbatch keys in `engineering_selection.equivalence_certificate.models.{vit,semantic}.weighted_losses`. JSON deserialization converts those keys to strings. Sorted canonical hashing then uses a different order. The disk-plan hash is `69650ba8de00232a8ac48bd5da67b47033746f04b66c54f5c453df17f50b18e0`; restoring only these key types reproduces the original registered hash `7895ab32cb465f383dda44802b821b8fab2cc927109c1670563233e28869c046` exactly.

`notebooks/04_Followup_Exact_Resume.ipynb` supplies the narrow recovery adapter. It proves JSON-observable value equality, unchanged arm protocols/source hashes/deadline, and acceptance by the frozen runner. It temporarily patches only the exact plan-path read and restores the reader afterward. Fresh-kernel CPU checks also verified retained full-state preflight evidence and the tagged detached-worker import. Evidence: `artifacts/notebook_04/semantic-followup-001/exact_resume_identity_certificate.json`.

## Evaluation output directory

All 12 arms finished 300 successful updates each, with no failed attempts and identical within-seed data/augmentation/mask traces. Evaluation then stopped at its first prediction CSV export because the frozen CSV helper does not create parent directories. The missing directory was `artifacts/notebook_04/semantic-followup-001/predictions/`. The original traceback is retained in `worker.log`.

The directory was created, and the exact-resume worker reloaded the existing completed training and resumable evaluation state under the same run ID, frozen definitions and original deadline. No new training budget was added. The recovery notebook now creates this output directory in its untagged launch cell before reconnecting or launching. This consumer-only preparation leaves the frozen training, evaluation and recovery exports unchanged.

## Stale status during controller startup

The first recovery controller read the previous worker failure before the newly launched worker published its own status. The detached recovery worker continued successfully; the controller exception did not terminate it.

The recovery notebook now waits, only after a new launch, for a status timestamp at or after that launch began before calling the frozen monitor. This wait is limited to 55 seconds and the original absolute deadline. A fresh failure is still reported, and worker exit or failure to publish a fresh status is an error. Existing status files are not rewritten to manufacture success. The original independent watchdog remains active.

The modified consumer cells were compiled and inspected without executing a launch or interfering with the active worker. Frozen recovery export and bootstrap hashes were checked before and after the change; both are identical. A completed-run fresh-kernel replay is the remaining end-to-end consumer validation.

## Statistical dependency

The across-seed Student t intervals require SciPy. SciPy 1.18.1 was installed before evaluation, without changing NumPy, PyTorch or frozen notebook definitions; `pyproject.toml` records the dependency. This supplies the existing statistical calculation and does not change the registered training or evaluation policy.

The original wall-clock limit remains **2026-09-27 02:25:06 UTC**. Recovery never resets it. Official Food101 test data remain unused.

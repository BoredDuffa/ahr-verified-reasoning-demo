# Publication audit — 16 October 2026

## Destination and baseline

This update targets **[BoredDuffa/ahr-verified-reasoning-demo](https://github.com/BoredDuffa/ahr-verified-reasoning-demo)**, the public research output linked in the fellowship application. The previous `main` commit was `c1d1dc13a1bd2fd06695db6325040f8050f6e74d`, dated 9 June 2026. Its five-file static demo remains present and its exact-answer core is unchanged.

The original private `BoredDuffa/ahr` repository was also checked: its HEAD remains `722d820c1e098b0a7f85efc5735cbb5eecff3620`, with a non-truncated tree of 1,155 files. Its old notebooks and source are contextualised through the later consolidated research archive. This publication does not change the visibility of that repository or rewrite its history.

## Evidence reviewed

- Existing application record to identify the intended public repository.
- Relevant architecture, theory, Go, HRM/JEPA comparison, verification-integration and protocol chats, used as discovery leads rather than substitutes for experiment receipts.
- The accessible live Go/FMM research workspace, inspected and copied read-only on 16 October. No remote code, process, model, data or checkpoint was modified.
- The consolidated verified-reasoning archive through late June, including ranker and Lean milestone records.
- The existing July mathematical monograph, its source/provenance record and earlier arithmetic milestone report.
- The local 16-notebook AHR v2 XL draft and its static-delivery manifest.
- The independent local mini-Go implementation, protocol, saved training records and existing tests.

The live files resolve important changes beyond older chat summaries: the economic-credit pilot has completed guided training and screens; the base was selected for confirmation; the confirmation/promotion gate failed; and the 16 October follow-up stopped on insufficient fresh rows. The newer FMM two-seed development pilot completed, while its larger registered follow-up remains an unfinished snapshot.

## Published coverage

The [source manifest](../research/SOURCE_MANIFEST.json) records **1,118 copied source/report/support files**, including **246 notebooks**, with original and public SHA-256 hashes. It also identifies filtered generated records. The notebook collection contains **979 code cells**. Scientific reports retain original measurements; they are not regenerated experiment results.

New overview, result, economics, reproduction and audit documents, a source index, a checked result ledger, and a static research page make the snapshot navigable. Existing historical Markdown may retain original paths and obsolete next-step instructions; it is archive evidence, not an instruction to start those jobs now.

### Notebook transformations

Code and Markdown source cells are preserved. Saved outputs, execution counters, execution timing and widget state are cleared; missing schema cell IDs may be normalised by `nbformat`. This avoids publishing bulky embedded images and stale execution displays. Both original-file and published-file hashes are recorded, alongside a code-cell hash and counts of saved executed cells/errors before clearing. A recorded original error is not hidden as successful execution: the source index exposes its count.

Original source hashes inside old reports may refer to earlier versions. A current source snapshot and a historical result do not imply that the result was produced by the current source. Reproducing a frozen run requires its exact original pins and external artifacts.

### Omitted or external

- Large checkpoint weights, optimizer states, generated datasets and training caches; their existence or hash metadata does not make this a portable model release.
- Large per-game/per-position traces and repeated fixture, smoke, browser, launcher and temporary-session records. Go/FMM discovery copies used size limits of 4/5 MiB per file; legacy selected JSON used 2 MB. These limits do not define scientific success or failure.
- The 17,455,360-byte KataGo confirmation report is represented by a source-hashed summary/protocol extract. It records its original path, digest, byte length and transformation. Full traces remain in original research storage.
- Private company code and healthcare integration records, patient/customer information, correspondence, application materials and contact details.
- Third-party books/paper collections, vendored engine binaries, virtual environments, credentials, SSH keys and unrelated projects.

Consequently this is **a curated public source-and-evidence update, not a byte-for-byte backup of every private workspace**. Branches with unfinished or failed experiments retain those states. No new official ARC score, clinical validation, general proof-solving result or Go rank is inferred.

## Validation performed for this publication

| Check | Result |
|---|---|
| Exact browser-demo route regression | Passed existing supported-family/fraction and abstention checks |
| Historical exact mathematical-search unit suite | 18 tests passed |
| Mini-Go rules/model/search suite | 21 tests and 629 subtests passed; four GNU Go integration tests skipped because the local vendored binary is absent |
| Notebook schema and transformed Python syntax | 246 notebooks / 979 code cells passed; no syntax issues |
| Archived Python source syntax | 93 source files passed |
| Copied JSON parsing | 728 files passed |
| Source/published provenance and notebook code hashes | Checked against the publication manifest |
| Headline results | 22 JSON values checked directly against the published primary report fields |
| Documentation and source-index links | Relative targets checked; archival internal paths are not claimed portable |
| Credential-pattern scan | No matching token, private-key, credential URL or hard-coded long credential assignment found in the staged text snapshot |
| Research page | Opened and visually inspected in a browser; existing demo checked separately |

[STATIC_CHECKS.json](../research/STATIC_CHECKS.json) and [RESULTS_LEDGER.json](../research/RESULTS_LEDGER.json) retain machine-readable validation details. These checks establish packaging and bounded software behaviour. **No full scientific notebook, training run, tournament or Lean proof corpus was rerun for publication.**

The repository push and Pages deployment are verified separately after this audit is written. No application resubmission or external correspondence is part of the update.

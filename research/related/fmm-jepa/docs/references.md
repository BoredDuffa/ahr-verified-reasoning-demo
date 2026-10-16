# Primary-source provenance and implementation decisions

Sources below were accessed on 2026-09-23. Paper versions and the official I-JEPA commit are pinned. `configs/reference_sources.json` records machine-readable provenance. These references motivate and check the foundation; none is evidence of FMM-JEPA's own learning or performance.

## I-JEPA: paper and official code are not identical specifications

**Assran et al., “Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture,” arXiv:2301.08243v3, 13 April 2023.** Sections 3 and Appendix A describe latent block prediction, full-image teacher targets, and the context/target arrangement. Section 3 writes squared L2 distance, whereas the official implementation uses SmoothL1. The paper allows overlapping target blocks and removes them from context; its representative mask ranges are target area 0.15–0.20, target aspect 0.75–1.5, and context area 0.85–1.0. [Versioned paper](https://arxiv.org/html/2301.08243v3).

**Official implementation:** `facebookresearch/ijepa`, commit **`52c1ae95d05f743e000e8f10a1f3a79b10cff048`**, resolved by `git ls-remote` and inspected through commit-pinned source URLs. This is a provenance pin, not a claim to reproduce the original full training recipe.

| Source | Verified behavior | FMM-JEPA decision |
| --- | --- | --- |
| [`src/train.py`, target/loss/EMA](https://github.com/facebookresearch/ijepa/blob/52c1ae95d05f743e000e8f10a1f3a79b10cff048/src/train.py#L269-L306) | Encode the full image without gradients, feature-normalize, select target positions, then `smooth_l1_loss` with defaults; update teacher parameters after the optimizer. | Explicit beta 1 SmoothL1 and valid-element averaging. Keep full-image teacher and stop-gradient. |
| [`src/train.py`, initialization](https://github.com/facebookresearch/ijepa/blob/52c1ae95d05f743e000e8f10a1f3a79b10cff048/src/train.py#L147-L206) | Teacher starts as a deep copy and its parameters are frozen. The script does not explicitly put it in eval mode; EMA loops over parameters, not buffers. | Freeze teacher, define eval/buffer policy explicitly, and test the numerical update. |
| [`src/models/vision_transformer.py`, predictor](https://github.com/facebookresearch/ijepa/blob/52c1ae95d05f743e000e8f10a1f3a79b10cff048/src/models/vision_transformer.py#L262-L309) | Project context; add positions; append learned mask-plus-position tokens; run predictor blocks; return only target slots in mask order. | Conventional small Transformer predictor; never pass true target embeddings into it. |
| [`src/masks/multiblock.py`](https://github.com/facebookresearch/ijepa/blob/52c1ae95d05f743e000e8f10a1f3a79b10cff048/src/masks/multiblock.py#L69-L156) | Samples multiple targets, restricts context through complements, may relax restrictions after retries, and truncates batch masks to shared lengths. | Bounded retries and explicit padding/validity. Never relax target exclusion. Log fallback use. |

Thus “I-JEPA-style” here means an implementation-following beta 1 SmoothL1 latent-prediction objective with documented project-specific masking and shape handling. It does not mean exact reproduction of every paper hyperparameter or implementation behavior.

## Classical interpolation and kernel-independent FMM

**William Fong and Eric Darve, “The black-box fast multipole method,” Journal of Computational Physics 228(23), 8712–8725 (2009), DOI `10.1016/j.jcp.2009.08.031`.** The journal PDF was inspected, particularly the Chebyshev construction and multilevel algorithm in Sections 2–3. It provides the relevant interpolation-based chain: source anterpolation, upward transfer, well-separated kernel evaluations at interpolation nodes, downward transfer, and interpolation to receivers. Its later SVD compression is not implemented in this foundation. [Author-hosted journal PDF](https://mc.stanford.edu/cgi-bin/images/f/fa/Darve_bbfmm_2009.pdf).

**Lexing Ying, George Biros, and Denis Zorin, “A kernel-independent adaptive fast multipole algorithm in two and three dimensions,” Journal of Computational Physics 196(2), 591–626 (2004), DOI `10.1016/j.jcp.2003.11.021`.** This is a distinct kernel-independent construction using equivalent densities on enclosing surfaces. It supplies historical/methodological context; FMM-JEPA's Chebyshev transfers are not an implementation of those boundary-value solves. The publisher abstract and indexed author-PDF excerpt were accessible; direct PDF opening failed, so a full derivation audit of this distinct method remains pending. [Author-hosted journal PDF](https://web.stanford.edu/~lexing/fmm.pdf), [publisher record](https://www.sciencedirect.com/science/article/pii/S0021999103006090).

Classical interpolation assumptions concern sufficiently regular kernels on well-separated cells and numerical truncation/roundoff. Learned channel maps, gates, nonlinearities, and semantic inputs do not automatically satisfy those assumptions. Notebook 00 therefore measures a separate regularized scalar-kernel diagnostic and never transfers its error certificate to representation quality.

## Existing learned FMM-inspired attention

**Tan M. Nguyen, Vai Suliafu, Stanley J. Osher, Long Chen, and Bao Wang, “FMMformer: Efficient and Flexible Transformer via Decomposed Near-field and Far-field Attention,” NeurIPS 2021; arXiv:2108.02347v1.** This work separates banded near-field and low-rank far-field attention. It is relevant prior art, but that decomposition alone is not the explicit geometric P2M → M2M → M2L → L2L → L2P chain required here. No published performance number is adopted as an FMM-JEPA result. [Versioned preprint](https://arxiv.org/abs/2108.02347v1), [proceedings PDF](https://papers.nips.cc/paper_files/paper/2021/file/f621585df244e9596dc70a39b579efb1-Paper.pdf).

**Yanming Kang, Giang Tran, and Hans De Sterck, “Fast Multipole Attention: A Scalable Multilevel Attention Mechanism for Text and Images,” arXiv:2310.11960v4, 18 September 2025.** Version 1 used the title “A Divide-and-Conquer Attention Mechanism for Long Sequences”; this reference deliberately pins the later version. Section 3 distinguishes a multilevel O(n log n) construction from a query-downsampled linear variant and includes a 2-D quadtree formulation with learned summaries. These claims concern that paper's operators. They do not establish this project's complexity or accuracy. [Versioned full text](https://arxiv.org/html/2310.11960v4).

The existence of these methods prevents treating “FMM-inspired neural token mixing” as a novelty claim. A dedicated related-work and novelty review remains future work. This source review verifies conceptual and implementation details; it does not reproduce those papers' experiments or review every public implementation.

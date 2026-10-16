# Core geometry and operators

The reusable source lives in `fmm_jepa/geometry.ipynb` and `fmm_jepa/operators.ipynb`. These definition notebooks export only explicitly tagged `ahr-export` cells. The project bootstrap makes them available through ordinary imports after installing the notebook finder. Importing never runs an experiment. This layout follows the user's notebook-only working agreement.

## Contracts and coordinate system

`FMMMixer(dim, order=2, leaf_size=2, receiver_gate=True)` accepts `forward(x, indices, valid, grid)`:

- `x`: floating tensor `[B,V,D]`, where `V` includes batch padding.
- `indices`: int64 `[B,V]`, original row-major patch positions.
- `valid`: bool `[B,V]`; valid indices must be distinct per image and in range.
- `grid`: `(rows, columns)`, positive integers, independent of visible count.
- Result: `[B,V,D]` in exactly the supplied index order; invalid outputs are exactly zero.

A patch at row `y`, column `x` has coordinate `((x+.5)/columns, (y+.5)/rows)`. Invalid entries may repeat a harmless index; even NaN invalid feature values are removed using `where` before aggregation. All-empty samples produce zero mixer output; the model/masking contract may impose the stronger condition that every image has nonempty context.

`return_branches(...)` returns `(near, far)` before the final shared output projection. `last_metrics` records detached near/far norms. `last_stats` additionally records geometry counts, visibility, and work estimates. Reading these metrics incurs host synchronization on CUDA; current timing includes this instrumentation overhead. `near_enabled` and `far_enabled` support controlled diagnostics, defaulting to `True`.

`build_geometry(grid, order=2, leaf_size=2, device='cpu', dtype=torch.float32)` returns immutable field bindings with cached geometric tensors. The bounded LRU key includes grid, order, leaf size, device and dtype. It contains no feature moments or visibility, so geometry is safe to share between steps and masks. Treat returned tensors as read-only. `clear_geometry_cache()` releases cache references when changing many configurations.

## Hierarchy and unique interaction ownership

`leaf_size=b` is patches per cell side, so each occupied leaf contains at most `b²` original positions. Let `S` be the next power of two greater than or equal to `max(ceil(rows/b), ceil(columns/b))`. The leaf topology is `S×S`; depth is `L=log2(S)`. Every level allocates all `4^level` cells, including empty rectangular/padded regions. Root side lengths in physical coordinates are `b*S/columns` and `b*S/rows`. Padding may extend the root beyond `[0,1]`, while original patch coordinates remain strictly inside `[0,1]²`.

Cell ID at level `ell` is `row*2^ell+column`. A child maps to `(row//2, column//2)`, with quadrant `q=2*(row mod 2)+(column mod 2)`. Leaf source cells at Chebyshev cell distance at most one are direct near neighbors. Learned near messages include the diagonal self-pair. Classical direct evaluation excludes it.

At every level `ell>=2`, a receiver's interaction list consists of children of its parent's neighboring cells that are not adjacent to the receiver cell itself. Its directed M2L list has at most 27 source cells. A non-near ordered point pair belongs to the first level where its ancestors become nonadjacent; their parents are still adjacent. All such pairs consequently have exactly one owner. No pair can reappear in a deeper interaction list because its parents are already separated there.

`ownership_table(geometry, include_self=True)` enumerates a dense count table only for tests with at most 256 points. `pair_owner(receiver_idx, source_idx, grid, leaf_size=2)` returns `0` for direct ownership or the unique M2L level without allocating a table. Main production operators never call the dense diagnostic.

## Chebyshev definitions and transfer direction

Order `p` means `p` first-kind Chebyshev nodes per axis; coefficients per cell are `r=p²`. The sorted one-dimensional nodes are `cos((2k+1)*pi/(2p))`. Let `l_a(t)` be their cardinal Lagrange polynomials. A two-dimensional basis is `l_(ay,ax)(u,v)=l_ay(v)*l_ax(u)`; node and coefficient order is row-major in the node grid. Polynomial products evaluate the basis without a removable `0/0` at node locations.

Moments and local coefficients at a level have shape `[B,C_ell,r,D]`:

\[
M_{c,a}=\sum_{j\in c}l_a(\xi_c(x_j))v_j,
\]

where `xi_c` maps a physical cell to `[-1,1]²`. These are interpolation-weighted feature charges, not pooled averages.

For child quadrant `q`, let `eta_q` map child-local coordinates to parent-local coordinates. Define `T[q,a,b]=l_a(eta_q(node_b))`, with shape `[4,r_parent,r_child]`. Then

\[
M_{parent,a}=\sum_{children\ q,b}T_{q,a,b}M_{child,b},\qquad
L_{child,b}=\sum_aT_{q,a,b}L_{parent,a}.
\]

The first identity reproduces parent polynomial moments from child moments. The second evaluates a parent interpolation polynomial at child nodes. The transfer matrices are transposes in their action on moments and local values. The node-basis polynomial identity and the corresponding adjoint identity are tested in float64 for orders 2, 4 and 6. Neither identity says that interpolation of an arbitrary kernel is exact.

After upward M2M, M2L forms local nodal values using a matrix with axes `[receiver_node, source_node]`. L2L accumulates parent local values before L2P evaluates the leaf basis:

\[
L_{c,a}\mathrel{+}=\sum_{d\in I(c),b}K_{cd,ab}M_{d,b},\qquad
m_i^{far}=\sum_a l_a(\xi_{c(i)}(x_i))L_{c(i),a}.
\]

All five stages, P2M → M2M → M2L → L2L → L2P, are exercised in the shared implementation. Even when a pair owns a coarse-level interaction, its feature charges travel upward from leaves and its local values travel downward to receivers.

## Learned departures from the classical kernel

The learned source charge is a bias-free linear projection `far_value(h)`. For each integer relative source-cell offset `(dx,dy)` in `[-3,3]²`, the model owns a trainable `r×r` translation matrix shared across levels. There are 49 stored matrices; offsets excluded by the well-separation rule are unused parameters. Initial matrices evaluate `1/sqrt(distance²+.1²)` between nodes of unit cells, translated by the offset. Training can change every used matrix entry. Offsets use cell-normalized topology, so physical aspect ratio and physical cell scale do not impose classical-kernel equivariance on the neural model.

The far result is divided by the image's valid source count. If enabled, a receiver gate `sigmoid(W_g h_i+b_g)` multiplies it channel-wise. This limits simple token-count growth while retaining receiver dependence. It does not preserve the classical field's magnitude or error guarantee.

Near edges use

\[
\phi(h_i,h_j,\Delta x)=\sigma((W_qh_i)^T(W_kh_j)/\sqrt D+w_{pos}^T\Delta x)W_vh_j,
\]

then divide each receiver's sum by its number of valid near neighbors. This is a content-dependent bounded-neighborhood sigmoid operator; it is not softmax attention. The near and far branches are summed and passed through a shared bias-free output projection. There is no global-attention fallback. Residual/MLP scaffolding lives in the encoder, outside this mixer.

Feature-valued moments and fixed geometric separation impose an information bottleneck. Spatial separation does not establish semantic compressibility, preserved remote object identities, representation quality, or equivalence to self-attention. The receiver gate and residual path are design choices to investigate. Classical numerical certificates must not be presented as guarantees for the learned translations.

## Classical diagnostic

`classical_field(charges, grid, order=4, leaf_size=2, epsilon=.05, valid=None)` accepts `[N]` or `[B,N]` scalar charges and evaluates the hierarchy using the exact regularized physical kernel at cell nodes. It uses the same geometry, point basis, transfers, interaction lists, P2M, M2M, M2L, L2L and L2P primitives as the neural mixer. Its near kernel is exact, its self-interaction is excluded, and it does not apply neural normalization or receiver gates.

`direct_field` constructs the dense reference only for at most 1024 positions. Absolute error is the L2 norm of the difference. Stabilized relative error divides it by `max(norm(exact),1e-12)`. Fixed signed-charge tests compare orders 2/4/6 on `8×8` and `7×9` grids; the ledger stores actual values and tolerances. Strict per-realization monotonicity is deliberately not a gate.

## Cost of the implementation

Let `N_grid=rows*columns`, `N_vis` be valid input tokens, `D` the feature width, `b` the leaf side, `r=p²`, `C=sum_(ell=0)^L 4^ell`, and `E=sum_ell |I_ell|<=27*C`. Scatter/gather input handling costs `O(B*N_vis*D)`, but projection and hierarchy work traverse the allocated full lattice/cells, including empty cells:

| Operation | Work, suppressing batch factor |
|---|---|
| Full-lattice query/key/value/far projections | `O(N_grid*D²)` |
| Point basis P2M and L2P | `O(N_grid*r*D)` |
| Direct near messages | `O(N_grid*b²*D)`; at most `9*b²` sources per point |
| Upward and downward transfers | `O(C*r²*D)` |
| M2L translations | `O(E*r²*D)` |
| Receiver gate and output projection | `O(N_grid*D² + N_vis*D²)` |

Current vectorization materializes edge features rather than a fused sparse kernel, so near memory is `O(B*N_grid*b²*D)` and M2L temporary memory is `O(E*r² + B*E*r*D)` alongside cell coefficients `O(B*C*r*D)`. Autograd retains intermediate tensors. Geometry stores per-cell node locations and bounded edge lists. Highly elongated grids can allocate substantially more square-topology cells than occupied leaves. Fixing `b`, `p`, and moderate grid aspect ratio gives favorable dependence on `N_grid`; that statement is conditional, not a measured speedup. Explicit `r²=p^4` dependence can be expensive.

Python orchestration, intermediate tensors, index-add operations, cache construction on first use, and detached metric synchronization can dominate small runs. The test suite combines bounded edge/cell operation counts with a TorchDispatch trace of forward/backward tensor shapes on `8×8`, `12×12`, and `16×16` grids. Absence of an `N×N` shape alone would be insufficient; the source implementation's enumerated work bounds also matter.

Dense attention has an `O(N_vis²*D)` pairwise component, plus projections and MLPs. A fair comparison states visible and full-grid counts and includes actual parameter counts. A conventional attention predictor remains in JEPA; favorable encoder scaling never establishes whole-system linear time.

## Executable certificates

`fmm_tests.test_geometry.run_geometry_tests()` returns a JSON-safe map from gate names to `status` and `details`. Tests report exceptions as FAIL rather than stopping or fabricating a result. The gates cover rectangular coordinates, hierarchy and exhaustive ordered-pair ownership, interpolation and transfer identities, classical approximation, near/distant controlled impulses including a depth-three path, NaN invalid features and empty cells, finite-difference transfer/kernel/neural-input checks, production structural bounds and tensor traces, and repeatable deterministic CPU evaluation. Notebook 00 writes this evidence into the run's certificate ledger.

Input feature/kernel gradient checks use float64; model smoke work may use float32. The fixed CPU checks do not certify CUDA atomic reduction determinism, reduced precision, arbitrary order, pathological aspect ratios, distributed execution, or real-data learning.

# Native AHR constraint-to-mini-Go pilot

## Locked question and scope

Can a small, freshly initialized Queen/Worker/Scout network learn local quantified
constraint-tree backup operations from synthetic problems, and transfer those
operations to unseen Go game trees without training on any Go positions?

This is a deliberately narrow mechanism test. It is **not** training a general
SAT/CSP solver on massive problems, reusing the earlier AHR v0.2 checkpoint, or
demonstrating the full formal AHR soundness/completeness contract. Generic tree
backup is much easier than learning Go strategy, variable ordering, or proofs.

## Training and controls

- Native learned Scout, Worker and slower Queen states, plus bounded memory.
- Quantified curriculum: bounded child values with existential/max and
  universal/min constraints. Analytic labels are used only offline for training.
- Existential-only control: same architecture and training budget, only max
  examples. Its exposure to min roles at evaluation is deliberately out of
  distribution; this tests whether quantifier coverage matters.
- Untrained control: the exact initialization used for each trained seed.
- Seeds 17, 29 and 43; train/validation/test generator seeds are separate.
- No Go positions, engine moves, Go outcomes or Go test labels enter training.
- Validation selects the lowest validation MSE checkpoint within each curriculum.
- Complete recursive Go inference uses learned backups internally; only root move
  selection uses the normal argmax/argmin action rule. No analytic internal
  minimax backup is substituted for network predictions.

## Go rules and search

3x3 primary board; 5x5 secondary engine challenge. Black moves first, komi 0.5,
no suicide, positional superko, two consecutive passes terminate. Area scoring
counts stones and empty regions touching exactly one color. Stones left on the
board count as present; there is no implicit dead-stone adjudication. Passing is
exempt from positional superko. Terminal games are scored exactly under these
declared rules. A 100-ply safety limit creates a separately reported cutoff,
never silently a normal completed game.

All search contestants receive the same complete depth-3 tree on 3x3 and depth-2
tree on 5x5. Frontier area score is a heuristic for the eventual game outcome.
The exhaustive reference establishes **finite-horizon area-score minimax**, not
perfect full-game Go. Terminal leaves are exact; horizon leaves are not proofs.
No pruning by neural score removes a legal move. The root action is always checked
by the independent rules layer before use.

## Evaluation

1. New synthetic quantified trees: MSE and tolerance accuracy.
2. Unseen 3x3 positions: optimal-action agreement and regret relative to the
   exhaustive fixed-horizon reference, excluding trivial tied-action positions
   in an additional informative-position summary. Report all positions as well.
3. Paired mini-Go games versus random legal play, greedy area score, exhaustive
   shallow minimax, and GNU Go level 1 if successfully available. The same opening
   is played once with each contestant color. Store every move and score.
4. A small 5x5 GNU Go challenge as an exploratory out-of-board-size check.

The tournament uses equal search depth for AHR and the internal exact baseline,
not equal wall time. Time and node costs are recorded separately. GNU Go has a
different algorithm and its level-1 configuration is not a compute-matched control.
Paired openings, games and training seeds are distinct evidence units. Bootstrap
intervals resample opening pairs and seeds; they describe this small suite only.

## Integrity and recovery

Rules unit tests and independent randomized move/scoring checks precede results.
Training saves model, optimizer, RNG, epoch, configuration and selection metrics
atomically at each epoch under checkpoints/<run_id>/. It retains latest two and
best. A short exact-resume smoke test is required before training. Game records
are saved after every game and are replayed to recompute results independently.
No paid compute is provisioned. This run is local CPU work.

## Interpretation gate

Report whether quantified training improves over untrained and existential-only
controls. Even success would support only transfer of the tested backup operation.
Compare to analytic minimax cost: a neural approximation can reduce accuracy and
be slower than an exact operation in this deliberately small domain. A negative
result is retained, and the locked held-out data are never used to retune a model.


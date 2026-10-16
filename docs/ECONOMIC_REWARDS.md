# Economic incentives and AHR

Author: Benjamin Falkenburg. Evidence snapshot: 16 October 2026.

There are **two distinct projects**: an implemented Go economic-credit pilot, and a proposed AI-agent oversight study. Neither establishes a general solution to reward hacking or moral agency.

## Implemented: opportunity cost and economic credit in Go

[Notebook 20](../research/go/experiments/ahr_v2/20_economic_rewards_and_teacher_guided_improvement.ipynb) compares ordinary offline teacher guidance with bounded opportunity-cost weighting and a separate economic-credit head. KataGo supplies offline labels; it does not choose AHR's moves at inference. Both arms use the same frozen buffer, starting weights, examples, seeds and update counts. The control has the same architecture but disables regret weighting and credit loss.

The actual terminal objective remains win `+1`, loss `−1`, draw `0`. Captures, stone counts and passing receive no flat bonuses or penalties.

Let `s_t` include the complete legal Go history, rules, komi and player to move. Let `c_t ∈ {−1,+1}` denote mover colour, and let `Φᴮ(s)` be a frozen Black-perspective forecast, with zero terminal potential. For the undiscounted episodic case:

```text
r'_t = c_t [rᴮ_t + η(Φᴮ(s_{t+1}) − Φᴮ(s_t))]
G'_t = r'_t − G'_{t+1}
```

For a completed game with alternating turns, the sum telescopes to the mover-perspective terminal outcome minus the fixed state baseline, `G'_t = z_t − η Φᵐᵒᵛᵉʳ(s_t)`. This accounting identity requires the declared signs and terminal condition. It does not prove convergence of neural training, quality of the forecast, full-game optimality, or invariant behaviour under approximate optimisation.

The auxiliary credit head predicts this shaped outcome; it does not replace the Worker's ordinary win/outcome target. A separate bounded multiplier increases **policy-imitation** weight for moves with higher teacher-estimated opportunity cost. That cost is the estimated best-action win probability minus the selected-action estimate, clipped to `[0,1]`; the multiplier is capped. Estimates are not certificates. Incomplete games remain UNKNOWN and receive no observed terminal-return target.

The [recorded pilot](../research/go/reports/ahr_v2/ahr-v2-katago-econ-20261011-v1/results.json) completed 1,000 guided updates per arm. Screen wins were economics 4/40, control 2/40 and base 5/40 on shared openings. The base was selected; its separate confirmation was 3/100 against the one-visit KataGo profile. The [promotion record](../research/go/reports/ahr_v2/ahr-v2-katago-econ-20261011-v1-eval-recovery-v1/promotion.json) retained the incumbent. These results support an implemented, auditable comparison, **not a confirmed economic-credit advantage**. Full context, timing overruns and uncertainty limits are in [RESULTS.md](RESULTS.md).

The method draws on established [potential-based reward shaping](https://people.eecs.berkeley.edu/~russell/papers/icml99-shaping.pdf), its [stochastic-game extension](https://arxiv.org/abs/1401.3907), and [dataset aggregation](https://proceedings.mlr.press/v15/ross11a.html). The use of economic terminology is not itself a novelty claim.

## Proposed: incentive-focused oversight of AI agents

The research question is whether explicit analysis of proxy incentives and manipulation opportunities improves the detection and mitigation of exploitation on unfamiliar tasks, beyond ordinary critique with the same resources. This remains a proposal: the Go experiment does not validate an LLM critic or a reward-hacking intervention.

In a finite single-decision reference model, an agent actually maximises

```text
U(a) = r(a) − c(a) − P p(a),   P ≥ 0,
```

where `r` is expected proxy reward, `c` expected cost, `p` the flag probability under an oversight mechanism, and `P` an externally enforced penalty. For an honest action `h` and specified exploit `a`, define:

```text
G(a) = [r(a) − c(a)] − [r(h) − c(h)]
δ(a) = p(a) − p(h)
U(a) − U(h) = G(a) − P δ(a).
```

Thus `h` is strictly preferred when `P δ(a) > G(a)`. If `G(a)>0` and `δ(a)≤0`, no nonnegative penalty makes `h` strictly preferred. False positives matter through `p(h)`. Valid uniform bounds `G(a)≤Gmax` and `δ(a)≥d>0` across a specified exploit set imply that `P>Gmax/d` suffices within this model. Establishing those bounds, the correct beliefs/objective, and reliable enforcement is substantive work. This elementary calculation is not a new alignment theorem and does not automatically transfer to learned LLM objectives, long horizons or manipulable monitors.

The proposed evaluation would:

1. Compare ordinary and incentive-focused critics on identical fixed traces, with labels from an independent evaluator.
2. Use a matched two-by-two intervention design: ordinary/economic critique, each with/without actual enforced penalties. Include no-critic and oracle-monitor controls.
3. Hold information, model capacity, token/tool budgets, candidate counts and revision opportunities constant.
4. Measure exploit rate alongside legitimate task success, false accusations, refusals, runtime and total computation. Reduced proxy score or greater refusal alone is not success.
5. Tune on development families, then freeze penalties, success margins, sample size and clustered/paired analysis before final held-out families.
6. Stress-test adaptive exploits, correlated actor/critic errors, reward rescaling and attacks on enforcement.

A wrapper that selects safer candidates demonstrates the combined system's behaviour, not that the base model internalised different incentives. AHR's relevant design connection is the distinction between proposals, learned rankings and an independent acceptance condition. Hard admissibility constraints and learned/economic preferences must remain distinct.

Relevant starting literature includes [Concrete Problems in AI Safety](https://arxiv.org/abs/1606.06565), [Reward Tampering Problems and Solutions](https://arxiv.org/abs/1908.04734), [Self-Refine](https://arxiv.org/abs/2303.17651), [Constitutional AI](https://arxiv.org/abs/2212.08073) and [reward-tampering generalisation experiments](https://www.anthropic.com/research/reward-tampering). A full literature review and baseline reproduction are needed before claiming novelty.

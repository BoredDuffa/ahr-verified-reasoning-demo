"""Board-independent quantified-tree CSP training examples and exact oracle.

Each example is a feasible domain of child continuations with bounded utility,
and either an existential or universal choice over that domain. This is the
local backup subproblem of a quantified game tree. It is deliberately much
narrower than general quantified constraint satisfaction: the network is not
learning arbitrary constraint encodings, proof certificates, or Go rules.

Utilities mix continuous, discrete, clustered, decisive-branch and nested-tree
distributions. The input contains no Go positions, Go games, Go-engine outputs,
or Go-derived training labels. The exact oracle below supplies supervision and
must not be called by the neural policy when choosing game moves.
"""

from __future__ import annotations

import torch
from torch import Tensor


def exact_backup(values: Tensor, mask: Tensor, roles: Tensor) -> Tensor:
    """Independently specified masked max/min supervision, never a model layer."""
    if values.ndim != 2 or mask.shape != values.shape:
        raise ValueError("values and mask must have matching [B, K] shapes")
    if roles.shape != (values.shape[0],):
        raise ValueError("roles must have shape [B]")
    mask = mask.to(dtype=torch.bool, device=values.device)
    if not bool(mask.any(dim=1).all()):
        raise ValueError("every node must have at least one feasible child")
    maxima = values.masked_fill(~mask, -torch.inf).amax(dim=1)
    minima = values.masked_fill(~mask, torch.inf).amin(dim=1)
    return torch.where(roles > 0, maxima, minima)


def generate_data(
    n: int, seed: int, max_children: int = 10, mode: str = "quantified",
) -> dict[str, Tensor]:
    """Return deterministic CPU tensors: values, mask, roles, target.

    Corresponding existential and quantified cohorts have identical child
    values and feasibility masks for a given seed. Only the choice quantifier
    and its exact target differ. Padding location is randomized, preventing
    the first/last slot from acting as an implicit answer or legality cue.

    ``existential`` trains only +1 choices. ``quantified`` balances independent
    +1/-1 choices in expectation. Batch size and child count are unconstrained
    at inference, subject to the caller's own compute limits.
    """
    if n < 1 or max_children < 1:
        raise ValueError("n and max_children must be positive")
    if mode not in {"quantified", "existential"}:
        raise ValueError("mode must be 'quantified' or 'existential'")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    shape = (n, max_children)
    values = 2 * torch.rand(shape, generator=generator) - 1
    family = torch.randint(0, 5, (n,), generator=generator)

    discrete = torch.randint(-4, 5, shape, generator=generator).float() / 4
    center = 1.6 * torch.rand((n, 1), generator=generator) - 0.8
    spread = 0.02 + 0.35 * torch.rand((n, 1), generator=generator)
    clustered = (center + spread * torch.randn(shape, generator=generator)).clamp(-1, 1)
    decisive = 0.5 * (2 * torch.rand(shape, generator=generator) - 1)
    decisive_slot = torch.randint(max_children, (n,), generator=generator)
    decisive_sign = 2 * torch.randint(2, (n,), generator=generator).float() - 1
    decisive[torch.arange(n), decisive_slot] = decisive_sign

    # Exact child-subtree utilities broaden distributions without introducing
    # any game-specific encoding. Each child is a depth-two alternating tree.
    subtree_leaves = 2 * torch.rand((n, max_children, 3, 3), generator=generator) - 1
    nested_max_min = subtree_leaves.amin(dim=-1).amax(dim=-1)
    nested_min_max = subtree_leaves.amax(dim=-1).amin(dim=-1)
    nested_polarity = torch.randint(2, shape, generator=generator).bool()
    nested = torch.where(nested_polarity, nested_max_min, nested_min_max)
    values = torch.where((family == 1)[:, None], discrete, values)
    values = torch.where((family == 2)[:, None], clustered, values)
    values = torch.where((family == 3)[:, None], decisive, values)
    values = torch.where((family == 4)[:, None], nested, values)

    counts = torch.randint(1, max_children + 1, (n,), generator=generator)
    slot_order = torch.rand(shape, generator=generator).argsort(dim=1).argsort(dim=1)
    mask = slot_order < counts[:, None]
    # Preserve a rare decisive branch in its designated family.
    mask[torch.arange(n)[family == 3], decisive_slot[family == 3]] = True
    values = values.masked_fill(~mask, 0).to(dtype=torch.float32)
    random_roles = 2 * torch.randint(2, (n,), generator=generator).float() - 1
    roles = random_roles if mode == "quantified" else torch.ones(n)
    target = exact_backup(values, mask, roles)
    return {"values": values, "mask": mask, "roles": roles, "target": target}

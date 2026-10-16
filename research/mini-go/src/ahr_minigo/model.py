"""A small, explicit Queen/Worker/Scout neural constraint-tree backup.

This is a bounded research probe, not a general CSP solver or a claim of a
faithful implementation of every mechanism in the AHR theory. The network
learns one local quantified-tree operation. Applying it recursively is the
caller's responsibility. It never calls a max/min value oracle.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class AHRBackupNet(nn.Module):
    """Learn an adversarial backup from an unordered, masked set of children.

    ``values`` is [batch, children] with utilities in [-1, 1]. ``mask`` has the
    same shape and marks legal/feasible children. ``roles`` is [batch], +1 for
    an existential/maximizing choice and -1 for a universal/minimizing choice.
    An empty feasible set is an encoding error and is rejected explicitly.

    Scouts recurrently inspect children. A Worker recurrently pools their
    evidence. A slower Queen updates shared context and a soft refinement
    budget. Pheromone evidence is bounded in [0, 1] and local to each forward
    call. The budget is a learned refinement gate, not a wall-clock controller.

    Legal-child masking is exact; strategic values and attention are learned.
    The returned value is a learned attention mixture with a bounded learned
    correction. Neither selecting extremal values nor any exact minimax
    operation is built into the network.
    """

    def __init__(
        self, width: int = 32, queen_steps: int = 2, scout_steps: int = 2
    ) -> None:
        super().__init__()
        if width < 4 or queen_steps < 1 or scout_steps < 1:
            raise ValueError("width >= 4 and positive recurrence counts required")
        self.width = width
        self.queen_steps = queen_steps
        self.scout_steps = scout_steps
        self.child_encoder = nn.Sequential(
            nn.Linear(2, width), nn.Tanh(), nn.Linear(width, width), nn.Tanh()
        )
        self.queen_init = nn.Linear(1, width)
        self.scout_input = nn.Linear(3 * width + 1, width)
        self.scout_cell = nn.GRUCell(width, width)
        self.scout_score = nn.Sequential(
            nn.Linear(3 * width + 2, width), nn.Tanh(), nn.Linear(width, 1)
        )
        self.worker_cell = nn.GRUCell(2 * width, width)
        self.queen_cell = nn.GRUCell(2 * width + 2, width)
        self.budget_head = nn.Linear(width, 1)
        self.correction = nn.Sequential(
            nn.Linear(2 * width + 2, width), nn.Tanh(), nn.Linear(width, 1)
        )
        self.log_attention_scale = nn.Parameter(torch.tensor(0.7))
        self.pheromone_strength = nn.Parameter(torch.tensor(0.1))
        self.pheromone_retention_logit = nn.Parameter(torch.tensor(0.0))
        # Begin near a neutral attention mixture; the correction must be learned.
        nn.init.zeros_(self.correction[-1].weight)
        nn.init.zeros_(self.correction[-1].bias)

    def forward(
        self, values: Tensor, mask: Tensor, roles: Tensor,
        return_details: bool = False,
    ) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
        if values.ndim != 2 or mask.shape != values.shape:
            raise ValueError("values and mask must have matching [B, K] shapes")
        if roles.shape != (values.shape[0],):
            raise ValueError("roles must have shape [B]")
        mask = mask.to(device=values.device, dtype=torch.bool)
        if not bool(mask.any(dim=1).all()):
            raise ValueError("every node must have at least one feasible child")
        roles = roles.to(device=values.device, dtype=values.dtype)
        # Mask before encoding: even a NaN in padding must never affect a result.
        clean_values = torch.where(mask, values, torch.zeros_like(values))
        batch, children = values.shape
        width = self.width
        role_children = roles[:, None].expand(-1, children)
        encoded = self.child_encoder(
            torch.stack((clean_values, role_children), dim=-1)
        )
        child_mask = mask[..., None].to(values.dtype)
        encoded = encoded * child_mask
        scout = encoded
        worker = values.new_zeros(batch, width)
        queen = torch.tanh(self.queen_init(roles[:, None]))
        pheromone = values.new_zeros(batch, children)
        retention = torch.sigmoid(self.pheromone_retention_logit)
        scale = torch.exp(self.log_attention_scale.clamp(-2.0, 4.0))
        weights = mask.to(values.dtype) / mask.sum(dim=1, keepdim=True)
        budget = torch.sigmoid(self.budget_head(queen))
        # Count the feasible domain independently of how much padding a caller
        # uses; appending masked slots must preserve the computation.
        count_scaled = torch.log1p(mask.sum(dim=1).to(values.dtype)) / 3.0

        for _ in range(self.queen_steps):
            for _ in range(self.scout_steps):
                queen_children = queen[:, None, :].expand(-1, children, -1)
                worker_children = worker[:, None, :].expand(-1, children, -1)
                scout_input = torch.tanh(self.scout_input(torch.cat(
                    (encoded, queen_children, worker_children,
                     pheromone[..., None]), dim=-1,
                )))
                scout = self.scout_cell(
                    scout_input.reshape(-1, width), scout.reshape(-1, width)
                ).reshape(batch, children, width) * child_mask
                energy = self.scout_score(torch.cat(
                    (scout, queen_children, worker_children,
                     clean_values[..., None], role_children[..., None]), dim=-1,
                )).squeeze(-1)
                logits = scale * energy + self.pheromone_strength * pheromone
                weights = torch.softmax(logits.masked_fill(~mask, -torch.inf), dim=1)
                pheromone = (
                    retention * pheromone + (1 - retention) * torch.sigmoid(energy)
                ) * mask.to(values.dtype)

            pooled = (weights[..., None] * scout).sum(dim=1)
            proposed_worker = self.worker_cell(torch.cat((pooled, queen), dim=1), worker)
            worker = worker + budget * (proposed_worker - worker)
            entropy = -(weights * weights.clamp_min(1e-12).log()).sum(dim=1)
            queen_input = torch.cat(
                (worker, pooled, entropy[:, None], count_scaled[:, None]), dim=1
            )
            proposed_queen = self.queen_cell(queen_input, queen)
            queen = queen + budget * (proposed_queen - queen)
            budget = torch.sigmoid(self.budget_head(queen))

        mixture = (weights * clean_values).sum(dim=1)
        correction = 0.25 * torch.tanh(self.correction(torch.cat(
            (worker, queen, mixture[:, None], roles[:, None]), dim=1
        )).squeeze(-1))
        prediction = (mixture + correction).clamp(-1.0, 1.0)
        if not return_details:
            return prediction
        return prediction, {
            "attention": weights,
            "pheromone": pheromone,
            "queen_state": queen,
            "worker_state": worker,
            "scout_state": scout,
            "refinement_budget": budget.squeeze(-1),
            "attention_entropy": entropy,
            "attention_mixture": mixture,
            "learned_correction": correction,
        }

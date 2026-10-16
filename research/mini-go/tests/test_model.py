"""Meaningful checks of masks, symmetry, gradients, and supervision boundaries."""

import pytest
import torch

from ahr_minigo.model import AHRBackupNet
from ahr_minigo.synthetic import exact_backup, generate_data


def test_oracle_uses_quantifier_and_ignores_padding():
    values = torch.tensor([[0.3, -0.8, 99.0], [0.3, -0.8, -99.0]])
    mask = torch.tensor([[True, True, False], [True, True, False]])
    target = exact_backup(values, mask, torch.tensor([1.0, -1.0]))
    torch.testing.assert_close(target, torch.tensor([0.3, -0.8]))


def test_synthetic_cohorts_share_values_and_are_reproducible():
    quantified = generate_data(128, seed=731)
    again = generate_data(128, seed=731)
    existential = generate_data(128, seed=731, mode="existential")
    for key in quantified:
        torch.testing.assert_close(quantified[key], again[key])
    for key in ("values", "mask"):
        torch.testing.assert_close(quantified[key], existential[key])
    assert (existential["roles"] == 1).all()
    assert set(quantified["roles"].tolist()) == {-1.0, 1.0}
    assert quantified["mask"].any(dim=1).all()
    assert ((quantified["target"] >= -1) & (quantified["target"] <= 1)).all()


def test_network_ignores_padding_values_and_is_permutation_equivariant():
    torch.manual_seed(4)
    model = AHRBackupNet().eval()
    data = generate_data(12, seed=91)
    expected, details = model(data["values"], data["mask"], data["roles"], True)
    poisoned = data["values"].clone().masked_fill(~data["mask"], float("nan"))
    actual = model(poisoned, data["mask"], data["roles"])
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
    appended = model(
        torch.cat((poisoned, torch.full((12, 3), float("nan"))), dim=1),
        torch.cat((data["mask"], torch.zeros(12, 3, dtype=torch.bool)), dim=1),
        data["roles"],
    )
    torch.testing.assert_close(appended, expected, atol=1e-6, rtol=1e-6)
    permutation = torch.tensor([4, 2, 1, 9, 3, 8, 6, 5, 0, 7])
    permuted, permuted_details = model(
        poisoned[:, permutation], data["mask"][:, permutation], data["roles"], True
    )
    torch.testing.assert_close(permuted, expected, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(
        permuted_details["attention"], details["attention"][:, permutation],
        atol=1e-6, rtol=1e-6,
    )
    assert torch.isfinite(actual).all()
    assert (details["attention"][~data["mask"]] == 0).all()
    assert ((details["pheromone"] >= 0) & (details["pheromone"] <= 1)).all()
    assert ((details["refinement_budget"] > 0) & (details["refinement_budget"] < 1)).all()


def test_all_three_recurrent_roles_receive_training_gradients():
    torch.manual_seed(45)
    model = AHRBackupNet()
    data = generate_data(24, seed=72)
    prediction = model(data["values"], data["mask"], data["roles"])
    loss = (prediction - data["target"]).square().mean()
    loss.backward()
    for module in (model.scout_cell, model.worker_cell, model.queen_cell):
        gradients = [p.grad for p in module.parameters() if p.grad is not None]
        assert gradients
        assert all(torch.isfinite(gradient).all() for gradient in gradients)
        assert sum(gradient.abs().sum().item() for gradient in gradients) > 0
    assert 30_000 <= sum(p.numel() for p in model.parameters()) <= 60_000


def test_empty_domains_rejected_and_single_child_supported():
    model = AHRBackupNet()
    with pytest.raises(ValueError, match="feasible child"):
        model(torch.zeros(1, 3), torch.zeros(1, 3, dtype=torch.bool), torch.ones(1))
    data = generate_data(5, seed=2, max_children=1)
    assert model(data["values"], data["mask"], data["roles"]).shape == (5,)

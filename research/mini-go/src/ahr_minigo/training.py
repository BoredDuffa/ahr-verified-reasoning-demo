"""Bounded CPU training with validated exact epoch-boundary continuation."""
from __future__ import annotations
import copy
import json
import os
import platform
import random
import time
from pathlib import Path
import numpy as np
import torch
from .io import atomic_json, file_hash
from .model import AHRBackupNet
from .synthetic import generate_data


DEFAULT_CONFIG = dict(seeds=[17, 29, 43], epochs=24, train_n=8192,
                      validation_n=1024, test_n=2048, batch_size=256,
                      width=32, max_children=26, learning_rate=0.003,
                      threads=2)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_model(config):
    return AHRBackupNet(width=config["width"], queen_steps=2, scout_steps=2)


def atomic_torch(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(),
                torch=torch.get_rng_state())


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])


def train_epoch(model, optimizer, data, batch_size):
    model.train()
    order = torch.randperm(len(data["target"]))
    loss_sum = 0.0
    for ix in order.split(batch_size):
        optimizer.zero_grad(set_to_none=True)
        prediction = model(data["values"][ix], data["mask"][ix], data["roles"][ix])
        loss = ((prediction - data["target"][ix]) ** 2).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        optimizer.step()
        loss_sum += float(loss.detach()) * len(ix)
    return loss_sum / len(order)


@torch.no_grad()
def evaluate(model, data, batch_size=512):
    model.eval()
    predictions = []
    for start in range(0, len(data["target"]), batch_size):
        ix = slice(start, start + batch_size)
        predictions.append(model(data["values"][ix], data["mask"][ix], data["roles"][ix]))
    prediction = torch.cat(predictions)
    error = prediction - data["target"]
    return dict(n=len(error), mse=float(error.square().mean()),
                mae=float(error.abs().mean()), within_005=float((error.abs() <= .05).float().mean()))


def resume_smoke(output, config=None):
    config = dict(DEFAULT_CONFIG if config is None else config)
    torch.set_num_threads(config["threads"])
    seed_all(719)
    data = generate_data(128, 8101, config["max_children"], "quantified")
    model = make_model(config)
    opt = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    train_epoch(model, opt, data, 64)
    payload = dict(model=copy.deepcopy(model.state_dict()), optimizer=copy.deepcopy(opt.state_dict()), rng=rng_state())
    checkpoint = Path(output) / "checkpoints/resume_smoke/epoch_001.pt"
    atomic_torch(checkpoint, payload)
    expected_loss = train_epoch(model, opt, data, 64)
    expected = copy.deepcopy(model.state_dict())
    restarted = make_model(config)
    restarted_opt = torch.optim.Adam(restarted.parameters(), lr=config["learning_rate"])
    recovered = torch.load(checkpoint, map_location="cpu", weights_only=False)
    restarted.load_state_dict(recovered["model"])
    restarted_opt.load_state_dict(recovered["optimizer"])
    restore_rng(recovered["rng"])
    actual_loss = train_epoch(restarted, restarted_opt, data, 64)
    equal = expected_loss == actual_loss and all(torch.equal(value, restarted.state_dict()[key]) for key, value in expected.items())
    report = dict(exact_continuation=equal, expected_loss=expected_loss,
                  resumed_loss=actual_loss, compared_tensors=len(expected), checkpoint_sha256=file_hash(checkpoint))
    atomic_json(Path(output) / "results/resume_smoke.json", report)
    if not equal:
        raise RuntimeError("Exact continuation smoke failed")
    return report


def train_one(output, seed, mode, config=None):
    config = dict(DEFAULT_CONFIG if config is None else config)
    output = Path(output)
    run_id = f"{mode}_seed{seed}"
    folder = output / "checkpoints" / run_id
    folder.mkdir(parents=True, exist_ok=True)
    result_path = output / "results/training" / f"{run_id}.json"
    if result_path.exists():
        existing = json.loads(result_path.read_text())
        if existing["config"] != config:
            raise ValueError("Configuration changed: use a new output directory")
        if file_hash(folder / "best.pt") != existing["checkpoint_sha256"]:
            raise ValueError("Completed checkpoint digest mismatch")
        return existing
    torch.set_num_threads(config["threads"])
    seed_all(seed)
    model = make_model(config)
    opt = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    # Same initialization and child samples across curricula; roles are the intervention.
    init_path = output / "checkpoints" / f"untrained_seed{seed}.pt"
    if not init_path.exists():
        atomic_torch(init_path, dict(model=model.state_dict(), config=config, seed=seed))
    data = generate_data(config["train_n"], 10000 + seed, config["max_children"], mode)
    val = generate_data(config["validation_n"], 20000 + seed, config["max_children"], mode)
    test = generate_data(config["test_n"], 30000 + seed, config["max_children"], "quantified")
    first_epoch, best, best_epoch, history, elapsed_prior = 0, float("inf"), None, [], 0.0
    manifest_path = folder / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        recovery_path = folder / manifest["latest"]
        if file_hash(recovery_path) != manifest["latest_sha256"]:
            raise ValueError("Latest checkpoint digest mismatch")
        payload = torch.load(recovery_path, map_location="cpu", weights_only=False)
        if payload["config"] != config:
            raise ValueError("Resume config mismatch")
        model.load_state_dict(payload["model"])
        opt.load_state_dict(payload["optimizer"])
        restore_rng(payload["rng"])
        first_epoch, best, best_epoch = payload["epoch"], payload["best"], payload["best_epoch"]
        history, elapsed_prior = payload["history"], payload["elapsed_s"]
        print(f"Resuming {run_id} at epoch {first_epoch}", flush=True)
    started = time.perf_counter()
    for epoch in range(first_epoch, config["epochs"]):
        train_loss = train_epoch(model, opt, data, config["batch_size"])
        val_report = evaluate(model, val)
        improved = val_report["mse"] < best
        if improved:
            best, best_epoch = val_report["mse"], epoch + 1
        history.append(dict(epoch=epoch + 1, train_mse=train_loss, validation_mse=val_report["mse"]))
        payload = dict(model=model.state_dict(), optimizer=opt.state_dict(), rng=rng_state(),
                       epoch=epoch + 1, best=best, best_epoch=best_epoch, history=history,
                       elapsed_s=elapsed_prior + time.perf_counter() - started,
                       config=config, mode=mode, seed=seed,
                       scheduler=None, scaler=None, sampler_position="epoch_boundary")
        latest = folder / f"epoch_{epoch + 1:03d}.pt"
        atomic_torch(latest, payload)
        if improved:
            atomic_torch(folder / "best.pt", payload)
        atomic_json(manifest_path, dict(latest=latest.name, latest_sha256=file_hash(latest),
                                       best="best.pt", best_epoch=best_epoch,
                                       resume="train_one with the same config and output folder"))
        # Exact targets owned by this run; preserve two latest plus separate best.
        if epoch >= 2:
            obsolete = folder / f"epoch_{epoch - 1:03d}.pt"
            if obsolete.exists():
                obsolete.unlink()
        if (epoch + 1) % 4 == 0:
            print(f"{run_id} epoch {epoch + 1}/{config['epochs']} train={train_loss:.5f} val={val_report['mse']:.5f}", flush=True)
    checkpoint = torch.load(folder / "best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"])
    report = dict(run_id=run_id, config=config, seed=seed, mode=mode,
                  parameters=sum(p.numel() for p in model.parameters()), best_epoch=best_epoch,
                  training_s=elapsed_prior + time.perf_counter() - started,
                  history=history, quantified_test=evaluate(model, test),
                  checkpoint_sha256=file_hash(folder / "best.pt"),
                  environment=dict(python=platform.python_version(), torch=torch.__version__, device="cpu"))
    atomic_json(result_path, report)
    return report


def load_model(output, mode, seed, config=None):
    config = dict(DEFAULT_CONFIG if config is None else config)
    path = (Path(output) / "checkpoints" / f"untrained_seed{seed}.pt" if mode == "untrained"
            else Path(output) / "checkpoints" / f"{mode}_seed{seed}" / "best.pt")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = make_model(config)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model

#!/usr/bin/python3
"""
(D)iversit(y) i(n) (A)dversarial (M)odel-based (O)ptimization (DynAMO).

Author(s):
    Michael Yao @michael-s-yao

Licensed under the MIT License. Copyright University of Pennsylvania 2024.
"""
import click
import design_bench
import numpy as np
import pytorch_lightning as pl
import sys
import torch
import yaml
from botorch.utils.transforms import unnormalize
from pathlib import Path
from typing import Optional, Union

import dynamo


@click.command()
@click.option(
    "--task",
    "-t",
    required=True,
    type=click.Choice([
        task.task_name for task in design_bench.registry.all()
    ]),
    help="Offline optimization task."
)
@click.option(
    "--optimizer",
    "-o",
    required=True,
    type=click.Choice(dynamo.optim.get_optimizers()),
    help="Backbone optimizer."
)
@click.option(
    "--transform",
    "-f",
    required=True,
    type=click.Choice(dynamo.core.get_transforms()),
    help="Forward surrogate model transform."
)
@click.option(
    "--batch-size",
    type=int,
    default=64,
    show_default=True,
    help="Sampling batch size."
)
@click.option(
    "--sobol-init/--best-init",
    default=True,
    help="Policy initialization strategy."
)
@click.option(
    "--dual-step-size",
    type=float,
    default=0.01,
    show_default=True,
    help="Step size for dual variable optimization."
)
@click.option(
    "--W0",
    type=float,
    default=0.0,
    show_default=True,
    help="Wasserstein distance constraint threshold."
)
@click.option(
    "--num-restarts",
    type=int,
    default=2,
    show_default=True,
    help="The number of allowed optimization restarts."
)
@click.option(
    "--patience",
    type=int,
    default=10,
    show_default=True,
    help="Patience before restarting the optimizer."
)
@click.option(
    "--beta",
    type=float,
    default=1.0,
    show_default=True,
    help="Diversity hyperparameter."
)
@click.option(
    "--tau",
    type=float,
    default=1.0,
    show_default=True,
    help="Temperature hyperparameter."
)
@click.option(
    "--gamma",
    type=float,
    default=1.0,
    show_default=True,
    help="Proposed sample weighting over time."
)
@click.option(
    "--eta",
    type=float,
    default=0.01,
    show_default=True,
    help="Step size for first-order optimization methods."
)
@click.option(
    "--pretraining-strategy",
    type=click.Choice(["None", "COMs", "RoMA"], case_sensitive=False),
    default=None,
    show_default=True,
    help="Optional forward surrogate model pretraining strategy."
)
@click.option(
    "--mixed-chi-squared-weighting",
    type=float,
    default=None,
    show_default=True,
    help="The optional weighting of the Chi-squared divergence penalty."
)
@click.option(
    "--ablate-critic/--no-ablate-critic",
    default=False,
    show_default=True,
    help="Whether to ablate source critic feedback."
)
@click.option(
    "--oracle-evaluation-budget",
    "-k",
    type=int,
    default=128,
    show_default=True,
    help="Oracle evaluation budget."
)
@click.option(
    "--seed", type=int, default=0, show_default=True, help="Random seed."
)
@click.option(
    "--device", type=str, default="auto", show_default=True, help="Device."
)
@click.option(
    "--savedir",
    type=str,
    default="results",
    show_default=True,
    help="Directory to log the optimization results to."
)
@click.option(
    "--fast-dev-run/--full-run",
    default=False,
    help="Whether to run a fast debugging run."
)
def main(
    task: str,
    optimizer: str,
    transform: str,
    bounds_fn: Union[Path, str] = "bounds.yaml",
    batch_size: int = 64,
    sobol_init: bool = True,
    dual_step_size: float = 0.01,
    w0: float = 0.0,
    num_restarts: int = 2,
    patience: int = 10,
    beta: float = 1.0,
    tau: float = 1.0,
    gamma: float = 1.0,
    eta: float = 0.01,
    pretraining_strategy: Optional[str] = None,
    mixed_chi_squared_weighting: Optional[float] = None,
    ablate_critic: bool = False,
    oracle_evaluation_budget: int = 128,
    seed: int = 0,
    device: str = "auto",
    savedir: Union[Path, str] = "results",
    fast_dev_run: bool = False
):
    """Diversity-optimized generative adversarial model-based optimization."""
    pl.seed_everything(seed)
    rng = np.random.default_rng(seed)

    task_name, oracle_kwargs = task, {}
    if task == "StoryGen-Exact-v0" and transform == "GAMBOTransform":
        oracle_kwargs["device_id"] = -1 + (2 * (torch.cuda.device_count() > 1))
    task = dynamo.make(task_name, oracle_kwargs=oracle_kwargs)
    if not task.is_discrete:
        task.map_normalize_x()

    dm = dynamo.data.DesignBenchDataModule(
        task, batch_size=batch_size, seed=seed
    )
    dm.prepare_data()
    dm.setup()

    if str(pretraining_strategy).strip().lower() == "roma":
        model = dynamo.models.EncDecPropModule.load_from_RoMA_checkpoint(
            dynamo.utils.get_model_ckpt(task_name), task=task
        )
    elif str(pretraining_strategy).strip().lower() == "coms":
        model = dynamo.models.EncDecPropModule.load_from_COMs_checkpoint(
            dynamo.utils.get_model_ckpt(task_name), task=task
        )
    else:
        model = dynamo.models.EncDecPropModule.load_from_checkpoint(
            dynamo.utils.get_model_ckpt(task_name), task=task
        )
    vae, surrogate = model.vae, model.surrogate
    device = next(model.parameters()).device

    with open(bounds_fn) as f:
        bounds = yaml.safe_load(f)[task_name]
    bounds = torch.tensor(bounds)
    if bounds.ndim == 1:
        bounds = torch.hstack([bounds.unsqueeze(dim=-1)] * vae.latent_size)

    xp = [
        vae.encode(batch.x.to(device))[1].detach().cpu()
        for batch in dm.val_dataloader()
    ]
    xp = torch.cat(xp).flatten(start_dim=1).to(device)
    y = torch.cat([batch.y for batch in dm.val_dataloader()]).to(device)

    if device == "auto" and torch.cuda.is_available():
        xp, bounds = xp.cuda(), bounds.cuda()
    else:
        device = torch.device(device)
        xp, bounds = xp.to(device), bounds.to(device)
    xp, bounds = xp.double(), bounds.double()

    forward_model = getattr(dynamo.core, transform)(
        surrogate,
        vae=vae,
        xp=xp,
        yp=y,
        beta=beta,
        tau=tau,
        W0=w0,
        dual_step_size=dual_step_size,
        task=task,
        pool=dm.train_dataloader(),
        mixed_chi_squared_weighting=mixed_chi_squared_weighting,
        ablate_critic=ablate_critic,
        seed=seed
    )

    policy = getattr(dynamo.optim, optimizer)(
        batch_size=batch_size,
        ndim=xp.size(dim=-1),
        sampling_bounds=bounds,
        seed=seed,
        eta=eta,
        num_actors=torch.cuda.device_count(),
        device=device
    )

    state = dynamo.optim.OptimizerState(
        task_name,
        task,
        model,
        savedir,
        num_restarts=num_restarts,
        patience=patience,
        logger=dynamo.metrics.get_logger(),
        seed=seed
    )

    qtt = None
    if sobol_init:
        sobol = torch.quasirandom.SobolEngine(
            dimension=xp.size(dim=-1), scramble=True, seed=seed
        )

    while not state.has_converged:
        if state.designs is None and sobol_init:
            new_xq = unnormalize(sobol.draw(batch_size).to(bounds), bounds)
            qtt = None
        elif state.designs is None:
            best_idxs = np.argsort(y.detach().cpu().numpy().squeeze())
            best_idxs = best_idxs[-batch_size:]
            rng.shuffle(best_idxs)
            new_xq = xp[best_idxs]
            qtt = None
        else:
            new_xq = policy(func=forward_model).to(device)

        new_yq = forward_model(new_xq).detach()
        new_qt = np.ones(new_yq.size(dim=0))
        if qtt is None:
            qtt = new_qt
        else:
            qtt = np.concatenate((new_qt * qtt[0] * np.exp(-gamma), qtt))

        state.log(new_xq.detach().cpu(), new_yq.detach().cpu())

        if state.designs is None:
            continue
        forward_model.fit(xp=xp, xq=state.designs.to(device), yp=y, qpi=qtt)
        policy.fit(state.designs, state.predictions, seed=seed)

        if fast_dev_run:
            return

    state.save(oracle_evaluation_budget, cli=" ".join(["python"] + sys.argv))


if __name__ == "__main__":
    main()

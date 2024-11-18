#!/usr/bin/python3
"""
Diversity-optimized generative adversarial model-based optimization (DO-GAMBO).

Author(s):
    Michael Yao @michael-s-yao

Licensed under the MIT License. Copyright University of Pennsylvania 2024.
"""
import click
import design_bench
import logging
import numpy as np
import pytorch_lightning as pl
import torch
import yaml
from botorch.utils.transforms import unnormalize
from pathlib import Path
from typing import Union

import dogambo


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
    "--seed", type=int, default=0, show_default=True, help="Random seed."
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
    seed: int = 0,
    savedir: Union[Path, str] = "results",
    fast_dev_run: bool = False
):
    """Diversity-optimized generative adversarial model-based optimization."""
    pl.seed_everything(seed)
    rng = np.random.default_rng(seed)

    task, task_name = design_bench.make(task), task

    dm = dogambo.data.DesignBenchDataModule(
        task, batch_size=batch_size, seed=seed
    )
    dm.prepare_data()
    dm.setup()

    model = dogambo.models.EncDecPropModule.load_from_checkpoint(
        dogambo.utils.get_model_ckpt(task_name), task=task
    )
    vae, surrogate = model.vae, model.surrogate
    kld = dogambo.metrics.KLDivergence()
    device = next(model.parameters()).device

    with open(bounds_fn) as f:
        bounds = yaml.safe_load(f)[task_name]
    bounds = torch.tensor(bounds)
    if bounds.ndim == 1:
        bounds = torch.hstack([bounds.unsqueeze(dim=-1)] * vae.latent_size)

    xp = [
        vae.encode(batch.x.to(device))[1].detach().cpu()
        for batch in dm.train_dataloader()
    ]
    xp = torch.cat(xp).flatten(start_dim=1).to(device)
    y = torch.cat([batch.y for batch in dm.train_dataloader()]).to(device)

    if sobol_init:
        sobol = torch.quasirandom.SobolEngine(
            dimension=xp.size(dim=-1), scramble=True, seed=seed
        )
        xq = unnormalize(sobol.draw(batch_size), bounds)
    else:
        best_idxs = np.argsort(y.squeeze())[-batch_size:]
        rng.shuffle(best_idxs)
        xq = xp[best_idxs]

    ptau_ref = dogambo.utils.p_tau_ref(y, tau=tau)

    if torch.cuda.is_available():
        xp, xq, bounds = xp.cuda(), xq.cuda(), bounds.cuda()
    xp, xq, bounds = xp.double(), xq.double(), bounds.double()
    qtt = np.ones(xq.size(dim=0), dtype=np.float32)

    g = dogambo.models.ExplicitDual(
        Xp=xp,
        ptau=ptau_ref,
        critic=dogambo.models.LipschitzMLP(xp.size(dim=-1), 1, [512, 512]),
        W0=w0,
        dual_step_size=dual_step_size,
        seed=seed
    )
    g.fit(xp, xq, qtt)

    policy = dogambo.optim.qEIPolicy(
        batch_size=batch_size,
        ndim=xp.size(dim=-1),
        sampling_bounds=bounds,
        seed=seed
    )

    state = dogambo.optim.OptimizerState(
        task_name,
        task,
        model,
        savedir,
        num_restarts=num_restarts,
        patience=patience,
        logger=logging.getLogger(__name__)
    )
    Ecp = g.critic(xp).mean() - w0
    yq = surrogate(xq).detach()
    yq -= (beta / tau) * kld(
        xq.mean() - xp.mean(), torch.log(xq.std().square() + xp.std().square())
    )
    yq -= (beta * g.lambd * (Ecp - g.critic(xq)))
    state.log(xq.detach(), yq.detach())

    while not state.has_converged:
        g.fit(xp, state.designs, qtt)
        policy.fit(state.designs, state.predictions, seed=seed)
        new_xq = policy()
        new_yq = surrogate(new_xq).detach()
        new_yq -= (beta / tau) * kld(
            new_xq.mean() - xp.mean(),
            torch.log(new_xq.std().square() + xp.std().square())
        )
        new_yq -= (beta * g.lambd * (Ecp - g.critic(xq)))
        new_qt = np.ones(new_yq.size(dim=0)) * qtt[0] * np.exp(-gamma)

        state.log(new_xq.detach(), new_yq.detach())
        qtt = np.concatenate((new_qt, qtt))
        if fast_dev_run:
            return

    state.save()


if __name__ == "__main__":
    main()

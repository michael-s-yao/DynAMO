#!/usr/bin/python3
"""
Diversity-optimized generative adversarial model-based optimization (DO-GAMBO).

Author(s):
    Michael Yao @michael-s-yao

Licensed under the MIT License. Copyright University of Pennsylvania 2024.
"""
import click
import design_bench
import dogambo
import lightning.pytorch as pl
from typing import Optional


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
    default=128,
    show_default=True,
    help="Batch size."
)
@click.option(
    "--num-epochs",
    type=int,
    default=100,
    show_default=True,
    help="Number of training epochs."
)
@click.option(
    "--lr",
    type=float,
    default=0.0003,
    show_default=True,
    help="Learning rate."
)
@click.option(
    "--seed",
    type=int,
    default=0,
    show_default=True,
    help="Random seed."
)
@click.option(
    "--fast-dev-run/--full-run",
    default=False,
    help="Whether to run a fast debugging run."
)
def main(
    task: str,
    batch_size: int = 128,
    num_epochs: int = 100,
    lr: float = 0.0003,
    alpha: float = 1e-4,
    beta: float = 1.0,
    seed: Optional[int] = None,
    fast_dev_run: bool = True
):
    """Train the VAE and surrogate objective models."""
    pl.seed_everything(seed)
    task, task_name = design_bench.make(task), task
    task.map_normalize_y()
    if not task.is_discrete:
        task.map_normalize_x()

    dm = dogambo.data.DesignBenchDataModule(
        task, batch_size=batch_size, seed=seed
    )

    model = dogambo.models.EncDecPropModule(
        task, lr=lr, alpha=alpha, beta=beta
    )

    callbacks = [
        pl.callbacks.ModelCheckpoint(
            monitor="val_loss", mode="min", filename=f"{task_name}"
        ),
        pl.callbacks.EarlyStopping(
            monitor="val_loss",
            min_delta=0.0,
            patience=10,
            mode="min",
            check_finite=True
        )
    ]

    trainer = pl.Trainer(
        accelerator="auto",
        devices="auto",
        fast_dev_run=fast_dev_run,
        max_epochs=num_epochs,
        callbacks=callbacks,
        deterministic=True
    )

    trainer.fit(model, datamodule=dm)


if __name__ == "__main__":
    main()

#!/usr/bin/python3
"""
Optimizer evaluation script for analyzing experimental results.

Author(s):
    Michael Yao @michael-s-yao

Licensed under the MIT License. Copyright University of Pennsylvania 2024.
"""
import dogambo
import click
import design_bench
import numpy as np
import os
import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional, Tuple, Union


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
    "--oracle-budget",
    "-b",
    required=True,
    type=int,
    help="Oracle evaluation budget. Common choices are 1, 128, and 256."
)
@click.option(
    "--savedir",
    type=str,
    default="results",
    show_default=True,
    help="Path to the saved experimental results."
)
@click.option(
    "--diversity-metric",
    "-d",
    type=click.Choice(dogambo.metrics.get_diversity_metric_options()),
    multiple=True,
    help="Diversity metric(s) to use."
)
@click.option(
    "--pretraining-strategy",
    type=click.Choice(["None", "COMs", "RoMA"], case_sensitive=False),
    default=None,
    show_default=True,
    help="Optional forward surrogate model pretraining strategy."
)
@click.option(
    "--seed",
    type=int,
    default=0,
    show_default=True,
    help="Random seed."
)
@click.option(
    "--max-samples",
    type=int,
    default=-1,
    show_default=True,
    help="Maximum number of reference samples. Default None."
)
def main(
    task: str,
    oracle_budget: int,
    savedir: Union[Path, str],
    diversity_metric: Tuple[str],
    pretraining_strategy: Optional[str] = None,
    seed: int = 0,
    max_samples: Optional[int] = -1,
):
    """Optimizer evaluation script for analyzing experimental results."""
    task, task_name = design_bench.make(task), task
    dm = dogambo.data.DesignBenchDataModule(task, seed=0)
    dm.prepare_data()
    dm.setup()

    if task_name == "StoryGen-Exact-v0":
        tmp = task
    else:
        tmp = design_bench.make(
            task_name, dataset_kwargs={
                "max_percentile": 100.0, "min_percentile": 0.0
            }
        )
    ymin, ymax = tmp.y.min(), tmp.y.max()

    results = [
        np.load(os.path.join(savedir, fn)) for fn in filter(
            lambda fn: fn.startswith(task_name), os.listdir(savedir)
        )
    ]
    designs, scores = [], []
    for data in results:
        predictions = data["predictions"].squeeze()
        idxs = np.argsort(predictions)[-min(oracle_budget, len(predictions)):]
        designs.append(torch.from_numpy(data["designs"][idxs]))
        scores.append((data["scores"].squeeze()[idxs] - ymin) / (ymax - ymin))
    if torch.cuda.is_available():
        designs = [x.cuda() for x in designs]

    click.echo(f"Max Score: {[np.max(y) for y in scores]}")
    click.echo(f"Median Score: {[np.median(y) for y in scores]}")

    reference = dm.val.x
    if max_samples is not None and len(reference) > max_samples > 0:
        reference = reference[
            np.random.default_rng(seed).choice(
                len(reference), size=max_samples, replace=False
            )
        ]

    for metric in diversity_metric:
        if metric.lower() == "l1-coverage":
            if task_name in [
                "TFBind8-Exact-v0", "TFBind10-Exact-v0", "UTR-ResNet-v0"
            ]:
                embedder = dogambo.embed.DNABERT()
            elif task_name in ["GFP-Transformer-v0"]:
                embedder = dogambo.embed.ESM2()
            elif task_name in [
                "ChEMBL_MCHC_CHEMBL3885882_MorganFingerprint-RandomForest-v0",
                "PenalizedLogP-Exact-v0"
            ]:
                embedder = dogambo.embed.ChemBERT()
            else:
                embedder = nn.Identity()
            _designs = map(embedder, designs)
        else:
            _designs = designs

        diversity = []
        for x in _designs:
            y = dogambo.metrics.compute_diversity(x, reference, metric=metric)
            if task.is_discrete and metric.lower() != "l1-coverage":
                y = y / task.input_size
            diversity.append(y.item())
        click.echo(f"Diversity ({metric}): {diversity}")


if __name__ == "__main__":
    main()

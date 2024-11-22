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
import statsmodels.stats.api as sms
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Final, Sequence, Union


class ExperimentalResult:
    def __init__(
        self,
        task_name: str,
        task: design_bench.task.Task,
        results_fn: Union[Path, str],
        oracle_eval_budget: int,
        vae: nn.Module
    ):
        """
        Args:
            task_name: the name of the offline optimization task.
            task: the offline optimization task.
            results_fn: the path to the results file.
            oracle_eval_budget: the oracle evaluation budget.
            vae: the trained VAE model.
        """
        self.task_name: Final[str] = task_name
        self.task: Final[design_bench.task.Task] = task
        self.results_fn: Final[Union[Path, str]] = results_fn
        self.oracle_eval_budget: Final[int] = oracle_eval_budget
        self.vae: Final[nn.Module] = vae

        self.data: Final[Dict[str, Union[Sequence[str], np.ndarray]]] = (
            np.load(self.results_fn)
        )

        self.designs = torch.from_numpy(self.data["designs"]).to(
            next(self.vae.parameters())
        )
        self.designs: Final[torch.Tensor] = self.designs.reshape(
            -1, self.vae.bottleneck_size, self.vae.model_dim
        )

        self.predictions: Final[np.ndarray] = np.squeeze(
            self.data["predictions"], axis=-1
        )

        self.scores: Final[np.ndarray] = np.squeeze(
            self.data["scores"], axis=-1
        )
        if self.task_name in [
            "ChEMBL_MCHC_CHEMBL3885882_MorganFingerprint-RandomForest-v0"
        ]:
            tmp = design_bench.make(
                task_name, dataset_kwargs={
                    "max_percentile": 100.0, "min_percentile": 0.0
                }
            )
            self.scores = (self.scores - tmp.y.min()) / (
                tmp.y.max() - tmp.y.min()
            )

        self.best_idxs = np.argsort(self.predictions.flatten())[
            -self.oracle_eval_budget:
        ]
        self.observed_scores = self.scores.flatten()[self.best_idxs]
        self.observed_designs = self.designs[self.best_idxs]

    def best_scores(self, top_k: int) -> np.ndarray:
        """
        Returns the best oracle scores from the sampled designs.
        Input:
            top_k: the number of designs to evaluate with the oracle.
        Returns:
            The top_k best oracle scores.
        """
        assert 0 < top_k <= self.observed_scores.shape[0]
        return np.sort(self.observed_scores)[-top_k:]

    @property
    def best_designs(self) -> torch.Tensor:
        """
        Returns the observed designs evaluated using the oracle function.
        Input:
            None.
        Returns:
            The observed designs evaluated using the oracle function.
        """
        return self.vae.sample(z=self.observed_designs)[..., 1:]

    @property
    def max_score(self) -> float:
        """
        Returns the best oracle score from the sampled designs.
        Input:
            None.
        Returns:
            The best oracle score attained by the sampled designs.
        """
        return self.best_scores(top_k=1).item()

    @property
    def mean_score(self) -> float:
        """
        Returns the mean oracle score from the sampled designs.
        Input:
            None.
        Returns:
            The mean oracle score attained by the sampled designs.
        """
        return np.mean(self.observed_scores)

    @property
    def median_score(self) -> float:
        """
        Returns the median oracle score from the sampled designs.
        Input:
            None.
        Returns:
            The median oracle score attained by the sampled designs.
        """
        return np.median(self.observed_scores)


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
def main(
    task: str, savedir: Union[Path, str], oracle_budget: int
):
    """Optimizer evaluation script for analyzing experimental results."""
    task, task_name = design_bench.make(task), task
    dm = dogambo.data.DesignBenchDataModule(task, seed=0)
    dm.prepare_data()
    dm.setup()
    model = dogambo.models.EncDecPropModule.load_from_checkpoint(
        dogambo.utils.get_model_ckpt(task_name), task=task
    )

    results = [
        ExperimentalResult(
            task_name,
            task,
            os.path.join(savedir, fn),
            oracle_budget,
            model.vae
        )
        for fn in filter(
            lambda fn: fn.startswith(task_name), os.listdir(savedir)
        )
    ]

    max_int = sms.DescrStatsW([r.max_score for r in results]).tconfint_mean()
    max_mean = max_int[0] + ((max_int[-1] - max_int[0]) / 2.0)
    click.echo(f"Max Score: {max_mean} [{max_int[0]} - {max_int[-1]}]")

    me_int = sms.DescrStatsW([r.median_score for r in results]).tconfint_mean()
    me_mean = me_int[0] + ((me_int[-1] - me_int[0]) / 2.0)
    click.echo(f"Median Score: {me_mean} [{me_int[0]} - {me_int[-1]}]")

    mu_int = sms.DescrStatsW(
        np.concatenate([r.best_scores(oracle_budget) for r in results])
    )
    mu_int = mu_int.tconfint_mean()
    mu_mean = mu_int[0] + ((mu_int[-1] - mu_int[0]) / 2.0)
    click.echo(f"All Scores: {mu_mean} [{mu_int[0]} - {mu_int[-1]}]")

    if task_name in ["TFBind8-Exact-v0", "TFBind10-Exact-v0", "UTR-ResNet-v0"]:
        embedder = dogambo.embed.DNABERT()
    elif task_name in ["GFP-Transformer-v0"]:
        embedder = dogambo.embed.ESM2()
    elif task_name in [
        "ChEMBL_MCHC_CHEMBL3885882_MorganFingerprint-RandomForest-v0"
    ]:
        embedder = dogambo.embed.ChemBERT()

    designs = [embedder(r.best_designs) for r in results]
    reference = embedder(dm.val.x)
    diversity = [
        dogambo.metrics.compute_diversity(x, reference, metric="fid")
        for x in designs
    ]

    div_int = sms.DescrStatsW(diversity).tconfint_mean()
    div_mean = div_int[0] + ((div_int[-1] - div_int[0]) / 2.0)
    click.echo(f"Diversity: {div_mean} [{div_int[0]} - {div_int[-1]}]")


if __name__ == "__main__":
    main()

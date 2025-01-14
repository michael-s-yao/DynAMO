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
import pickle
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Final, Optional, Sequence, Tuple, Union


class ExperimentalResult:
    def __init__(
        self,
        task_name: str,
        task: design_bench.task.Task,
        results_fn: Union[Path, str],
        oracle_eval_budget: int,
        vae: nn.Module,
        ymin: Optional[float] = None,
        ymax: Optional[float] = None,
        cache_fn: Optional[Union[Path, str]] = ".dogambo.cache"
    ):
        """
        Args:
            task_name: the name of the offline optimization task.
            task: the offline optimization task.
            results_fn: the path to the results file.
            oracle_eval_budget: the oracle evaluation budget.
            vae: the trained VAE model.
            ymin: an optional minimum oracle value to normalize the scores.
            ymax: an optional maximum oracle value to normalize the scores.
            cache_fn: an optional cache file of experimental results.
        """
        self.task_name: Final[str] = task_name
        self.task: Final[design_bench.task.Task] = task
        self.results_fn: Final[Union[Path, str]] = os.path.abspath(results_fn)
        self.oracle_eval_budget: Final[int] = oracle_eval_budget
        self.vae: Final[nn.Module] = vae
        self.ymin: Final[Optional[float]] = ymin
        self.ymax: Final[Optional[float]] = ymax
        self.cache_fn: Final[Optional[Union[Path, str]]] = cache_fn

        self.data: Final[Dict[str, Union[Sequence[str], np.ndarray]]] = (
            np.load(self.results_fn)
        )

        if self.results_fn.lower().endswith(".npy"):
            self.designs = torch.from_numpy(self.data)
        else:
            self.designs = torch.from_numpy(self.data["designs"])

        if self.task.is_discrete:
            self.designs = self.designs.to(
                next(self.vae.parameters()).device
            )
            self.designs: Final[torch.Tensor] = self.designs.reshape(
                -1, self.vae.bottleneck_size, self.vae.model_dim
            )
        else:
            self.designs: Final[torch.Tensor] = self.designs.reshape(
                -1, self.designs.size(dim=-1)
            )

        if self.results_fn.lower().endswith(".npy"):
            self.predictions = np.ones((self.data.shape[0], 1), np.float32)
            self.scores = self.task.predict(
                self.designs.detach().cpu().numpy()
            )
            self.scores = np.squeeze(self.scores, axis=-1)
        else:
            self.predictions: Final[np.ndarray] = np.squeeze(
                self.data["predictions"]
            )
            self.scores: Final[np.ndarray] = np.squeeze(
                self.data["scores"]
            )

        if self.ymin is not None and self.ymax is not None:
            self.scores = (self.scores - self.ymin) / (self.ymax - self.ymin)

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
        if self.task.is_discrete:
            if self.cache_fn is not None:
                if os.path.isfile(self.cache_fn):
                    with open(self.cache_fn, "rb") as f:
                        data = pickle.load(f)
                    designs = data.get(self.cache_key, None)
                    if designs is not None:
                        return designs
                else:
                    data = {}
            designs = self.vae.sample(z=self.observed_designs)[..., 1:]
            if self.cache_fn is not None:
                data[self.cache_key] = designs
                with open(self.cache_fn, "wb") as f:
                    pickle.dump(data, f)
            return designs
        return self.observed_designs

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

    @property
    def cache_key(self) -> str:
        """
        Returns the cache key associated with the experimental results.
        Input:
            None.
        Returns:
            The cache key associated with the experimental results.
        """
        return self.results_fn + "-" + str(self.oracle_eval_budget)


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
    model = dogambo.models.EncDecPropModule.load_from_checkpoint(
        dogambo.utils.get_model_ckpt(task_name), task=task
    )

    if task_name == "StoryGen-Exact-v0":
        tmp = task
    else:
        tmp = design_bench.make(
            task_name, dataset_kwargs={
                "max_percentile": 100.0, "min_percentile": 0.0
            }
        )

    results = [
        ExperimentalResult(
            task_name,
            task,
            os.path.join(savedir, fn),
            oracle_budget,
            model.vae,
            ymin=tmp.y.min(),
            ymax=tmp.y.max()
        )
        for fn in filter(
            lambda fn: fn.startswith(task_name), os.listdir(savedir)
        )
    ]

    click.echo(f"Max Score: {[r.max_score for r in results]}")
    click.echo(f"Median Score: {[r.median_score for r in results]}")

    designs = [r.best_designs for r in results]
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
            _designs = [embedder(x) for x in designs]
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

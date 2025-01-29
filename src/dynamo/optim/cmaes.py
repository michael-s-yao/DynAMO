#!/usr/bin/python3
"""
Implements naive CMA-ES generative optimization policy.

Author(s):
    Michael Yao @michael-s-yao

Citation(s):
    [1] Hansen N. The CMA evolution strategy: A tutorial. arXiv Preprint.
        (2016). doi: 10.48550/arXiv.1604.00772
    [2] Hansen N, Ostermeier A. Adapting arbitrary normal mutation
        distributions in evolution strategies: The covariance matrix
        adaptation. Proc IEEE Intern Conf Evolutionary Computation: 312-7.
        (1996). doi: 10.1109/ICEC.1996.542381

Licensed under the MIT License. Copyright University of Pennsylvania 2024.
"""
import torch
from evotorch import Problem
from evotorch.algorithms.cmaes import CMAES
from evotorch.decorators import on_cuda, vectorized
from evotorch.logging import StdOutLogger
from typing import Any, Callable, Dict, Final, Optional, Union

from .base import BaseGenerativePolicy


class CMAESPolicy(BaseGenerativePolicy):
    """Covariance matrix adaptation evolutionary strategy generative policy."""

    def __init__(
        self,
        batch_size: int,
        sampling_bounds: torch.Tensor,
        optimizer_kwargs: Optional[Dict[str, Any]] = {"separable": True},
        num_steps_per_acq: int = 4,
        eta: float = 0.01,
        seed: int = 0,
        quiet: bool = True,
        num_actors: int = 4,
        num_gpus_per_actor: Final[Union[float, int]] = 1,
        **kwargs
    ):
        """
        Args:
            batch_size: the number of new designs to return per sampling step.
            sampling_bounds: the sampling bounds of shape 2D, where D is the
                number of design dimensions.
            optimizer_kwargs: optional keyword arguments for the optimizer.
            num_steps_per_acq: the number of algorithm iterations to run per
                acquisition step. Default 4.
            eta: step size. Default 0.01.
            seed: random seed. Default 0.
            quiet: whether to not print ouputs to stdout. Default False.
            num_actors: number of actors for parallelization. Default 4.
            num_gpus_per_actor: number of GPUs for each actor. Default 1.
        """
        assert sampling_bounds is not None
        super().__init__(
            batch_size=batch_size,
            sampling_bounds=sampling_bounds,
            seed=seed,
            **kwargs
        )
        self.optimizer_kwargs = optimizer_kwargs
        self.num_steps_per_acq: Final[int] = num_steps_per_acq
        self.eta: Final[float] = eta
        self.quiet: Final[bool] = quiet
        self.solution_length: Final[int] = self.sampling_bounds.size(dim=-1)
        self.searcher: CMAES = None
        self.num_actors: Final[int] = num_actors
        self.num_gpus_per_actor: Final[Union[float, int]] = num_gpus_per_actor

    def forward(
        self, func: Callable[[torch.Tensor], torch.Tensor], **kwargs
    ) -> torch.Tensor:
        """
        Returns a new batch of candidates to evaluate.
        Input:
            func: The surrogate function to optimize against.
        Returns:
            A batch of candidates to evaluate of shape BD, where B is the batch
            size and D is the number of design dimensions.
        """
        @on_cuda
        @vectorized
        def wrapper(x: torch.Tensor) -> torch.Tensor:
            return func(x.clone()).squeeze(dim=-1)

        kwargs = {"device": self.sampling_bounds.device}
        if self.num_actors > 1:
            kwargs.update({
                "device": "cpu",
                "num_actors": self.num_actors,
                "num_gpus_per_actor": self.num_gpus_per_actor
            })
        problem = Problem(
            "max",
            wrapper,
            solution_length=self.solution_length,
            initial_bounds=self.sampling_bounds,
            dtype=self.sampling_bounds.dtype,
            **kwargs
        )
        if self.searcher is None:
            self.searcher = CMAES(
                problem,
                stdev_init=self.eta,
                popsize=self.batch_size,
                **self.optimizer_kwargs
            )
            if not self.quiet:
                StdOutLogger(self.searcher)
        else:
            self.searcher._problem = problem

        self.searcher.run(self.num_steps_per_acq)
        return self.searcher.population.values.clone()

    def fit(self, X: torch.Tensor, y: torch.Tensor, **kwargs) -> None:
        """
        Performs any pre-acquisition steps.
        Input:
            X: a tensor of shape ND of all prior evaluated designs, where N is
                the number of designs and D is the number of design dimensions.
            y: a tensor of shape N1 of all objective evaluations, where N is
                the number of designs.
        Returns:
            None.
        """
        return

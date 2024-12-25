#!/usr/bin/python3
"""
Implements accelerated neural evolution through cooperatively coevolved
synapses generative optimization policy.

Author(s):
    Michael Yao @michael-s-yao

Citation(s):
    [1] Gomez F, Schmidhuber J, Miikkulainen R. Accelerated neural evolution
        through cooperatively coevolved synapses. J Mach Learn Res 9: 937-65.
        (2008). doi: 10.5555/1390681.1390712

Licensed under the MIT License. Copyright University of Pennsylvania 2024.
"""
import torch
from evotorch import Problem
from evotorch.algorithms.ga import Cosyne
from evotorch.logging import StdOutLogger
from typing import Any, Callable, Dict, Final, Optional

from .base import BaseGenerativePolicy


class CoSyNEPolicy(BaseGenerativePolicy):
    """CoSyNE generative policy."""

    def __init__(
        self,
        batch_size: int,
        sampling_bounds: torch.Tensor,
        optimizer_kwargs: Optional[Dict[str, Any]] = {},
        num_steps_per_acq: int = 4,
        eta: float = 0.01,
        seed: int = 0,
        quiet: bool = True,
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
        self.searcher: Cosyne = None

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
        problem = Problem(
            "max",
            lambda x: func(x.clone()).squeeze(dim=-1),
            solution_length=self.solution_length,
            initial_bounds=self.sampling_bounds,
            device=self.sampling_bounds.device,
            dtype=self.sampling_bounds.dtype
        )
        if self.searcher is None:
            self.searcher = Cosyne(
                problem,
                popsize=self.batch_size,
                tournament_size=self.num_steps_per_acq,
                mutation_stdev=self.eta,
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

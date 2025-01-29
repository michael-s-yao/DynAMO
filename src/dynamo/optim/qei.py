#!/usr/bin/python3
"""
Implements the standard quasi-Expected Improvement (qEI) generative policy.

Author(s):
    Michael Yao @michael-s-yao

Citation(s):
    [1] Jones DR, Schonlau M, Welch WJ. Efficient global optimization of
        expensive black-box functions. J Glob Opt 13:455-92. (1998). doi:
        10.1023/A:1008306431147

Licensed under the MIT License. Copyright University of Pennsylvania 2024.
"""
import torch
from botorch import fit_gpytorch_mll
from botorch.acquisition import qExpectedImprovement
from botorch.models import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from botorch.optim import optimize_acqf
from botorch.utils.transforms import normalize, unnormalize
from gpytorch.mlls import ExactMarginalLogLikelihood
from typing import Any, Dict, Final, Optional

from .base import BaseGenerativePolicy


class qEIPolicy(BaseGenerativePolicy):
    """Standard quasi-Expected Improvement (qEI) baseline policy."""

    def __init__(
        self,
        batch_size: int,
        sampling_bounds: torch.Tensor,
        seed: int = 0,
        num_restarts: int = 10,
        raw_samples: int = 512,
        **kwargs
    ):
        """
        Args:
            batch_size: batch size to use for Bayesian sampling per iteration.
            sampling_bounds: the sampling bounds of shape 2D, where D is the
                number of design dimensions.
            seed: random seed. Default 0.
            num_restarts: the number of starting points for multistart
                acquisition function optimization. Default 10.
            raw_samples: the number of samples for initialization. Default 512.
        """
        assert sampling_bounds is not None
        super().__init__(
            batch_size=batch_size,
            sampling_bounds=sampling_bounds,
            seed=seed,
            **kwargs
        )
        self.num_restarts: Final[int] = num_restarts
        self.raw_samples: Final[int] = raw_samples

    def forward(
        self,
        options: Optional[Dict[str, Any]] = {"batch_limit": 5, "maxiter": 200},
        **kwargs
    ) -> torch.Tensor:
        """
        Optimizes the quasi-Expected Improvement (qEI) acquisition function and
        returns a new batch of candidates to evaluate.
        Input:
            options: optional keyword arguments for the acquisition function
                optimization call.
        Returns:
            A batch of candidates to evaluate of shape BD, where B is the batch
            size and D is the number of design dimensions.
        """
        candidates, _ = optimize_acqf(
            acq_function=self.acqf,
            bounds=self.normalized_bounds,
            q=self.batch_size,
            num_restarts=self.num_restarts,
            raw_samples=self.raw_samples,
            options=options
        )
        return unnormalize(candidates, self.sampling_bounds).detach()

    def fit(self, X: torch.Tensor, y: torch.Tensor, **kwargs) -> None:
        """
        Fits a GP surrogate model and then optimizes the acquisition function
        based on the updated posterior.
        Input:
            X: a tensor of shape ND of all prior evaluated designs, where N is
                the number of designs and D is the number of design dimensions.
            y: a tensor of shape N1 of all objective evaluations, where N is
                the number of designs.
        Returns:
            The optimized acquisition function.
        """
        z = normalize(X, self.sampling_bounds.to(X.device)).detach()
        self.model = SingleTaskGP(
            z, y.to(z), outcome_transform=Standardize(m=1)
        )
        self.model = self.model.double()
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
        fit_gpytorch_mll(mll)

        self.acqf = qExpectedImprovement(model=self.model, best_f=y.max())

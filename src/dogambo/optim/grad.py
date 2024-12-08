#!/usr/bin/python3
"""
Implements naive first-order generative policies.

Author(s):
    Michael Yao @michael-s-yao

Licensed under the MIT License. Copyright University of Pennsylvania 2024.
"""
import torch
import torch.optim as optim
from typing import Any, Callable, Dict, Optional, Type

from .base import BaseGenerativePolicy


class FirstOrderPolicy(BaseGenerativePolicy):
    """Iterative first-order baseline generative policy."""

    def __init__(
        self,
        batch_size: int,
        sampling_bounds: torch.Tensor,
        optimizer: Type[optim.Optimizer],
        optimizer_kwargs: Optional[Dict[str, Any]] = {},
        num_steps_per_acq: int = 4,
        eta: float = 0.01,
        seed: int = 0,
        device: torch.device = torch.device("cpu"),
        **kwargs
    ):
        """
        Args:
            batch_size: the number of new designs to return per sampling step.
            sampling_bounds: the sampling bounds of shape 2D, where D is the
                number of design dimensions.
            optimizer: the iterative optimizer.
            optimizer_kwargs: optional keyword arguments for the optimizer.
            num_steps_per_acq: the number of iterative gradient steps to take
                per acquisition step. Default 4.
            eta: step size. Default 0.01.
            seed: random seed. Default 0.
            device: device. Default CPU.
        """
        super().__init__(
            batch_size=batch_size,
            sampling_bounds=sampling_bounds,
            seed=seed,
            **kwargs
        )
        self.optimizer_cls = optimizer
        self.optimizer_kwargs = optimizer_kwargs
        self.num_steps_per_acq = num_steps_per_acq
        self.eta = eta
        self.optimizer, self.X = None, None
        self.device = device

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
        assert self.optimizer is not None and self.X is not None
        for _ in range(self.num_steps_per_acq):
            self.optimizer.zero_grad()
            func(self.X).sum().backward(retain_graph=True)
            self.optimizer.step()
            self.X = torch.clamp(
                self.X.detach(),
                min=self.sampling_bounds[0],
                max=self.sampling_bounds[-1]
            )
        candidates = self.X.detach()
        self.optimizer, self.X = None, None
        return candidates

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
        self.X = X[torch.argsort(y.squeeze())[-self.batch_size:]].detach()
        self.X = self.X.to(self.device)
        self.X = self.X.requires_grad_(True)
        self.optimizer = self.optimizer_cls(
            [self.X], lr=self.eta, maximize=True, **self.optimizer_kwargs
        )


class GradientAscentPolicy(FirstOrderPolicy):
    """Iterative first-order gradient ascent generative policy."""

    def __init__(
        self,
        batch_size: int,
        optimizer_kwargs: Optional[Dict[str, Any]] = {},
        num_steps_per_acq: int = 4,
        eta: float = 0.01,
        seed: int = 0,
        device: torch.device = torch.device("cpu"),
        **kwargs
    ):
        """
        Args:
            batch_size: the number of new designs to return per sampling step.
            optimizer: the iterative optimizer.
            num_steps_per_acq: the number of iterative gradient steps to take
                per acquisition step. Default 4.
            eta: step size. Default 0.01.
            seed: random seed. Default 0.
            device: device. Default CPU.
        """
        super().__init__(
            batch_size=batch_size,
            optimizer=optim.SGD,
            optimizer_kwargs=optimizer_kwargs,
            num_steps_per_ac=num_steps_per_acq,
            eta=eta,
            seed=seed,
            device=device,
            **kwargs
        )


class AdamAscentPolicy(FirstOrderPolicy):
    """Iterative first-order generative policy with Adam optimizer."""

    def __init__(
        self,
        batch_size: int,
        optimizer_kwargs: Optional[Dict[str, Any]] = {},
        num_steps_per_acq: int = 4,
        eta: float = 0.01,
        seed: int = 0,
        device: torch.device = torch.device("cpu"),
        **kwargs
    ):
        """
        Args:
            batch_size: the number of new designs to return per sampling step.
            optimizer: the iterative optimizer.
            num_steps_per_acq: the number of iterative gradient steps to take
                per acquisition step. Default 4.
            eta: step size. Default 0.01.
            seed: random seed. Default 0.
            device: device. Default CPU.
        """
        super().__init__(
            batch_size=batch_size,
            optimizer=optim.Adam,
            optimizer_kwargs=optimizer_kwargs,
            num_steps_per_ac=num_steps_per_acq,
            eta=eta,
            seed=seed,
            device=device,
            **kwargs
        )

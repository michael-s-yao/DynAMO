#!/usr/bin/python3
"""
Conservative objective models (COMs) objective transform implementation.

Author(s):
    Michael Yao @michael-s-yao

Citation(s):
    [1] Trabucco B*, Kumar A*, Geng X, Levine S. Conservative objective
        models for effective offline model-based optimization. Proc ICML:
        10358-68. (2021). URL: https://proceedings.mlr.press/v139/trabucco21a

Licensed under the MIT License. Copyright University of Pennsylvania 2024.
"""
import numpy as np
import torch
from copy import deepcopy
from typing import Final, Optional

from .base import BaseObjectiveTransform


class COMsTransform(BaseObjectiveTransform):
    name: Final[str] = "COMsTransform"

    def __init__(
        self,
        surrogate: torch.Tensor,
        alpha: float = 1.0,
        T: int = 5,
        steps_per_update: int = 20,
        inner_lr: float = 0.001,
        **kwargs
    ):
        """
        Args:
            surrogate: the original forward surrogate model.
            alpha: RoMA regularization strength. Default 1.0.
            T: maximum number of solution updates. Default 5.
            steps_per_update: maximum number of weight steps per update.
                Default 20.
            inner_lr: the step size for weight updates. Default 0.001.
        """
        super().__init__(
            surrogate=surrogate,
            alpha=alpha,
            T=T,
            steps_per_update=steps_per_update,
            inner_lr=inner_lr,
            **kwargs
        )

    def forward(self, xq: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Forward pass through the RoMA objective transform.
        Input:
            xq: the input batch of designs to the surrogate model.
        Returns:
            The forward model predictions of the input designs.
        """
        return self.surrogate(xq)

    def fit(
        self,
        xp: torch.Tensor,
        xq: torch.Tensor,
        qpi: Optional[np.ndarray] = None,
        **kwargs
    ) -> None:
        """
        Updates the weights of the surrogate model.
        Input:
            Xp: a dataset of real reference designs of shape ND, where N is
                the number of designs and D the number of design dimensions.
            Xq: a dataset of generated designs of shape MD, where M is the
                number of designs and D the number of design dimensions.
            qpi: an optional array of shape N specifying the sampling
                probability over the generated designs.
        Returns:
            None.
        """
        for _ in range(self.T):
            tmp_model = deepcopy(self.surrogate)
            self.surrogate.zero_grad()
            for _ in range(self.steps_per_update):
                xq = xq.requires_grad_(True)
                yq = self.surrogate(xq)
                loss = torch.linalg.norm(
                    torch.autograd.grad(yq.sum(), xq)[0], dim=-1
                )
                loss += self.alpha * torch.square(
                    (self.surrogate(xq) - tmp_model(xq)).squeeze(dim=-1)
                )
                loss.mean().backward(retain_graph=True)
                for param in self.surrogate.parameters():
                    param = param - (self.inner_lr * param.grad)

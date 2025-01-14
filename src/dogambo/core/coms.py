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
import torch
import torch.nn.functional as F
from typing import Final

from .base import BaseObjectiveTransform


class COMsTransform(BaseObjectiveTransform):
    name: Final[str] = "COMsTransform"

    def __init__(
        self,
        surrogate: torch.Tensor,
        alpha: float = 0.1,
        _beta: float = 0.9,
        eta: float = 0.01,
        steps_per_update: int = 100,
        lambd: float = 0.01,
        **kwargs
    ):
        """
        Args:
            surrogate: the original forward surrogate model.
            alpha: a hyperparameter controlling the tradeoff between
                conservatism and regression. Default 0.1.
            _beta: a hyperparameter controlling the weighted penalty of the
                hallucinated, lookahead gradient-ascient iterate. Default 0.9.
            eta: a hyperparameter controlling the step size of the
                hallucinated, lookahead gradient-ascient iterate. Default 0.01.
            steps_per_update: maximum number of weight steps per update.
                Default 100.
            lambd: the step size for weight updates. Default 0.001.
        """
        kwargs.pop("beta", None)
        super().__init__(
            surrogate=surrogate,
            alpha=alpha,
            beta=_beta,
            eta=eta,
            steps_per_update=steps_per_update,
            lambd=lambd,
            **kwargs
        )

    def forward(self, xq: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Forward pass through the COMs objective transform.
        Input:
            xq: the input batch of designs to the surrogate model.
        Returns:
            The forward model predictions of the input designs.
        """
        xq = xq.requires_grad_(True)
        try:
            grad = torch.autograd.grad(self.surrogate(xq).sum(), xq)[0]
        except RuntimeError:
            y = self.surrogate(xq).squeeze()
            grad = torch.vstack([
                torch.gradient(y, spacing=(xq[..., dim],))[0]
                for dim in range(xq.size(dim=-1))
            ])
            grad = grad.T
        xupdt = xq + (self.eta * grad)
        return self.surrogate(xq) - self.beta * self.surrogate(xupdt)

    def fit(
        self,
        xp: torch.Tensor,
        yp: torch.Tensor,
        **kwargs
    ) -> None:
        """
        Updates the weights of the surrogate model.
        Input:
            xp: a dataset of real reference designs of shape ND, where N is
                the number of designs and D the number of design dimensions.
            yp: a dataset of the corresponding reference design scores of shape
                N1, where N is the number of designs.
        Returns:
            None.
        """
        xt = xp.clone()
        for _ in range(self.steps_per_update):
            self.surrogate.zero_grad()
            xt = xt.requires_grad_(True)
            yt = self.surrogate(xt)
            ell = F.mse_loss(yt.squeeze(), yp.squeeze())
            ell = ell - (self.alpha * (self.surrogate(xp) - yt))
            try:
                ell.sum().backward(retain_graph=True)
            except RuntimeError:
                return
            for param in self.surrogate.parameters():
                param = param - (self.lambd * param.grad)
            xt = xt + (
                self.eta * torch.autograd.grad(self.surrogate(xt).sum(), xt)[0]
            )

#!/usr/bin/python3
"""
Generative adversarial model-based optimization (GAMBO) offline objective
transform module implementation.

Author(s):
    Michael Yao @michael-s-yao

Citation(s):
    [1] Yao MS, Zeng Y, Bastani H, Garder JR, Gee JC, Bastani O. Generative
        adversarial model-based optimization via source critic regularization.
        Proc NeurIPS. (2024). URL: https://openreview.net/forum?id=3RxcarQFRn

Portions of this code were adapted from the gabo repository from @michael-s-yao
at https://github.com/michael-s-yao/gabo.

Licensed under the MIT License. Copyright University of Pennsylvania 2024.
"""
import numpy as np
import torch
import torch.nn.functional as F
from typing import Final, Optional, Sequence

from .base import BaseObjectiveTransform
from ..models import LipschitzMLP


class GAMBOTransform(BaseObjectiveTransform):
    name: Final[str] = "GAMBOTransform"

    def __init__(
        self,
        surrogate: torch.Tensor,
        xp: torch.Tensor,
        yp: torch.Tensor,
        critic_hidden_dims: Sequence[int] = [512, 512],
        alpha: Optional[float] = None,
        c: float = 0.01,
        search_budget: int = 1024,
        norm_thresh: float = 0.001,
        seed: Optional[int] = 0,
        **kwargs
    ):
        """
        Args:
            surrogate: the original forward surrogate model.
            xp: a dataset of real offline designs of shape ND, where N is the
                number of offline designs and D the number of design
                dimensions.
            yp: the corresponding oracle design scores of the real offline
                designs of shape N1.
            critic_hidden_dims: the hidden dimensions of the source critic.
                Default [512, 512].
            alpha: an optional constant value for alpha.
            c: weight clipping parameter. Default 0.01.
            search_budget: number of samples to sample from the normal
                distribution in solving for alpha. Default 4096.
            norm_thresh: threshold value for the norm of the Lagrangian
                gradient. Default 0.001.
            seed: optional random seed. Default 0.
        """
        super().__init__(
            surrogate=surrogate,
            xp=xp,
            critic_hidden_dims=critic_hidden_dims,
            _alpha=alpha,
            c=c,
            search_budget=search_budget,
            norm_thresh=norm_thresh,
            seed=seed,
            **kwargs
        )
        self.critic = LipschitzMLP(
            self.xp.size(dim=-1), 1, self.critic_hidden_dims
        )
        if torch.cuda.is_available():
            self.xp, self.critic = self.xp.cuda(), self.critic.cuda()
        self.critic.eval()

    def forward(
        self,
        xq: torch.Tensor,
        alpha: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Forward pass through the GAMBO objective transform.
        Input:
            xq: the input batch of designs to the surrogate model.
            alpha: an optional tensor of specific alpha value(s) to use.
        Returns:
            The forward model predictions of the input designs.
        """
        if alpha is None:
            alpha = self.alpha()
        return ((1.0 - alpha) * self.surrogate(xq, flatten=False)) - F.relu(
            alpha * self.wasserstein(self.xp, xq).unsqueeze(dim=-1)
        )

    def fit(self, xp: torch.Tensor, xq: torch.Tensor, **kwargs) -> None:
        """
        Fits the Lipschitz-constrained source critic model.
        Input:
            xp: a dataset of real reference designs of shape ND, where N is
                the number of designs and D the number of design dimensions.
            xq: a dataset of generated designs of shape MD, where M is the
                number of designs and D the number of design dimensions.
        Returns:
            None.
        """
        return self.critic.fit(xp, xq, **kwargs)

    def wasserstein(self, P: torch.Tensor, Q: torch.Tensor) -> torch.Tensor:
        """
        Calculates the Wasserstein distance contribution from each datum in the
        batch of generated samples Q.
        Input:
            P: a reference dataset of shape BN, where B is the batch size and
                N is the input dimension into the critic function.
            Q: a dataset of generated samples of shape BN, where B is the batch
                size and N is the input dimension into the critic function.
        Returns:
            The Wasserstein distance contribution from each datum in Q.
        """
        P = P.reshape(P.size(dim=0), -1)
        return torch.mean(self.critic(P, flatten=False)) - torch.squeeze(
            self.critic(Q, flatten=False), dim=-1
        )

    def alpha(self, **kwargs) -> torch.Tensor:
        """
        Returns the optimal value of alpha according to the dual optimization
        problem.
        Input:
            None.
        Returns:
            The optimal value of alpha.
        """
        if self._alpha is not None:
            return self._alpha
        alpha = torch.from_numpy(np.linspace(0.0, 1.0, num=201))
        if torch.cuda.is_available():
            alpha = alpha.cuda()
        return alpha[torch.argmax(self._score(alpha))]

    def _score(self, alpha: torch.Tensor) -> torch.Tensor:
        """
        Scores alpha values according to the Lagrange dual function g(alpha).
        Input:
            alpha: the particular values of alpha to score.
        Returns:
            g(alpha) as a tensor with the same shape as the alpha input.
        """
        xstar = self._search(alpha).detach()
        return torch.where(
            torch.linalg.norm(xstar, dim=-1) > 0.0,
            ((alpha - 1.0) * self.surrogate(xstar).squeeze(dim=-1)) + (
                alpha * self.wasserstein(self.xp, xstar)
            ),
            -np.inf
        )

    def _search(self, alpha: torch.Tensor) -> torch.Tensor:
        """
        Approximates z* for the Lagrange dual function by searching over
        the standard normal distribution.
        Input:
            alpha: values of alpha to find z* for.
        Returns:
            The optimal z* from the sampled latent space points. The returned
            tensor has shape AD, where A is the number of values of alpha
            tested and D is the dimensions of the latent space points.
        """
        alpha = alpha.repeat(self.search_budget, 1).unsqueeze(dim=-1)
        alpha = alpha.permute(1, 0, 2)
        z = self.prior(alpha.size(dim=0))
        if torch.cuda.is_available():
            alpha, z = alpha.cuda(), z.cuda()
        z = z.detach().requires_grad_(True)

        DL = torch.autograd.grad(self(z, alpha).sum(), z)[0]
        DL = torch.linalg.norm(DL, dim=-1) / DL.size(dim=-1)
        norm_DL, idxs = torch.min(DL, dim=-1)
        z = z[torch.arange(z.size(dim=0)).to(idxs), idxs]
        idxs = torch.where(norm_DL < self.norm_thresh, idxs, -1)
        with torch.no_grad():
            for bad_idx in torch.where(idxs < 0)[0]:
                z[bad_idx] = torch.zeros_like(z[bad_idx])
        return z

    def prior(self, batch_size: int) -> torch.Tensor:
        """
        Returns samples over the prior of the latent space distribution.
        Input:
            batch_size: number of batches to sample.
        Returns:
            Samples from the prior of the latent space distribution.
        """
        eps = torch.randn(
            (batch_size, self.search_budget, self.xp.size(dim=-1))
        )
        mu = self.xp.mean(dim=0, keepdim=True).mean(dim=1, keepdim=True)
        sigma = self.xp.std(dim=0, keepdim=True).mean(dim=1, keepdim=True)
        return mu + (sigma * eps.to(sigma))

#!/usr/bin/python3
"""
Defines the explicit dual function for DO-GAMBO.

Author(s):
    Michael Yao @michael-s-yao

Licensed under the MIT License. Copyright University of Pennsylvania 2024.
"""
import numpy as np
import torch
import torch.nn as nn
from typing import Final, Optional


class ExplicitDual(nn.Module):
    def __init__(
        self,
        Xp: torch.Tensor,
        ptau: torch.Tensor,
        critic: nn.Module,
        batch_size: int = 1024,
        dual_step_size: float = 0.001,
        W0: float = 0.0,
        seed: Optional[int] = 0,
        **kwargs
    ):
        """
        Args:
            Xp: the offline reference dataset with shape ND, where N is the
                number of reference designs and D is the number of design
                dimensions.
            ptau: the relative weighting of each design with shape N.
            critic: the fitted source critic function.
            batch_size: the number of datums to use to approximate the
                reference distribution empirically. Default 1024.
            dual_step_size: the step size for dual variable optimization.
            W0: the 1-Wasserstein distance threshold hyperparameter.
                Default 0.0.
            seed: random seed. Default 0.
        """
        super().__init__()
        self._Xp = Xp
        self._ptau = ptau
        self.critic = critic.to(Xp)
        self._W0: Final[float] = W0
        self._batch_size: Final[int] = batch_size
        self._dual_step_size: Final[float] = dual_step_size
        self._seed: Final[Optional[int]] = seed
        self._rng = np.random.default_rng(seed=self._seed)
        self._lambd = torch.ones(1).to(next(self.critic.parameters()))

        for key, val in kwargs.items():
            setattr(self, key, val)

    def forward(self, lambd: torch.Tensor) -> torch.Tensor:
        """
        Computes the value of the explicit dual function.
        Input:
            lambd: a value(s) of lambda to evaluate the dual function at,
                with shape B1 where B is the number of values to evaluate.
        Returns:
            The value(s) of the explicit dual function g(lambd).
        """
        n = self._Xp.size(dim=0)
        idxs = self._rng.choice(n, min(self._batch_size, n), replace=False)

        xp, ptau = self._Xp[idxs], self._ptau[idxs]
        ptau = ptau / ptau.sum()

        g = -1.0 * (ptau * self.fs_star(lambd * self.critic(xp))).sum(dim=0)
        return g + (lambd * ((ptau * self.critic(xp)).sum(dim=0) - self._W0))

    @staticmethod
    def fs_star(v: torch.Tensor) -> torch.Tensor:
        """
        Computes the Fenchel conjugate of the function fs(u) = u log u.
        Input:
            v: the dual input to the Fenchel conjugate.
        Returns:
            The Fenchel conjugate of fs(u) evaluated at the point(s) v.
        """
        return torch.exp(v - 1.0)

    @property
    def lambd(self) -> torch.Tensor:
        """
        Computes the optimal value of the Lagrange multiplier lambda.
        Input:
            None.
        Returns:
            The optimal value of the Lagrange multiplier lambda.
        """
        self._lambd = self._lambd.requires_grad_(True)
        lambd_optimizer = torch.optim.Adam(
            [self._lambd], lr=self._dual_step_size, maximize=True
        )
        dl = -np.inf * torch.ones(1)
        while not torch.isclose(
            torch.abs(dl), torch.zeros_like(dl), atol=1e-4
        ):
            lambd_optimizer.zero_grad()
            self(self._lambd).backward(retain_graph=True)
            dl = self._lambd.grad
            lambd_optimizer.step()
        self._lambd = self._lambd.detach()
        return self._lambd.clone()

    def fit(
        self,
        Xp: torch.Tensor,
        Xq: torch.Tensor,
        qpi: Optional[np.ndarray] = None,
        lr: float = 0.001,
        batch_size: int = 128,
        patience: int = 100,
        **kwargs
    ) -> None:
        """
        Fits the Lipschitz-constrained source critic model.
        Input:
            Xp: a dataset of real reference designs of shape ND, where N is
                the number of designs and D the number of design dimensions.
            Xq: a dataset of generated designs of shape MD, where M is the
                number of designs and D the number of design dimensions.
            qpi: an optional array of shape N specifying the sampling
                probability over the generated designs.
            lr: learning rate. Default 0.001.
            batch_size: batch size. Default 128.
            patience: patience. Default 100.
        Returns:
            None.
        """
        self.critic.fit(
            Xp=Xp,
            Xq=Xq,
            p_sampling_prob=self._ptau.detach().cpu().numpy(),
            q_sampling_prob=qpi,
            lr=lr,
            batch_size=batch_size,
            rng=self._rng,
            patience=patience
        )

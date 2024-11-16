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


class ExplicitDual(nn.Module):
    def __init__(
        self,
        beta: float,
        Xp: torch.Tensor,
        ptau: torch.Tensor,
        critic: nn.Module,
        W0: float,
        batch_size: int,
        seed: int = 0,
        **kwargs
    ):
        """
        Args:
            beta: the relative importance of the KL divergence- based reward.
            Xp: the offline reference dataset with shape ND, where N is the
                number of reference designs and D is the number of design
                dimensions.
            ptau: the relative weighting of each design with shape N.
            critic: the fitted source critic function.
            W0: the 1-Wasserstein distance threshold hyperparameter.
            batch_size: the number of datums to use to approximate the
                reference distribution empirically.
            seed: random seed. Default 0.
        """
        super().__init__()
        self._beta = beta
        self._Xp = Xp
        self._ptau = ptau
        self._critic = critic
        self._W0 = W0
        self._batch_size = batch_size
        self._seed = seed
        self._rng = np.random.default_rng(seed=self._seed)

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

        t1 = self._beta * (ptau * self.fs_star(lambd * self._critic(xp))).sum()
        t2 = ((ptau - 1.0) * self.fs_star(lambd * self._critic(xp))).sum()
        t3 = self._beta * lambd * (ptau * self._critic(xp)).sum()
        t4 = -self._beta * lambd * self._W0
        return t1 + t2 + t3 + t4

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

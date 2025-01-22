#!/usr/bin/python3
"""
Diversity-optimized generative adversarial model-based optimization
(DO-GAMBO) offline objective transform module implementation.

Author(s):
    Michael Yao @michael-s-yao

Licensed under the MIT License. Copyright University of Pennsylvania 2024.
"""
import numpy as np
import torch
import torch.nn.functional as F
from typing import Final, Optional, Sequence

from .base import BaseObjectiveTransform
from ..metrics import KLDivergence, ChiSquaredDivergence
from ..models import ExplicitDual, LipschitzMLP
from ..utils import p_tau_ref


class DOGAMBOTransform(BaseObjectiveTransform):
    name: Final[str] = "DOGAMBOTransform"

    def __init__(
        self,
        surrogate: torch.Tensor,
        xp: torch.Tensor,
        yp: torch.Tensor,
        critic_hidden_dims: Sequence[int] = [512, 512],
        beta: float = 1.0,
        tau: float = 1.0,
        W0: float = 0.0,
        dual_step_size: float = 0.01,
        mixed_chi_squared_weighting: Optional[float] = None,
        ablate_critic: bool = False,
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
            beta: the relative strength of the KL-divergence term. Default 1.0.
            tau: the temperature hyperparameter. Default 1.0.
            W0: the Wasserstein threshold constant. Default 0.0.
            dual_step_size: the step size for solving the dual problem.
                Default 0.01.
            mixed_chi_squared_weighting: the weighting of the Chi-squared
                divergence term. Default not used.
            ablate_critic: whether to turn off source critic feedback.
            seed: optional random seed. Default 0.
        """
        super().__init__(
            surrogate=surrogate,
            critic_hidden_dims=critic_hidden_dims,
            beta=beta,
            tau=tau,
            W0=W0,
            dual_step_size=dual_step_size,
            mixed_chi_squared_weighting=mixed_chi_squared_weighting,
            ablate_critic=ablate_critic,
            seed=seed,
            **kwargs
        )
        self.kld = KLDivergence()
        if self.mixed_chi_squared_weighting is not None:
            self.xsd = ChiSquaredDivergence()

        self.eps: Final[float] = np.finfo(np.float32).eps
        self.xp = xp
        self.yp = yp
        self.g = ExplicitDual(
            Xp=self.xp,
            ptau=p_tau_ref(self.yp, tau=self.tau),
            critic=LipschitzMLP(
                self.xp.size(dim=-1), 1, self.critic_hidden_dims
            ),
            W0=self.W0,
            dual_step_size=self.dual_step_size,
            mixed_chi_squared_weighting=self.mixed_chi_squared_weighting,
            ablate_critic=self.ablate_critic,
            seed=self.seed
        )

    def forward(self, xq: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Forward pass through the DOGAMBO objective transform.
        Input:
            xq: the input batch of designs to the surrogate model.
        Returns:
            The forward model predictions of the input designs.
        """
        yq = self.surrogate(xq)
        yq -= (self.beta / (self.tau + self.eps)) * self.kld(
            xq.mean() - self.xp.mean(),
            torch.log(xq.std().square() + self.xp.std().square())
        )
        gamma = 1.0
        if self.mixed_chi_squared_weighting is not None:
            coeff = self.mixed_chi_squared_weighting * (
                self.beta / (self.tau + self.eps)
            )
            yq -= coeff * self.xsd(
                xq.mean() - self.xp.mean(),
                torch.log(xq.std().square() + self.xp.std().square())
            )
            gamma += self.mixed_chi_squared_weighting
        return yq - self.beta * gamma * self.g.lambd * F.relu(
            self.g.critic(self.xp).mean() - self.g.critic(xq) - self.W0
        )

    def fit(
        self,
        xp: torch.Tensor,
        xq: torch.Tensor,
        qpi: Optional[np.ndarray] = None,
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
        Returns:
            None.
        """
        return self.g.fit(xp, xq, qpi, **kwargs)

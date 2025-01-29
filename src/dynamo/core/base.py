#!/usr/bin/python3
"""
Base class implementation for offline objective transform modules.

Author(s):
    Michael Yao @michael-s-yao

Licensed under the MIT License. Copyright University of Pennsylvania 2024.
"""
import abc
import numpy as np
import torch
import torch.nn as nn
from typing import Final, Optional


class BaseObjectiveTransform(nn.Module, abc.ABC):
    name: str = "BaseObjectiveTransform"

    def __init__(self, surrogate: nn.Module, **kwargs):
        """
        Args:
            surrogate: the original forward surrogate model.
        """
        super().__init__()
        self.surrogate = surrogate
        for key, val in kwargs.items():
            setattr(self, key, val)

    @abc.abstractmethod
    def forward(self, xq: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Forward pass through the objective transform.
        Input:
            xq: the input batch of designs to the surrogate model.
        Returns:
            The offline predictions for the input designs.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def fit(
        self,
        xp: torch.Tensor,
        xq: torch.Tensor,
        qpi: Optional[np.ndarray] = None,
        **kwargs
    ) -> None:
        """
        Performs any model updating/fitting steps.
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
        raise NotImplementedError


class IdentityTransform(BaseObjectiveTransform):
    name: Final[str] = "IdentityTransform"

    def forward(self, xq: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Forward pass through the identity regularizer (i.e., no regularization
        of the offline objective is applied).
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
        Performs any model updating/fitting steps.
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
        return

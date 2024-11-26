#!/usr/bin/python3
"""
Base generative policy class.

Author(s):
    Michael Yao @michael-s-yao

Licensed under the MIT License. Copyright University of Pennsylvania 2024.
"""
import abc
import numpy as np
import torch
import torch.nn as nn
from typing import Optional


class BaseGenerativePolicy(abc.ABC, nn.Module):
    """Base generative policy class."""

    def __init__(
        self,
        batch_size: int,
        sampling_bounds: Optional[torch.Tensor] = None,
        seed: int = 0,
        **kwargs
    ):
        """
        Args:
            batch_size: batch size to use for sampling per iteration.
            sampling_bounds: the (optional) sampling bounds of shape 2D, where
                D is the number of design dimensions.
            seed: random seed. Default 0.
        """
        super().__init__()
        self.batch_size = batch_size
        if self.sampling_bounds:
            self.sampling_bounds = sampling_bounds.double()
            if torch.cuda.is_available():
                self.sampling_bounds = self.sampling_bounds.cuda()
            self.normalized_bounds = torch.zeros_like(self.sampling_bounds)
            self.normalized_bounds[-1] = 1.0
        self.seed = seed
        self._rng = np.random.RandomState(seed=self.seed)

        for key, val in kwargs.items():
            setattr(self, key, val)

    @abc.abstractmethod
    def forward(self, *args, **kwargs) -> torch.Tensor:
        """
        Returns a new batch of candidates to evaluate.
        Input:
            Varies by the generative sampling policy.
        Returns:
            A batch of candidates to evaluate of shape BD, where B is the batch
            size and D is the number of design dimensions.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def fit(self, X: torch.Tensor, y: torch.Tensor) -> None:
        """
        Fits the generative policy and performs any pre-acquisition steps.
        Input:
            X: a tensor of shape ND of all prior evaluated designs, where N is
                the number of designs and D is the number of design dimensions.
            y: a tensor of shape N1 of all objective evaluations, where N is
                the number of designs.
        Returns:
            None.
        """
        raise NotImplementedError

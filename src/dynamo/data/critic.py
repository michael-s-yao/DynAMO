#!/usr/bin/python3
"""
Iterable dataset for training adversarial source critics.

Author(s):
    Michael Yao @michael-s-yao

Licensed under the MIT License. Copyright University of Pennsylvania 2024.
"""
from __future__ import annotations
import numpy as np
import torch
from torch.utils.data import IterableDataset
from typing import Sequence, Tuple, Union


class SourceCriticDataset(IterableDataset):
    def __init__(
        self,
        *args: Sequence[torch.Tensor],
        seed: int = 0,
        batch_size: int = 64
    ):
        """
        Args:
            args: the input dataset(s) to sample from.
            seed: random seed. Default 0.
            batch_size: batch size. Default 64.
        """
        super().__init__()
        self._datasets = args
        assert len(self._datasets) > 0
        assert all([isinstance(X, torch.Tensor) for X in self._datasets])
        self._rng = np.random.default_rng(seed=seed)
        self._batch_size = batch_size

    def __iter__(self) -> SourceCriticDataset:
        """
        Returns the dataset as an iterable.
        Input:
            None.
        Returns:
            The dataset as an iterable.
        """
        return self

    def __next__(self) -> Union[torch.Tensor, Tuple[torch.Tensor]]:
        """
        Returns a batch of datums from the dataset.
        Input:
            None.
        Returns:
            A batch of datums from the dataset.
        """
        samples = []
        for X in self._datasets:
            idxs = self._rng.choice(
                X.size(dim=0), size=self._batch_size, replace=False
            )
            samples.append(X[idxs])

        if len(samples) == 1:
            return samples[0]
        else:
            return tuple(samples)

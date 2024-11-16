#!/usr/bin/python
"""
Pytorch Lightning DataModules for Design-Bench optimization tasks.

Author(s):
    Michael Yao @michael-s-yao

License under the MIT License. Copyright University of Pennsylvania 2024.
"""
import numpy as np
import pytorch_lightning as pl
import torch
from design_bench.task import Task
from torch.utils.data import Dataset, DataLoader
from typing import Final, NamedTuple, Optional


class DesignBenchBatch(NamedTuple):
    x: torch.Tensor
    y: torch.Tensor


class DesignBenchDataset(Dataset):
    def __init__(self, x: torch.Tensor, y: torch.Tensor):
        """
        Args:
            x: the input designs of shape ND, where N is the number of designs
                and D the number of design dimensions.
            y: the output objectives of shape N1.
        """
        self.x: Final[torch.Tensor] = x
        self.y: Final[torch.Tensor] = y

    def __len__(self) -> int:
        """
        Returns the number of elements in the dataset.
        Input:
            None.
        Returns:
            The number of elements in the dataset.
        """
        return self.y.size(dim=0)

    def __getitem__(self, idx: int) -> DesignBenchBatch:
        """
        Returns a specified datum from the dataset.
        Input:
            idx: the index of the datum to retrieve.
        Returns:
            The specified datum from the dataset.
        """
        return DesignBenchBatch(self.x[idx], self.y[idx])


class DesignBenchDataModule(pl.LightningDataModule):
    def __init__(
        self,
        task: Task,
        val_frac: float = 0.2,
        batch_size: int = 128,
        num_workers: int = 0,
        seed: Optional[int] = 42,
    ):
        """
        Args:
            task: the Design-Bench offline optimization task.
            val_frac: the fraction of data to use for model validation.
            batch_size: batch size. Default 128.
            num_workers: number of workers. Default 0.
            seed: optional random seed. Default 42.
        """
        super().__init__()
        self.task: Final[Task] = task
        self.val_frac: Final[float] = val_frac
        self.batch_size: Final[int] = batch_size
        self.num_workers: Final[int] = num_workers
        self.seed: Final[Optional[int]] = seed
        self.is_discrete: Final[bool] = self.task.is_discrete

        self.x = torch.from_numpy(self.task.x)
        if self.is_discrete:
            self.x = self.x.to(torch.long)
        self.y = torch.from_numpy(self.task.y).double()

    def setup(self, *args, **kwargs) -> None:
        """
        Perform training and validation splits.
        Input:
            None.
        Returns:
            None.
        """
        self.num_val = int(self.val_frac * self.y.size(dim=0))
        idxs = np.arange(self.y.size(dim=0))
        np.random.default_rng(seed=self.seed).shuffle(idxs)

        self.train = DesignBenchDataset(
            self.x[idxs[self.num_val:]], self.y[idxs[self.num_val:]]
        )
        self.val = DesignBenchDataset(
            self.x[idxs[:self.num_val]], self.y[idxs[:self.num_val]]
        )

    def train_dataloader(self) -> DataLoader:
        """
        Returns the dataloader for model training.
        Input:
            None.
        Returns:
            The dataloader for model training.
        """
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True
        )

    def val_dataloader(self) -> DataLoader:
        """
        Returns the dataloader for model validation.
        Input:
            None.
        Returns:
            The dataloader for model validation.
        """
        return DataLoader(
            self.val, batch_size=self.batch_size, num_workers=self.num_workers
        )

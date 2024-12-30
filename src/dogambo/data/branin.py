#!/usr/bin/python3
"""
Implements a toy dataset for the Branin task.

Author(s):
    Michael Yao @michael-s-yao

Licensed under the MIT License. Copyright University of Pennsylvania 2024.
"""
import numpy as np
from datasets import load_dataset
from datasets.exceptions import DatasetNotFoundError
from design_bench.datasets.continuous_dataset import ContinuousDataset


class BraninDataset(ContinuousDataset):
    name: str = "branin/branin"

    x_name: str = "design"

    y_name: str = "score"

    hf_repo_name: str = "michaelsyao/Branin"

    def __init__(self, **kwargs):
        """
        Args:
            None.
        """
        try:
            data = load_dataset(self.hf_repo_name)["train"].to_pandas()
        except DatasetNotFoundError:
            super().__init__(
                np.array([[0.0, 0.0]]), np.array([[0.0]]), **kwargs
            )
            return
        x = data[["x1", "x2"]].to_numpy()
        y = data["y"].to_numpy()[..., np.newaxis]
        super().__init__(x, y, **kwargs)

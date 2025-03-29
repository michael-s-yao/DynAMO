#!/usr/bin/python3
"""
Implements a toy dataset for the multiobjective-derived tasks.

Author(s):
    Michael Yao @michael-s-yao

Licensed under the MIT License. Copyright University of Pennsylvania 2024.
"""
import numpy as np
from datasets import load_dataset
from design_bench.datasets.continuous_dataset import ContinuousDataset


class VehicleSafetyDataset(ContinuousDataset):
    name: str = "vehicle_safety/vehicle_safety"

    x_name: str = "designs"

    y_name: str = "scores"

    hf_repo_name: str = "michaelsyao/VehicleSafetyDataset"

    def __init__(self, **kwargs):
        """
        Args:
            None.
        """
        data = load_dataset(self.hf_repo_name)["train"].to_pandas()
        x = np.array([xx for xx in data[self.x_name].to_numpy()])
        y = np.array([yy for yy in data[self.y_name].to_numpy()])
        super().__init__(x, y, **kwargs)


class WeldedBeamDataset(ContinuousDataset):
    name: str = "welded_beam/welded_beam"

    x_name: str = "designs"

    y_name: str = "scores"

    hf_repo_name: str = "michaelsyao/WeldedBeamDataset"

    def __init__(self, **kwargs):
        """
        Args:
            None.
        """
        data = load_dataset(self.hf_repo_name)["train"].to_pandas()
        x = np.array([xx for xx in data[self.x_name].to_numpy()])
        y = np.array([yy for yy in data[self.y_name].to_numpy()])
        super().__init__(x, y, **kwargs)

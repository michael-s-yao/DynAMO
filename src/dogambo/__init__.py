#!/usr/bin/python3
"""
Diversity-optimized generative adversarial model-based optimization (DO-GAMBO).

Author(s):
    Michael Yao @michael-s-yao

Licensed under the MIT License. Copyright University of Pennsylvania 2024.
"""
import numpy as np
import torch


def override_deprecations() -> None:
    """
    Overrides package deprecation issues due to outdated package requirements
    from the design-bench repository.
    Input:
        None.
    Returns:
        None.
    """
    import collections
    import pickle
    from gym.envs import registration as gym_reg

    setattr(np, "bool", bool)
    setattr(np, "float", np.float64)
    setattr(np, "int", np.int32)
    setattr(np, "loads", pickle.loads)
    setattr(
        torch.optim.lr_scheduler,
        "LRScheduler",
        torch.optim.lr_scheduler._LRScheduler
    )

    class Registry:
        env_specs = {}

        def __iter__(self):
            return iter(self.env_specs)

        def __setitem__(self, key, val):
            self.env_specs[key] = val

        def values(self):
            return self.env_specs.values()

    gym_reg.registry = Registry()

    for attr in collections.abc.__all__:
        setattr(collections, attr, getattr(collections.abc, attr))


override_deprecations()


torch.set_default_dtype(torch.float64)


import design_bench
from typing import Any, Callable, Dict, Optional, Union
from . import data, core, embed, models, metrics, utils, optim, oracle


__all__ = [
    "data",
    "core",
    "embed",
    "models",
    "metrics",
    "utils",
    "optim",
    "oracle",
    "register",
    "make"
]


def register(
    task_name: str,
    dataset: Union[str, Callable],
    oracle: Union[str, Callable],
    dataset_kwargs: Optional[Dict[str, Any]] = None,
    oracle_kwargs: Optional[Dict[str, Any]] = None
):
    """
    Calls the Design-Bench `register()` function to register a new task.
    Input:
        task_name: the name of the MBO task.
        dataset: the import path to the target dataset class.
        oracle: the import path to the target oracle class.
        dataset_kwargs: optional additional keyword arguments that are provided
            to the dataset class when it is initialized.
        oracle_kwargs: optional additional keyword arguments that are provided
            to the oracle class when it is initialized.
    Returns:
        None.
    """
    return design_bench.registration.register(
        task_name,
        dataset,
        oracle,
        dataset_kwargs=dataset_kwargs,
        oracle_kwargs=oracle_kwargs
    )


def make(
    task_name: str,
    dataset_kwargs: Optional[Dict[str, Any]] = None,
    oracle_kwargs: Optional[Dict[str, Any]] = None,
    **kwargs
) -> design_bench.task.Task:
    """
    Instantiates the intended task.
    Input:
        task_name: the name of the MBO task.
        dataset_kwargs: optional additional keyword arguments for initializing
            the offline dataset.
        oracle_kwargs: optional additional keyword arguments for initializing
            the offline oracle function.
    Returns:
        The specified offline MBO task.
    """
    return design_bench.registration.make(
        task_name,
        dataset_kwargs=dataset_kwargs,
        oracle_kwargs=oracle_kwargs,
        **kwargs
    )


register(
    "StoryGen-Exact-v0",
    "dogambo.data:StoryGenerationDataset",
    "dogambo.oracle:StoryGenerationOracle",
    dataset_kwargs={
        "max_samples": None,
        "distribution": None,
        "max_percentile": 100.0,
        "min_percentile": 0.0
    },
    oracle_kwargs={}
)


register(
    "PenalizedLogP-Exact-v0",
    "dogambo.data:PenalizedLogPDataset",
    "dogambo.oracle:PenalizedLogPOracle",
    dataset_kwargs={
        "max_samples": None,
        "distribution": None,
        "max_percentile": 50.0,
        "min_percentile": 0.0
    },
    oracle_kwargs={}
)


register(
    "Branin-Exact-v0",
    "dogambo.data:BraninDataset",
    "dogambo.oracle:BraninOracle",
    dataset_kwargs={
        "max_samples": None,
        "distribution": None,
        "max_percentile": 100.0,
        "min_percentile": 0.0
    },
    oracle_kwargs={}
)

#!/usr/bin/python3
"""
Diversity-optimized generative adversarial model-based optimization (DO-GAMBO).

Author(s):
    Michael Yao @michael-s-yao

Licensed under the MIT License. Copyright University of Pennsylvania 2024.
"""
from design_bench.registration import register
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


from . import data, core, embed, models, metrics, utils, optim, oracle


__all__ = [
    "data", "core", "embed", "models", "metrics", "utils", "optim", "oracle"
]


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

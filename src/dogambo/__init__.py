#!/usr/bin/python3
"""
Diversity-optimized generative adversarial model-based optimization (DO-GAMBO).

Author(s):
    Michael Yao @michael-s-yao

Licensed under the MIT License. Copyright University of Pennsylvania 2024.
"""
import collections
import numpy as np
import pickle
import torch
from gym.envs import registration as gym_reg


def override_deprecations() -> None:
    """
    Overrides package deprecation issues due to outdated package requirements
    from the design-bench repository.
    Input:
        None.
    Returns:
        None.
    """
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
    return


override_deprecations()


torch.set_default_dtype(torch.float64)


from . import data, models, utils, policy  # noqa


__all__ = ["data", "models", "utils", "policy"]

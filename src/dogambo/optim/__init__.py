#!/usr/bin/python3
"""
Generative sampling policy implementations.

Author(s):
    Michael Yao @michael-s-yao

Licensed under the MIT License. Copyright University of Pennsylvania 2024.
"""
from typing import Sequence
from .base import BaseGenerativePolicy
from .qei import qEIPolicy
from .qucb import qUCBPolicy
from .state import OptimizerState
from .grad import FirstOrderPolicy, GradientAscentPolicy, AdamAscentPolicy
from .cmaes import CMAESPolicy


__all__ = [
    "BaseGenerativePolicy",
    "qEIPolicy",
    "qUCBPolicy",
    "OptimizerState",
    "FirstOrderPolicy",
    "GradientAscentPolicy",
    "AdamAscentPolicy",
    "CMAESPolicy",
    "get_optimizers"
]


def get_optimizers() -> Sequence[str]:
    """
    Returns a list of the implemented optimizer algorithms.
    Input:
        None.
    Returns:
        A list of the implemented optimizer algorithms.
    """
    return list(
        filter(
            lambda module: module != "BaseGenerativePolicy" and (
                module.endswith("Policy")
            ),
            __all__
        )
    )

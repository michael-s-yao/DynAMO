#!/usr/bin/python3
"""
Core offline objective transform implementations.

Author(s):
    Michael Yao @michael-s-yao

Licensed under the MIT License. Copyright University of Pennsylvania 2024.
"""
from typing import Sequence
from .base import BaseObjectiveTransform, IdentityTransform
from .dogambo_core import DOGAMBOTransform
from .gambo import GAMBOTransform
from .roma import RoMATransform
from .coms import COMsTransform


__all__ = [
    "BaseObjectiveTransform",
    "IdentityTransform",
    "DOGAMBOTransform",
    "GAMBOTransform",
    "RoMATransform",
    "COMsTransform",
    "get_transforms"
]


def get_transforms() -> Sequence[str]:
    """
    Returns a list of the implemented forward model transforms.
    Input:
        None.
    Returns:
        A list of the implemented forward surrogate model transforms.
    """
    return list(
        filter(
            lambda module: module != "BaseObjectiveTransform" and (
                module.endswith("Transform")
            ),
            __all__
        )
    )

#!/usr/bin/python3
"""
Custom oracle function(s) for new Design-Bench tasks.

Author(s):
    Michael Yao @michael-s-yao

Licensed under the MIT License. Copyright University of Pennsylvania 2024.
"""
from .llm import StoryGenerationOracle
from .molecule import PenalizedLogPOracle
from .branin import BraninOracle


__all__ = [
    "StoryGenerationOracle",
    "PenalizedLogPOracle",
    "BraninOracle"
]

#!/usr/bin/python3
"""
Utility functions for diversity-optimized generative adversarial model-based
optimization (DO-GAMBO).

Author(s):
    Michael Yao @michael-s-yao

Licensed under the MIT License. Copyright University of Pennsylvania 2024.
"""
import design_bench
import numpy as np
import torch
from design_bench.task import Task
from typing import Callable, Union


def p_tau_ref(task: Task, tau: float) -> np.ndarray:
    """
    Constructs the tau-weighted reference distribution from the offline
    reference distribution.
    Input:
        task: the offline optimization task.
        tau: the temperature hyperparameter. Increasing tau is equivalent
            to weighting reward maximization greater than diversity.
    Returns:
        The tau-weighted probabilities for each of the offline designs.
    """
    p = np.exp(-tau * (task.y.max() - task.y.squeeze(axis=-1)))
    return (p / np.sum(p))[..., np.newaxis]


def normalize_y(task: str) -> Callable[
    [Union[torch.Tensor, np.ndarray]], Union[torch.Tensor, np.ndarray]
]:
    """
    Returns a function to min-max normalize objective values.
    Input:
        task: the name of the offline optimization task.
    Returns:
        A callable function to normalize a set of y values.
    """
    task = design_bench.make(
        task, dataset_kwargs={"max_percentile": 100.0, "min_percentile": 0.0}
    )
    return lambda x: (x - task.y.min()) / (task.y.max() - task.y.min())


def unnormalize_y(task: str) -> Callable[
    [Union[torch.Tensor, np.ndarray]], Union[torch.Tensor, np.ndarray]
]:
    """
    Returns a function to min-max un-normalize objective values.
    Input:
        task: the name of the offline optimization task.
    Returns:
        A callable function to un-normalize a set of y values.
    """
    task = design_bench.make(
        task, dataset_kwargs={"max_percentile": 100.0, "min_percentile": 0.0}
    )
    return lambda z: task.y.min() + ((task.y.max() - task.y.min()) * z)

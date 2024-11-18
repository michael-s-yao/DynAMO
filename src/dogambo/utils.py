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
import os
import torch
from pathlib import Path
from typing import Callable, Optional, Union


def p_tau_ref(y: torch.Tensor, tau: float) -> torch.Tensor:
    """
    Constructs the tau-weighted reference distribution from the offline
    reference distribution.
    Input:
        y: a tensor of the reference objective values.
        tau: the temperature hyperparameter. Increasing tau is equivalent
            to weighting reward maximization greater than diversity.
    Returns:
        The tau-weighted probabilities for each of the offline designs.
    """
    p = torch.exp(-tau * (y.max() - y.squeeze(axis=-1)))
    return (p / torch.sum(p)).unsqueeze(dim=-1)


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


def get_model_ckpt(
    task_name: str, path: Union[Path, str] = "lightning_logs"
) -> Optional[Union[Path, str]]:
    """
    Retrieves the path of the most recent model checkpoint for a task.
    Input:
        task_name: the name of the offline optimization task.
        path: the path to the model checkpoints.
    Returns:
        The path of the most recent model checkpoint for the task. If no
        checkpoint is available, returns None.
    """
    files = []
    for root, _, filenames in os.walk(path):
        for filename in filenames:
            if filename != f"{task_name}.ckpt":
                continue
            filepath = os.path.join(root, filename)
            files.append((filepath, os.path.getmtime(filepath)))

    files.sort(key=lambda x: x[1], reverse=True)
    if len(files) == 0:
        return None
    return files[0][0]

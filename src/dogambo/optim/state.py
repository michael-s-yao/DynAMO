#!/usr/bin/python3
"""
Object to track the optimization state.

Author(s):
    Michael Yao @michael-s-yao

Licensed under the MIT License. Copyright University of Pennsylvania 2024.
"""
import logging
import numpy as np
import os
import torch
from datetime import datetime
from design_bench.task import Task
from pathlib import Path
from typing import Final, Optional, Union

from ..models.joint import EncDecPropModule


class OptimizerState:
    def __init__(
        self,
        task_name: str,
        task: Task,
        model: EncDecPropModule,
        savedir: Optional[Union[Path, str]] = None,
        num_restarts: int = 2,
        patience: int = 10,
        logger: Optional[logging.Logger] = None,
        **kwargs
    ):
        """
        Args:
            task_name: the name of the offline optimization task.
            task: the offline optimization task.
            model: the trained joint VAE and surrogate model.
            savedir: the directory to save the optimization results to.
            num_restarts: the number of allowed restarts. Default 2.
            patience: patience before restarting. Default 10.
            logger: an optional logger specification.
        """
        self.task_name: Final[str] = task_name
        self.task = task
        self.model: Final[EncDecPropModule] = model
        self.savedir: Final[Optional[Union[Path, str]]] = savedir
        self.max_restarts: Final[int] = num_restarts
        self.num_restarts = 0
        self.patience: Final[int] = patience
        self.num_fails = 0
        self.best_yq = -np.inf
        self.logger = logger

        for key, val in kwargs.items():
            setattr(self, key, val)

        self.xq, self.yq = None, None

    def log(self, xq: torch.Tensor, yq: torch.Tensor) -> None:
        """
        Evaluates and records a set of proposed designs.
        Input:
            xq: a tensor of proposed designs of shape ND, where N is the number
                of proposed designs and D is the number of design dimensions.
            yq: a tensor of corresponding surrogate predictions of shape N1.
        Returns:
            None.
        """
        if self.best_yq > yq.max():
            self.num_fails += 1
        else:
            self.best_yq = yq.max()
            self.num_fails = 0

        if self.num_fails >= self.patience:
            self.num_fails = 0
            self.num_restarts += 1
            self.best_yq = -np.inf
            if self.logger is not None:
                self.logger.info(
                    f"Restart [{self.num_restarts}/{self.max_restarts}]"
                )

        if self.xq is None:
            self.xq = xq.unsqueeze(dim=0)
        else:
            self.xq = torch.cat((self.xq, xq.unsqueeze(dim=0)))

        if self.yq is None:
            self.yq = yq.unsqueeze(dim=0)
        else:
            self.yq = torch.cat((self.yq, yq.unsqueeze(dim=0)))

        if self.logger is not None:
            self.logger.info(
                f"Best Observed Prediction: {self.best_yq:.3f}"
            )
            self.logger.info(f"Number of Samples: {torch.numel(self.yq)}")

    @property
    def designs(self) -> torch.Tensor:
        """
        Returns a tensor of all of the previously sampled designs.
        Input:
            None.
        Returns:
            A tensor of all the previously sampled designs of shape ND, where N
            is the number of previously sampled designs and D the number of
            design dimensions.
        """
        return self.xq.reshape(-1, self.xq.size(dim=-1))

    @property
    def predictions(self) -> torch.Tensor:
        """
        Returns a tensor of all of the previous surrogate predictions.
        Input:
            None.
        Returns:
            A tensor of all the previous surrogate predictions of shape N1,
            where N is the number of previously sampled designs.
        """
        return self.yq.reshape(-1, 1)

    @property
    def has_converged(self) -> bool:
        """
        Returns whether the optimization has converged.
        Input:
            None.
        Returns:
            Whether the optimization has converged.
        """
        return (self.num_restarts >= self.max_restarts) and (
            self.num_fails >= self.patience
        )

    def save(self) -> None:
        """
        Saves all of the recorded designs and scores.
        Input:
            None.
        Returns:
            None.
        """
        if self.savedir is None:
            return
        elif len(self.savedir) > 0 and self.savedir != ".":
            os.makedirs(self.savedir, exist_ok=True)
            np.savez(
                os.path.join(
                    self.savedir,
                    "{task_name}-{date:%Y-%m-%d_%H:%M:%S}.npz".format(
                        task_name=self.task_name, date=datetime.now()
                    )
                ),
                designs=self.xq.detach().cpu().numpy(),
                predictions=self.yq.detach().cpu().numpy(),
                scores=np.stack(self.scores)
            )

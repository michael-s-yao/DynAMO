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
from math import isclose
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
        max_samples_per_restart: Optional[int] = 2048,
        logger: Optional[logging.Logger] = None,
        seed: Optional[int] = None,
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
            max_samples_per_restart: maximum samples per restart.
                Default 2048.
            logger: an optional logger specification.
            seed: optional random seed. Default None.
        """
        self.task_name: Final[str] = task_name
        self.task = task
        self.model: Final[EncDecPropModule] = model
        self.savedir: Final[Optional[Union[Path, str]]] = savedir
        self.max_restarts: Final[int] = num_restarts
        self.num_restarts = 0
        self.patience: Final[int] = patience
        self.max_samples_per_restart: Final[Optional[int]] = (
            max_samples_per_restart
        )
        self.num_fails = 0
        self.best_yq = -np.inf
        self.logger = logger
        self.seed = seed

        for key, val in kwargs.items():
            setattr(self, key, val)

        self.xq, self.yq = None, None
        self.scores = []

        self.curr_xq, self.curr_yq = None, None
        self.curr_scores = []

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
        if self.curr_xq is None:
            self.curr_xq = xq.unsqueeze(dim=0)
        else:
            self.curr_xq = torch.cat((self.curr_xq, xq.unsqueeze(dim=0)))

        if self.curr_yq is None:
            self.curr_yq = yq.unsqueeze(dim=0)
        else:
            self.curr_yq = torch.cat((self.curr_yq, yq.unsqueeze(dim=0)))

        self.curr_scores.append(self.__predict(xq)[np.newaxis])

        if self.best_yq > yq.max() or isclose(self.best_yq, yq.max()):
            self.num_fails += 1
        else:
            self.best_yq = yq.max()
            self.num_fails = 0

        if self.logger is not None:
            self.logger.info(
                f"Best Observed Prediction: {self.best_yq:.3f}"
            )
            self.logger.info(f"Number of Samples: {torch.numel(self.curr_yq)}")

        if (self.num_fails < self.patience) and (
            self.max_samples_per_restart is None or
            torch.numel(self.curr_yq) < self.max_samples_per_restart
        ):
            return
        self.num_fails = 0
        self.num_restarts += 1

        if self.xq is None:
            self.xq, self.curr_xq = self.curr_xq, None
        else:
            self.xq, self.curr_xq = torch.cat((self.xq, self.curr_xq)), None

        if self.yq is None:
            self.yq, self.curr_yq = self.curr_yq, None
        else:
            self.yq, self.curr_yq = torch.cat((self.yq, self.curr_yq)), None

        self.scores = self.scores + self.curr_scores
        self.curr_scores = []

        self.best_yq = -np.inf
        if self.logger is not None and not self.has_converged:
            self.logger.info(
                f"Restart [{self.num_restarts}/{self.max_restarts}]"
            )

    @property
    def designs(self) -> torch.Tensor:
        """
        Returns a tensor of all of the previously sampled designs after the
        most recent restart.
        Input:
            None.
        Returns:
            A tensor of all the previously sampled designs of shape ND, where N
            is the number of previously sampled designs and D the number of
            design dimensions.
        """
        if self.curr_xq is None:
            return self.curr_xq
        return self.curr_xq.reshape(-1, self.curr_xq.size(dim=-1))

    @property
    def predictions(self) -> Optional[torch.Tensor]:
        """
        Returns a tensor of all of the previous surrogate predictions.
        Input:
            None.
        Returns:
            A tensor of all the previous surrogate predictions of shape N1,
            where N is the number of previously sampled designs.
        """
        if self.curr_yq is None:
            return self.curr_yq
        return self.curr_yq.reshape(-1, 1)

    @property
    def has_converged(self) -> bool:
        """
        Returns whether the optimization has converged.
        Input:
            None.
        Returns:
            Whether the optimization has converged.
        """
        return self.num_restarts > self.max_restarts

    def __predict(self, xq: torch.Tensor) -> np.ndarray:
        """
        Evaluates a design(s) according to the true oracle function.
        Input:
            xq: a tensor of proposed designs of shape ND, where N is the number
                of proposed designs and D is the number of design dimensions.
        Returns:
            An array of predictions of shape N1.
        """
        if self.task.is_discrete:
            device = next(self.model.vae.parameters()).device
            xq = self.model.vae.sample(z=xq.to(device)).detach().cpu().numpy()
            xq = xq[..., 1:]  # Remove start token.
        else:
            xq = self.task.denormalize_x(xq.detach().cpu().numpy())
        y = self.task.predict(xq)
        if y.ndim < 2:
            y = y[..., np.newaxis]
        return y

    def save(self, **kwargs) -> None:
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

        assert self.curr_xq is None and self.curr_yq is None
        np.savez(
            os.path.join(
                self.savedir,
                "{task_name}-{date:%Y-%m-%d_%H:%M:%S}-{seed}.npz".format(
                    task_name=self.task_name,
                    date=datetime.now(),
                    seed=self.seed
                )
            ),
            designs=self.xq.detach().cpu().numpy(),
            predictions=self.yq.detach().cpu().numpy(),
            scores=np.concatenate(self.scores),
            **kwargs
        )

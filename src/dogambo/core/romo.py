#!/usr/bin/python3
"""
Retrieval-enhanced offline model-based optimization (ROMO) objective
transform implementation.

Author(s):
    Michael Yao @michael-s-yao

Citation(s):
    [1] Chen M, Zhao H, Zhao Y, Fan H, Gao H, Yu Y, Tian Z. ROMO: Retrieval-
        enhanced offline model-based optimization. Proc DAI 10: 1-9. (2023).
        doi: 10.1145/3627676.3627685

Licensed under the MIT License. Copyright University of Pennsylvania 2024.
"""
import numpy as np
import os
import sys
import torch
import torch.nn as nn
from design_bench.task import Task
from pathlib import Path
from torch.utils.data import DataLoader
from typing import Optional

from .base import BaseObjectiveTransform
from ..utils import get_task_name_from_task


class ROMOTransform(BaseObjectiveTransform):
    """
    Usage Notes:
        [1] The path to the directory of ROMO-pretrained models should be
            specified using the ROMO_MODELDIR environmental variable. By
            default, it is set to `~/ROMO/design-bench`.
        [2] We use the ROMO model training scripts provided by the original
            authors at github.com/cmciris/ROMO/blob/main/hartmann/romo.py.
            The forward surrogate model should be saved according to the
            following directory structure:
                ROMO_MODELDIR
                |-- task_name (e.g., TFBind8-Exact-v0)
                    |-- model (i.e., the saved PyTorch forward surrogate
                        model from a call to `torch.save()`.)
            The entire model (including the architecture itself - not just
            the model weights) should be saved after training here at
                - github.com/cmciris/ROMO/blob/main/design-bench/romo.py#L215
    """

    name: str = "ROMOTransform"

    def __init__(
        self,
        surrogate: nn.Module,
        vae: nn.Module,
        task: Task,
        pool: DataLoader,
        _beta: float = 0.5,
        **kwargs
    ):
        """
        Args:
            surrogate: the original forward surrogate model.
            vae: the original VAE model.
            task: the offline optimization task.
            pool: the dataset of prior examples [x, y] to use for retrieval
                enhancement.
            _beta: the relative weighting of the original forward surrogate
                model predictions compared to the weight of the retrieval-
                enhanced network. Default 0.5.
        """
        assert 0.0 <= _beta <= 1.0
        kwargs.pop("beta", None)
        super().__init__(
            surrogate=surrogate,
            vae=vae,
            task=task,
            pool=pool,
            beta=_beta,
            **kwargs
        )

        x, y = [], []
        for _x, _y in pool:
            if self.task.is_discrete:
                _x = self.task.to_logits(_x.detach().cpu().numpy())
                _x = torch.from_numpy(_x).reshape(-1, _x.shape[1], _x.shape[2])
                x.append(_x.to(_y))
            else:
                x.append(_x)
            y.append(_y)
        self.pool = [torch.cat(x), torch.cat(y)]

        task_name = get_task_name_from_task(task)
        assert task_name is not None
        surrogate_path = os.path.join(
            os.environ.get(
                "ROMO_MODELDIR", Path.home() / "ROMO" / "design-bench"
            ),
            task_name,
            "model"
        )
        sys.path.append(os.path.dirname(os.path.dirname(surrogate_path)))
        self.romo_surrogate = torch.load(surrogate_path)

    def forward(self, xq: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Forward pass through the identity regularizer (i.e., no regularization
        of the offline objective is applied).
        Input:
            xq: the input batch of designs to the surrogate model.
        Returns:
            The forward model predictions of the input designs.
        """
        if self.task.is_discrete:
            design = self.vae.sample(z=xq).detach().cpu().numpy()
            design = torch.from_numpy(self.task.to_logits(design[..., 1:]))
            design = design.to(xq)
        else:
            design = xq
        self.pool = [tsr.to(xq) for tsr in self.pool]

        y = self.surrogate(xq)
        param = next(self.romo_surrogate.forward_model.parameters())
        xr = self.romo_surrogate.search_engine(design, self.pool).to(param)
        romo_y, _ = self.romo_surrogate.forward_model(design.to(param), xr)
        return (self.beta * y) + ((1.0 - self.beta) * romo_y.to(xq))

    def fit(
        self,
        xp: torch.Tensor,
        xq: torch.Tensor,
        qpi: Optional[np.ndarray] = None,
        **kwargs
    ) -> None:
        """
        Performs any model updating/fitting steps.
        Input:
            Xp: a dataset of real reference designs of shape ND, where N is
                the number of designs and D the number of design dimensions.
            Xq: a dataset of generated designs of shape MD, where M is the
                number of designs and D the number of design dimensions.
            qpi: an optional array of shape N specifying the sampling
                probability over the generated designs.
        Returns:
            None.
        """
        return

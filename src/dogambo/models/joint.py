#!/usr/bin/python3
"""
Joint VAE-surrogate PyTorch Lightning Module.

Author(s):
    Michael Yao @michael-s-yao

Licensed under the MIT License. Copyright University of Pennsylvania 2024.
"""
from __future__ import annotations
import design_bench
import lightning.pytorch as pl
import os
import torch
import torch.nn as nn
import torch.optim as optim
import tensorflow as tf
from design_bench.task import Task
from typing import Any, Dict, Final, Optional
from tensorflow.keras.models import load_model
from pathlib import Path
from typing import Union

from ..data import DesignBenchBatch
from .mlp import MLP
from .vae import IdentityVAE, InfoTransformerVAE
from ..metrics import KLDivergence


class EncDecPropModule(pl.LightningModule):
    """Joint Encoder-Decoder-PropertyPredictor Model."""

    def __init__(
        self,
        task: Task,
        vae_kwargs: Optional[Dict[str, Any]] = {},
        mlp_kwargs: Optional[Dict[str, Any]] = {},
        lr: float = 0.0003,
        alpha: float = 1e-4,
        beta: float = 1.0,
        **kwargs
    ):
        """
        Args:
            task: the Design-Bench offline optimization task.
            vae_kwargs: keyword arguments for instantiating the VAE.
            mlp_kwargs: keyword arguments for instantiating the MLP surrogate.
            lr: learning rate. Default 0.0003.
            alpha: relative weighting of the KL Divergence loss term.
            beta: relative weighting of the surrogate predictor loss term.
        """
        super().__init__()
        self.task: Final[Task] = task
        self.vae_kwargs: Final[Optional[Dict[str, Any]]] = vae_kwargs
        self.mlp_kwargs: Final[Optional[Dict[str, Any]]] = mlp_kwargs
        self.lr: Final[float] = lr
        self.alpha: Final[float] = alpha
        self.beta: Final[float] = beta

        if self.task.is_discrete:
            self.vae = InfoTransformerVAE(
                self.task.num_classes,
                max_string_length=self.task.input_size,
                **self.vae_kwargs
            )
            self.recon_loss = nn.CrossEntropyLoss()
            self.kld_loss = KLDivergence()
        else:
            self.vae = IdentityVAE(self.task.input_size)

        self.surrogate = MLP(
            self.vae.latent_size, 1, **self.mlp_kwargs
        )
        self.surrogate_loss = nn.MSELoss()

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the joint surrogate-VAE model.
        Input:
            x: the input to the joint surrogate-VAE model.
        Returns:
            A dictionary with the final and intermediate model outputs.
        """
        z, mu, logvar = self.vae.encode(x)
        logits = self.vae.decode(z, tokens=x)
        z = z.reshape(-1, self.vae.latent_size).to(
            next(self.surrogate.parameters()).dtype
        )
        ypred = self.surrogate(z)
        return {
            "x": x,
            "z": z,
            "mu": mu,
            "logvar": logvar,
            "logits": logits,
            "y": ypred
        }

    def training_step(
        self, batch: DesignBenchBatch, batch_idx: int
    ) -> torch.Tensor:
        """
        Implements a single model training step.
        Input:
            batch: an input training batch.
            batch_idx: the index of the input training batch.
        Returns:
            The training loss averaged over the batch.
        """
        out = self(batch.x)
        mse_loss = self.surrogate_loss(out["y"], batch.y)
        if not self.task.is_discrete:
            self.log(
                "train_loss",
                mse_loss,
                prog_bar=True,
                sync_dist=True,
                batch_size=batch.y.size(dim=0)
            )
            return mse_loss
        recon_loss = self.recon_loss(
            out["logits"].reshape(-1, out["logits"].size(dim=-1)),
            batch.x.flatten()
        )
        kld_loss = self.kld_loss(out["mu"], out["logvar"])
        vae_loss = recon_loss + (self.alpha * kld_loss)
        train_loss = vae_loss + (self.beta * mse_loss)
        self.log(
            "train_loss",
            train_loss,
            prog_bar=True,
            sync_dist=True,
            batch_size=batch.y.size(dim=0)
        )
        return train_loss

    def validation_step(
        self, batch: DesignBenchBatch, batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        """
        Implements a single model validation step.
        Input:
            batch: an input validation batch.
            batch_idx: the index of the input validation batch.
        Returns:
            A map of the validation metrics averaged over the batch.
        """
        batch_size = batch.y.size(dim=0)
        out = self(batch.x)
        mse_loss = self.surrogate_loss(out["y"], batch.y)
        if not self.task.is_discrete:
            self.log(
                "val_loss",
                mse_loss,
                sync_dist=True,
                batch_size=batch_size
            )
            return {"mse_loss": mse_loss, "val_loss": mse_loss}
        self.log(
            "mse_loss",
            mse_loss,
            prog_bar=True,
            sync_dist=True,
            batch_size=batch_size
        )
        recon_loss = self.recon_loss(
            out["logits"].reshape(-1, out["logits"].size(dim=-1)),
            batch.x.flatten()
        )
        kld_loss = self.kld_loss(out["mu"], out["logvar"])
        self.log(
            "kld_loss",
            kld_loss,
            prog_bar=True,
            sync_dist=True,
            batch_size=batch_size
        )
        self.log(
            "recon_loss",
            recon_loss,
            prog_bar=True,
            sync_dist=True,
            batch_size=batch_size
        )
        vae_loss = recon_loss + (self.alpha * kld_loss)
        val_loss = vae_loss + (self.beta * mse_loss)
        self.log(
            "val_loss",
            val_loss,
            sync_dist=True,
            batch_size=batch_size
        )
        return {
            "mse_loss": mse_loss,
            "recon_loss": recon_loss,
            "kld_loss": kld_loss,
            "vae_loss": vae_loss,
            "val_loss": val_loss
        }

    def test_step(
        self, batch: DesignBenchBatch, batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        """
        Implements a single model testing step.
        Input:
            batch: an input test batch.
            batch_idx: the index of the input test batch.
        Returns:
            A map of the test metrics averaged over the batch.
        """
        out = self(batch.x)
        mse_loss = self.surrogate_loss(out["y"], batch.y)
        if not self.task.is_discrete:
            return {"mse_loss": mse_loss, "val_loss": mse_loss}
        recon_loss = self.recon_loss(
            out["logits"].reshape(-1, out["logits"].size(dim=-1)),
            batch.x.flatten()
        )
        kld_loss = self.kld_loss(out["mu"], out["logvar"])
        vae_loss = recon_loss + (self.alpha * kld_loss)
        return {
            "mse_loss": mse_loss,
            "recon_loss": recon_loss,
            "kld_loss": kld_loss,
            "vae_loss": vae_loss,
            "test_loss": vae_loss + (self.beta * mse_loss)
        }

    def configure_optimizers(self) -> optim.Optimizer:
        """
        Configures and return the model optimizer.
        Input:
            None.
        Returns:
            The configured model optimizer.
        """
        params = list(self.surrogate.parameters())
        if self.task.is_discrete:
            params += list(self.vae.parameters())
        return optim.Adam(params, lr=self.lr)

    @classmethod
    def load_from_RoMA_checkpoint(
        cls, ckpt_fn: Union[Path, str], task: Task, **kwargs
    ) -> EncDecPropModule:
        """
        Loads the RoMA pretrained model. Importantly, using the RoMA model
        (written in TensorFlow) means that gradient information will not be
        available, and this method should only be used with non-first-order
        optimization methods.
        Input:
            task: the Design-Bench offline optimization task.
            ckpt_fn: the checkpoint of the original VAE-surrogate model.
        Returns:
            The loaded model.
        Notes:
            [1] The path to the directory of RoMA-pretrained models should be
                specified using the ROMA_MODELDIR environmental variable. By
                default, it is set to `~/RoMA`.
            [2] We use the RoMA model training scripts provided by the original
                authors at https://github.com/sihyun-yu/RoMA. Both the VAE
                encoder (only for discrete tasks) and forward surrogate models
                should be saved according to the following directory structure:
                    ROMA_MODELDIR
                    |-- task_name (e.g., TFBind8-Exact-v0)
                        |-- model (i.e., the saved TensorFlow forward surrogate
                            model directory.)
                        |-- encoder (i.e., the saved TensorFlow VAE encoder.)
                The VAE encoder should be saved after training here:
                    - github.com/sihyun-yu/RoMA/
                        design_baselines/safeweight_latent/__init__.py#L125
                The model should be saved after training here:
                    - github.com/sihyun-yu/RoMA/
                        design_baselines/safeweight_latent/__init__.py#L198
                    - github.com/sihyun-yu/RoMA/
                        design_baselines/safeweight/__init__.py#L142
        """
        model = cls.load_from_checkpoint(ckpt_fn, task=task, **kwargs)

        task_spec = filter(
            lambda x: isinstance(x.dataset, str) and (
                x.dataset.split(":")[-1] in str(type(task.dataset))
            ),
            design_bench.registry.all()
        )
        task_spec = filter(
            lambda x: isinstance(x.oracle, str) and (
                x.oracle.split(":")[-1] in str(type(task.oracle))
            ),
            task_spec
        )
        task_name = next(task_spec).task_name
        surrogate_path = os.path.join(
            os.environ.get("ROMA_MODELDIR", Path.home() / "RoMA"),
            task_name,
            "model"
        )
        surrogate = load_model(surrogate_path)
        surrogate.compile()
        if task.is_discrete:
            encoder_path = os.path.join(
                os.environ.get("ROMA_MODELDIR", Path.home() / "RoMA"),
                task_name,
                "encoder"
            )
            encoder = load_model(encoder_path)
            encoder.compile()

        class Wrapper(nn.Module):
            def __init__(self, *args, **kwargs):
                super().__init__()
                self.dummy = nn.Linear(
                    1, 1, device=next(model.parameters()).device
                )

            def forward(self, X: torch.Tensor) -> torch.Tensor:
                if model.task.is_discrete:
                    design = model.vae.sample(z=X).detach().cpu().numpy()
                    design = design[..., 1:]
                    design, _ = tf.split(encoder(design), 2, axis=-1)
                else:
                    design = X.detach().cpu().numpy()
                y = torch.from_numpy(surrogate(design).numpy()[..., 0])
                return y.to(X)

        model.surrogate = Wrapper()
        return model

    @classmethod
    def load_from_COMs_checkpoint(
        cls, ckpt_fn: Union[Path, str], task: Task, **kwargs
    ) -> EncDecPropModule:
        """
        Loads the COMs pretrained model. Importantly, using the COMs model
        (written in TensorFlow) means that gradient information will not be
        available, and this method should only be used with non-first-order
        optimization methods.
        Input:
            task: the Design-Bench offline optimization task.
            ckpt_fn: the checkpoint of the original VAE-surrogate model.
        Returns:
            The loaded model.
        Notes:
            [1] The path to the directory of COMs-pretrained models should be
                specified using the COMS_MODELDIR environmental variable. By
                default, it is set to `~/design-baselines`.
            [2] We use the COMs model training scripts provided by the original
                authors at https://github.com/brandontrabucco/design-baselines.
                Both the VAE encoder (only for discrete tasks) and forward
                surrogate models should be saved according to the following
                directory structure:
                    COMS_MODELDIR
                    |-- task_name (e.g., TFBind8-Exact-v0)
                        |-- model (i.e., the saved TensorFlow forward surrogate
                            model directory.)
                        |-- encoder (i.e., the saved TensorFlow VAE encoder.)
                The VAE encoder should be saved after training here:
                    - github.com/brandontrabucco/design-baselines/
                        design_baselines/coms_cleaned/__init__.py#L266
                The model should be saved after training here:
                    - github.com/brandontrabucco/design-baselines/
                        design_baselines/coms_cleaned/__init__.py#L303
        """
        model = cls.load_from_checkpoint(ckpt_fn, task=task, **kwargs)

        task_spec = filter(
            lambda x: isinstance(x.dataset, str) and (
                x.dataset.split(":")[-1] in str(type(task.dataset))
            ),
            design_bench.registry.all()
        )
        task_spec = filter(
            lambda x: isinstance(x.oracle, str) and (
                x.oracle.split(":")[-1] in str(type(task.oracle))
            ),
            task_spec
        )
        task_name = next(task_spec).task_name
        surrogate_path = os.path.join(
            os.environ.get("COMS_MODELDIR", Path.home() / "design-baselines"),
            task_name,
            "model"
        )
        surrogate = load_model(surrogate_path)
        surrogate.compile()

        class Wrapper(nn.Module):
            def __init__(self, *args, **kwargs):
                super().__init__()
                self.dummy = nn.Linear(
                    1, 1, device=next(model.parameters()).device
                )

            def forward(self, X: torch.Tensor) -> torch.Tensor:
                if model.task.is_discrete:
                    design = model.vae.sample(z=X).detach().cpu().numpy()
                    design = model.task.to_logits(design[..., 1:])
                else:
                    design = X.detach().cpu().numpy()
                y = torch.from_numpy(surrogate(design).numpy()[..., 0])
                return y.to(X)

        model.surrogate = Wrapper()
        return model

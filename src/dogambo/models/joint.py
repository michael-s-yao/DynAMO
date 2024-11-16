#!/usr/bin/python3
"""
Joint VAE-surrogate PyTorch Lightning Module.

Author(s):
    Michael Yao @michael-s-yao

Licensed under the MIT License. Copyright University of Pennsylvania 2024.
"""
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
from design_bench.task import Task
from typing import Any, Dict, Final, Optional

from ..data import DesignBenchBatch
from .mlp import MLP
from .vae import IdentityVAE, InfoTransformerVAE


class KLDLoss(nn.Module):
    def forward(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        return -0.5 * torch.sum(
            1.0 + logvar - torch.pow(mu, 2) - torch.exp(logvar).sum()
        )


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
                self.task.num_classes, **self.vae_kwargs
            )
            self.recon_loss = nn.CrossEntropyLoss()
            self.kld_loss = KLDLoss()
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
        x = x.to(device=next(self.vae.parameters()).device)
        if not self.task.is_discrete:
            return {"x": x, "y": self.surrogate(x)}
        z, mu, logvar = self.vae.encode(x)
        logits = self.vae.decode(z, tokens=x)
        z = z.reshape(-1, self.vae.latent_size)
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
        self.log(
            "mse_loss",
            mse_loss,
            prog_bar=True,
            sync_dist=True,
            batch_size=batch_size
        )
        if not self.task.is_discrete:
            self.log(
                "val_loss",
                mse_loss,
                sync_dist=True,
                batch_size=batch_size
            )
            return {"mse_loss": mse_loss, "val_loss": mse_loss}
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
            "val_loss": vae_loss + (self.beta * mse_loss)
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

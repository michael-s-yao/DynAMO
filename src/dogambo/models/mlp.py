#!/usr/bin/python3
"""
Base and Lipschitz-regularized Multilayer Perceptron (MLP) implementation.

Author(s):
    Michael Yao @michael-s-yao

Licensed under the MIT License. Copyright University of Pennsylvania 2024.
"""
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from typing import Sequence, Optional, Tuple


def Block(
    in_dim: int,
    out_dim: int,
    normalize: bool,
    activation: Optional[str] = None
) -> nn.Module:
    """
    Generates a layer of a network consisting of a linear transformation,
    optional batch normalization, and activation.
    Input:
        in_dim: number of input dimensions.
        out_dim: number of output dimensions.
        normalize: whether to apply batch normalization.
        activation: activation function. One of [`LeakyReLU`, `Tanh`,
            `Sigmoid`, `ReLU`, None].
    Output:
        Layer consisting of a linear transformation, optional batch
            normalization, and activation.
    """
    layer = [nn.Linear(in_dim, out_dim)]

    if normalize:
        layer.append(nn.BatchNorm1d(out_dim))

    if activation is None:
        pass
    elif activation.lower() == "relu":
        layer.append(nn.ReLU(inplace=False))
    elif activation.lower() == "leakyrelu":
        layer.append(nn.LeakyReLU(negative_slope=0.2, inplace=False))
    elif activation.lower() == "gelu":
        layer.append(nn.GELU())
    elif activation.lower() == "tanh":
        layer.append(nn.Tanh())
    elif activation.lower() == "sigmoid":
        layer.append(nn.Sigmoid())
    else:
        raise NotImplementedError(
            "`activation` must be one of [`LeakyReLU`, `Tanh`, `Sigmoid`]."
        )

    return nn.Sequential(*layer)


class WeightClipper:
    """Object to clip the weights of a neural network to a finite range."""

    def __init__(self, c: float = 0.01):
        """
        Args:
            c: weight clipping parameter to clip all weights between [-c, c].
        """
        self.c = c

    def __call__(self, module: nn.Module) -> None:
        """
        Clips the weights of an input neural network to between [-c, c].
        Input:
            module: neural network to clip the weights of.
        Returns:
            None.
        """
        _ = [p.data.clamp_(-self.c, self.c) for p in module.parameters()]


class MLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dims: Sequence[int] = [2048, 2048],
        dropout: float = 0.0,
        final_activation: Optional[str] = None,
        hidden_activation: str = "LeakyReLU",
        use_batch_norm: bool = False
    ):
        """
        Args:
            in_dim: dimensions of input data.
            out_dim: dimensions of mode output.
            hidden_dims: dimensions of the hidden intermediate layers.
            dropout: dropout. Default 0.1.
            final_activation: final activation function. One of [`Sigmoid`,
                `LeakyReLU`, None].
            hidden_activation: hidden activation functions. One of [`ReLU`,
                `LeakyReLU`, `GELU`].
            use_batch_norm: whether to apply batch normalization.
        """
        super().__init__()
        layers, dims = [], [in_dim] + hidden_dims + [out_dim]
        for i in range(len(dims) - 1):
            func = hidden_activation if i < len(dims) - 2 else final_activation
            layers.append(
                Block(
                    dims[i],
                    dims[i + 1],
                    normalize=use_batch_norm,
                    activation=func
                )
            )
            if i < len(dims) - 2 and dropout > 0.0:
                layers.append(nn.Dropout(p=dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the MLP model.
        Input:
            X: input tensor of shape Bx(in_dim), where B is the batch size.
        Returns:
            Output tensor of shape Bx(out_dim), where B is the batch size.
        """
        return self.model(X)


class LipschitzMLP(MLP):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dims: Sequence[int] = [2048, 2048],
        dropout: float = 0.0,
        final_activation: Optional[str] = None,
        hidden_activation: str = "LeakyReLU",
        use_batch_norm: bool = False,
        c: float = 0.01
    ):
        """
        Args:
            in_dim: dimensions of input data.
            out_dim: dimensions of mode output.
            hidden_dims: dimensions of the hidden intermediate layers.
            dropout: dropout. Default 0.0.
            final_activation: final activation function. One of [`Sigmoid`,
                `LeakyReLU`, None].
            hidden_activation: hidden activation functions. One of [`ReLU`,
                `LeakyReLU`, `GELU`].
            use_batch_norm: whether to apply batch normalization.
            c: weight clipping parameter. Default 0.01.
        """
        super().__init__(
            in_dim=in_dim,
            out_dim=out_dim,
            hidden_dims=hidden_dims,
            dropout=dropout,
            final_activation=final_activation,
            hidden_activation=hidden_activation,
            use_batch_norm=use_batch_norm
        )
        self._c = c
        self.clipper = WeightClipper(c=self._c)

    def fit(
        self,
        Xp: torch.Tensor,
        Xq: torch.Tensor,
        p_sampling_prob: Optional[torch.Tensor] = None,
        lr: float = 0.001,
        batch_size: int = 128,
        seed: int = 0,
        patience: int = 100
    ) -> nn.Module:
        """
        Fits the Lipschitz MLP as a source critic model.
        Input:
            Xp: a dataset of real reference designs of shape ND, where N is
                the number of designs and D the number of design dimensions.
            Xq: a dataset of generated designs of shape MD, where M is the
                number of designs and D the number of design dimensions.
            p_sampling_prob: an optional tensor of shape N specifying the
                sampling probability over the real reference designs.
            lr: learning rate. Default 0.001.
            batch_size: batch size. Default 128.
            seed: seed. Default 0.
            patience: patience. Default 100.
        Returns:
            The fitted source critic model.
        """
        self.train()
        cache, Wd = [-float("inf")] * patience, 0.0
        if p_sampling_prob is not None and isinstance(
            p_sampling_prob, torch.Tensor
        ):
            p_sampling_prob = p_sampling_prob.detach().cpu().numpy()
        if p_sampling_prob is not None:
            p_sampling_prob = p_sampling_prob.squeeze()

        def generator():
            while not np.isclose(Wd, min(cache), rtol=1e-3) or Wd < min(cache):
                yield

        optimizer = optim.Adam(self.parameters(), lr=lr)

        rng = np.random.default_rng(seed=(seed + sum([ord(c) for c in "fit"])))
        with tqdm(generator(), desc="Fitting Source Critic") as pbar:
            for _ in pbar:
                optimizer.zero_grad()

                pi = rng.choice(
                    Xp.size(dim=0),
                    min(Xp.size(dim=0), batch_size),
                    replace=False,
                    p=p_sampling_prob
                )
                qi = rng.choice(
                    Xq.size(dim=0),
                    min(Xq.size(dim=0), batch_size),
                    replace=False
                )

                Ecp, Ecq = self(Xp[pi]).mean(), self(Xq[qi]).mean()
                (-Ecp + Ecq + (Ecp + Ecq).abs()).backward()
                optimizer.step()
                self.clipper(self)
                Wd = (Ecp - Ecq).item()
                cache = cache[1:] + [Wd]
                pbar.set_postfix(Wd=Wd)
        self.eval()
        return self

    def K(self, max_iters: float = 100, atol: float = 1e-6) -> float:
        k = 1.0
        for name, A in self.named_parameters():
            if "weight" not in name:
                continue
            x = torch.randn(A.size(dim=1), 1, device=A.device)
            x = x / torch.norm(x)
            ATA = A.T @ A

            for _ in range(max_iters):
                x = ATA @ x
                x = x / torch.norm(x)

                sigma = torch.norm(A @ x)
                if torch.abs(sigma - torch.norm(ATA @ x)) < atol:
                    break
            k = k * sigma
        return k.item()


class MLPLightningModule(pl.LightningModule):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dims: Sequence[int],
        dropout: float = 0.0,
        final_activation: Optional[str] = None,
        hidden_activation: str = "LeakyReLU",
        use_batch_norm: bool = False,
        learning_rate: float = 0.001,
    ):
        """
        Args:
            in_dim: dimensions of input data.
            out_dim: dimensions of mode output.
            hidden_dims: dimensions of the hidden intermediate layers.
            dropout: dropout. Default 0.1.
            final_activation: final activation function. One of [`Sigmoid`,
                `LeakyReLU`, None].
            hidden_activation: hidden activation functions. One of [`ReLU`,
                `LeakyReLU`, `GELU`].
            use_batch_norm: whether to apply batch normalization.
            learning_rate: learning rate.
        """
        super().__init__()
        self.save_hyperparameters()
        self.model = MLP(
            in_dim=self.hparams.in_dim,
            out_dim=self.hparams.out_dim,
            hidden_dims=self.hparams.hidden_dims,
            dropout=self.hparams.dropout,
            final_activation=self.hparams.final_activation,
            hidden_activation=self.hparams.hidden_activation,
            use_batch_norm=self.hparams.use_batch_norm
        )
        self.loss = nn.MSELoss()

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the MLP model.
        Input:
            X: input tensor of shape Bx(in_dim), where B is the batch size.
        Returns:
            Output tensor of shape Bx(out_dim), where B is the batch size.
        """
        return self.model(X)

    def training_step(
        self, batch: Tuple[torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        X, y = batch
        train_loss = self.loss(self.model(X), y)
        self.log("train_loss", train_loss)
        return train_loss

    def validation_step(
        self, batch: Tuple[torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        X, y = batch
        val_loss = self.loss(self.model(X), y)
        self.log("val_loss", val_loss)
        return val_loss

    def configure_optimizers(self) -> optim.Optimizer:
        return optim.Adam(self.model.parameters(), lr=self.hparams.lr)

#!/usr/bin/python3
"""
Metric implementations.

Author(s):
    Michael Yao @michael-s-yao

Licensed under the MIT License. Copyright University of Pennsylvania 2024.
"""
import logging
import numpy as np
import scipy
import torch
import torch.nn as nn
from typing import Optional


class KLDivergence(nn.Module):
    def forward(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Computes the KL Divergence with respect to the multivariate normal
        distribution.
        Input:
            mu: the mean of the input distribution.
            logvar: the logarithm of the variance of the input distribution.
        Returns:
            The KL Divergence.
        """
        return -0.5 * torch.sum(
            1.0 + logvar - torch.pow(mu, 2) - torch.exp(logvar).sum()
        )


def get_logger() -> logging.Logger:
    """
    Returns the logger object associated with the package.
    Input:
        None.
    Returns:
        The logger object associated with the package.
    """
    logger = logging.getLogger("dogambo")
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler())
    return logger


def FID(xq: torch.Tensor, xp: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Computes the FID between a set of generated designs and a set of reference
    real designs. Portions of this code was adapted from the pytorch-fid
    repository by @mseitzer at https://github.com/mseitzer/pytorch-fid
    Input:
        xq: the batch of generated designs of shape ND, where N is the number
            of generated designs and D the number of design dimensions.
        xp: a batch of reference designs of shape MD, where M is the number of
            true reference designs.
        eps: jitter to add if the product between covariances is singular.
    Returns:
        The FID between the set of generated and real designs.
    """
    mu_p = np.atleast_1d(torch.mean(xp, axis=0).detach().cpu().numpy())
    mu_q = np.atleast_1d(torch.mean(xq, axis=0).detach().cpu().numpy())
    covar_p = np.atleast_2d(torch.cov(xp.T).detach().cpu().numpy())
    covar_q = np.atleast_2d(torch.cov(xq.T).detach().cpu().numpy())

    dmu = mu_p - mu_q

    covar_mean, _ = scipy.linalg.sqrtm(covar_p.dot(covar_q), disp=False)
    if not np.isfinite(covar_mean).all():
        offset = np.eye(covar_p.shape[0]) * eps
        covar_mean = scipy.linalg.sqrtm(
            (covar_p + offset).dot(covar_q + offset)
        )

    if np.iscomplexobj(covar_mean):
        if not np.allclose(np.diagonal(covar_mean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covar_mean.imag))
            raise ValueError("Imaginary component {}".format(m))
        covar_mean = covar_mean.real

    return dmu.dot(dmu) + np.trace(covar_p) + np.trace(covar_q) - (
        2.0 * np.trace(covar_mean)
    )


def compute_diversity(
    xq: torch.Tensor, xp: Optional[torch.Tensor] = None, metric: str = ""
) -> torch.Tensor:
    """
    Computes the diversity of a set of generated designs.
    Input:
        xq: the batch of generated designs of shape ND, where N is the number
            of generated designs and D the number of design dimensions.
        xp: an optional batch of reference designs of shape MD, where M is the
            number of true reference designs.
        metric: the diversity metric to use.
    Returns:
        The diversity of the generated designs.
    """
    if metric.lower() == "fid":
        assert xp is not None
        return FID(xq, xp)

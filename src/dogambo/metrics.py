#!/usr/bin/python3
"""
Metric implementations.

Author(s):
    Michael Yao @michael-s-yao

Licensed under the MIT License. Copyright University of Pennsylvania 2024.
"""
import torch
import torch.nn as nn


class KLDivergence(nn.Module):
    def forward(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        return -0.5 * torch.sum(
            1.0 + logvar - torch.pow(mu, 2) - torch.exp(logvar).sum()
        )

#!/usr/bin/python3
"""
Metric implementations.

Author(s):
    Michael Yao @michael-s-yao

Citation(s):
    [1] Kim M, Berto F, Ahn S, Park J. Bootstrapped training of score-
        conditioned generator for offline design of biological sequences.
        Proc NeurIPS 2958: 67643-61. (2023). doi: 10.5555/3666122.3669080

Licensed under the MIT License. Copyright University of Pennsylvania 2024.
"""
import logging
import multiprocess as mp
import torch
import torch.nn as nn
from torchmetrics.text import EditDistance
from tqdm import tqdm
from typing import Optional, Sequence


def get_diversity_metric_options() -> Sequence[str]:
    """
    Returns a list of the implemented diversity metric options.
    Input:
        None.
    Returns:
        A list of the implemented diversity metric options.
    """
    return ["l1-coverage", "pairwise-diversity", "minimum-novelty"]


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


class ChiSquaredDivergence(nn.Module):
    def forward(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Computes the Chi-squared Divergence with respect to the multivariate
        normal distribution.
        Input:
            mu: the mean of the input distribution.
            logvar: the logarithm of the variance of the input distribution.
        Returns:
            The Chi-squared Divergence.
        """
        var = torch.exp(logvar)
        coeff = torch.where(
            var > 0.5,
            var / torch.sqrt(2.0 * var - 1.0),
            1.0 / (torch.sqrt(var * (2.0 - var)))
        )
        arg = torch.square(mu - 1.0) * torch.where(
            var > 0.5,
            2.0 * var - 1.0,
            2.0 - var
        )
        return (coeff * torch.exp(arg)) - 1.0


class NMSELoss(nn.Module):
    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        Computes the normalized mean square error loss (NMSE).
        Input:
            a: an input tensor of size D, where D is the number of design
                dimensions.
            b: an input tensor of size BD, where B is the number of reference
                dimensions.
        Returns:
            The NMSE loss with respect to a of shape B.
        """
        return torch.square(a - b).mean(dim=-1) / torch.square(a).mean()


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


def L1_coverage(xq: torch.Tensor) -> torch.Tensor:
    """
    Computes the L1 coverage (i.e., sum_D max(xq[i]) - min(xq[i]) / D, where
    D is the number of dimensions) of a set of generated designs.
    Input:
        xq: the batch of generated designs of shape ND, where N is the number
            of generated designs and D the number of design dimensions.
    Returns:
        The L1 coverage of the generated designs.
    """
    xq = xq.reshape(xq.size(dim=0), -1)
    return torch.abs(xq.max(dim=0).values - xq.min(dim=0).values).sum() / (
        xq.size(dim=-1)
    )


def pairwise_diversity(xq: torch.Tensor) -> torch.Tensor:
    """
    Computes the average Levenshtein distance between any two unique discrete
    sequences from a dataset of generated designs, or the average L2 distance
    between any two continuous designs in a dataset of generated designs.
    Input:
        xq: the batch of generated designs of shape ND, where N is the number
            of generated designs and D is the sequence length.
    Returns:
        The average Levenshtein distance over the dataset.
    Citation(s):
        [1] Kim M, Berto F, Ahn S, Park J. Bootstrapped training of score-
            conditioned generator for offline design of biological sequences.
            Proc NeurIPS 2958: 67643-61. (2023). doi: 10.5555/3666122.3669080
    """
    metric = NMSELoss()
    if not torch.is_floating_point(xq):
        metric = EditDistance(reduction=None)
        xq = ["".join([chr(65 + x) for x in seq]) for seq in xq]
    else:
        results = [
            metric(xq[i], xq).sum() / (len(xq) - 1)
            for i in tqdm(range(len(xq)))
        ]
        return sum(results) / float(len(xq))

    def novelty(idx: int, q: mp.Queue) -> None:
        return q.put(
            metric([xq[idx] for _ in range(len(xq))], xq).sum() / (len(xq) - 1)
        )

    jobs = []
    q = mp.Queue()
    for i in tqdm(range(len(xq))):
        jobs.append(mp.Process(target=novelty, args=(i, q)))
    for job in jobs:
        job.start()
    for job in jobs:
        job.join()
    results = [q.get() for i in range(len(xq))]
    return sum(results) / float(len(xq))


def minimum_novelty(xq: torch.Tensor, xp: torch.Tensor) -> torch.Tensor:
    """
    Computes the novelty of a set of generated compared to previous designs.
    Input:
        xq: the batch of generated designs of shape ND, where N is the number
            of generated designs and D the number of design dimensions.
        xp: a batch of reference designs of shape MD, where M is the number of
            true reference designs.
    Returns:
        The average minimum novelty of each of the generated designs.
    Citation(s):
        [1] Kim M, Berto F, Ahn S, Park J. Bootstrapped training of score-
            conditioned generator for offline design of biological sequences.
            Proc NeurIPS 2958: 67643-61. (2023). doi: 10.5555/3666122.3669080
    """
    metric = NMSELoss()
    if not torch.is_floating_point(xq):
        metric = EditDistance(reduction=None)
        xq = ["".join([chr(65 + x) for x in seq]) for seq in xq]
        xp = ["".join([chr(65 + x) for x in seq]) for seq in xp]
    else:
        results = [
            metric(xq[i], xp.to(xq)).min() for i in tqdm(range(len(xq)))
        ]
        return sum(results) / float(len(xq))

    def novelty(idx: int, q: mp.Queue) -> None:
        return q.put(metric([xq[idx] for _ in range(len(xp))], xp).min())

    jobs = []
    q = mp.Queue()
    for i in tqdm(range(len(xq))):
        jobs.append(mp.Process(target=novelty, args=(i, q)))
    for job in jobs:
        job.start()
    for job in jobs:
        job.join()
    results = [q.get() for i in range(len(xq))]
    return sum(results) / float(len(xq))


def compute_diversity(
    xq: torch.Tensor, xp: Optional[torch.Tensor] = None, metric: str = ""
) -> Optional[torch.Tensor]:
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
    metric = metric.lower().replace("-", "_")
    if metric.replace("_", "-") not in get_diversity_metric_options():
        raise ValueError(f"Unrecognized evaluation metric {metric}")
    elif metric == "l1_coverage":
        return L1_coverage(xq)
    elif metric == "pairwise_diversity":
        return pairwise_diversity(xq)
    elif metric == "minimum_novelty":
        assert xp is not None
        return minimum_novelty(xq, xp)
    return None

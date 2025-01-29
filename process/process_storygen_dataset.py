#!/usr/bin/python3
"""
Script to construct the StoryGen custom MBO offline dataset.

Author(s):
    Michael Yao @michael-s-yao

Citation(s):
    [1] Lovelace J, Kishore V, Wan C, Shekhtman ES, Weinberger KQ. Latent
        diffusion for language generation. Proc NeurIPS. (2023). URL:
        https://openreview.net/forum?id=NKdtztladR
    [2] Mostafazadeh N, Chambers N, He X, Parikh D, Batra D, Vanderwende L,
        Kohli P, Allen J. A corpus and cloze evaluation for deeper
        understanding of commonsense stories. Proc NAACL: Human Lang Tech:
        839-49. (2016). doi: 10.18653/v1/N16-1098

Licensed under the MIT License. Copyright University of Pennsylvania 2024.
"""
import numpy as np
from dynamo.data import StoryGenerationDataset
from dynamo.models import StoryGenerationOracle, DiffusionLM
from math import ceil
from pathlib import Path
from typing import Union


def main(
    n: int = 1000,
    batch_size: int = 4,
    savepath: Union[Path, str] = "stories.npz"
):
    """Construct the StoryGen custom MBO offline dataset."""
    oracle = StoryGenerationOracle(StoryGenerationDataset())
    model = DiffusionLM.from_pretrained("michaelsyao/DiffusionLM-ROCStories")

    designs, scores = [], []
    for _ in range(ceil(n / batch_size)):
        zp, _ = model.diffusion.sample(batch_size, 64)
        zp = zp[:, :32, :]
        designs.append(zp.detach().cpu().numpy())
        scores.append(oracle.predict(zp))
    np.savez(
        savepath,
        designs=np.concatenate(designs),
        scores=np.concatenate(scores)
    )


if __name__ == "__main__":
    main()

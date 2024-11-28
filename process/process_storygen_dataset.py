#!/usr/bin/python3
import numpy as np
from dogambo.data import StoryGenerationDataset
from dogambo.models import StoryGenerationOracle, DiffusionLM
from math import ceil
from pathlib import Path
from typing import Union


def main(
    n: int = 1000,
    batch_size: int = 4,
    savepath: Union[Path, str] = "stories.npz"
):
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

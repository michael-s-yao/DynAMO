#!/usr/bin/python3
"""
Script to construct the toy Branin custom MBO offline dataset.

Author(s):
    Michael Yao @michael-s-yao

Licensed under the MIT License. Copyright University of Pennsylvania 2024.
"""
import numpy as np
import pandas as pd
import torch
import yaml
from botorch.utils.transforms import unnormalize
from dogambo.oracle import BraninOracle
from dogambo.data import BraninDataset
from pathlib import Path
from typing import Optional, Union


def main(
    n: int = 1000,
    savepath: Union[Path, str] = "branin.parquet",
    bounds_fn: Union[Path, str] = "bounds.yaml",
    seed: Optional[int] = 0
):
    """Construct the custom toy Branin dataset for offline MBO."""
    oracle = BraninOracle(BraninDataset())
    rng = np.random.default_rng(seed)

    with open(bounds_fn) as f:
        bounds = torch.tensor(yaml.safe_load(f)["Branin-Exact-v0"])

    designs = unnormalize(
        torch.from_numpy(rng.uniform(size=(n, 2))), bounds=bounds
    )
    scores = oracle.predict(designs)
    data = pd.DataFrame(
        np.hstack((designs, scores[..., np.newaxis])),
        columns=["x1", "x2", "y"]
    )
    data.to_parquet(savepath)


if __name__ == "__main__":
    main()

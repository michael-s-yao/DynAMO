#!/usr/bin/python3
"""
Implements a story generation dataset based off the ROCStories dataset.

Author(s):
    Michael Yao @michael-s-yao

Licensed under the MIT License. Copyright University of Pennsylvania 2024.
"""
import numpy as np
import pickle
from datasets import load_dataset
from design_bench.datasets.continuous_dataset import ContinuousDataset


class StoryGenerationDataset(ContinuousDataset):
    name: str = "story_generation/story_generation"

    x_name: str = "latent_embedding"

    y_name: str = "flesch_reading_ease_score"

    hf_repo_name: str = "michaelsyao/StoryGen"

    def __init__(self, **kwargs):
        """
        Args:
            None.
        """
        data = load_dataset(self.hf_repo_name)["train"]
        x = np.concatenate([
            pickle.loads(design)[np.newaxis] for design in data["designs"]
        ])
        y = np.array(data["scores"])[..., np.newaxis]
        super().__init__(x, y, **kwargs)

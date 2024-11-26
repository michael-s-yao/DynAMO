#!/usr/bin/python
"""
Oracle function for creative story generation using language models.

Author(s):
    Michael Yao @michael-s-yao

Licensed under the MIT License. Copyright University of Pennsylvania 2024.
"""
import jsonlines
import numpy as np
import os
import torch
from design_bench.oracles.exact_oracle import ExactOracle
from mauve import compute_mauve
from mauve.utils import featurize_tokens_from_model
from transformers import pipeline
from typing import Final, Optional, Set, Union

from .difflm import DiffusionLM, DIFFUSIONLM_PATH
from ..data.stories import StoryGenerationDataset


class StoryGenerationOracle(ExactOracle):
    llm_hf_repo_name: str = "meta-llama/Llama-3.2-3B"

    difflm_hf_repo_name: str = "michaelsyao/DiffusionLM-ROCStories"

    stories_dataset_path: str = os.path.join(
        DIFFUSIONLM_PATH, "datasets/ROCstory/roc_valid.json"
    )

    name: str = "mauve_score"

    def __init__(
        self,
        dataset: StoryGenerationDataset,
        num_reference: int = 64,
        seed: Optional[int] = 0,
        **kwargs
    ):
        """
        Args:
            dataset: the offline dataset for the offline optimization problem.
            num_reference: number of reference stories to use.
            seed: random seed. Default 0.
        """
        super().__init__(
            dataset,
            is_batched=False,
            internal_batch_size=1,
            internal_measurements=1,
            expect_normalized_y=False,
            expect_normalized_x=False,
            expect_logits=None,
            **kwargs
        )
        self.pipe = pipeline(
            "text-generation",
            model=self.llm_hf_repo_name,
            device=(-1 + torch.cuda.is_available()),
            dtype=torch.bfloat16
        )
        self.pipe.tokenizer.pad_token = self.pipe.tokenizer.eos_token
        self.difflm = DiffusionLM.from_pretrained(self.difflm_hf_repo_name)

        self.seed: Final[int] = seed
        self._rng = np.random.default_rng(self.seed)

        with open(self.stories_dataset_path, "r") as f:
            with jsonlines.Reader(f) as reader:
                self.reference_stories = sum(list(reader), [])
        self.reference_stories = self._rng.choice(
            self.reference_stories,
            size=num_reference,
            replace=(num_reference > len(self.reference_stories))
        )
        self.p_features = self.pipe.tokenizer(
            self.reference_stories.tolist(), padding=True
        )
        self.p_features = self.p_features.convert_to_tensors("pt").input_ids
        self.p_features = featurize_tokens_from_model(
            self.pipe.model,
            self.p_features.to(next(self.pipe.model.parameters()).device),
            batch_size=len(self.p_features),
            name="reference_stories"
        )

    @classmethod
    def supported_datasets(cls) -> Set:
        """
        An attribute that defines the set of dataset classes which this oracle
        can be applied to forming a valid ground truth score function for a
        model-based optimization problem.
        Input:
            None.
        Returns:
            An attribute that defines the set of dataset classes which this
            oracle can be applied to.
        """
        return {StoryGenerationDataset}

    @classmethod
    def fully_characterized(cls) -> bool:
        """
        An attribute of whether all possible inputs to the model-based
        optimizaiton problem have been evaluated and are returned via lookup.
        Input:
            None.
        Returns:
            Whether all possible inputs to the model-based optimization
            problem have been evaluated and are returned via lookup.
        """
        return False

    @classmethod
    def is_simulated(cls) -> bool:
        """
        An attribute that defines whether the values returned by the oracle
        were obtained by running a computer simulation rather than performing
        physical experiments with real data.
        Input:
            None.
        Returns:
            Whether the values returned by the oracle were obtained by running
            a computer simulation rather than performing physical experiments
            with real data.
        """
        return True

    def protected_predict(
        self, x: Union[torch.Tensor, np.ndarray]
    ) -> Union[torch.Tensor, np.ndarray]:
        """
        Design scoring function that computes the MAUVE score of a generated
        text output seeded by the input design(s).
        Input:
            x: a batch or single design given as input to the oracle model.
        Returns:
            A batch or single ground-truth prediction made by the oracle model.
        """
        is_np = isinstance(x, np.ndarray)
        if is_np:
            x = torch.from_numpy(x)
        x = x.to(next(self.difflm.parameters()).device)
        text_seeds = self.difflm.decode(self.difflm(x))
        stories = self.pipe(text_seeds)
        scores = []
        for story in stories:
            out = compute_mauve(
                p_feat=self.p_features,
                q_text=story,
                featurize_model_name=self.llm_hf_repo_name
            )
            scores.append(out.mauve)

        if is_np:
            return np.array(scores)
        return torch.tensor(scores).to(x)

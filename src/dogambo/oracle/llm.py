#!/usr/bin/python
"""
Oracle function for creative story generation using language models.

Author(s):
    Michael Yao @michael-s-yao

Licensed under the MIT License. Copyright University of Pennsylvania 2024.
"""
import numpy as np
import torch
from design_bench.oracles.exact_oracle import ExactOracle
from textstat import flesch_reading_ease
from transformers import pipeline
from typing import Set, Union

from ..models.difflm import DiffusionLM
from ..data.stories import StoryGenerationDataset


class StoryGenerationOracle(ExactOracle):
    llm_hf_repo_name: str = "meta-llama/Llama-3.2-3B"

    difflm_hf_repo_name: str = "michaelsyao/DiffusionLM-ROCStories"

    name: str = "flesch_reading_ease_score"

    max_new_tokens: int = 64

    num_samples: int = 128

    def __init__(self, dataset: StoryGenerationDataset, **kwargs):
        """
        Args:
            dataset: the offline dataset for the offline optimization problem.
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
        self.device_id = -1 + torch.cuda.is_available()

        self.pipe = pipeline(
            "text-generation",
            model=self.llm_hf_repo_name,
            device=self.device_id,
            max_new_tokens=self.max_new_tokens
        )
        self.pipe.tokenizer.pad_token = self.pipe.tokenizer.eos_token
        self.difflm = DiffusionLM.from_pretrained(self.difflm_hf_repo_name)

        for key, val in kwargs.items():
            setattr(self, key, val)

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
    ) -> np.ndarray:
        """
        Design scoring function that computes the MAUVE score of a generated
        text output seeded by the input design(s).
        Input:
            x: a batch or single design given as input to the oracle model.
        Returns:
            A batch or single ground-truth prediction made by the oracle model.
        """
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        x = x.to(next(self.difflm.parameters()).device)

        scores = []
        for story in self.difflm.decode(self.difflm(x)):
            stories = self.pipe(
                [story],
                repetition_penalty=2.0,
                eos_token_id=self.pipe.tokenizer.eos_token_id,
                pad_token_id=self.pipe.tokenizer.eos_token_id,
                return_full_text=True,
                continue_final_message=True,
                do_sample=False,
                top_p=None,
                temperature=None,
            )
            stories = [st[0]["generated_text"] for st in stories]
            scores.append(np.mean([flesch_reading_ease(st) for st in stories]))

        return np.array(scores)

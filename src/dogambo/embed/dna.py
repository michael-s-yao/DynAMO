#!/usr/bin/python3
"""
DNA embedding model.

Author(s):
    Michael Yao @michael-s-yao

Citation(s):
    [1] Zhou Z, Ji Y, Li W, Dutta P, Davuluri RV, Liu H. DNABERT-2: Efficient
        Foundation Model and Benchmark for Multi-Species Genomes. Proc ICLR.
        (2024). doi: 10.48550/arXiv.2306.15006

Licensed under the MIT License. Copyright University of Pennsylvania 2024.
"""
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from transformers.models.bert.configuration_bert import BertConfig
from typing import Final, Sequence

from ..utils import import_flash_attn


class DNABERT(nn.Module):
    hf_repo_name: Final[str] = "zhihan1996/DNABERT-2-117M"

    trust_remote_code: bool = True

    vocabulary: Final[Sequence[str]] = ["A", "C", "G", "T"]

    def __init__(self):
        """
        Args:
            None.
        """
        super().__init__()
        self.config = BertConfig.from_pretrained(self.hf_repo_name)
        self.config.use_flash_attn = False
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.hf_repo_name, trust_remote_code=self.trust_remote_code
        )

        self.dtype = torch.float16
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            self.dtype = torch.bfloat16
        attn_and_autocast = import_flash_attn()
        self.autocast_context = attn_and_autocast["autocast_context"]

        self.model = AutoModel.from_pretrained(
            self.hf_repo_name,
            trust_remote_code=self.trust_remote_code,
            config=self.config,
            torch_dtype=self.dtype
        )
        if torch.cuda.is_available():
            self.model = self.model.cuda()

    def forward(self, dna: torch.Tensor) -> torch.Tensor:
        """
        Embeds a DNA sequence or list of DNA sequences.
        Input:
            dna: a DNA sequence or list of DNA sequences to embed.
        Returns:
            A tensor of embeddings of shape ND, where N is the number of
            input sequences and D is the number of embedding dimensions.
        """
        idx_to_char = lambda idx: self.vocabulary[idx]  # noqa
        dna = dna.detach().cpu().numpy()
        dna = np.vectorize(idx_to_char)(dna)
        embeddings = []
        with torch.inference_mode():
            with self.autocast_context:
                for seq in ["".join([bp for bp in seq]) for seq in dna]:
                    inputs = self.tokenizer(seq, return_tensors="pt")
                    inputs = inputs["input_ids"]
                    inputs = inputs.to(next(self.model.parameters()).device)
                    embeddings.append(
                        torch.mean(self.model(inputs)[0][0], dim=0)
                    )
        return torch.stack(embeddings)

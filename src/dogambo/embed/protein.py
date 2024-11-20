#!/usr/bin/python3
"""
Protein embedding model.

Author(s):
    Michael Yao @michael-s-yao

Citation(s):
    [1] Lin Z, Akin H, Rao R, Hie B, Zhu Z, Lu W, Smetanin N, Verkuil R,
        Kabeli O, Shmueli Y, Dos Santas Costa A, Fazel-Zarandi M, Sercu T,
        Candido S, Rives A. Evolutionary-scale prediction of atomic-level
        protein structure with a language model. Science 379(6637):1123-30.
        (2023). doi: 10.1126/science.ade2574

Licensed under the MIT License. Copyright University of Pennsylvania 2024.
"""
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer, EsmModel
from typing import Sequence

from ..utils import import_flash_attn


class ESM2(nn.Module):
    hf_repo_name: str = "facebook/esm2_t33_650M_UR50D"

    trust_remote_code: bool = True

    def __init__(self, **kwargs):
        """
        Args:
            None.
        """
        super().__init__()
        for key, val in kwargs.items():
            setattr(self, key, val)

        self.dtype = torch.float16
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            self.dtype = torch.bfloat16
        attn_and_autocast = import_flash_attn()
        self.autocast_context = attn_and_autocast["autocast_context"]

        self.tokenizer = AutoTokenizer.from_pretrained(self.hf_repo_name)
        self.model = EsmModel.from_pretrained(
            self.hf_repo_name, torch_dtype=self.dtype
        )
        if torch.cuda.is_available():
            self.model = self.model.cuda()

    def forward(self, prot: torch.Tensor) -> torch.Tensor:
        """
        Embeds a protein sequence or list of protein sequences.
        Input:
            prot: a protein sequence or list of protein sequences to embed.
        Returns:
            A tensor of embeddings of shape ND, where N is the number of
            input sequences and D is the number of embedding dimensions.
        """
        idx_to_char = lambda idx: self.vocabulary[idx]  # noqa
        prot = prot.detach().cpu().numpy()
        prot = np.vectorize(idx_to_char)(prot)
        embeddings = []
        with torch.inference_mode():
            with self.autocast_context:
                for seq in ["".join([bp for bp in seq]) for seq in prot]:
                    inputs = self.tokenizer(seq, return_tensors="pt")
                    for key, val in inputs.items():
                        if not isinstance(val, torch.Tensor):
                            continue
                        inputs[key] = val.to(
                            next(self.model.parameters()).device
                        )
                    embeddings.append(self.model(**inputs).last_hidden_state)
        return torch.stack(embeddings).to(torch.float32)

    @property
    def vocabulary(self) -> Sequence[str]:
        """
        Defines the amino acid vocabulary. Uses the same vocabulary as defined
        in the Design-Bench repository at the following URL:
            https://github.com/brandontrabucco/design-bench/blob/
            e52939588421b5433f6f2e9b359cf013c542bd89/process/process_raw_gfp.py
        Input:
            None.
        Returns:
            The amino acid vocabulary.
        """
        return list("ARNDCQEGHILKMFPSTWYV")

#!/usr/bin/python3
"""
Molecule embedding model.

Author(s):
    Michael Yao @michael-s-yao

Citation(s):
    [1] Zhang XC, Wu CK, Yi JC, Zeng XX, Yang CQ, Lu AP, Hou TJ, Cao DS.
        Pushing the boundaries of molecular property prediction for drug
        discovery with multitask learning BERT enhanced by SMILES enumeration.
        Research 2022(0004). (2022). doi: 10.34133/research.0004

Licensed under the MIT License. Copyright University of Pennsylvania 2024.
"""
import design_bench
import numpy as np
import os
import torch
import torch.nn as nn
from deepchem.feat.smiles_tokenizer import SmilesTokenizer
from transformers import AutoTokenizer, AutoModel
from typing import Final, Sequence

from ..utils import import_flash_attn


class ChemBERT(nn.Module):
    hf_repo_name: Final[str] = "jonghyunlee/ChemBERT_ChEMBL_pretrained"

    def __init__(self):
        """
        Args:
            None.
        """
        super().__init__()
        self.dtype = torch.float16
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            self.dtype = torch.bfloat16
        attn_and_autocast = import_flash_attn()
        self.autocast_context = attn_and_autocast["autocast_context"]

        self.tokenizer = AutoTokenizer.from_pretrained(self.hf_repo_name)
        self.model = AutoModel.from_pretrained(
            self.hf_repo_name, torch_dtype=self.dtype
        )
        self.max_seq_length = self.model.embeddings.token_type_ids.size(dim=-1)

        if torch.cuda.is_available():
            self.model = self.model.cuda()

    def forward(self, molecule: torch.Tensor) -> torch.Tensor:
        """
        Embeds a SMILES molecule sequence or list of molecule sequences.
        Input:
            molecule: a molecule or list of molecules to embed.
        Returns:
            A tensor of embeddings of shape ND, where N is the number of
            input sequences and D is the number of embedding dimensions.
        """
        idx_to_char = lambda idx: self.vocabulary[idx]  # noqa
        molecule = molecule.detach().cpu().numpy()
        molecule = np.vectorize(idx_to_char)(molecule)
        embeddings = []
        with torch.inference_mode():
            with self.autocast_context:
                for seq in ["".join([tok for tok in seq]) for seq in molecule]:
                    inputs = self.tokenizer(seq, return_tensors="pt")
                    inputs = inputs["input_ids"]
                    inputs = inputs.to(next(self.model.parameters()).device)
                    if inputs.size(dim=-1) > self.max_seq_length:
                        inputs = inputs[..., :self.max_seq_length]
                    embeddings.append(self.model(inputs)[0][0].mean(dim=0))
        return torch.stack(embeddings).to(torch.float32)

    @property
    def vocabulary(self) -> Sequence[str]:
        """
        Defines the SMILES molecule vocabulary. Uses the same vocabulary as
        defined in the Design-Bench repository at the following URL:
            https://github.com/brandontrabucco/design-bench/blob/e529395884
            21b5433f6f2e9b359cf013c542bd89/process/process_raw_chembl.py
        Input:
            None.
        Returns:
            The SMILES molecule vocabulary.
        """
        tokenizer = SmilesTokenizer(
            os.path.join(
                os.path.dirname(os.path.dirname(design_bench.__file__)),
                "design_bench_data",
                "smiles_vocab.txt"
            )
        )
        return sorted(tokenizer.vocab.keys(), key=tokenizer.vocab.__getitem__)

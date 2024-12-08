#!/usr/bin/python3
"""
Pytorch Lightning Data Module for Molecule Generation.

Author(s):
    Michael Yao @michael-s-yao

Citations(s):
    [1] Krenn M, Hase F, Nigam AK, Friederich P, Aspuru-Guzik A. Self-
        referencing embedded strings (SELFIES): A 100% robust molecular string
        representation. Machine Learning: Science and Technology 1(4): 045024.
        (2020). https://doi.org/10.1088/2632-2153/aba947

Licensed under the MIT License. Copyright University of Pennsylvania 2024.
"""
import json
import numpy as np
import selfies as sf
from datasets import load_dataset
from design_bench.datasets.discrete_dataset import DiscreteDataset
from pathlib import Path
from transformers import PreTrainedTokenizer
from typing import Dict, Optional, Sequence, Tuple, Union


class SELFIESTokenizer(PreTrainedTokenizer):
    def __init__(self, vocab: Dict[str, int] = {}, **kwargs):
        """
        Args:
            vocab: a vocab mapping SELFIES tokens to integer IDs.
        """
        self.__token_ids = vocab
        self.__id_tokens: Dict[int, str] = {
            value: key for key, value in vocab.items()
        }
        super().__init__(**kwargs)

    def _tokenize(self, text: str, **kwargs) -> Sequence[str]:
        """
        Tokenizes an input molecule using the SELFIES tokenizer.
        Input:
            None.
        Returns:
            The tokenized molecule.
        """
        return list(sf.split_selfies(text))

    def _convert_token_to_id(self, token: str) -> int:
        """
        Converts a string token to an integer ID.
        Input:
            token: the string token.
        Returns:
            The corresponding integer ID of the string token in the vocabulary.
        """
        if token in self.__token_ids.keys():
            return self.__token_ids[token]
        return self.unk_token_id

    def _convert_id_to_token(self, idx: int) -> str:
        """
        Converts an ID to a string token.
        Input:
            idx: the token ID.
        Returns:
            The corresponding string token in the vocabulary.
        """
        if idx in self.__id_tokens.keys():
            return self.__id_tokens[idx]
        return self.unk_token

    def get_vocab(self) -> Dict[str, int]:
        """
        Retrieves a copy of the vocabulary of the tokenizer.
        Input:
            None.
        Returns:
            A copy of the vocabulary of the tokenizer.
        """
        return self.__token_ids.copy()

    def save_vocabulary(
        self,
        save_directory: Union[Path, str],
        filename_prefix: Optional[str] = None
    ) -> Tuple[str]:
        """
        Saves the vocabulary of the tokenizer to disk.
        Input:
            save_directory: the directory to save the vocabulary to.
            filename_prefix: an optional prefix for the vocabulary filename.
        Returns:
            A tuple of the saved vocabulary filepath.
        """
        filename_prefix = "" if filename_prefix is None else filename_prefix
        vocab_path = Path(save_directory, filename_prefix + "vocab.json")
        with open(vocab_path, "w") as f:
            json.dump(self.__token_ids, f)
        return (str(vocab_path),)

    @property
    def vocab_size(self) -> int:
        """
        Returns the size of the vocabulary of the tokenizer.
        Input:
            None.
        Returns:
            The size of the vocabulary of the tokenizer.
        """
        return len(self.__token_ids)


class PenalizedLogPDataset(DiscreteDataset):
    name: str = "selfies_molecule/selfies_molecule"

    x_name: str = "selfies_strings"

    y_name: str = "penalized_logp_score"

    hf_repo_name: str = "michaelsyao/PenalizedLogPDataset"

    def __init__(self, partition: str = "train", **kwargs):
        """
        Args:
            partition: the partition to load. One of [`train`, `val`].
        """
        data = load_dataset(self.hf_repo_name)[partition]
        x = np.array(data["designs"])
        y = np.array(data["scores"])
        idxs = np.intersect1d(
            np.where(np.isfinite(y.flatten()))[0],
            np.where(y.flatten() > -1e12)[0]
        )
        x, y = x[idxs], y[idxs]
        super().__init__(
            x, y, num_classes=(1 + x.max()), is_logits=False, **kwargs
        )

#!/usr/bin/python3
"""
Defines the oracle objective for the Molecule LogP task.

Author(s):
    Michael Yao @michael-s-yao

Citation(s):
    [1] Brown N, Fiscato M, Segler MHS, Vaucher AC. GuacaMol: Benchmarking
        models for de novo molecular design. J Chem Inf Model 59(3):1096-08.
        (2019). https://doi.org/10.1021/acs.jcim.8b00839

Licensed under the MIT License. Copyright University of Pennsylvania 2024.
"""
import networkx as nx
import numpy as np
import selfies as sf
import torch
from design_bench.oracles.exact_oracle import ExactOracle
from rdkit import Chem
from rdkit.Chem import Crippen
from rdkit.Chem import rdmolops
from typing import Final, Optional, Set, Union

from . import sascorer
from ..data import PenalizedLogPDataset, SELFIESTokenizer


class PenalizedLogPOracle(ExactOracle):
    name: str = "penalized_logp_score"

    tokenizer_hf_repo_name: Final[str] = "michaelsyao/SELFIESTokenizer"

    def __init__(
        self, dataset: Optional[PenalizedLogPDataset] = None, **kwargs
    ):
        """
        Args:
            dataset: the offline dataset for the offline optimization problem.
        """
        self.__tokenizer = SELFIESTokenizer.from_pretrained(
            self.tokenizer_hf_repo_name
        )
        if dataset is None:
            return
        super().__init__(
            dataset,
            is_batched=False,
            internal_batch_size=1,
            internal_measurements=1,
            expect_normalized_y=False,
            expect_normalized_x=False,
            expect_logits=False,
            **kwargs
        )

    @property
    def tokenizer(self) -> SELFIESTokenizer:
        """
        Returns the tokenizer associated with the oracle.
        Input:
            None.
        Returns:
            The tokenizer associated with the oracle.
        """
        return self.__tokenizer

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
        return {PenalizedLogPDataset}

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
        self, smile: Union[torch.Tensor, np.ndarray]
    ) -> np.ndarray:
        """
        Computes the ground-truth penalized LogP score for a molecule.
        Input:
            smile: a tokenized representation of the molecule.
        Returns:
            The penalized LogP score of the molecule.
        """
        if not isinstance(smile, torch.Tensor):
            smile = torch.from_numpy(smile)
        if smile.ndim < 2:
            smile = smile.unsqueeze(dim=0)
        smile = self.tokenizer.batch_decode(smile)
        scores = []
        for i, seq in enumerate(smile):
            seq = seq.replace(" ", "")
            eos_idx = seq.find(self.tokenizer.eos_token)
            if eos_idx > -1:
                seq = seq[:eos_idx]
            bos_idx = seq.rfind(self.tokenizer.bos_token)
            if bos_idx > -1:
                seq = seq[bos_idx + len(self.tokenizer.bos_token):]
            smile[i] = sf.decoder(seq)

            mol = Chem.MolFromSmiles(smile[i])
            if mol is None:
                scores.append(-float("inf"))
                continue
            logp = Crippen.MolLogP(mol)
            sa = sascorer.calculateScore(mol)
            cycle_length = self._cycle_score(mol)
            # Calculate the final penalized score. The magic numbers below
            # are the empirical means and standard deviations of the dataset.
            z_logp = (logp - 2.45777691) / 1.43341767
            z_sa = (sa - 3.05352042) / 0.83460587
            z_cycle_length = (cycle_length - 0.04861121) / 0.28746695
            penalized_logp = max(z_logp - z_sa - z_cycle_length, -float("inf"))
            scores.append(
                -float("inf") if penalized_logp is None else penalized_logp
            )
        return np.array(scores)

    def _cycle_score(self, mol: Chem.Mol) -> int:
        """
        Calculates the cycle score for an input molecule.
        Input:
            mol: input molecule to calculate the cycle score for.
        Returns:
            The cycle score for the input molecule.
        """
        cycle_list = nx.cycle_basis(nx.Graph(rdmolops.GetAdjacencyMatrix(mol)))
        if len(cycle_list) == 0:
            cycle_length = 0
        else:
            cycle_length = max([len(j) for j in cycle_list])
        if cycle_length <= 6:
            cycle_length = 0
        else:
            cycle_length = cycle_length - 6
        return cycle_length

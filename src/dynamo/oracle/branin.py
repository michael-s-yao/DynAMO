#!/usr/bin/python3
"""
Defines the oracle objective for the toy Branin task.

Author(s):
    Michael Yao @michael-s-yao

Citation(s):
    [1] Branin FH. Widely convergent method for finding multiple solutions
        of simultaneous nonlinear equations. IBM J Res Dev 16(5): 504-22.
        (1972). doi: 10.1147/rd.165.0504

Licensed under the MIT License. Copyright University of Pennsylvania 2024.
"""
import numpy as np
import torch
from design_bench.oracles.exact_oracle import ExactOracle
from botorch.test_functions.synthetic import Branin
from typing import Optional, Set, Sequence, Tuple, Union

from ..data import BraninDataset


class BraninOracle(ExactOracle):
    name: str = "negative_branin_function"

    def __init__(
        self,
        dataset: Optional[BraninDataset] = None,
        bounds: Optional[Union[float, Sequence[float]]] = None,
        **kwargs
    ):
        """
        Args:
            dataset: the offline dataset for the offline optimization problem.
            bounds: an optional specification for the oracle evaluation bounds.
        """
        if dataset is None:
            return
        self.__func = Branin(negate=True, bounds=bounds)
        super().__init__(
            dataset,
            is_batched=False,
            internal_batch_size=1,
            internal_measurements=1,
            expect_normalized_y=False,
            expect_normalized_x=False,
            **kwargs
        )

    @property
    def optimizers(self) -> Sequence[Tuple[float]]:
        """
        Returns the optimal designs in the usual Branin function bounds.
        Input:
            None.
        Returns:
            The optimal designs in the usual Branin function bounds.
        """
        return self.__func._optimizers

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
        return {BraninDataset}

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
        return True

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
        return False

    def protected_predict(
        self, x: Union[torch.Tensor, np.ndarray]
    ) -> np.ndarray:
        """
        Computes the (negative) Branin function value for a design(s).
        Input:
            x: an input design or set of designs.
        Returns:
            The (negative) Branin score(s) of the design(s).
        """
        if not isinstance(x, torch.Tensor):
            x = torch.from_numpy(x)
        return self.__func(x).detach().cpu().numpy()

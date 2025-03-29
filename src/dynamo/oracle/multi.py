#!/usr/bin/python3
"""
Defines the oracle objectives derived from multi-objective optimization
problems.

Author(s):
    Michael Yao @michael-s-yao

Citation(s):
    [1] Tanabe R, Ishibuchi H. An easy-to-use real-world multi-objective
        optimization problem suite. Appl Soft Compu 89(C). (2020). doi:
        10.1016/j.asoc.2020.106078
    [2] Ray T, Liew KM. A swarm metaphor for multiobjective design
        optimization. Eng Optim 34(2): 141-53. (2002). doi:
        10.1080/03052150210915
    [3] Liao X, Li Q, Yang X, Zhang W, Li W. Multiobjective optimization
        for crash safety design of vehicles using stepwise regression model.
        Struct Multidisc Opt 35:561-9. (2008). doi: 10.1007/s00158-007-0163-x

Licensed under the MIT License. Copyright University of Pennsylvania 2024.
"""
import numpy as np
import torch
from design_bench.datasets.discrete.utr_dataset import UTRDataset
from design_bench.oracles.exact_oracle import ExactOracle
from botorch.test_functions.multi_objective import VehicleSafety
from pymoo.problems.multi import WeldedBeam
from typing import Optional, Set, Sequence, Tuple, Union

from ..data import VehicleSafetyDataset, WeldedBeamDataset


class VehicleSafetyOracle(ExactOracle):
    name: str = "negative_vehicle_mass"

    def __init__(
        self,
        dataset: Optional[VehicleSafetyDataset] = None,
        objective_idx: int = 0,
        **kwargs
    ):
        """
        Args:
            dataset: the offline dataset for the offline optimization problem.
            objective_idx: the index of the objective to return.
        """
        if dataset is None:
            return

        self.__func = VehicleSafety(negate=True)
        self.objective_idx = objective_idx
        assert 0 <= self.objective_idx < self.__func.num_objectives

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
        Returns the optimal designs in the usual function bounds.
        Input:
            None.
        Returns:
            The optimal designs in the usual function bounds.
        """
        return [(1.0, 1.0, 1.0, 1.0, 1.0)]

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
        return {VehicleSafetyDataset}

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
        return True

    def protected_predict(
        self, x: Union[torch.Tensor, np.ndarray]
    ) -> np.ndarray:
        """
        Computes the (negative) vehicle safety function value for a design(s).
        Input:
            x: an input design or set of designs.
        Returns:
            The (negative) vehicle safety objective score(s) of the design(s).
        """
        if not isinstance(x, torch.Tensor):
            x = torch.from_numpy(x)
        assert x.size(dim=-1) == self.__func.dim
        y = self.__func(x)[..., self.objective_idx].unsqueeze(dim=-1)
        return y.detach().cpu().numpy()


class WeldedBeamOracle(ExactOracle):
    name: str = "negative_welded_beam_cost"

    def __init__(
        self,
        dataset: Optional[WeldedBeamDataset] = None,
        objective_idx: int = 0,
        **kwargs
    ):
        """
        Args:
            dataset: the offline dataset for the offline optimization problem.
            objective_idx: the index of the objective to return.
        """
        if dataset is None:
            return

        self.__func = WeldedBeam()
        self.objective_idx = objective_idx
        assert 0 <= self.objective_idx < self.__func.n_obj

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
        Returns the optimal designs in the usual function bounds.
        Input:
            None.
        Returns:
            The optimal designs in the usual function bounds.
        """
        return [(0.125, 0.1, 0.1, 0.125)]

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
        return {WeldedBeamDataset}

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
        return True

    def protected_predict(
        self, x: Union[torch.Tensor, np.ndarray]
    ) -> np.ndarray:
        """
        Computes the (negative) welded beam cost function value for a design.
        Input:
            x: an input design or set of designs.
        Returns:
            The (negative) welded beam cost objective score of the design(s).
        """
        is_batched = len(x.shape) > 1
        if not isinstance(x, torch.Tensor):
            x = torch.from_numpy(x)
        if not is_batched:
            x = x.unsqueeze(dim=0)
        assert x.size(dim=-1) == self.__func.n_var
        y = -1.0 * np.array([
            self.__func.evaluate(xx.unsqueeze(dim=0))[0] for xx in x
        ])
        y = y[..., self.objective_idx, np.newaxis]
        if not is_batched:
            y = np.array([y.item()])
        return y


class GCContentOracle(ExactOracle):
    name: str = "negative_gc_content"

    def __init__(
        self,
        dataset: Optional[UTRDataset] = None,
        **kwargs
    ):
        """
        Args:
            dataset: the offline dataset for the offline optimization problem.
            bounds: an optional specification for the oracle evaluation bounds.
        """
        if dataset is None:
            return

        super().__init__(
            dataset,
            is_batched=False,
            internal_batch_size=1,
            internal_measurements=1,
            expect_normalized_y=False,
            expect_normalized_x=False,
            **kwargs
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
        return {UTRDataset}

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
        Computes the (negative) GC content for a design(s).
        Input:
            x: an input design or set of designs.
        Returns:
            The (negative) GC content of the design(s).
        """
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        assert x.dtype in [np.int8, np.int16, np.int32, np.int64]
        gc_count = np.logical_or(x == 1, x == 2).sum(axis=-1)
        return -1.0 * gc_count.astype(np.float32) / x.shape[-1]

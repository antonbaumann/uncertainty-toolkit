import torch
from typing import List

from lib.losses.regression import RegressionLoss
from lib.losses.classification import HeteroscedasticSoftmax
from lib.wrappers.base import BaseWrapper


class Ensemble(torch.nn.Module):
    """
    A module for computing the ensemble prediction using deep ensembles.
    """

    def __init__(
            self, 
            *,
            models: List[torch.nn.Module],
            criterion: RegressionLoss | HeteroscedasticSoftmax,
        ):
        """
        Initializes the `Ensemble` module.

        Arguments:
            models (List[torch.nn.Module]): A list of models to be used in the ensemble.
            criterion (RegressionLoss | HeteroscedasticSoftmax): The criterion to be used for computing probabilistic outputs.
        """
        super(Ensemble, self).__init__()
        self.wrapper = BaseWrapper(models=models, criterion=criterion, monte_carlo_samples=1)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.wrapper(input)

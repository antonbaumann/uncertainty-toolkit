import torch

from utools.losses.regression import RegressionLoss
from utools.losses.classification import HeteroscedasticSoftmax
from utools.wrappers.base import BaseWrapper


class MonteCarlo(torch.nn.Module):
    """
    A module for computing the Monte Carlo prediction.
    """

    def __init__(
            self, 
            *,
            model: torch.nn.Module,
            criterion: RegressionLoss | HeteroscedasticSoftmax,
            monte_carlo_samples: int,
        ):
        """
        Initializes the `MonteCarlo` module.

        Arguments:
            model (torch.nn.Module): The model to be used for Monte Carlo prediction.
            criterion (RegressionLoss | HeteroscedasticSoftmax): The criterion to be used for computing probabilistic outputs.
            monte_carlo_samples (int): The number of samples to use for Monte Carlo simulation.
        """
        super(MonteCarlo, self).__init__()
        self.wrapper = BaseWrapper(models=[model], criterion=criterion, monte_carlo_samples=monte_carlo_samples)

        # Activate MC Dropout for the model
        for model in self.wrapper.models:
            self._activate_mc_dropout(model)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Computes the Monte Carlo prediction.
        """
        return self.wrapper(input)
    
    @staticmethod
    def _activate_mc_dropout(model: torch.nn.Module):
        """
        Activates MC Dropout for all Dropout layers in the given module.
        Recursively iterates through all submodules.

        Args:
            module: Module to activate MC Dropout for.
        """
        for submodule in model.modules():
            if submodule.__class__.__name__.startswith('Dropout'):
                submodule.train()
                print(f"Activated MC Dropout for {submodule}")

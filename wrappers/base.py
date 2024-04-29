import torch
from typing import List

from losses.regression import RegressionLoss
from losses.classification import HeteroscedasticSoftmax

class BaseWrapper(torch.nn.Module):
    """
    A base wrapper class for computing deep ensemble predictions or Monte Carlo predictions.
    """

    def __init__(
            self, 
            *,
            models: List[torch.nn.Module],
            criterion: RegressionLoss | HeteroscedasticSoftmax,
            monte_carlo_samples: int,
        ):
        super(BaseWrapper, self).__init__()
        self.models: List[torch.nn.Module] = models
        """A list of models to be used in the ensemble."""
        self.criterion: torch.nn.Module = criterion
        """The criterion to be used for computing the mean and variance of the ensemble prediction."""
        self.monte_carlo_samples: int = monte_carlo_samples
        """The number of samples to use for Monte Carlo simulation."""

        # Set the models to evaluation mode
        for model in models:
            model.eval()
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to compute the ensemble prediction.

        Arguments:
            input (torch.Tensor): Input tensor to be used for prediction.

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing the mean, aleatoric variance, and epistemic variance.
        """
        if issubclass(type(self.criterion), HeteroscedasticSoftmax):
            return self._forward_classification(input)
        elif issubclass(type(self.criterion, RegressionLoss)):
            return self._forward_regression(input)
        else:
            raise ValueError(f"Unsupported criterion: {type(self.criterion)}")
        
    def _forward_classification(self, input: torch.Tensor) -> torch.Tensor:
        probas = []

        with torch.no_grad():
            for model in self.models:
                for _ in range(self.monte_carlo_samples):
                    out = model(input)
                    proba = self.criterion(out).detach().cpu()
                    probas.append(proba)

        probas = torch.stack(probas, dim=0).mean(dim=0)
        return dict(probas=probas)

    def _forward_regression(self, input: torch.Tensor) -> torch.Tensor:
        means, vars = [], []

        with torch.no_grad():
            for model in self.models:
                for _ in range(self.monte_carlo_samples):
                    out = model(input)
                    mean = self.criterion.mean(out).detach().cpu()
                    var = self.criterion.var(out).detach().cpu()
                    means.append(mean)
                    vars.append(var)

        means = torch.stack(means, dim=0)
        vars = torch.stack(vars, dim=0)

        mean = means.mean(dim=0)
        aleatoric_var = vars.mean(dim=0)
        epistemic_var = means.var(dim=0)

        return dict(mean=mean, aleatoric_var=aleatoric_var, epistemic_var=epistemic_var)

import torch
from typing import Optional

class GaussianNLL(torch.nn.Module):
    """
    This module computes the negative log-likelihood for a Gaussian distribution.
    """

    num_params: int = 2
    """Number of parameters per channel expected in the input tensor"""

    def __init__(self, eps_min: float = 1e-5, eps_max: float = 1e3):
        """Initializes the `GaussianNLL` module with specified parameters for epsilon bounds.

        Args:
            eps_min (float): Minimum allowed value for variance to prevent division by zero errors.
            eps_max (float): Maximum allowed value for variance to ensure numerical stability.
        """
        super().__init__()
        self.eps_min: float = eps_min
        """Minimum allowed value for variance to prevent division by zero errors"""

        self.eps_max: float = eps_max
        """Maximum allowed value for variance to ensure numerical stability"""

    def forward(self, input: torch.Tensor, target: torch.Tensor, *, mask: Optional[torch.Tensor] = None, reduce_mean: bool = True) -> torch.Tensor:
        """Forward pass for computing the Gaussian negative log-likelihood.

        Args:
            input (torch.Tensor): Tensor containing both mean and log variance. Shape should be `(B, C*2, H, W)`.
            target (torch.Tensor): Tensor containing target values. Shape should be `(B, C, H, W)`.
            mask (Optional[torch.Tensor]): Tensor for masking specific elements. Defaults to None. Shape should be `(B, 1, H, W)`.
            reduce_mean (bool): If True, returns the mean of the losses; otherwise returns the loss per element. Defaults to True.

        Returns:
            `torch.Tensor`: Calculated loss. Scalar if `reduce_mean` is True, otherwise tensor of shape `(B, C, H, W)`.
        """
        y_hat, log_variance = torch.chunk(input, self.num_params, dim=-3)
        diff = y_hat - target
        variance = torch.exp(log_variance).clone()
        variance.clamp_(min=self.eps_min, max=self.eps_max)

        loss = torch.log(variance) + (diff ** 2) / variance
        if mask is not None:
            loss *= mask
        if reduce_mean:
            return torch.mean(loss)
        return loss

    def std(self, input: torch.Tensor) -> torch.Tensor:
        """Calculate the standard deviation from the input tensor's predicted log variance.

        Args:
            input (torch.Tensor): The input tensor containing mean and log variance.
        
        Returns:
            `torch.Tensor`: Standard deviation of the Gaussian distribution.
        """
        _, log_variance = torch.chunk(input, self.num_params, dim=-3)
        return torch.exp(log_variance) ** 0.5

    def var(self, input: torch.Tensor) -> torch.Tensor:
        """Calculate the variance from the input tensor's predicted log variance.

        Args:
            input (torch.Tensor): The input tensor containing mean and log variance.
        
        Returns:
            `torch.Tensor`: Variance of the Gaussian distribution.
        """
        _, log_variance = torch.chunk(input, self.num_params, dim=-3)
        return torch.exp(log_variance)

    def mean(self, input: torch.Tensor) -> torch.Tensor:
        """Extract the mean from the input tensor.

        Args:
            input (torch.Tensor): The input tensor containing mean and log variance.
        
        Returns:
            `torch.Tensor`: Mean of the Gaussian distribution.
        """
        y_hat, _ = torch.chunk(input, self.num_params, dim=-3)
        return y_hat

class LaplacianNLL(torch.nn.Module):
    """
    This module computes the negative log-likelihood for a Laplace distribution.
    """

    num_params: int = 2
    """Number of parameters per channel expected in the input tensor"""

    def __init__(self, eps_min: float = 1e-5, eps_max: float = 1e3):
        """
        Initializes the `LaplacianNLL` module with specified parameters for epsilon bounds.
        """
        super().__init__()
        self.eps_min = eps_min
        """Minimum allowed value for scale to prevent division by zero errors"""
        self.eps_max = eps_max
        """Maximum allowed value for scale to ensure numerical stability"""

    def forward(self, input: torch.Tensor, target: torch.Tensor, *, mask: Optional[torch.Tensor] = None, reduce_mean: bool = True) -> torch.Tensor:
        """
        Forward pass for computing the Laplacian negative log-likelihood.

        Parameters:
            input (torch.Tensor): Tensor containing both mean and log scale. Shape should be (B, C*2, H, W).
            target (torch.Tensor): Tensor containing target values. Shape should be (B, C, H, W).
            mask (Optional[torch.Tensor]): Tensor for masking specific elements. Defaults to None. Shape should be (B, 1, H, W).
            reduce_mean (bool): If True, returns the mean of the losses; otherwise returns the loss per element. Defaults to True.

        Returns:
            `torch.Tensor`: Calculated loss. Scalar if `reduce_mean` is True, otherwise tensor of shape `(B, C, H, W)`.
        """
        y_hat, log_scale = torch.chunk(input, self.num_params, dim=-3)
        diff = y_hat - target
        scale = torch.exp(log_scale).clone()
        scale.clamp_(min=self.eps_min, max=self.eps_max)

        loss = torch.log(scale) + torch.abs(diff) / scale
        if mask is not None:
            loss *= mask
        if reduce_mean:
            return torch.mean(loss)
        return loss

    def std(self, input: torch.Tensor) -> torch.Tensor:
        """
        Calculate the standard deviation for a Laplace distribution.

        Parameters:
            input (torch.Tensor): The input tensor containing mean and log scale.
        
        Returns:
            `torch.Tensor`: Standard deviation of the Laplace distribution, calculated as sqrt(2) times the scale.
        """
        _, log_scale = torch.chunk(input, self.num_params, dim=-3)
        return torch.exp(log_scale) * (2 ** 0.5)

    def var(self, input: torch.Tensor) -> torch.Tensor:
        """
        Calculate the variance for a Laplace distribution.

        Parameters:
            input (torch.Tensor): The input tensor containing mean and log scale.
        
        Returns:
            `torch.Tensor`: Variance of the Laplace distribution, calculated as twice the square of the scale.
        """
        return self.std(input) ** 2

    def mean(self, input: torch.Tensor) -> torch.Tensor:
        """
        Extract the mean from the input tensor.

        Parameters:
            input (torch.Tensor): The input tensor containing mean and log scale.
        
        Returns:
            `torch.Tensor`: Mean of the Laplace distribution.
        """
        y_hat, _ = torch.chunk(input, self.num_params, dim=-3)
        return y_hat

class EvidentialLoss(torch.nn.Module):
    """
    This module implements the evidential regression loss for deep learning models introduced in https://arxiv.org/abs/1910.02600.
    """
    num_params: int = 4
    """Number of parameters per channel expected in the input tensor"""
    
    def __init__(self, coeff: float = 1.0):
        """
        Initializes the `EvidentialLoss` module with a specified coefficient for the loss calculation.
        """
        super().__init__()
        self.coeff: float = coeff
        """Coefficient for the loss calculation"""

    def forward(self, input: torch.Tensor, target: torch.Tensor, *, mask: Optional[torch.Tensor] = None, reduce_mean: bool = True) -> torch.Tensor:
        """
        Forward pass for calculating the evidential loss given the model's output and the targets.

        Parameters:
            input (torch.Tensor): The input tensor containing the parameters for the loss calculation (mu, v, alpha, beta).
            target (torch.Tensor): The tensor containing the target values.
            mask (Optional[torch.Tensor]): An optional tensor for masking specific elements. Defaults to None.
            reduce_mean (bool): If True, returns the mean of the losses; otherwise returns the loss per element. Defaults to True.

        Returns:
            torch.Tensor: The calculated loss. Scalar if `reduce_mean` is True, otherwise tensor of shape (B, C, H, W).
        """
        mu, v, alpha, beta = torch.chunk(input, self.num_params, dim=-3)
        loss = self.evidential_loss(mu, v, alpha, beta, target)

        if mask is not None:
            loss *= mask

        if reduce_mean:
            return torch.mean(loss)
        return loss

    @staticmethod
    def evidential_loss(mu: torch.Tensor, v: torch.Tensor, alpha: torch.Tensor, beta: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute the Sum of Squared Error loss for an evidential deep regression model based on the input parameters.
        Code from https://github.com/aamini/chemprop

        Parameters:
            mu (torch.Tensor): Predicted mean.
            v (torch.Tensor): Predicted v parameter.
            alpha (torch.Tensor): Predicted alpha parameter.
            beta (torch.Tensor): Predicted beta parameter.
            target (torch.Tensor): The target tensor.

        Returns:
            `torch.Tensor`: Computed loss value for each element.
        """
        def Gamma(x: torch.Tensor) -> torch.Tensor:
            return torch.exp(torch.lgamma(x))

        coeff_denom = 4 * Gamma(alpha) * v * torch.sqrt(beta)
        coeff_num = Gamma(alpha - 0.5)
        coeff = coeff_num / coeff_denom

        second_term = 2 * beta * (1 + v)
        second_term += (2 * alpha - 1) * v * torch.pow((target - mu), 2)
        L_SOS = coeff * second_term
        L_REG = torch.pow((target - mu), 2) * (2 * alpha + v)

        loss_val = L_SOS + L_REG
        return loss_val

    def mean(self, input: torch.Tensor) -> torch.Tensor:
        """
        Extract the mean parameter from the input tensor.

        Parameters:
            input (torch.Tensor): The input tensor containing the parameters for the evidential distribution.

        Returns:
            `torch.Tensor`: Extracted mean of the distribution.
        """
        mu, _, _, _ = torch.chunk(input, self.num_params, dim=-3)
        return mu
    
    def aleatoric_var(self, input: torch.Tensor) -> torch.Tensor:
        """
        Calculate the aleatoric variance from the input tensor's predicted parameters.

        Parameters:
            input (torch.Tensor): The input tensor containing the parameters for the evidential distribution.

        Returns:
            `torch.Tensor`: Calculated aleatoric variance.
        """
        _, _, alpha, beta = torch.chunk(input, self.num_params, dim=-3)
        return beta / (alpha - 1)
    
    def epistemic_var(self, input: torch.Tensor) -> torch.Tensor:
        """
        Calculate the epistemic variance from the input tensor's predicted parameters.

        Parameters:
            input (torch.Tensor): The input tensor containing the parameters for the evidential distribution.

        Returns:
            `torch.Tensor`: Calculated epistemic variance.
        """
        _, v, alpha, beta = torch.chunk(input, self.num_params, dim=-3)
        return v * (beta / (alpha - 1))

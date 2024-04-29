import torch
from typing import Optional

class GaussianNLL(torch.nn.Module):
    """
    A module for computing the negative log-likelihood for a Gaussian distribution.

    Attributes:
        eps_min (float): Minimum allowed value for variance to avoid division by zero.
        eps_max (float): Maximum allowed value for variance for numerical stability.

    Shape:
        - Input: (B, C*2, H, W), where `B` is the batch size, `C` is the number of channels,
          `H` is the height, and `W` is the width. Input should contain predicted mean and
          log variance stacked along the channel dimension.
        - Target: (B, C, H, W), same shape as each half of the input channels (mean or variance).
        - Mask (optional): (B, 1, H, W), same spatial dimensions as the input and target,
          with a single channel, used to specify elements to include in the loss calculation.
        - Output: Scalar if `reduce_mean` is True, otherwise (B, C, H, W) aligned with input/target dimensions.
    """
    num_params = 2

    def __init__(self, eps_min: float = 1e-5, eps_max: float = 1e3):
        super().__init__()
        self.eps_min = eps_min
        self.eps_max = eps_max

    def forward(self, input: torch.Tensor, target: torch.Tensor, *, mask: Optional[torch.Tensor] = None, reduce_mean: bool = True) -> torch.Tensor:
        """
        Forward pass for computing the Gaussian negative log-likelihood.

        Parameters:
            input (torch.Tensor): The input tensor containing both mean and log variance.
            target (torch.Tensor): The tensor containing the target values.
            mask (Optional[torch.Tensor]): An optional tensor for masking specific elements.
            reduce_mean (bool): If True, returns the mean of the losses; otherwise returns the loss per element.

        Returns:
            torch.Tensor: The calculated loss. Scalar if `reduce_mean` is True, otherwise (B, C, H, W).
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
        """Calculate the standard deviation from the input tensor."""
        _, log_variance = torch.chunk(input, self.num_params, dim=-3)
        return torch.exp(log_variance) ** 0.5
    
    def var(self, input: torch.Tensor) -> torch.Tensor:
        """Calculate the variance from the input tensor."""
        _, log_variance = torch.chunk(input, self.num_params, dim=-3)
        return torch.exp(log_variance)

    def mean(self, input: torch.Tensor) -> torch.Tensor:
        """Extract the mean from the input tensor."""
        y_hat, _ = torch.chunk(input, self.num_params, dim=-3)
        return y_hat
    

class LaplacianNLL(torch.nn.Module):
    """
    A module for computing the negative log-likelihood for a Laplace distribution.

    Attributes:
        eps_min (float): Minimum allowed value for scale to avoid division by zero.
        eps_max (float): Maximum allowed value for scale for numerical stability.

    Shape:
        - Input: (B, C*2, H, W), where `B` is the batch size, `C` is the number of channels,
          `H` is the height, and `W` is the width. Input should contain predicted mean and
          log scale stacked along the channel dimension.
        - Target: (B, C, H, W), same shape as each half of the input channels (mean or scale).
        - Mask (optional): (B, 1, H, W), same spatial dimensions as the input and target,
          with a single channel, used to specify elements to include in the loss calculation.
        - Output: Scalar if `reduce_mean` is True, otherwise (B, C, H, W) aligned with input/target dimensions.
    """
    num_params = 2

    def __init__(self, eps_min: float = 1e-5, eps_max: float = 1e3):
        super().__init__()
        self.eps_min = eps_min
        self.eps_max = eps_max

    def forward(self, input: torch.Tensor, target: torch.Tensor, *, mask: Optional[torch.Tensor] = None, reduce_mean: bool = True) -> torch.Tensor:
        """
        Forward pass for computing the Laplacian negative log-likelihood.

        Parameters:
            input (torch.Tensor): The input tensor containing both mean and log scale.
            target (torch.Tensor): The tensor containing the target values.
            mask (Optional[torch.Tensor]): An optional tensor for masking specific elements.
            reduce_mean (bool): If True, returns the mean of the losses; otherwise returns the loss per element.

        Returns:
            torch.Tensor: The calculated loss. Scalar if `reduce_mean` is True, otherwise (B, C, H, W).
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
        """Calculate the standard deviation for a Laplace distribution."""
        _, log_scale = torch.chunk(input, self.num_params, dim=-3)
        return torch.exp(log_scale) * (2 ** 0.5)
    
    def var(self, input: torch.Tensor) -> torch.Tensor:
        """Calculate the variance for a Laplace distribution."""
        return self.std(input) ** 2

    def mean(self, input: torch.Tensor) -> torch.Tensor:
        """Extract the mean from the input tensor."""
        y_hat, _ = torch.chunk(input, self.num_params, dim=-3)
        return y_hat
    

class EvidentialLoss(torch.nn.Module):
    """
    A module implementing the evidential regression loss for deep learning models.

    Attributes:
        coeff (float): A coefficient used for loss calculation.

    Shape:
        - Input: (B, C*4, H, W), where `B` is the batch size, `C` is the number of channels,
          `H` is the height, and `W` is the width. Input should contain the parameters for
          the evidential loss stacked along the channel dimension.
        - Target: (B, C, H, W), same shape as a quarter of the input channels.
        - Mask (optional): (B, 1, H, W), same spatial dimensions as the input and target,
          with a single channel, used to specify elements to include in the loss calculation.
        - Output: Scalar if `reduce_mean` is True, otherwise (B, C, H, W) aligned with input/target dimensions.
    """
    num_params = 4
    
    def __init__(self, coeff: float) -> None:
        super().__init__()
        self.coeff = coeff

    @staticmethod
    def evidential_loss(mu: torch.Tensor, v: torch.Tensor, alpha: torch.Tensor, beta: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute the Sum of Squared Error loss for an evidential deep regression model.
        
        Parameters:
            mu (torch.Tensor): Predicted mean.
            v (torch.Tensor): Predicted variance.
            alpha (torch.Tensor): Predicted alpha parameter.
            beta (torch.Tensor): Predicted beta parameter.
            target (torch.Tensor): The target tensor.
        
        Returns:
            torch.Tensor: Computed loss value for each element.
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

    def forward(self, input: torch.Tensor, target: torch.Tensor, *, mask: Optional[torch.Tensor] = None, reduce_mean: bool = True) -> torch.Tensor:
        """
        Forward pass for calculating the evidential loss given the model's output and the targets.

        Parameters:
            input (torch.Tensor): The input tensor containing the parameters for the loss calculation.
            target (torch.Tensor): The tensor containing the target values.
            mask (Optional[torch.Tensor]): An optional tensor for masking specific elements.
            reduce_mean (bool): If True, returns the mean of the losses; otherwise returns the loss per element.

        Returns:
            torch.Tensor: The calculated loss. Scalar if `reduce_mean` is True, otherwise (B, C, H, W).
        """
        gamma, v, alpha, beta = torch.chunk(input, self.num_params, dim=-3)
        loss = self.evidential_loss(
            mu=gamma,
            v=v,
            alpha=alpha,
            beta=beta,
            target=target,
        )

        if mask is not None:
            loss *= mask

        if reduce_mean:
            return torch.mean(loss)
        return loss
        
    def mean(self, input: torch.Tensor) -> torch.Tensor:
        """Extract the mean parameter from the input tensor."""
        gamma, _, _, _ = torch.chunk(input, self.num_params, dim=-3)
        return gamma
    
    def aleatoric_var(self, input: torch.Tensor) -> torch.Tensor:
        """Calculate the aleatoric variance from the input tensor."""
        _, _, alpha, beta = torch.chunk(input, self.num_params, dim=-3)
        return beta / (alpha - 1)
    
    def epistemic_var(self, input: torch.Tensor) -> torch.Tensor:
        """Calculate the epistemic variance from the input tensor."""
        _, v, alpha, beta = torch.chunk(input, self.num_params, dim=-3)
        return beta / (v * (alpha - 1))
import torch


class HeteroscedasticSoftmax(torch.nn.Module):
    """
    A module for computing the softmax of Gaussian modeled logits using Monte Carlo sampling.
    Refer to the mathematical details provided in https://arxiv.org/abs/1703.04977.

    Attributes:
        num_samples (int): Number of samples to use for Monte Carlo sampling.

    Shape:
        - Input: (*batch, 2*C, H, W), where `*batch` denotes any number of preceding dimensions,
          `C` is the number of channels, `H` is the height, and `W` is the width.
          The input tensor should contain predicted logits and log standard deviation stacked along
          the second to last dimension.
        - Output: (*batch, C, H, W), the softmax probabilities with the same spatial dimensions
          and batch size as the input.
    """

    def __init__(self, num_samples: int = 100):
        """
        Initializes the HeteroscedasticSoftmax module.

        Parameters:
            num_samples (int): The number of samples to use for Monte Carlo simulation.
        """
        super(HeteroscedasticSoftmax, self).__init__()
        self.num_samples = num_samples

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to compute the softmax of Gaussian modeled logits using Monte Carlo sampling.

        Parameters:
            input (torch.Tensor): Input tensor containing predicted logits and log standard deviation.

        Returns:
            torch.Tensor: The computed softmax probabilities across the channel dimension.
        """
        logits, log_std = torch.chunk(input, 2, dim=-3)

        std = torch.exp(log_std)
        softmax_sum = torch.zeros_like(logits)
        for _ in range(self.num_samples):
            epsilon = torch.randn_like(std)
            logit_samples = logits + epsilon * std
            softmax_sum += torch.nn.functional.softmax(logit_samples, dim=-3)

        softmax = softmax_sum / self.num_samples
        return softmax
    

class HeteroscedasticCrossEntropy(torch.nn.Module):
    """
    A module for computing the cross-entropy loss of Gaussian modeled logits using Monte Carlo sampling.
    Refer to the mathematical details provided in https://arxiv.org/abs/1703.04977.

    Attributes:
        num_samples (int): Number of samples to use for Monte Carlo sampling.
        label_smoothing (float): The label smoothing factor to be applied.

    Shape:
        - Input: (*batch, 2*C, H, W), where `*batch` denotes any number of preceding dimensions,
          `C` is the number of channels, `H` is the height, and `W` is the width.
          The input tensor should contain predicted logits and log standard deviation stacked along
          the second to last dimension.
        - Target: (*batch, H, W), where target labels are provided for each spatial dimension.
        - Output: Scalar, representing the computed loss over the batch.
    """

    def __init__(self, num_samples: int = 100, label_smoothing: float = 0.0):
        """
        Initializes the HeteroscedasticCrossEntropy module.

        Parameters:
            num_samples (int): The number of samples to use for Monte Carlo simulation.
            label_smoothing (float): Label smoothing factor to be used during loss computation.
        """
        super(HeteroscedasticCrossEntropy, self).__init__()
        self.num_samples = num_samples
        self.label_smoothing = label_smoothing

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to compute the cross-entropy loss of Gaussian modeled logits using Monte Carlo sampling.

        Parameters:
            input (torch.Tensor): Input tensor containing predicted logits and log standard deviation.
            target (torch.Tensor): Target tensor containing the true class labels.

        Returns:
            torch.Tensor: Scalar value representing the computed cross-entropy loss.
        """
        logits, log_std = torch.chunk(input, 2, dim=-3)

        std = torch.exp(log_std)
        loss = 0
        for _ in range(self.num_samples):
            epsilon = torch.randn_like(std)
            logit_samples = logits + epsilon * std

            loss += torch.nn.functional.cross_entropy(
                input=logit_samples,
                target=target,
                label_smoothing=self.label_smoothing
            ) / self.num_samples

        return loss
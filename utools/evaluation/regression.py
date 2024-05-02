import torch
from typing import Literal, Optional
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor


def regression_precision_recall_df(
    y_pred: torch.Tensor, 
    var_pred: torch.Tensor, 
    y_true: torch.Tensor, 
    n_bins: int = 50,
    n_samples: Optional[int] = None,
) -> pd.DataFrame:
    """
    Compute a precision-recall dataframe.

    Arguments:
        y_pred (torch.Tensor): The predicted values. Shape: `(n_samples,)`.
        var_pred (torch.Tensor): The predicted variance. Shape: `(n_samples,)`.
        y_true (torch.Tensor): The true values. Shape: `(n_samples,)`.
        n_bins (int): The number of bins.
        n_samples (int): The number of samples to use for calibration.

    Returns:
        `pd.DataFrame`: The precision-recall dataframe.
    """
    if n_samples is not None:
        # Randomly sample `num_samples` samples
        indices = torch.randperm(len(y_pred))[:n_samples]
        y_pred, var_pred, y_true = y_pred[indices], var_pred[indices], y_true[indices]

    perm = torch.argsort(var_pred, descending=False)
    y_pred, var_pred, y_true = y_pred[perm], var_pred[perm], y_true[perm]
    diff = y_pred - y_true

    cutoff_percentiles = torch.arange(n_bins)[1:] / (n_bins - 1)
    cutoff_indices = (cutoff_percentiles * len(y_pred)).int()

    mae = [diff[:i].abs().mean().item() for i in cutoff_indices]
    mse = [(diff[:i]**2).mean().item() for i in cutoff_indices]

    return pd.DataFrame({
        'percentile': cutoff_percentiles,
        'mae': mae,
        'mse': mse,
        'rmse': [m**0.5 for m in mse],
    })


def regression_calibration_df(
    y_pred: torch.Tensor, 
    var_pred: torch.Tensor, 
    y_true: torch.Tensor,
    distribution: Literal['normal', 'laplace'] = 'normal',
    n_bins: int = 50,
    n_samples: Optional[int] = None,
    max_workers: int = 1,
) -> pd.DataFrame:
    r"""
    Compute the calibration dataframe for regression models.

    For each expected probability, the observed probability $p_{obs}$ is computed by comparing the true values to the inverse CDF of the predicted values evaluated at the expected probability $p_{exp}$:
    $$ p_{obs} = \frac{\sum_{i=1}^{N}\mathbb I\left[y_i \leq F_i^{-1}(p_{exp})\right]}{N}  $$

    Refer to https://arxiv.org/abs/1807.00263 for mathematical details.

    Args:
        y_pred (torch.Tensor): The predicted values. Shape: `(n_samples,)`.
        var_pred (torch.Tensor): The predicted variance. Shape: `(n_samples,)`.
        y_true (torch.Tensor): The true values. Shape: `(n_samples,)`.
        distribution (Distribution): The distribution module (e.g. Normal, if `GaussianNLL` was used or Laplace, if `LaplaceNLL` was used).
        n_bins (int): The number of bins.
        n_samples (int): The number of samples to use for calibration.
        max_workers (int): The number of workers to use for parallel processing. Default: 1. WARNING: Can get memory intensive if set to a high value.
    
    Returns:
        `pd.DataFrame`: The calibration dataframe.
    """
    if n_samples is not None:
        # Randomly sample `num_samples` samples
        indices = torch.randperm(len(y_pred))[:n_samples]
        y_pred, var_pred, y_true = y_pred[indices], var_pred[indices], y_true[indices]

    if distribution == 'normal':
        icdf = _normal_icdf
    elif distribution == 'laplace':
        icdf = _laplace_icdf
    else:
        raise ValueError(f"Unknown distribution: {distribution}")
    
    confidence_levels = torch.linspace(0, 1, n_bins)

    def process_bin(p):
        icdf_values = icdf(p, loc=y_pred, variance=var_pred)
        observed_p = (y_true <= icdf_values).float().mean()
        return {'expected_p': p.item(), 'observed_p': observed_p.item()}

    # Use ThreadPoolExecutor to parallelize the bin processing
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        calibration_data = list(tqdm(executor.map(process_bin, confidence_levels), total=n_bins))

    return pd.DataFrame(calibration_data)


def _normal_icdf(p: torch.Tensor, loc: torch.Tensor, variance: torch.Tensor) -> torch.Tensor:
    """
    Compute the inverse CDF of the Normal distribution.

    Args:
        p (torch.Tensor): The probabilities. Shape: `(n_samples,)`.
        loc (torch.Tensor): The location parameter. Shape: `(n_samples,)`.
        scale (torch.Tensor): The scale parameter. Shape: `(n_samples,)`.
    
    Returns:
        `torch.Tensor`: The inverse CDF values. Shape: `(n_samples,)`.
    """
    scale = variance.sqrt()
    return loc + scale * torch.erfinv(2 * p - 1) * 2**0.5


def _laplace_icdf(p: torch.Tensor, loc: torch.Tensor, variance: torch.Tensor) -> torch.Tensor:
    """
    Compute the inverse CDF of the Laplace distribution.

    Args:
        p (torch.Tensor): The probabilities. Shape: `(n_samples,)`.
        loc (torch.Tensor): The location parameter. Shape: `(n_samples,)`.
        scale (torch.Tensor): The scale parameter. Shape: `(n_samples,)`.
    
    Returns:
        `torch.Tensor`: The inverse CDF values. Shape: `(n_samples,)`.
    """
    scale = variance.sqrt() / 2**0.5
    return loc - scale * torch.sign(p - 0.5) * torch.log(1 - 2 * torch.abs(p - 0.5))
    
import torch
from typing import Literal
import pandas as pd
from tqdm import tqdm


def compute_precision_recall_dataframe(
    y_pred: torch.Tensor, 
    var_pred: torch.Tensor, 
    y_true: torch.Tensor, 
    n_bins: int = 50,
) -> pd.DataFrame:
    """
    Compute a precision-recall dataframe.

    Arguments:
        y_pred (torch.Tensor): The predicted values. Shape: `(n_samples,)`.
        var_pred (torch.Tensor): The predicted variance. Shape: `(n_samples,)`.
        y_true (torch.Tensor): The true values. Shape: `(n_samples,)`.
        n_bins (int): The number of bins.
    """
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


def compute_calibration_dataframe(
    y_pred: torch.Tensor, 
    var_pred: torch.Tensor, 
    y_true: torch.Tensor,
    distribution: Literal['normal', 'laplace'],
    n_bins: int = 50,
) -> pd.DataFrame:
    """
    Compute the calibration dataframe for regression models.
    Refer to https://arxiv.org/abs/1807.00263 for mathematical details.

    Args:
        y_pred (torch.Tensor): The predicted values. Shape: `(n_samples,)`.
        var_pred (torch.Tensor): The predicted variance. Shape: `(n_samples,)`.
        y_true (torch.Tensor): The true values. Shape: `(n_samples,)`.
        distribution (Distribution): The distribution module (e.g. Normal, if `GaussianNLL` was used or Laplace, if `LaplaceNLL` was used).
        n_bins (int): The number of bins.
    
    Returns:
        pd.DataFrame: The calibration dataframe.
    """
    if distribution == 'normal':
        icdf = _normal_icdf
    elif distribution == 'laplace':
        icdf = _laplace_icdf
    else:
        raise ValueError(f"Unknown distribution: {distribution}")
    
    confidence_levels = torch.arange(n_bins) / (n_bins - 1)

    calibration_data = []
    for p in tqdm(confidence_levels):
        icdf_values = icdf(p, loc=y_pred, variance=var_pred)
        observed_p = (y_true <= icdf_values).float().mean()
        calibration_data.append({
            'expected_p': p.item(),
            'observed_p': observed_p.item(),
        })
    return pd.DataFrame(calibration_data)


def expected_calibration_error(calibration_df: pd.DataFrame) -> float:
    """
    Compute the expected calibration error.

    Args:
        calibration_df (pd.DataFrame): The calibration dataframe.
    
    Returns:
        float: The expected calibration error.
    """
    return (calibration_df['observed_p'] - calibration_df['expected_p']).abs().mean()


def _normal_icdf(p: torch.Tensor, loc: torch.Tensor, variance: torch.Tensor) -> torch.Tensor:
    """
    Compute the inverse CDF of the Normal distribution.

    Args:
        p (torch.Tensor): The probabilities. Shape: `(n_samples,)`.
        loc (torch.Tensor): The location parameter. Shape: `(n_samples,)`.
        scale (torch.Tensor): The scale parameter. Shape: `(n_samples,)`.
    
    Returns:
        torch.Tensor: The inverse CDF values. Shape: `(n_samples,)`.
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
        torch.Tensor: The inverse CDF values. Shape: `(n_samples,)`.
    """
    scale = variance.sqrt() / 2**0.5
    return loc - scale * torch.sign(p - 0.5) * torch.log(1 - 2 * torch.abs(p - 0.5))
    
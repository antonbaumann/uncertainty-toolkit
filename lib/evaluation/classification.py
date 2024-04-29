import torch
import pandas as pd

def classification_calibration_df(
        y_pred: torch.Tensor,
        y_true: torch.Tensor, 
        positive_class_index: int,
        n_bins: int = 10,
    ) -> pd.DataFrame:
    """
    Compute the calibration dataframe for classification models.
    
    Args:
        y_pred (torch.Tensor): The predicted probabilities. Shape: `(n_samples, C)`.
        y_true (torch.Tensor): The true labels. Shape: `(n_samples, 1)`.
        n_bins (int): The number of bins to use for calibration.
    
    Returns:
        pd.DataFrame: The calibration dataframe.
    """
    bin_edges = torch.linspace(0, 1, n_bins + 1)

    calibration_data = []
    for i in range(n_bins):
        mask = (y_pred[:, positive_class_index] >= bin_edges[i]) & (y_pred[:, positive_class_index] < bin_edges[i+1])
        prob_true = (y_true[mask, 0] == positive_class_index).float().mean().item()
        prob_pred = y_pred[mask, positive_class_index].float().mean().item()

        calibration_data.append({
            'expected_p': prob_pred,
            'observed_p': prob_true,
            'count': mask.sum().item(),
        })
        
    return pd.DataFrame(calibration_data)

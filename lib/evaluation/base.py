import pandas as pd


def expected_calibration_error(calibration_df: pd.DataFrame) -> float:
    """
    Compute the expected calibration error.

    Args:
        calibration_df (pd.DataFrame): The calibration dataframe.
    
    Returns:
        float: The expected calibration error.
    """

    abs_diff = (calibration_df['observed_p'] - calibration_df['expected_p']).abs()

    if 'count' in calibration_df.columns:
        total_count = calibration_df['count'].sum()
        weight = calibration_df['count'] / total_count
        return (abs_diff * weight).sum()

    return  abs_diff.mean()
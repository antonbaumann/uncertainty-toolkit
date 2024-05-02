import torch.nn as nn


def create_module_list(module: nn.Module, num_subnetworks: int, **kwargs):
    """
    Utility function to create a list of identical modules.
    
    Args:
        module: The PyTorch module to be repeated.
        num_subnetworks: Number of times the module needs to be repeated.
        **kwargs: Arguments to be passed to the module during initialization.
    
    Returns:
        A module list containing the repeated modules.
    """
    return nn.ModuleList([module(**kwargs) for _ in range(num_subnetworks)])

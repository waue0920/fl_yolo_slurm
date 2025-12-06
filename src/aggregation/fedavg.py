
import torch

def aggregate(client_weights, **kwargs):
    """
    FedAvg Aggregation: Simple average of all client weights.
    Args:
        client_weights: List of model weights (list of dict or list of np.array)
    Returns:
        aggregated_weights: Average weights with the same structure as client_weights[0]
    """
    if not client_weights:
        raise ValueError("client_weights is empty!")

    # Assume each client_weights[i] is a dict (key: param name, value: torch.Tensor)
    keys = client_weights[0].keys()
    aggregated = {}
    for k in keys:
        # Stack all client weights for the same key (torch.stack)
        stacked = torch.stack([cw[k].float() for cw in client_weights], dim=0)
        aggregated[k] = torch.mean(stacked, dim=0)
    return aggregated

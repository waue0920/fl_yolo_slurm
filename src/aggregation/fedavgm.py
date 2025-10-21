import torch
from collections import OrderedDict

def _weighted_average(client_weights, client_sizes):
    """
    Helper function to perform weighted averaging (standard FedAvg).
    This is used to calculate the average model update direction.

    Args:
        client_weights (list[dict]): A list of model state_dicts from clients.
        client_sizes (list[int]): A list of the number of samples for each client.

    Returns:
        dict: The aggregated (averaged) model state_dict.
    """
    if not client_weights:
        raise ValueError("client_weights cannot be empty.")
    if len(client_weights) != len(client_sizes):
        raise ValueError(f"Mismatch in lengths: {len(client_weights)} weights vs {len(client_sizes)} sizes.")

    total_samples = sum(client_sizes)
    if total_samples == 0:
        # Fallback to simple average if all client sizes are zero
        print("[Warning] All client sizes are zero. Falling back to simple average.")
        keys = client_weights[0].keys()
        avg_weights = OrderedDict()
        for k in keys:
            stacked = torch.stack([cw[k].float() for cw in client_weights], dim=0)
            avg_weights[k] = torch.mean(stacked, dim=0)
        return avg_weights

    # Get the keys from the first model
    keys = client_weights[0].keys()
    avg_weights = OrderedDict()

    for k in keys:
        # Sum of weighted weights for the current layer
        # Ensure all tensors are on the same device and dtype
        ref_tensor = client_weights[0][k]
        weighted_sum = torch.zeros_like(ref_tensor, dtype=torch.float32, device=ref_tensor.device)
        for i, weights in enumerate(client_weights):
            weighted_sum += (client_sizes[i] / total_samples) * weights[k].to(weighted_sum.device, dtype=weighted_sum.dtype)
        avg_weights[k] = weighted_sum
        
    return avg_weights

def aggregate(client_weights, client_sizes, global_weights, server_momentum, 
              **kwargs):
    import os
    server_lr = float(os.environ.get('SERVER_FEDAVGM_LR', 1.0))
    momentum = float(os.environ.get('SERVER_FEDAVGM_MOMENTUM', 0.9))
    """
    Performs Federated Averaging with Server Momentum (FedAvgM).

    This function updates the global model by applying momentum to the average
    of client updates.

    Args:
        client_weights (list[dict]): A list of model state_dicts from clients.
        client_sizes (list[int]): The number of samples for each client, used for weighted averaging.
        global_weights (dict): The global model weights from the previous round.
        server_momentum (dict): The server's momentum vector from the previous round.
        server_lr (float, optional): The learning rate on the server side. Defaults to 1.0.
        momentum (float, optional): The momentum factor. Defaults to 0.9.

    Returns:
        tuple[dict, dict]: A tuple containing:
            - new_global_weights (dict): The updated global model weights for the new round.
            - new_momentum (dict): The updated server momentum vector to be saved for the next round.
    """
    if not client_weights:
        raise ValueError("client_weights is empty!")

    # 1. Perform weighted averaging of client models to get the consensus model
    avg_weights = _weighted_average(client_weights, client_sizes)

    # 2. Calculate the pseudo-gradient (the direction of the server update)
    # This is the difference between the old global model and the new average model
    pseudo_grad = OrderedDict()
    for k in global_weights:
        if k in avg_weights:
            pseudo_grad[k] = global_weights[k].to(avg_weights[k].device) - avg_weights[k]
        else:
            print(f"[Warning] Key {k} not found in averaged weights. Skipping.")

    # 3. Update the server momentum
    # new_momentum = momentum * server_momentum + pseudo_grad
    new_momentum = OrderedDict()
    for k in pseudo_grad:
        if k in server_momentum:
            new_momentum[k] = momentum * server_momentum[k].to(pseudo_grad[k].device) + pseudo_grad[k]
        else:
            # This can happen in the first round where momentum is empty
            new_momentum[k] = pseudo_grad[k]

    # 4. Apply the momentum to update the global weights
    # new_global_weights = global_weights - server_lr * new_momentum
    new_global_weights = OrderedDict()
    for k in global_weights:
        if k in new_momentum:
            new_global_weights[k] = global_weights[k].to(new_momentum[k].device) - server_lr * new_momentum[k]
        else:
            new_global_weights[k] = global_weights[k] # No update for this key

    return new_global_weights, new_momentum

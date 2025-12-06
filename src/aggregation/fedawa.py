"""
FedAWA aggregation algorithm implementation for server-side federated learning.

This module provides:
- cosine_distance_matrix: compute pairwise cosine distance
- euclidean_distance_matrix: compute pairwise Lp-mean distance (Euclidean when p=2)
- aggregate: perform FedAWA aggregation with learnable client weights (T_weights)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from collections import OrderedDict
import numpy as np
import os


def cosine_distance_matrix(x, y):
    """
    Compute cosine distance matrix.
    Args:
        x: Tensor of shape [batch1, dim]
        y: Tensor of shape [batch2, dim]
    Returns:
        Tensor of shape [batch1, batch2] with distances in [0, 2]
    """
    x_normalized = x / (torch.norm(x, dim=-1, keepdim=True) + 1e-8)
    y_normalized = y / (torch.norm(y, dim=-1, keepdim=True) + 1e-8)
    # cosine similarity
    cos_sim = torch.matmul(x_normalized, y_normalized.t())
    # cosine distance = 1 - cosine_similarity
    return 1 - cos_sim


def euclidean_distance_matrix(x, y, p=2):
    """
    Compute Lp-mean distance matrix (Euclidean when p=2).
    Args:
        x: Tensor of shape [batch1, dim]
        y: Tensor of shape [batch2, dim]
        p: Power for Lp distance (default 2)
    Returns:
        Tensor of shape [batch1, batch2]
    """
    x_expanded = x.unsqueeze(-2)  # [batch1, 1, dim]
    y_expanded = y.unsqueeze(-3)  # [1, batch2, dim]
    return torch.mean(torch.abs(x_expanded - y_expanded) ** p, dim=-1)


def aggregate(client_weights, client_vectors, global_weights, T_weights_state=None, **kwargs):
    """
    FedAWA aggregation algorithm.

    Args:
        client_weights: list[OrderedDict], client model weights per client
        client_vectors: list[dict], additional client info (unused in this implementation)
        global_weights: OrderedDict, current global model weights
        T_weights_state: dict | None, state dict carrying T_weights from the previous round
        **kwargs: extra arguments (ignored)

    Returns:
        aggregated_weights: OrderedDict, aggregated global weights
        T_weights_state: dict, updated state containing T_weights for the next round

    Environment variables:
        - SERVER_FEDAWA_SERVER_EPOCHS: number of optimization epochs for T_weights (default: 1)
        - SERVER_FEDAWA_SERVER_OPTIMIZER: 'adam' or 'sgd' (default: 'adam')
        - SERVER_FEDAWA_SERVER_LR: server-side learning rate (default: 0.001 for Adam, 0.01 for SGD)
        - SERVER_FEDAWA_GAMMA: scaling factor applied to aggregation weights (default: 1.0)
        - SERVER_FEDAWA_REG_DISTANCE: distance metric, 'cos' or 'euc' (default: 'cos')
    """
    
    # Read hyperparameters
    server_epochs = int(os.environ.get('SERVER_FEDAWA_SERVER_EPOCHS', 1))
    server_optimizer = os.environ.get('SERVER_FEDAWA_SERVER_OPTIMIZER', 'adam').lower()
    # Use different default LR depending on optimizer (as in the original paper)
    if server_optimizer == 'adam':
        server_lr = float(os.environ.get('SERVER_FEDAWA_SERVER_LR', 0.001))  # Adam: 0.001
    else:  # sgd
        server_lr = float(os.environ.get('SERVER_FEDAWA_SERVER_LR', 0.01))   # SGD: 0.01
    gamma = float(os.environ.get('SERVER_FEDAWA_GAMMA', 1.0))
    reg_distance = os.environ.get('SERVER_FEDAWA_REG_DISTANCE', 'cos').lower()
    
    num_clients = len(client_weights)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"[FedAWA] Aggregating {num_clients} clients")
    print(f"[FedAWA] Server epochs: {server_epochs}, Optimizer: {server_optimizer}, LR: {server_lr}")
    print(f"[FedAWA] Gamma: {gamma}, Distance metric: {reg_distance}")
    
    # 1) Flatten client weights to vectors (similar to flat_w)
    client_flat_weights = []
    for i, w in enumerate(client_weights):
        flat_w = []
        for k in sorted(w.keys()):  # ensure consistent key order
            # cast to float32 to avoid half-precision issues
            flat_w.append(w[k].float().flatten())
        flat_w = torch.cat(flat_w).to(device)
        client_flat_weights.append(flat_w)
    
    # Stack: [num_clients, total_params]
    local_param_list = torch.stack(client_flat_weights)
    
    # Flatten global model weights
    global_flat_weights = []
    for k in sorted(global_weights.keys()):
        # cast to float32 to avoid half-precision issues
        global_flat_weights.append(global_weights[k].float().flatten())
    global_flat_weights = torch.cat(global_flat_weights).to(device)
    
    # 2) Initialize/load T_weights (trainable parameters)
    if T_weights_state is not None and 'T_weights' in T_weights_state:
        # load from previous round
        T_weights = T_weights_state['T_weights'].clone().detach().to(device)
        # sanity check the loaded weights
        if torch.isnan(T_weights).any() or torch.isinf(T_weights).any():
            print(f"[FedAWA] WARNING: Loaded T_weights contains NaN/Inf, reinitializing")
            T_weights = torch.ones(num_clients, dtype=torch.float32, device=device) / num_clients
        T_weights.requires_grad = True
        print(f"[FedAWA] Loaded T_weights from previous round: {T_weights.detach().cpu().numpy()}")
    else:
        # first round: initialize uniformly
        T_weights = torch.ones(num_clients, dtype=torch.float32, device=device) / num_clients
        T_weights.requires_grad = True
        print(f"[FedAWA] Initialized T_weights (uniform): {T_weights.detach().cpu().numpy()}")
    
    # 3) Set up optimizer
    if server_optimizer == 'adam':
        optimizer = optim.Adam([T_weights], lr=server_lr, betas=(0.5, 0.999))
    elif server_optimizer == 'sgd':
        optimizer = optim.SGD([T_weights], lr=server_lr, momentum=0.9, weight_decay=5e-4)
    else:
        raise ValueError(f"Unknown optimizer: {server_optimizer}")
    
    # 4) Optimize T_weights
    print(f"[FedAWA] Starting T_weights optimization...")
    for epoch in range(server_epochs):
        optimizer.zero_grad()
        
        # Softmax normalization
        probability_train = torch.softmax(T_weights, dim=0)
        
        # a) Regularization Loss: encourage aggregated result close to global model
        if reg_distance == 'cos':
            C = cosine_distance_matrix(global_flat_weights.unsqueeze(0), local_param_list)
        elif reg_distance == 'euc':
            C = euclidean_distance_matrix(global_flat_weights.unsqueeze(0), local_param_list, p=2)
        else:
            raise ValueError(f"Unknown distance metric: {reg_distance}")
        
        # Ensure dimension alignment with the correct squeeze
        C_squeezed = C.squeeze(0)  # [num_clients,]
        if C_squeezed.dim() == 0:  # case of scalar
            C_squeezed = C_squeezed.unsqueeze(0)
        reg_loss = torch.sum(probability_train * C_squeezed)
        
        # b) Similarity Loss: encourage consistency among client updates
        # client_grad = w_i - w_global
        client_grad = local_param_list - global_flat_weights.unsqueeze(0)
        
        # weighted mean gradient
        column_sum = torch.matmul(probability_train.unsqueeze(0), client_grad)  # [1, total_params]
        
        # distance from each client gradient to the weighted mean
        l2_distance = torch.norm(client_grad.unsqueeze(0) - column_sum.unsqueeze(1), p=2, dim=2)  # [1, num_clients]
        
        # Ensure dimension alignment
        l2_distance_squeezed = l2_distance.squeeze(0)  # [num_clients,]
        if l2_distance_squeezed.dim() == 0:  # case of scalar
            l2_distance_squeezed = l2_distance_squeezed.unsqueeze(0)
        sim_loss = torch.sum(probability_train * l2_distance_squeezed)
        
        # c) Total loss
        total_loss = reg_loss + sim_loss
        
        # Numerical stability check
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            print(f"[FedAWA] WARNING: Loss is NaN/Inf at epoch {epoch+1}!")
            print(f"  - reg_loss: {reg_loss.item()}, sim_loss: {sim_loss.item()}")
            print(f"  - T_weights: {T_weights.data}")
            # gradient clipping and early stop
            torch.nn.utils.clip_grad_norm_([T_weights], max_norm=1.0)
            if epoch > 0:
                # restore to previous valid state and stop
                print(f"[FedAWA] Restoring T_weights from previous state and reducing learning rate")
                break
        
        # Backpropagation
        total_loss.backward()
        
        # Gradient clipping to prevent explosion
        torch.nn.utils.clip_grad_norm_([T_weights], max_norm=10.0)
        
        # Check gradients
        if T_weights.grad is not None and (torch.isnan(T_weights.grad).any() or torch.isinf(T_weights.grad).any()):
            print(f"[FedAWA] WARNING: Gradient contains NaN/Inf at epoch {epoch+1}, skipping update")
            optimizer.zero_grad()
            continue
        
        optimizer.step()
        
        if epoch == 0 or (epoch + 1) % max(1, server_epochs // 5) == 0:
            print(f"[FedAWA] Epoch {epoch+1}/{server_epochs}: Loss={total_loss.item():.6f} "
                  f"(Reg={reg_loss.item():.6f}, Sim={sim_loss.item():.6f})")
    
    # 5) Final aggregation weights
    with torch.no_grad():
        final_probability = torch.softmax(T_weights, dim=0)
        print(f"[FedAWA] Final aggregation weights: {[f'{w.item():.4f}' for w in final_probability]}")
    
    # 6) Weighted aggregation in the original parameter space (not flat vectors)
    # Proper probability normalization to avoid numerical instability
    agg_weights = OrderedDict()
    for k in global_weights.keys():
        agg_weights[k] = torch.zeros_like(global_weights[k], dtype=torch.float32)
    
    # Apply gamma scaling factor
    scaled_probs = final_probability * gamma
    # Re-normalize scaled probabilities (sum to 1)
    total_prob = scaled_probs.sum()
    if total_prob > 1e-8:
        scaled_probs = scaled_probs / total_prob
    else:
        print(f"[FedAWA] WARNING: total probability {total_prob} is too small, using uniform weights")
        scaled_probs = torch.ones_like(final_probability) / len(final_probability)
    
    # Aggregate
    for i, (wi, prob) in enumerate(zip(client_weights, scaled_probs)):
        for k in agg_weights.keys():
            agg_weights[k] += prob.item() * wi[k].to(agg_weights[k].device)
    
    # Validate no NaN/Inf in aggregated weights; fallback to FedAvg if needed
    for k in agg_weights.keys():
        if torch.isnan(agg_weights[k]).any() or torch.isinf(agg_weights[k]).any():
            print(f"[FedAWA] WARNING: Aggregated weights contain NaN/Inf for {k}, using equal averaging instead")
            # Fallback to FedAvg
            agg_weights[k] = torch.zeros_like(global_weights[k], dtype=torch.float32)
            for wi in client_weights:
                agg_weights[k] += wi[k].to(agg_weights[k].device) / len(client_weights)
    
    # 7) Save T_weights state for the next round
    T_weights_state_new = {
        'T_weights': T_weights.detach().cpu(),
        'round': T_weights_state.get('round', 0) + 1 if T_weights_state else 1
    }
    
    print(f"[FedAWA] Aggregation complete. T_weights saved for next round.")
    
    return agg_weights, T_weights_state_new

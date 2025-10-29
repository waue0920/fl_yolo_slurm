"""
FedYOGA aggregation algorithm implementation for server-side federated learning.

This module provides:
- cosine_distance_matrix: compute pairwise cosine distance
- aggregate: perform FedYOGA aggregation with learnable layer-wise client weights

Note: FedYOGA uses fixed optimizer='adam' and distance='cos' (not configurable)
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


def aggregate(client_weights, client_vectors, global_weights, T_weights_state=None, **kwargs):
    """
    FedYOGA aggregation algorithm.

    Args:
        client_weights: list[OrderedDict], client model weights per client
        client_vectors: list[dict], additional client training history
        global_weights: OrderedDict, current global model weights
        T_weights_state: dict | None, state dict carrying T_weights from the previous round
        **kwargs: extra arguments (ignored)

    Returns:
        aggregated_weights: OrderedDict, aggregated global weights
        T_weights_state: dict, updated state containing T_weights for the next round

    Environment variables:
        - SERVER_FEDYOGA_SERVER_EPOCHS: number of optimization epochs for T_weights (default: 1)
        - SERVER_FEDYOGA_SERVER_LR: server-side learning rate (default: 0.001)
        - SERVER_FEDYOGA_GAMMA: scaling factor applied to aggregation weights (default: 1.0)
        - SERVER_FEDYOGA_LAYER_GROUP_SIZE: group N layers together (default: 1, i.e., per-layer)
    """
    
    # Read hyperparameters
    server_epochs = int(os.environ.get('SERVER_FEDYOGA_SERVER_EPOCHS', 1))
    server_optimizer = 'adam'  # Fixed: always use Adam
    server_lr = float(os.environ.get('SERVER_FEDYOGA_SERVER_LR', 0.001))
    gamma = float(os.environ.get('SERVER_FEDYOGA_GAMMA', 1.0))
    layer_group_size = int(os.environ.get('SERVER_FEDYOGA_LAYER_GROUP_SIZE', 1))
    
    num_clients = len(client_weights)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"[FedYOGA] Aggregating {num_clients} clients (Layer-wise mode)")
    print(f"[FedYOGA] Server epochs: {server_epochs}, LR: {server_lr}, Gamma: {gamma}")
    print(f"[FedYOGA] Layer group size: {layer_group_size} (fixed: optimizer=adam, distance=cos)")
    
    # 1) Extract layer-wise weights for each client
    # Structure: client_layers[client_idx][layer_idx] = tensor
    layer_keys = sorted(global_weights.keys())
    num_total_layers = len(layer_keys)
    
    # Group layers if needed
    if layer_group_size > 1:
        num_groups = (num_total_layers + layer_group_size - 1) // layer_group_size
        print(f"[FedYOGA] Grouping {num_total_layers} layers into {num_groups} groups")
    else:
        num_groups = num_total_layers
    
    # Organize client weights by layer/group
    client_layers = []
    for i, w in enumerate(client_weights):
        layers = []
        for key in layer_keys:
            layers.append(w[key].float().flatten().to(device))
        client_layers.append(layers)
    
    # Organize global weights by layer/group
    global_layers = []
    for key in layer_keys:
        global_layers.append(global_weights[key].float().flatten().to(device))
    
    # 2) Initialize/load T_weights: [num_groups, num_clients]
    if T_weights_state is not None and 'T_weights' in T_weights_state:
        T_weights = T_weights_state['T_weights'].clone().detach().to(device)
        # Validate shape
        expected_shape = (num_groups, num_clients)
        if T_weights.shape != expected_shape:
            print(f"[FedYOGA] WARNING: T_weights shape mismatch {T_weights.shape} vs {expected_shape}, reinitializing")
            T_weights = torch.ones(num_groups, num_clients, dtype=torch.float32, device=device) / num_clients
        elif torch.isnan(T_weights).any() or torch.isinf(T_weights).any():
            print(f"[FedYOGA] WARNING: Loaded T_weights contains NaN/Inf, reinitializing")
            T_weights = torch.ones(num_groups, num_clients, dtype=torch.float32, device=device) / num_clients
        T_weights.requires_grad = True
        print(f"[FedYOGA] Loaded T_weights from previous round, shape: {T_weights.shape}")
    else:
        T_weights = torch.ones(num_groups, num_clients, dtype=torch.float32, device=device) / num_clients
        T_weights.requires_grad = True
        print(f"[FedYOGA] Initialized T_weights (uniform), shape: {T_weights.shape}")
    
    # 3) Set up optimizer
    if server_optimizer == 'adam':
        optimizer = optim.Adam([T_weights], lr=server_lr, betas=(0.5, 0.999))
    elif server_optimizer == 'sgd':
        optimizer = optim.SGD([T_weights], lr=server_lr, momentum=0.9, weight_decay=5e-4)
    else:
        raise ValueError(f"Unknown optimizer: {server_optimizer}")
    
    # 4) Optimize T_weights layer-by-layer (or group-by-group)
    print(f"[FedYOGA] Starting T_weights optimization (layer-wise)...")
    
    for epoch in range(server_epochs):
        optimizer.zero_grad()
        
        total_reg_loss = 0.0
        total_sim_loss = 0.0
        
        # Process each layer group
        for group_idx in range(num_groups):
            # Determine which layers belong to this group
            start_layer = group_idx * layer_group_size
            end_layer = min(start_layer + layer_group_size, num_total_layers)
            
            # Concatenate layers in this group
            # client_group_weights: [num_clients, total_params_in_group]
            client_group_weights = []
            for client_idx in range(num_clients):
                group_params = torch.cat([client_layers[client_idx][layer_idx] 
                                         for layer_idx in range(start_layer, end_layer)])
                client_group_weights.append(group_params)
            client_group_weights = torch.stack(client_group_weights)  # [num_clients, dim]
            
            # Global group weights
            global_group_weights = torch.cat([global_layers[layer_idx] 
                                             for layer_idx in range(start_layer, end_layer)])
            
            # Get T_weights for this group: [num_clients]
            T_group = T_weights[group_idx]
            
            # Softmax normalization
            probability_train = torch.softmax(T_group, dim=0)
            
            # a) Regularization Loss (using cosine distance)
            C = cosine_distance_matrix(global_group_weights.unsqueeze(0), client_group_weights)
            
            C_squeezed = C.squeeze(0)
            if C_squeezed.dim() == 0:
                C_squeezed = C_squeezed.unsqueeze(0)
            reg_loss = torch.sum(probability_train * C_squeezed)
            
            # b) Similarity Loss
            client_grad = client_group_weights - global_group_weights.unsqueeze(0)
            column_sum = torch.matmul(probability_train.unsqueeze(0), client_grad)
            l2_distance = torch.norm(client_grad.unsqueeze(0) - column_sum.unsqueeze(1), p=2, dim=2)
            l2_distance_squeezed = l2_distance.squeeze(0)
            if l2_distance_squeezed.dim() == 0:
                l2_distance_squeezed = l2_distance_squeezed.unsqueeze(0)
            sim_loss = torch.sum(probability_train * l2_distance_squeezed)
            
            total_reg_loss += reg_loss
            total_sim_loss += sim_loss
        
        # c) Total loss
        total_loss = total_reg_loss + total_sim_loss
        
        # Numerical stability check
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            print(f"[FedYOGA] WARNING: Loss is NaN/Inf at epoch {epoch+1}!")
            print(f"  - reg_loss: {total_reg_loss.item()}, sim_loss: {total_sim_loss.item()}")
            torch.nn.utils.clip_grad_norm_([T_weights], max_norm=1.0)
            if epoch > 0:
                print(f"[FedYOGA] Early stopping due to numerical instability")
                break
        
        # Backpropagation
        total_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_([T_weights], max_norm=10.0)
        
        # Check gradients
        if T_weights.grad is not None and (torch.isnan(T_weights.grad).any() or torch.isinf(T_weights.grad).any()):
            print(f"[FedYOGA] WARNING: Gradient contains NaN/Inf at epoch {epoch+1}, skipping update")
            optimizer.zero_grad()
            continue
        
        optimizer.step()
        
        if epoch == 0 or (epoch + 1) % max(1, server_epochs // 5) == 0:
            print(f"[FedYOGA] Epoch {epoch+1}/{server_epochs}: Loss={total_loss.item():.6f} "
                  f"(Reg={total_reg_loss.item():.6f}, Sim={total_sim_loss.item():.6f})")
    
    # 5) Final aggregation using layer-wise weights
    print(f"[FedYOGA] Computing final aggregation with layer-wise weights...")
    
    agg_weights = OrderedDict()
    
    with torch.no_grad():
        for group_idx in range(num_groups):
            # Get final probability for this group
            T_group = T_weights[group_idx]
            final_probability = torch.softmax(T_group, dim=0)
            
            # Apply gamma scaling
            scaled_probs = final_probability * gamma
            total_prob = scaled_probs.sum()
            if total_prob > 1e-8:
                scaled_probs = scaled_probs / total_prob
            else:
                scaled_probs = torch.ones_like(final_probability) / len(final_probability)
            
            # Determine which layers belong to this group
            start_layer = group_idx * layer_group_size
            end_layer = min(start_layer + layer_group_size, num_total_layers)
            
            # Print weights for first/last/middle groups
            if group_idx == 0 or group_idx == num_groups - 1 or group_idx == num_groups // 2:
                print(f"[FedYOGA] Group {group_idx} (layers {start_layer}-{end_layer-1}) weights: "
                      f"{[f'{w.item():.4f}' for w in scaled_probs]}")
            
            # Aggregate each layer in this group
            for layer_idx in range(start_layer, end_layer):
                key = layer_keys[layer_idx]
                agg_weights[key] = torch.zeros_like(global_weights[key], dtype=torch.float32)
                
                for client_idx, prob in enumerate(scaled_probs):
                    agg_weights[key] += prob.item() * client_weights[client_idx][key].to(agg_weights[key].device)
    
    # Validate no NaN/Inf in aggregated weights
    has_nan_inf = False
    for k in agg_weights.keys():
        if torch.isnan(agg_weights[k]).any() or torch.isinf(agg_weights[k]).any():
            print(f"[FedYOGA] WARNING: Aggregated weights contain NaN/Inf for {k}")
            has_nan_inf = True
    
    if has_nan_inf:
        print(f"[FedYOGA] WARNING: Using FedAvg fallback due to NaN/Inf")
        for k in agg_weights.keys():
            agg_weights[k] = torch.zeros_like(global_weights[k], dtype=torch.float32)
            for wi in client_weights:
                agg_weights[k] += wi[k].to(agg_weights[k].device) / len(client_weights)
    
    # 6) Save T_weights state for the next round
    T_weights_state_new = {
        'T_weights': T_weights.detach().cpu(),
        'round': T_weights_state.get('round', 0) + 1 if T_weights_state else 1,
        'num_groups': num_groups,
        'num_clients': num_clients
    }
    
    print(f"[FedYOGA] Aggregation complete. T_weights shape {T_weights.shape} saved for next round.")
    
    return agg_weights, T_weights_state_new

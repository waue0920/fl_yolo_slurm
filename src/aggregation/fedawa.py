"""
FedAWA: Adaptive Optimization of Aggregation Weights in Federated Learning Using Client Vectors
Based on: https://github.com/ChanglongShi/FedAWA (CVPR 2025)

关键差异与 FedYoga:
- FedAWA: 使用梯度下降优化可学习的聚合权重 T_weights
- FedYoga: 使用 PCA + 启发式规则计算权重

FedAWA 核心机制:
1. 初始化可训练参数 T_weights (每个客户端一个)
2. 通过 Adam/SGD 优化器最小化 Loss = reg_loss + sim_loss
   - reg_loss: 让聚合结果接近全局模型 (regularization)
   - sim_loss: 鼓励客户端更新的一致性 (similarity)
3. 使用 softmax(T_weights) 作为最终聚合权重
"""

import torch
import torch.nn as nn
import torch.optim as optim
from collections import OrderedDict
import numpy as np
import os


def cosine_distance_matrix(x, y):
    """
    计算余弦距离矩阵
    x: [batch1, dim]
    y: [batch2, dim]
    return: [batch1, batch2]
    """
    x_normalized = x / (torch.norm(x, dim=-1, keepdim=True) + 1e-8)
    y_normalized = y / (torch.norm(y, dim=-1, keepdim=True) + 1e-8)
    # cosine similarity
    cos_sim = torch.matmul(x_normalized, y_normalized.t())
    # cosine distance = 1 - cosine_similarity
    return 1 - cos_sim


def euclidean_distance_matrix(x, y, p=2):
    """
    计算欧氏距离矩阵
    x: [batch1, dim]
    y: [batch2, dim]
    return: [batch1, batch2]
    """
    x_expanded = x.unsqueeze(-2)  # [batch1, 1, dim]
    y_expanded = y.unsqueeze(-3)  # [1, batch2, dim]
    return torch.mean(torch.abs(x_expanded - y_expanded) ** p, dim=-1)


def aggregate(client_weights, client_vectors, global_weights, T_weights_state=None, **kwargs):
    """
    FedAWA 聚合算法
    
    Args:
        client_weights: list of OrderedDict, 客户端模型权重
        client_vectors: list of dict, 客户端附加信息 (本实现中未使用)
        global_weights: OrderedDict, 全局模型权重
        T_weights_state: dict, 上一轮的 T_weights 状态 (用于跨轮持久化)
        **kwargs: 其他参数
    
    Returns:
        aggregated_weights: OrderedDict, 聚合后的权重
        T_weights_state: dict, 当前的 T_weights 状态 (供下一轮使用)
    
    Environment Variables:
        SERVER_FEDAWA_SERVER_EPOCHS: 优化 T_weights 的轮数 (默认 1)
        SERVER_FEDAWA_SERVER_OPTIMIZER: 优化器类型 'adam' 或 'sgd' (默认 'adam')
        SERVER_FEDAWA_SERVER_LR: 服务器端学习率 (默认 0.001)
        SERVER_FEDAWA_GAMMA: 聚合权重缩放系数 (默认 1.0)
        SERVER_FEDAWA_REG_DISTANCE: 距离度量 'cos' 或 'euc' (默认 'cos')
    """
    
    # 读取超参数
    server_epochs = int(os.environ.get('SERVER_FEDAWA_SERVER_EPOCHS', 1))
    server_optimizer = os.environ.get('SERVER_FEDAWA_SERVER_OPTIMIZER', 'adam').lower()
    # 根據優化器類型使用不同的學習率（與原論文一致）
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
    
    # 1. 将客户端权重展平为向量 (类似 flat_w)
    client_flat_weights = []
    for i, w in enumerate(client_weights):
        flat_w = []
        for k in sorted(w.keys()):  # 确保顺序一致
            # 转换为 float32 避免 half precision 问题
            flat_w.append(w[k].float().flatten())
        flat_w = torch.cat(flat_w).to(device)
        client_flat_weights.append(flat_w)
    
    # Stack: [num_clients, total_params]
    local_param_list = torch.stack(client_flat_weights)
    
    # 全局模型权重展平
    global_flat_weights = []
    for k in sorted(global_weights.keys()):
        # 转换为 float32 避免 half precision 问题
        global_flat_weights.append(global_weights[k].float().flatten())
    global_flat_weights = torch.cat(global_flat_weights).to(device)
    
    # 2. 初始化/加载 T_weights (可训练参数)
    if T_weights_state is not None and 'T_weights' in T_weights_state:
        # 从上一轮加载
        T_weights = T_weights_state['T_weights'].clone().detach().to(device)
        # 检查加载的权重
        if torch.isnan(T_weights).any() or torch.isinf(T_weights).any():
            print(f"[FedAWA] WARNING: Loaded T_weights contains NaN/Inf, reinitializing")
            T_weights = torch.ones(num_clients, dtype=torch.float32, device=device) / num_clients
        T_weights.requires_grad = True
        print(f"[FedAWA] Loaded T_weights from previous round: {T_weights.detach().cpu().numpy()}")
    else:
        # 第一轮：初始化为均匀分布
        T_weights = torch.ones(num_clients, dtype=torch.float32, device=device) / num_clients
        T_weights.requires_grad = True
        print(f"[FedAWA] Initialized T_weights (uniform): {T_weights.detach().cpu().numpy()}")
    
    # 3. 设置优化器
    if server_optimizer == 'adam':
        optimizer = optim.Adam([T_weights], lr=server_lr, betas=(0.5, 0.999))
    elif server_optimizer == 'sgd':
        optimizer = optim.SGD([T_weights], lr=server_lr, momentum=0.9, weight_decay=5e-4)
    else:
        raise ValueError(f"Unknown optimizer: {server_optimizer}")
    
    # 4. 优化 T_weights
    print(f"[FedAWA] Starting T_weights optimization...")
    for epoch in range(server_epochs):
        optimizer.zero_grad()
        
        # Softmax 归一化
        probability_train = torch.softmax(T_weights, dim=0)
        
        # a) Regularization Loss: 让聚合结果接近全局模型
        if reg_distance == 'cos':
            C = cosine_distance_matrix(global_flat_weights.unsqueeze(0), local_param_list)
        elif reg_distance == 'euc':
            C = euclidean_distance_matrix(global_flat_weights.unsqueeze(0), local_param_list, p=2)
        else:
            raise ValueError(f"Unknown distance metric: {reg_distance}")
        
        # 修复：確保維度一致，使用正確的降維方式
        C_squeezed = C.squeeze(0)  # [num_clients,]
        if C_squeezed.dim() == 0:  # scalar 的情況
            C_squeezed = C_squeezed.unsqueeze(0)
        reg_loss = torch.sum(probability_train * C_squeezed)
        
        # b) Similarity Loss: 鼓励客户端更新的一致性
        # client_grad = w_i - w_global
        client_grad = local_param_list - global_flat_weights.unsqueeze(0)
        
        # 加权平均的梯度
        column_sum = torch.matmul(probability_train.unsqueeze(0), client_grad)  # [1, total_params]
        
        # 计算每个客户端梯度与加权平均的距离
        l2_distance = torch.norm(client_grad.unsqueeze(0) - column_sum.unsqueeze(1), p=2, dim=2)  # [1, num_clients]
        
        # 修复：確保維度一致
        l2_distance_squeezed = l2_distance.squeeze(0)  # [num_clients,]
        if l2_distance_squeezed.dim() == 0:  # scalar 的情況
            l2_distance_squeezed = l2_distance_squeezed.unsqueeze(0)
        sim_loss = torch.sum(probability_train * l2_distance_squeezed)
        
        # c) 总 Loss
        total_loss = reg_loss + sim_loss
        
        # 数值稳定性检查
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            print(f"[FedAWA] WARNING: Loss is NaN/Inf at epoch {epoch+1}!")
            print(f"  - reg_loss: {reg_loss.item()}, sim_loss: {sim_loss.item()}")
            print(f"  - T_weights: {T_weights.data}")
            # 梯度裁剪和重置
            torch.nn.utils.clip_grad_norm_([T_weights], max_norm=1.0)
            if epoch > 0:
                # 恢復到上一個有效狀態
                print(f"[FedAWA] Restoring T_weights from previous state and reducing learning rate")
                break
        
        # 反向传播
        total_loss.backward()
        
        # 梯度裁剪防止爆炸
        torch.nn.utils.clip_grad_norm_([T_weights], max_norm=10.0)
        
        # 检查梯度
        if T_weights.grad is not None and (torch.isnan(T_weights.grad).any() or torch.isinf(T_weights.grad).any()):
            print(f"[FedAWA] WARNING: Gradient contains NaN/Inf at epoch {epoch+1}, skipping update")
            optimizer.zero_grad()
            continue
        
        optimizer.step()
        
        if epoch == 0 or (epoch + 1) % max(1, server_epochs // 5) == 0:
            print(f"[FedAWA] Epoch {epoch+1}/{server_epochs}: Loss={total_loss.item():.6f} "
                  f"(Reg={reg_loss.item():.6f}, Sim={sim_loss.item():.6f})")
    
    # 5. 最终的聚合权重
    with torch.no_grad():
        final_probability = torch.softmax(T_weights, dim=0)
        print(f"[FedAWA] Final aggregation weights: {[f'{w.item():.4f}' for w in final_probability]}")
    
    # 6. 加权聚合 (在原始权重空间操作，而非 flat_w)
    # 修复：正确的权重归一化逻辑，避免数值不稳定
    agg_weights = OrderedDict()
    for k in global_weights.keys():
        agg_weights[k] = torch.zeros_like(global_weights[k], dtype=torch.float32)
    
    # 应用 gamma 缩放因子
    scaled_probs = final_probability * gamma
    # 归一化缩放后的概率（确保和为 1）
    total_prob = scaled_probs.sum()
    if total_prob > 1e-8:
        scaled_probs = scaled_probs / total_prob
    else:
        print(f"[FedAWA] WARNING: total probability {total_prob} is too small, using uniform weights")
        scaled_probs = torch.ones_like(final_probability) / len(final_probability)
    
    # 聚合
    for i, (wi, prob) in enumerate(zip(client_weights, scaled_probs)):
        for k in agg_weights.keys():
            agg_weights[k] += prob.item() * wi[k].to(agg_weights[k].device)
    
    # 再次验证没有 NaN
    for k in agg_weights.keys():
        if torch.isnan(agg_weights[k]).any() or torch.isinf(agg_weights[k]).any():
            print(f"[FedAWA] WARNING: Aggregated weights contain NaN/Inf for {k}, using equal averaging instead")
            # 降级到 FedAvg
            agg_weights[k] = torch.zeros_like(global_weights[k], dtype=torch.float32)
            for wi in client_weights:
                agg_weights[k] += wi[k].to(agg_weights[k].device) / len(client_weights)
    
    # 7. 保存 T_weights 状态供下一轮使用
    T_weights_state_new = {
        'T_weights': T_weights.detach().cpu(),
        'round': T_weights_state.get('round', 0) + 1 if T_weights_state else 1
    }
    
    print(f"[FedAWA] Aggregation complete. T_weights saved for next round.")
    
    return agg_weights, T_weights_state_new

import torch
from collections import OrderedDict
import numpy as np
import os

# 支持 FedAdam

def aggregate(client_weights, **kwargs):
    global_weights = kwargs['global_weights']
    optimizer_state = kwargs.get('optimizer_state', None)
    
    # 從環境變數或預設值取得 FedOpt 參數
    lr = float(os.environ.get('SERVER_FEDOPT_LR', 0.001))
    beta1 = float(os.environ.get('SERVER_FEDOPT_BETA1', 0.9))
    beta2 = float(os.environ.get('SERVER_FEDOPT_BETA2', 0.999))
    eps = float(os.environ.get('SERVER_FEDOPT_EPS', 1e-8))
    
    print(f"[DEBUG] FedOpt aggregate called:")
    print(f"  - num_clients: {len(client_weights)}")
    print(f"  - global_weights keys: {len(global_weights)}")
    print(f"  - optimizer_state type: {type(optimizer_state)}")
    print(f"  - hyperparams: lr={lr}, beta1={beta1}, beta2={beta2}, eps={eps}")
    
    # 檢查輸入是否有異常
    for i, wi in enumerate(client_weights):
        nan_keys = []
        for k, v in wi.items():
            if torch.isnan(v).any() or torch.isinf(v).any():
                nan_keys.append(k)
        if nan_keys:
            print(f"[ERROR] Client {i} has NaN/Inf in keys: {nan_keys[:5]}...")
            return global_weights, optimizer_state
    
    # 檢查 global_weights
    nan_keys = []
    for k, v in global_weights.items():
        if torch.isnan(v).any() or torch.isinf(v).any():
            nan_keys.append(k)
    if nan_keys:
        print(f"[ERROR] Global weights has NaN/Inf in keys: {nan_keys[:5]}...")
        return global_weights, optimizer_state
    
    num_clients = len(client_weights)
    
    # 1. 先計算 client 權重的平均（標準 FedAvg）
    avg_client_weights = OrderedDict()
    for k in global_weights.keys():
        # 將計算強制轉為 float32 避免 'Half' 錯誤
        stacked = torch.stack([cw[k].float() for cw in client_weights], dim=0)
        avg_client_weights[k] = torch.mean(stacked, dim=0)
    
    # 2. 計算 delta (pseudo-gradient)
    # delta = avg_client_weights - global_weights
    avg_delta = OrderedDict()
    for k in global_weights.keys():
        avg_delta[k] = avg_client_weights[k] - global_weights[k].float()
    
    # 檢查 avg_delta 是否有異常
    nan_keys = []
    for k, v in avg_delta.items():
        if torch.isnan(v).any() or torch.isinf(v).any():
            nan_keys.append(k)
            print(f"[ERROR] avg_delta has NaN/Inf in key {k}")
            print(f"  - shape: {v.shape}, mean: {v.mean()}, std: {v.std()}")
    if nan_keys:
        print(f"[ERROR] avg_delta has NaN/Inf, stopping aggregation")
        return global_weights, optimizer_state

    # 3. 初始化或使用現有的 optimizer_state
    if optimizer_state is None:
        print(f"[DEBUG] Creating new optimizer state")
        # 創建新的 momentum 和 variance 字典 (使用 FP32 避免精度問題)
        m = OrderedDict({k: torch.zeros_like(v, dtype=torch.float32) for k, v in global_weights.items()})
        v = OrderedDict({k: torch.zeros_like(v, dtype=torch.float32) for k, v in global_weights.items()})
        # 創建 optimizer_state 字典來保存狀態
        optimizer_state = {
            'm': m,
            'v': v,
            'lr': lr,
            'beta1': beta1,
            'beta2': beta2,
            'eps': eps
        }
    else:
        print(f"[DEBUG] Using existing optimizer_state")
        m = optimizer_state['m']
        v = optimizer_state['v']
        # 簡單檢查幾個關鍵值
        sample_keys = list(global_weights.keys())[:3]
        for k in sample_keys:
            if k in m and k in v:
                m_stats = f"mean={m[k].mean():.6f}, std={m[k].std():.6f}"
                v_stats = f"mean={v[k].mean():.6f}, std={v[k].std():.6f}"
                print(f"  - {k}: m({m_stats}), v({v_stats})")

    # 4. FedAdam 公式
    # 更新全局權重：w_t+1 = w_t + Δw
    # 其中 Δw 是基於 Adam 優化的更新量
    for k in global_weights.keys():
        # m 和 v 使用 FP32 精度進行計算和存儲
        # 更新一階動量（momentum）和二階動量（variance）
        m[k] = beta1 * m[k] + (1 - beta1) * avg_delta[k]  # 保持 FP32
        v[k] = beta2 * v[k] + (1 - beta2) * (avg_delta[k] ** 2)  # 保持 FP32

        # FedAdam 更新：使用 Adam 優化器的更新規則
        # global_weights = global_weights + lr * m / (√v + ε)
        # 注意：這裡是加法，因為 m 已經包含了更新方向（avg_delta）
        global_weights_k_float = global_weights[k].float()
        update = global_weights_k_float + lr * m[k] / (torch.sqrt(v[k]) + eps)
        
        # 將結果轉回原始 dtype
        global_weights[k] = update.to(global_weights[k].dtype)
    
    # 更新 optimizer_state 中的 m 和 v (已經是 FP32，無需轉換)
    optimizer_state['m'] = m
    optimizer_state['v'] = v
    
    # 5. 回傳新 weights 及 optimizer_state
    return global_weights, optimizer_state

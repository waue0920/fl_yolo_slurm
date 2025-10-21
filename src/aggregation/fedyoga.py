from sklearn.decomposition import PCA
import torch
from collections import OrderedDict, deque
import numpy as np
import os

def get_loss_drop(results):
    """
    計算 LossDrop，results 為 list of dict 或 numpy array，需包含 val_loss 欄位。
    例如 results = [{'val/box_loss': ...}, ...] 或 results = np.array([...])
    """
    if results is None or len(results) < 2:
        return 0.0
    # 假設 val/box_loss 欄位
    if isinstance(results, list):
        # 取最後兩個 epoch 的 val/box_loss
        losses = [r.get('val/box_loss', None) for r in results if 'val/box_loss' in r]
    elif isinstance(results, np.ndarray):
        # 假設第一欄是 val/box_loss
        losses = results[:, 0]
    else:
        return 0.0
    if len(losses) < 2 or losses[-2] is None or losses[-1] is None:
        return 0.0
    return float(losses[-2] - losses[-1])

def get_grad_var(weights_history):
    """
    計算 GradVariance，weights_history 為 list of OrderedDict，每個為一輪的 weights。
    """
    if weights_history is None or len(weights_history) < 2:
        return 0.0
    # 計算 weight delta
    deltas = []
    for i in range(1, len(weights_history)):
        delta = []
        for k in weights_history[i].keys():
            # 轉為 float32 進行計算，避免 numpy 處理 float16 的問題
            diff = (weights_history[i][k] - weights_history[i-1][k]).float().cpu().numpy().flatten()
            delta.append(diff)
        delta = np.concatenate(delta)
        deltas.append(delta)
    deltas = np.stack(deltas)
    # 取所有 delta 的變異數
    return float(np.var(deltas))

def pca_reduce(X, n_components=8, solver='full'):
    """
    X: numpy array, shape [n_samples, n_features]
    n_components: int, target dimension
    solver: PCA svd_solver
    return: numpy array, shape [n_samples, n_components]
    """
    pca = PCA(n_components=n_components, svd_solver=solver)
    X_reduced = pca.fit_transform(X)
    return X_reduced

def softmax(x, temperature=1.0):
    x = x / temperature
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def aggregate(client_weights, client_vectors, global_weights, **kwargs):
    history_window = int(os.environ.get('SERVER_FEDYOGA_HISTORY_WINDOW', 5))
    pca_dim = int(os.environ.get('SERVER_FEDYOGA_PCA_DIM', 4))
    softmax_temperature = float(os.environ.get('SERVER_FEDYOGA_SOFTMAX_TEMPERATURE', 1.0))
    lossdrop_weight = float(os.environ.get('SERVER_FEDYOGA_LOSSDROP_WEIGHT', 1.0))
    gradvar_weight = float(os.environ.get('SERVER_FEDYOGA_GRADVAR_WEIGHT', 1.0))
    pca_solver = os.environ.get('SERVER_FEDYOGA_PCA_SOLVER', 'full')
    norm_eps = float(os.environ.get('SERVER_FEDYOGA_NORM_EPS', 1e-8))
    # Add clipping threshold for Non-IID scenarios
    clip_threshold = float(os.environ.get('SERVER_FEDYOGA_CLIP_THRESHOLD', 10.0))
    
    num_clients = len(client_weights)
    
    print(f"[FedYOGA] Aggregating {num_clients} clients with PCA_DIM={pca_dim}, CLIP_THRESHOLD={clip_threshold}")
    
    # 1. 計算 Ucli_r = wi - global_weights
    Ucli_list = []
    for i, wi in enumerate(client_weights):
        u = []
        for k in global_weights.keys():
            # 轉為 float32 再轉成 numpy，以供 PCA 使用
            diff = (wi[k] - global_weights[k]).float().cpu().numpy().flatten()
            u.append(diff)
        u_concat = np.concatenate(u)
        
        # Check for NaN/Inf in individual client
        if np.isnan(u_concat).any() or np.isinf(u_concat).any():
            print(f"[WARNING] Client {i+1}: Found NaN/Inf in weight differences")
            u_concat = np.nan_to_num(u_concat, nan=0.0, posinf=0.0, neginf=0.0)
        
        Ucli_list.append(u_concat)
    
    Ucli_arr = np.stack(Ucli_list)  # shape: [num_clients, total_params]
    
    # Statistics for debugging Non-IID scenarios
    print(f"[FedYOGA] Weight difference statistics:")
    print(f"  - Shape: {Ucli_arr.shape}")
    print(f"  - Mean: {np.mean(Ucli_arr):.6f}, Std: {np.std(Ucli_arr):.6f}")
    print(f"  - Min: {np.min(Ucli_arr):.6f}, Max: {np.max(Ucli_arr):.6f}")
    
    # Handle Inf values first by converting them to NaN so they can be imputed
    if np.isinf(Ucli_arr).any():
        inf_count = np.isinf(Ucli_arr).sum()
        print(f"[WARNING] Detected {inf_count} Inf values in client weight differences! Converting to NaN for imputation.")
        Ucli_arr = np.where(np.isinf(Ucli_arr), np.nan, Ucli_arr)

    # Column-wise (feature-wise) mean imputation (ignore NaNs when computing the mean)
    # This is more robust than replacing all NaNs with 0, especially in Non-IID scenarios.
    col_mean = np.nanmean(Ucli_arr, axis=0)
    # If a column is all-NaN, nanmean returns NaN for that column: fallback to 0.0
    nan_cols = np.isnan(col_mean)
    if nan_cols.any():
        print(f"[WARNING] Found {nan_cols.sum()} all-NaN feature columns; filling their mean with 0.0")
        col_mean[nan_cols] = 0.0

    nan_total = np.isnan(Ucli_arr).sum()
    if nan_total > 0:
        print(f"[INFO] Imputing {nan_total} NaN entries using column means")
        # Broadcast column means to rows and replace NaNs
        Ucli_arr = np.where(np.isnan(Ucli_arr), col_mean[None, :], Ucli_arr)

    # Safety: if any NaN/Inf remain (edge cases), replace them with 0.0
    if np.isnan(Ucli_arr).any() or np.isinf(Ucli_arr).any():
        rem_nan = np.isnan(Ucli_arr).sum()
        rem_inf = np.isinf(Ucli_arr).sum()
        print(f"[WARNING] After imputation, remaining NaN: {rem_nan}, Inf: {rem_inf}. Replacing them with 0.0")
        Ucli_arr = np.nan_to_num(Ucli_arr, nan=0.0, posinf=0.0, neginf=0.0)

    # Clip extreme values for Non-IID stability
    max_abs_val = np.abs(Ucli_arr).max()
    if max_abs_val > clip_threshold:
        print(f"[WARNING] Extreme values detected (max: {max_abs_val:.2f}), clipping to ±{clip_threshold}")
        Ucli_arr = np.clip(Ucli_arr, -clip_threshold, clip_threshold)
    
    # 2. 降維 (PCA) + Normalize
    n_components = min(pca_dim, Ucli_arr.shape[0], Ucli_arr.shape[1])
    print(f"[FedYOGA] Performing PCA reduction: {Ucli_arr.shape[1]} → {n_components} dimensions")
    
    try:
        Vcli_arr = pca_reduce(Ucli_arr, n_components=n_components, solver=pca_solver)
    except Exception as e:
        print(f"[ERROR] PCA failed: {e}")
        print(f"[FALLBACK] Using simple averaging instead of FedYOGA")
        # Fallback to FedAvg
        agg_weights = OrderedDict()
        for k in global_weights.keys():
            agg_weights[k] = torch.zeros_like(global_weights[k], dtype=torch.float32)
        for wi in client_weights:
            for k in agg_weights.keys():
                agg_weights[k] += wi[k] / num_clients
        return agg_weights
    
    Vcli_arr = Vcli_arr / (np.linalg.norm(Vcli_arr, axis=1, keepdims=True) + norm_eps)
    # 3. 歷史窗口平均 (假設 client_vectors 裡有歷史)
    Hcli_arr = []
    for i, vec in enumerate(client_vectors):
        history = vec.get('history', [])
        history = history[-history_window:] + [Vcli_arr[i]]
        h = np.mean(np.stack(history), axis=0)
        Hcli_arr.append(h)
        vec['history'] = history
    # 4. 拼接 LossDrop, GradVariance，並加權
    ClientVector_arr = []
    for i, vec in enumerate(client_vectors):
        loss_drop = get_loss_drop(vec.get('results', None)) if 'results' in vec else vec.get('loss_drop', 0.0)
        grad_var = get_grad_var(vec.get('weights_history', None)) if 'weights_history' in vec else vec.get('grad_var', 0.0)
        client_vec = np.concatenate([
            Hcli_arr[i],
            [lossdrop_weight * loss_drop],
            [gradvar_weight * grad_var]
        ])
        ClientVector_arr.append(client_vec)
    ClientVector_arr = np.stack(ClientVector_arr)  # shape: [num_clients, n_components+2]
    # 5. 計算聚合權重 ai (softmax)
    scores = np.linalg.norm(ClientVector_arr, axis=1)
    ai = softmax(scores, temperature=softmax_temperature)
    # 6. 聚合 Wglo_r = Σ(Wcli_r × ai)
    agg_weights = OrderedDict()
    for k in global_weights.keys():
        agg_weights[k] = torch.zeros_like(global_weights[k], dtype=torch.float32)
    for wi, a in zip(client_weights, ai):
        for k in agg_weights.keys():
            agg_weights[k] += a * wi[k]
    return agg_weights

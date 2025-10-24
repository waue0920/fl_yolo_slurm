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


def compute_balanced_client_weights(client_weights, client_data_info=None):
    """
    方案 B: 根據客戶端的數據多樣性計算平衡權重
    
    目的: 降低過度代表某個類別的客戶端 (如 Client 4 的 91% Rider)
    
    Args:
        client_weights: list of OrderedDict，每個客戶端的權重
        client_data_info: list of dict，每個客戶端的數據統計信息
                         格式: [{'class_counts': {...}, 'total_samples': ...}, ...]
    
    Returns:
        numpy array, shape [num_clients], 正規化後的權重 (sum=1)
    """
    num_clients = len(client_weights)
    
    # 如果沒有提供數據信息，使用均等權重
    if client_data_info is None or len(client_data_info) == 0:
        print("[FedYOGA] No client data info provided, using uniform weights")
        return np.ones(num_clients) / num_clients
    
    # 計算每個客戶端的「數據多樣性評分」
    diversity_scores = np.ones(num_clients)
    
    for i, info in enumerate(client_data_info):
        if info is None or 'class_counts' not in info:
            continue
        
        class_counts = info['class_counts']
        total_samples = sum(class_counts.values()) if isinstance(class_counts, dict) else 1
        
        if total_samples == 0:
            continue
        
        # 找出最主導的類別比例
        max_class_count = max(class_counts.values()) if isinstance(class_counts, dict) else total_samples
        max_class_ratio = max_class_count / total_samples
        
        # 根據不平衡程度調整權重
        # 如果某類別 > 90% (極端不平衡): 權重 = 0.25
        # 如果某類別 > 70% (嚴重不平衡): 權重 = 0.5
        # 如果某類別 > 50% (不平衡):     權重 = 0.75
        # 否則 (平衡):                  權重 = 1.0
        
        if max_class_ratio > 0.90:
            diversity_scores[i] = 0.25
            print(f"[FedYOGA-BalanceWeight] Client {i+1}: Extreme imbalance ({max_class_ratio*100:.1f}%), weight=0.25")
        elif max_class_ratio > 0.70:
            diversity_scores[i] = 0.5
            print(f"[FedYOGA-BalanceWeight] Client {i+1}: Severe imbalance ({max_class_ratio*100:.1f}%), weight=0.5")
        elif max_class_ratio > 0.50:
            diversity_scores[i] = 0.75
            print(f"[FedYOGA-BalanceWeight] Client {i+1}: Imbalance ({max_class_ratio*100:.1f}%), weight=0.75")
        else:
            diversity_scores[i] = 1.0
            print(f"[FedYOGA-BalanceWeight] Client {i+1}: Balanced ({max_class_ratio*100:.1f}%), weight=1.0")
    
    # 正規化權重（sum = 1）
    total_score = np.sum(diversity_scores)
    balanced_weights = diversity_scores / total_score if total_score > 0 else np.ones(num_clients) / num_clients
    
    print(f"[FedYOGA-BalanceWeight] Final weights: {[f'{w:.4f}' for w in balanced_weights]}")
    print(f"[FedYOGA-BalanceWeight] Weight reduction for Client 4: {(1/num_clients - balanced_weights[3])/(1/num_clients)*100:.1f}%")
    
    return balanced_weights


def aggregate(client_weights, client_vectors, global_weights, **kwargs):
    history_window = int(os.environ.get('SERVER_FEDYOGA_HISTORY_WINDOW', 5))
    pca_dim = int(os.environ.get('SERVER_FEDYOGA_PCA_DIM', 2))  # 修正：應該遠小於客戶端數量
    softmax_temperature = float(os.environ.get('SERVER_FEDYOGA_SOFTMAX_TEMPERATURE', 1.0))
    lossdrop_weight = float(os.environ.get('SERVER_FEDYOGA_LOSSDROP_WEIGHT', 1.0))
    gradvar_weight = float(os.environ.get('SERVER_FEDYOGA_GRADVAR_WEIGHT', 1.0))
    pca_solver = os.environ.get('SERVER_FEDYOGA_PCA_SOLVER', 'full')
    norm_eps = float(os.environ.get('SERVER_FEDYOGA_NORM_EPS', 1e-8))
    # 修正：裁剪閾值應該在 PCA 後應用，且更寬鬆
    clip_threshold_pre = float(os.environ.get('SERVER_FEDYOGA_CLIP_THRESHOLD_PRE', 1000.0))  # PCA 前粗略裁剪
    clip_threshold_post = float(os.environ.get('SERVER_FEDYOGA_CLIP_THRESHOLD_POST', 10.0))  # PCA 後精細裁剪
    
    num_clients = len(client_weights)
    
    print(f"[FedYOGA] Aggregating {num_clients} clients with PCA_DIM={pca_dim}")
    print(f"[FedYOGA] Clipping thresholds: PRE={clip_threshold_pre}, POST={clip_threshold_post}")
    
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

    # 修正：PCA 前只做寬鬆的極端值保護，避免破壞信號
    max_abs_val = np.abs(Ucli_arr).max()
    if max_abs_val > clip_threshold_pre:
        print(f"[WARNING] Pre-PCA: Extreme values detected (max: {max_abs_val:.2f}), clipping to ±{clip_threshold_pre}")
        Ucli_arr = np.clip(Ucli_arr, -clip_threshold_pre, clip_threshold_pre)
    
    print(f"[FedYOGA] Pre-PCA statistics:")
    print(f"  - Mean: {np.mean(Ucli_arr):.6f}, Std: {np.std(Ucli_arr):.6f}")
    print(f"  - Min: {np.min(Ucli_arr):.6f}, Max: {np.max(Ucli_arr):.6f}")
    
    # 2. 降維 (PCA) + Normalize
    # 修正：確保 PCA 維度遠小於客戶端數量，避免過擬合
    n_components = min(pca_dim, max(1, Ucli_arr.shape[0] - 1), Ucli_arr.shape[1])
    print(f"[FedYOGA] Performing PCA reduction: {Ucli_arr.shape[1]} → {n_components} dimensions")
    
    try:
        Vcli_arr = pca_reduce(Ucli_arr, n_components=n_components, solver=pca_solver)
        
        # 修正：PCA 後再做精細裁剪，保護降維後的特徵空間
        pca_max = np.abs(Vcli_arr).max()
        if pca_max > clip_threshold_post:
            print(f"[WARNING] Post-PCA: Clipping features from ±{pca_max:.2f} to ±{clip_threshold_post}")
            Vcli_arr = np.clip(Vcli_arr, -clip_threshold_post, clip_threshold_post)
        
        print(f"[FedYOGA] Post-PCA statistics:")
        print(f"  - Mean: {np.mean(Vcli_arr):.6f}, Std: {np.std(Vcli_arr):.6f}")
        print(f"  - Min: {np.min(Vcli_arr):.6f}, Max: {np.max(Vcli_arr):.6f}")
        
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
    
    # 正規化：每個客戶端的 PCA 特徵向量
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
    
    # 方案 B: 應用數據平衡加權（可選）
    enable_balance_weight = os.environ.get('SERVER_FEDYOGA_ENABLE_BALANCE_WEIGHT', 'false').lower() == 'true'
    if enable_balance_weight:
        # 從 client_vectors 中提取數據分佈信息
        client_data_info = []
        for vec in client_vectors:
            if 'data_distribution' in vec:
                client_data_info.append(vec['data_distribution'])
            else:
                client_data_info.append(None)
        
        # 計算基於數據多樣性的權重
        balance_weights = compute_balanced_client_weights(client_weights, client_data_info)
        
        # 合併 FedYOGA 權重和平衡權重
        # ai = ai * balance_weights（逐元素乘積）
        ai = ai * balance_weights
        ai = ai / np.sum(ai)  # 重新正規化
        
        print(f"[FedYOGA] After balance weighting: {[f'{w:.4f}' for w in ai]}")
    
    # 6. 聚合 Wglo_r = Σ(Wcli_r × ai)
    print(f"[FedYOGA] Aggregation weights: {[f'{w:.4f}' for w in ai]}")
    
    agg_weights = OrderedDict()
    for k in global_weights.keys():
        agg_weights[k] = torch.zeros_like(global_weights[k], dtype=torch.float32)
    for wi, a in zip(client_weights, ai):
        for k in agg_weights.keys():
            agg_weights[k] += a * wi[k]
    
    # 修正：聚合後檢查並修復 BatchNorm 參數
    for k in agg_weights.keys():
        if 'running_mean' in k or 'running_var' in k or 'num_batches_tracked' in k:
            if torch.isnan(agg_weights[k]).any() or torch.isinf(agg_weights[k]).any():
                print(f"[FedYOGA-FIX] Detected NaN/Inf in {k}, resetting to safe defaults")
                if 'running_mean' in k:
                    agg_weights[k] = torch.zeros_like(agg_weights[k])
                elif 'running_var' in k:
                    agg_weights[k] = torch.ones_like(agg_weights[k])
                elif 'num_batches_tracked' in k:
                    agg_weights[k] = torch.zeros_like(agg_weights[k])
    
    return agg_weights

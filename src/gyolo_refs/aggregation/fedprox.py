
import torch
import os

def aggregate(client_weights, global_weights=None, **kwargs):
    """
    FedProx 聚合：FedAvg + Proximal term (mu)。
    參數：
        client_weights: List of model weights (list of dict or list of np.array)
        global_weights: 上一輪的全域權重 (dict)，可選
        mu: FedProx 的 proximal term 係數
    回傳：
        aggregated_weights: 與 client_weights[0] 結構相同的平均權重
    """

    mu = float(os.environ.get('SERVER_FEDPROX_MU', 0.01))
    if not client_weights:
        raise ValueError("client_weights is empty!")
    if global_weights is None:
        # 若沒給 global_weights，退化為 FedAvg
        keys = client_weights[0].keys()
        aggregated = {}
        for k in keys:
            stacked = torch.stack([cw[k].float() for cw in client_weights], dim=0)
            aggregated[k] = torch.mean(stacked, dim=0)
        return aggregated

    # FedProx: 平均 + proximal term
    keys = client_weights[0].keys()
    aggregated = {}
    for k in keys:
        stacked = torch.stack([cw[k].float() for cw in client_weights], dim=0)
        avg = torch.mean(stacked, dim=0)
        # proximal term: (1 - mu) * avg + mu * global
        aggregated[k] = (1 - mu) * avg + mu * global_weights[k].float()
    return aggregated

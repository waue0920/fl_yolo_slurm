
import torch

def aggregate(client_weights, **kwargs):
    """
    FedAvg 聚合：對所有 client 的權重做簡單平均。
    參數：
        client_weights: List of model weights (list of dict or list of np.array)
    回傳：
        aggregated_weights: 與 client_weights[0] 結構相同的平均權重
    """
    if not client_weights:
        raise ValueError("client_weights is empty!")

    # 假設每個 client_weights[i] 是 dict (key: param name, value: torch.Tensor)
    keys = client_weights[0].keys()
    aggregated = {}
    for k in keys:
        # 堆疊所有 client 的同一層權重 (torch.stack)
        stacked = torch.stack([cw[k].float() for cw in client_weights], dim=0)
        aggregated[k] = torch.mean(stacked, dim=0)
    return aggregated

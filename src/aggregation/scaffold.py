
import torch

def aggregate(client_weights, control_variate=None, **kwargs):
    """
    SCAFFOLD 聚合：FedAvg + control variate 校正。
    參數：
        client_weights: List of model weights (list of dict or list of np.array)
        control_variate: dict, 全域控制變數 (全域c)，可選
    回傳：
        aggregated_weights: 與 client_weights[0] 結構相同的平均權重
    """
    if not client_weights:
        raise ValueError("client_weights is empty!")
    keys = client_weights[0].keys()
    aggregated = {}
    for k in keys:
        stacked = torch.stack([cw[k].float() for cw in client_weights], dim=0)
        avg = torch.mean(stacked, dim=0)
        if control_variate is not None:
            # SCAFFOLD: 校正 global control variate
            cv = control_variate.get(k, 0)
            if not torch.is_tensor(cv):
                cv = torch.tensor(cv, dtype=avg.dtype, device=avg.device)
            aggregated[k] = avg + cv
        else:
            aggregated[k] = avg
    return aggregated


import torch
from collections import OrderedDict

def aggregate(client_weights, client_steps, server_weights):
    import os
    mu = float(os.environ.get('SERVER_FEDNOVA_MU', 0.0))
    lr = float(os.environ.get('SERVER_FEDNOVA_LR', 1.0))
    n_total = sum(client_steps)
    d_fednova = OrderedDict()
    # Initialize d_fednova as float type
    for k in server_weights.keys():
        d_fednova[k] = torch.zeros_like(server_weights[k], dtype=torch.float32)
    # Aggregate
    for wi, ni in zip(client_weights, client_steps):
        for k in server_weights.keys():
            di = wi[k] - server_weights[k]
            di_norm = di.float() / ni
            d_fednova[k] += ni * di_norm
    for k in d_fednova.keys():
        d_fednova[k] /= n_total
    w_server_new = OrderedDict()
    for k in server_weights.keys():
        w_server_new[k] = server_weights[k].float() + d_fednova[k]
    return w_server_new
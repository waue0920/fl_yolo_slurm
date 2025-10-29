import argparse
import torch
import glob
import sys
import os
import time
import pandas as pd
from collections import OrderedDict
from pathlib import Path
from aggregation import AGGREGATORS

import argparse
import torch
import glob
import sys
import os
import time
import pandas as pd
from collections import OrderedDict
from pathlib import Path
from aggregation import AGGREGATORS

# Add YOLOv9 directory to Python path for model imports
if 'WROOT' in os.environ:
    yolov9_path = os.path.join(os.environ['WROOT'], 'yolov9')
    if yolov9_path not in sys.path:
        sys.path.insert(0, yolov9_path)

def calculate_alg_complexity(algorithm=None, template_model=None, expected_clients=None, start_time=None):
    """
    計算聚合演算法的空間與通訊複雜度
    """
    BYTE = 8
    elapsed = None
    if start_time is not None:
        elapsed = time.time() - start_time
    num_params = sum(p.numel() for p in template_model.parameters()) if template_model is not None else 0
    param_bits = sum(p.numel() * p.element_size() * BYTE for p in template_model.parameters()) if template_model is not None else 0

    print(f"[SUMMARY]({algorithm}) Aggregation time (s): {elapsed:.3f}")

    # 基礎：Server 需要暫存 N 個客戶端權重
    base_server_bits = param_bits * expected_clients  # NP
    communication_bits = param_bits * expected_clients * 2  # 上傳 + 下載

    if algorithm == 'fedopt':
        # Server 額外需要：global_weights(P) + momentum(P) + variance(P) = 3P
        server_state_bits = param_bits * 3
        total_server_bits = base_server_bits + server_state_bits  # NP + 3P
        print(f"[SUMMARY]({algorithm}) SpaceComplexity <(N+3)P> (bit): {total_server_bits:,}")
        print(f"[SUMMARY]({algorithm}) CommunicationComplexity <2NP> (bit): {communication_bits:,}")
    elif algorithm == 'fedavgm':
        # Server 額外需要：global_weights(P) + server_momentum(P) = 2P  
        server_state_bits = param_bits * 2
        total_server_bits = base_server_bits + server_state_bits  # NP + 2P
        print(f"[SUMMARY]({algorithm}) SpaceComplexity <(N+2)P> (bit): {total_server_bits:,}")
        print(f"[SUMMARY]({algorithm}) CommunicationComplexity <2NP> (bit): {communication_bits:,}")
    elif algorithm == 'fedprox':
        # Server 額外需要：global_weights(P) = 1P
        server_state_bits = param_bits
        total_server_bits = base_server_bits + server_state_bits  # NP + 1P
        print(f"[SUMMARY]({algorithm}) SpaceComplexity <(N+1)P> (bit): {total_server_bits:,}")
        print(f"[SUMMARY]({algorithm}) CommunicationComplexity <2NP> (bit): {communication_bits:,}")
    elif algorithm == 'fednova':
        # Server 額外需要：global_weights(P) + client_steps(N*32 bits) ≈ 1P
        server_state_bits = param_bits 
        total_server_bits = base_server_bits + server_state_bits  # NP + 1P + N
        print(f"[SUMMARY]({algorithm}) SpaceComplexity <(N+1)P> (bit): {total_server_bits:,}")
        print(f"[SUMMARY]({algorithm}) CommunicationComplexity <2NP> (bit): {communication_bits:,}")
    elif algorithm == 'fedyoga':
        # Server 額外需要：global_weights(P) + client_history(N²P) + metadata(N*32)
        server_state_bits = param_bits + expected_clients**2 * param_bits 
        total_server_bits = base_server_bits + server_state_bits  # (N+1+N²)P
        print(f"[SUMMARY]({algorithm}) SpaceComplexity <(N+1+N²)P> (bit): {total_server_bits:,}")
        print(f"[SUMMARY]({algorithm}) CommunicationComplexity <2NP> (bit): {communication_bits:,}")
    else:
        # 一般算法：Server 額外需要 global_weights(P) = 1P
        server_state_bits = param_bits
        total_server_bits = base_server_bits + server_state_bits  # NP + 1P
        print(f"[SUMMARY]({algorithm}) SpaceComplexity <(N+1)P> (bit): {total_server_bits:,}")
        print(f"[SUMMARY]({algorithm}) CommunicationComplexity <2NP> (bit): {communication_bits:,}")

    print(f"[SUMMARY]({algorithm}) P= {num_params:,}, N= {expected_clients:,} (Counts)")

def federated_aggregate(input_dir: Path, output_file: Path, expected_clients: int, round_num: int, algorithm: str, **agg_kwargs):
    start_time = time.time()
    print("================================================================================")
    print(f"Starting Federated Aggregation: {algorithm}")
    print(f">> Input directory:  {input_dir}")
    print(f">> Output file:      {output_file}")
    print(f">> Expected clients: {expected_clients}")
    print(f">> Target round:     {round_num}")
    print(f">> Algorithm:        {algorithm}")
    print("================================================================================")

    # 1. Find all client weights for the specific round
    client_weights_pattern = str(input_dir / f"r{round_num}_c*" / "weights" / "best.pt")
    client_weights_paths = sorted(glob.glob(client_weights_pattern))
    print(f"Searching for pattern: r{round_num}_c*/weights/best.pt")
    num_found = len(client_weights_paths)
    if num_found != expected_clients:
        print(f"Error: Mismatch in number of clients!")
        print(f"  - Expected: {expected_clients}")
        print(f"  - Found:    {num_found} ({client_weights_paths})")
        print("  - Please check the training logs for failed clients before proceeding.")
        exit(1)
    if num_found == 0:
        print(f"Error: No client weights found in '{input_dir}'")
        exit(1)
    print(f"Found {num_found} client models for aggregation:")
    for path in client_weights_paths:
        print(f"  - {path}")

    # 2. Load all client model state_dicts and get template model
    all_state_dicts = []
    valid_client_paths = []  # Track which clients are valid
    template_model = None
    template_ckpt = None
    client_sizes = [1] * num_found  # Placeholder: assume equal sizes for now
    print(f"[INFO] Using placeholder for client sizes: {client_sizes}")

    for i, path in enumerate(client_weights_paths):
        try:
            path_obj = Path(path)  # Convert to Path object for .name attribute
            ckpt = torch.load(path, map_location='cpu')
            if 'model' in ckpt:
                state_dict = ckpt['model'].state_dict()
                # Check for NaN/Inf values and attempt to fix them
                fixed_params = []
                critical_params = []
                
                for key, param in state_dict.items():
                    has_nan = torch.isnan(param).any().item()
                    has_inf = torch.isinf(param).any().item()
                    
                    if has_nan or has_inf:
                        # Distinguish between fixable (BatchNorm stats) and critical parameters
                        is_bn_stat = 'running_mean' in key or 'running_var' in key or 'num_batches_tracked' in key
                        
                        if is_bn_stat:
                            # Fix BatchNorm statistics by resetting to safe defaults
                            if 'running_mean' in key:
                                state_dict[key] = torch.zeros_like(param)
                                fixed_params.append(f"{key} (reset to 0)")
                            elif 'running_var' in key:
                                state_dict[key] = torch.ones_like(param)
                                fixed_params.append(f"{key} (reset to 1)")
                            elif 'num_batches_tracked' in key:
                                state_dict[key] = torch.zeros_like(param)
                                fixed_params.append(f"{key} (reset to 0)")
                        else:
                            # Critical parameters (weights, biases) - these are serious issues
                            if has_nan:
                                print(f"[ERROR] Client {i+1} ({path_obj.parent.parent.name}): Critical parameter '{key}' contains NaN!")
                            if has_inf:
                                print(f"[ERROR] Client {i+1} ({path_obj.parent.parent.name}): Critical parameter '{key}' contains Inf!")
                            critical_params.append(key)
                
                # Report what was fixed
                if fixed_params:
                    print(f"[INFO] Client {i+1} ({path_obj.parent.parent.name}): Fixed {len(fixed_params)} BatchNorm statistics")
                    for fp in fixed_params[:3]:  # Show first 3
                        print(f"       - {fp}")
                    if len(fixed_params) > 3:
                        print(f"       - ... and {len(fixed_params)-3} more")
                
                # Only skip if critical parameters are affected
                if critical_params:
                    print(f"[ERROR] Client {i+1} has {len(critical_params)} critical parameters with NaN/Inf. Skipping this client.")
                    continue
                
                all_state_dicts.append(state_dict)
                valid_client_paths.append(path)  # Track this valid client
                # Use first valid client as template (also fix its state_dict)
                if template_model is None:
                    template_model = ckpt['model']
                    template_ckpt = ckpt
                    # Also fix the template model's state_dict if needed
                    if fixed_params:
                        template_model.load_state_dict(state_dict)
                        print(f"[INFO] Template model's BatchNorm statistics also fixed")
            else:
                all_state_dicts.append(ckpt)
                if i == 0:
                    template_ckpt = ckpt
        except Exception as e:
            print(f"\nError: Failed to load weight file: {path}")
            print(f"  - Reason: {e}")
            exit(1)

    # Check if we have at least one valid client
    if len(all_state_dicts) == 0:
        print("\n" + "="*80)
        print("ERROR: No valid client models found!")
        print("All clients were skipped due to NaN/Inf values in their weights.")
        print("="*80)
        print("\nPossible causes:")
        print("  1. Training diverged (gradient explosion)")
        print("  2. Learning rate too high for Non-IID data")
        print("  3. BatchNorm statistics corrupted")
        print("\nSuggestions:")
        print("  - Check client training logs for errors")
        print("  - Reduce learning rate")
        print("  - Retrain affected clients")
        print("  - Consider using different aggregation algorithm")
        exit(1)

    if template_model is None:
        print("Error: Could not load template model from client weights")
        exit(1)
    
    print(f"\n[INFO] Successfully loaded {len(all_state_dicts)} valid client models (skipped {expected_clients - len(all_state_dicts)})")

    # Special handling for fedyoga and fedawa: rebuild client_vectors with only valid clients
    if algorithm in ['fedyoga', 'fedawa']:
        print(f"[INFO] Rebuilding client_vectors for {len(valid_client_paths)} valid clients...")
        client_results_paths = []
        for w_path in valid_client_paths:
            # Convert weights/best.pt path to results.csv path
            path_obj = Path(w_path)
            r_path = path_obj.parent.parent / "results.csv"
            client_results_paths.append(str(r_path))
        
        agg_kwargs['global_weights'] = template_model.state_dict()
        client_vectors = []
        for w_path, r_path in zip(valid_client_paths, client_results_paths):
            ckpt = torch.load(w_path, map_location='cpu')
            if 'model' in ckpt:
                weights_history = [ckpt['model'].state_dict()]
            else:
                weights_history = [ckpt]
            try:
                if Path(r_path).exists():
                    df = pd.read_csv(r_path)
                    results_history = df.to_dict(orient='records')
                else:
                    print(f"[WARNING] Results file not found: {r_path}")
                    results_history = []
            except Exception as e:
                print(f"[WARNING] Failed to read {r_path}: {e}")
                results_history = []
            client_vectors.append({
                'weights_history': weights_history,
                'results': results_history,
                'history': []
            })
        agg_kwargs['client_vectors'] = client_vectors
        print(f"[INFO] Created {len(client_vectors)} client_vectors for {algorithm.upper()}")

    # 3. Perform aggregation
    agg_fn = AGGREGATORS.get(algorithm)
    if agg_fn is None:
        print(f"Error: Unknown aggregation algorithm: {algorithm}")
        exit(1)
    print(f"\nAggregating weights using {algorithm}...")
    
    # 根據演算法型態呼叫並處理返回值
    if algorithm == 'fedopt':
        # FedOpt 返回 (aggregated_weights, optimizer_state)
        aggregated, optimizer_state = agg_fn(all_state_dicts, **agg_kwargs)
        # 保存 optimizer_state 供下一轮使用
        opt_state_path = input_dir / f"fedopt_state.pt"
        torch.save(optimizer_state, opt_state_path)
        print(f"[INFO] Saved FedOpt optimizer state to: {opt_state_path}")
    elif algorithm == 'fedavgm':
        # FedAvgM 返回 (aggregated_weights, server_momentum)
        aggregated, server_momentum = agg_fn(all_state_dicts, **agg_kwargs)
    elif algorithm == 'fedawa':
        # FedAWA 返回 (aggregated_weights, T_weights_state)
        aggregated, T_weights_state = agg_fn(all_state_dicts, **agg_kwargs)
        # 保存 T_weights_state 供下一轮使用
        t_weights_path = input_dir / f"fedawa_state.pt"
        torch.save(T_weights_state, t_weights_path)
        print(f"[INFO] Saved FedAWA T_weights state to: {t_weights_path}")
    elif algorithm == 'fedyoga':
        # FedYOGA 返回 (aggregated_weights, T_weights_state)
        aggregated, T_weights_state = agg_fn(all_state_dicts, **agg_kwargs)
        # 保存 T_weights_state 供下一轮使用
        t_weights_path = input_dir / f"fedyoga_state.pt"
        torch.save(T_weights_state, t_weights_path)
        print(f"[INFO] Saved FedYOGA T_weights state to: {t_weights_path}")
    else:
        # 其他演算法返回 aggregated_weights dict
        aggregated = agg_fn(all_state_dicts, **agg_kwargs)
    
    print("Aggregation complete.")

    # 4. Save the aggregated model
    output_file.parent.mkdir(parents=True, exist_ok=True)
    print("Loading aggregated weights into template model...")
    template_model.load_state_dict(aggregated)
    
    # Calculate and display complexity
    calculate_alg_complexity(
        algorithm=algorithm,
        template_model=template_model,
        expected_clients=len(all_state_dicts),
        start_time=start_time
    )
    
    model_to_save = {
        'model': template_model,
        'optimizer': template_ckpt.get('optimizer', None),
        'epoch': template_ckpt.get('epoch', -1),
        'training_results': template_ckpt.get('training_results', None)
    }
    torch.save(model_to_save, output_file)
    print(f"\nSuccessfully saved aggregated model to:")
    print(f"  -> {output_file}")
    print("================================================================================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Federated Aggregation Script for YOLOv9 models.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--input-dir', type=Path, required=True, help="Directory containing the client training outputs for a single round.")
    parser.add_argument('--output-file', type=Path, required=True, help="Full path for the output aggregated model.")
    parser.add_argument('--expected-clients', type=int, required=True, help="The number of client models expected to be in the input directory.")
    parser.add_argument('--round', type=int, required=True, help="Round number to filter client weights (e.g., 1, 2, 3, ...).")
    parser.add_argument('--algorithm', type=str, default='fedavg', choices=AGGREGATORS.keys(), help="Aggregation algorithm to use.")
    # 可擴充：parser.add_argument('--mu', type=float, default=0.01, help="FedProx mu value")
    args = parser.parse_args()

    server_fedprox_mu_env = os.environ.get('SERVER_FEDPROX_MU')

    # 需要先載入 template_model 以取得權重結構
    client_weights_pattern = str(args.input_dir / f"r{args.round}_c*" / "weights" / "best.pt")
    client_weights_paths = sorted(glob.glob(client_weights_pattern))
    if not client_weights_paths:
        print(f"Error: No client weights found for aggregation parameter preparation.")
        exit(1)
    ckpt = torch.load(client_weights_paths[0], map_location='cpu')
    if 'model' in ckpt:
        template_model = ckpt['model']
    else:
        print("Error: aggregation requires a model object in checkpoint.")
        exit(1)

    # 各演算法額外參數準備
    agg_kwargs = {}
    
    if args.algorithm == 'fedprox':
        if server_fedprox_mu_env is not None:
            try:
                agg_kwargs['mu'] = float(server_fedprox_mu_env)
                print(f"[INFO] Using SERVER_FEDPROX_MU from env: {agg_kwargs['mu']}")
            except Exception as e:
                print(f"[WARN] SERVER_FEDPROX_MU env parse error: {e}, fallback to arg/default")
                agg_kwargs['mu'] = getattr(args, 'mu', 0.01)
        else:
            agg_kwargs['mu'] = getattr(args, 'mu', 0.01)
    
    if args.algorithm == 'fednova':
        agg_kwargs['server_weights'] = template_model.state_dict()
        agg_kwargs['client_steps'] = [1] * args.expected_clients  # 可改為真實步數
    
    if args.algorithm == 'fedopt':
        agg_kwargs['global_weights'] = template_model.state_dict()
        # 載入前一輪的 optimizer_state (若存在)
        if args.round > 1:
            opt_state_path = args.input_dir / f"fedopt_state.pt"
            if opt_state_path.exists():
                try:
                    agg_kwargs['optimizer_state'] = torch.load(opt_state_path, map_location='cpu')
                    print(f"[INFO] Loaded previous FedOpt optimizer state from: {opt_state_path}")
                except Exception as e:
                    print(f"[WARN] Failed to load FedOpt optimizer state: {e}, using fresh state")
            else:
                print(f"[WARN] FedOpt optimizer state not found at: {opt_state_path}, using fresh state")
    
    if args.algorithm == 'fedavgm':
        agg_kwargs['client_sizes'] = [1] * args.expected_clients
        agg_kwargs['global_weights'] = template_model.state_dict()
        agg_kwargs['server_momentum'] = {k: torch.zeros_like(v) for k, v in template_model.state_dict().items()}
    
    if args.algorithm == 'fedawa':
        agg_kwargs['global_weights'] = template_model.state_dict()
        # 载入前一轮的 T_weights 状态 (若存在)
        if args.round > 1:
            t_weights_path = args.input_dir / f"fedawa_state.pt"
            if t_weights_path.exists():
                try:
                    agg_kwargs['T_weights_state'] = torch.load(t_weights_path, map_location='cpu')
                    print(f"[INFO] Loaded previous FedAWA T_weights state from: {t_weights_path}")
                except Exception as e:
                    print(f"[WARN] Failed to load FedAWA T_weights state: {e}, using fresh state")
            else:
                print(f"[INFO] FedAWA T_weights state not found at: {t_weights_path}, using fresh state")
    
    if args.algorithm == 'fedyoga':
        agg_kwargs['global_weights'] = template_model.state_dict()
        # 载入前一轮的 T_weights 状态 (若存在)
        if args.round > 1:
            t_weights_path = args.input_dir / f"fedyoga_state.pt"
            if t_weights_path.exists():
                try:
                    agg_kwargs['T_weights_state'] = torch.load(t_weights_path, map_location='cpu')
                    print(f"[INFO] Loaded previous FedYOGA T_weights state from: {t_weights_path}")
                except Exception as e:
                    print(f"[WARN] Failed to load FedYOGA T_weights state: {e}, using fresh state")
            else:
                print(f"[INFO] FedYOGA T_weights state not found at: {t_weights_path}, using fresh state")
    
    federated_aggregate(args.input_dir, args.output_file, args.expected_clients, args.round, args.algorithm, **agg_kwargs)

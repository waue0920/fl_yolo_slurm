import argparse
import torch
import glob
import sys
import os
from collections import OrderedDict
from pathlib import Path
from aggregation import AGGREGATORS

# Add YOLOv9 directory to Python path for model imports
if 'WROOT' in os.environ:
    yolov9_path = os.path.join(os.environ['WROOT'], 'yolov9')
    if yolov9_path not in sys.path:
        sys.path.insert(0, yolov9_path)

def federated_aggregate(input_dir: Path, output_file: Path, expected_clients: int, round_num: int, algorithm: str, **agg_kwargs):
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
    template_model = None
    template_ckpt = None
    for i, path in enumerate(client_weights_paths):
        try:
            ckpt = torch.load(path, map_location='cpu')
            if 'model' in ckpt:
                all_state_dicts.append(ckpt['model'].state_dict())
                if i == 0:
                    template_model = ckpt['model']
                    template_ckpt = ckpt
            else:
                all_state_dicts.append(ckpt)
        except Exception as e:
            print(f"\nError: Failed to load weight file: {path}")
            print(f"  - Reason: {e}")
            exit(1)
    if template_model is None:
        print("Error: Could not load template model from client weights")
        exit(1)

    # 3. Perform aggregation
    agg_fn = AGGREGATORS.get(algorithm)
    if agg_fn is None:
        print(f"Error: Unknown aggregation algorithm: {algorithm}")
        exit(1)
    print(f"\nAggregating weights using {algorithm}...")
    # 可根據演算法需求傳遞額外參數
    aggregated = agg_fn(all_state_dicts, **agg_kwargs)
    print("Aggregation complete.")

    # 4. Save the aggregated model
    output_file.parent.mkdir(parents=True, exist_ok=True)
    print("Loading aggregated weights into template model...")
    template_model.load_state_dict(aggregated)
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

    # 可根據 args.algorithm 傳遞額外參數
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
    federated_aggregate(args.input_dir, args.output_file, args.expected_clients, args.round, args.algorithm, **agg_kwargs)

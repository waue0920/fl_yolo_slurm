import argparse
import torch
import glob
import sys
import os
from collections import OrderedDict
from pathlib import Path

# Add YOLOv9 directory to Python path for model imports
if 'WROOT' in os.environ:
    yolov9_path = os.path.join(os.environ['WROOT'], 'yolov9')
    if yolov9_path not in sys.path:
        sys.path.insert(0, yolov9_path)


def federated_average(input_dir: Path, output_file: Path, expected_clients: int, round_num: int):
    """
    Performs federated averaging of client model weights from a specified directory.

    Args:
        input_dir: The directory containing client outputs. This function will
                   search for 'r{round_num}_c*/weights/best.pt' within this directory.
        output_file: The full path to save the aggregated model weights.
        expected_clients: The number of client weights expected for the average.
        round_num: The specific round number to filter client weights.
    """
    print("================================================================================")
    print(">> Starting Federated Averaging")
    print(f">> Input directory:  {input_dir}")
    print(f">> Output file:      {output_file}")
    print(f">> Expected clients: {expected_clients}")
    print(f">> Target round:     {round_num}")
    print("================================================================================")

    # --- 1. Find all client weights for the specific round ---
    # New naming convention: r{round}_c{client} (e.g., r1_c1, r1_c2, r2_c1, ...)
    client_weights_pattern = str(input_dir / f"r{round_num}_c*" / "weights" / "best.pt")
    client_weights_paths = sorted(glob.glob(client_weights_pattern))
    
    print(f"Searching for pattern: r{round_num}_c*/weights/best.pt")
    
    num_found = len(client_weights_paths)

    # --- Safeguard Check ---
    if num_found != expected_clients:
        print(f"Error: Mismatch in number of clients!")
        print(f"  - Expected: {expected_clients}")
        print(f"  - Found:    {num_found} ({client_weights_paths})")
        print("  - Please check the training logs for failed clients before proceeding.")
        exit(1)

    if num_found == 0:
        print(f"Error: No client weights found in '{input_dir}'")
        exit(1)

    print(f"Found {num_found} client models for averaging:")
    for path in client_weights_paths:
        print(f"  - {path}")

    # --- 2. Load all client model state_dicts and get template model ---
    all_state_dicts = []
    template_model = None
    template_ckpt = None
    
    for i, path in enumerate(client_weights_paths):
        try:
            ckpt = torch.load(path, map_location='cpu')
            if 'model' in ckpt:
                all_state_dicts.append(ckpt['model'].state_dict())
                # Use the first client's model as template for the aggregated model
                if i == 0:
                    template_model = ckpt['model']
                    template_ckpt = ckpt
            else:
                # Handle cases where the saved file is the state_dict itself
                all_state_dicts.append(ckpt)
        except Exception as e:
            print(f"\nError: Failed to load weight file: {path}")
            print(f"  - Reason: {e}")
            exit(1)
    
    if template_model is None:
        print("Error: Could not load template model from client weights")
        exit(1)

    # --- 3. Perform federated averaging ---
    avg_state_dict = OrderedDict()
    
    print("\nAveraging weights...")
    for key in all_state_dicts[0].keys():
        if all_state_dicts[0][key].dtype.is_floating_point:
            layer_sum = sum(state_dict[key].float() for state_dict in all_state_dicts)
            avg_state_dict[key] = layer_sum / num_found
        else:
            avg_state_dict[key] = all_state_dicts[0][key]

    print("Averaging complete.")

    # --- 4. Save the aggregated model ---
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Load the averaged weights into the template model
    print("Loading averaged weights into template model...")
    template_model.load_state_dict(avg_state_dict)
    
    # Create checkpoint in the same format as original YOLOv9
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
        description="Federated Averaging Script for YOLOv9 models.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--input-dir',
        type=Path,
        required=True,
        help="Directory containing the client training outputs for a single round."
    )
    parser.add_argument(
        '--output-file',
        type=Path,
        required=True,
        help="Full path for the output aggregated model."
    )
    parser.add_argument(
        '--expected-clients',
        type=int,
        required=True,
        help="The number of client models expected to be in the input directory."
    )
    parser.add_argument(
        '--round',
        type=int,
        required=True,
        help="Round number to filter client weights (e.g., 1, 2, 3, ...)."
    )
    args = parser.parse_args()

    federated_average(args.input_dir, args.output_file, args.expected_clients, args.round)

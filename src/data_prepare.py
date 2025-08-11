import argparse
import os
import random
import yaml
import shutil
from pathlib import Path

def prepare_data(
    dataset_name: str,
    num_clients: int,
    seed: int,
    project_root: Path,
    source_dir: Path = None,
    output_dir: Path = None,
):
    """
    Splits a YOLOv5 format dataset into multiple partitions for federated learning.

    This function creates symlinks to the original data to save space.

    Args:
        dataset_name: The name of the dataset (e.g., 'kitti').
        num_clients: The number of clients to partition the data for.
        seed: The random seed for shuffling to ensure reproducibility.
        project_root: The absolute path to the project's root directory.
        source_dir: Optional. The path to the source dataset. 
                    Defaults to '{project_root}/datasets/{dataset_name}'.
        output_dir: Optional. The path to the output directory for federated data.
                    Defaults to '{project_root}/federated_data/{dataset_name}'.
    """
    print(f"--- Starting data preparation for dataset: {dataset_name} ---")
    print(f"--- Number of clients: {num_clients} ---")

    # If paths are not provided, construct them based on project_root
    if source_dir is None:
        source_dir = project_root / "datasets" / dataset_name
    if output_dir is None:
        output_dir = project_root / "federated_data" / f"{dataset_name}_{num_clients}"

    source_images_dir = source_dir / "images" / "train"
    source_labels_dir = source_dir / "labels" / "train"
    source_yaml_path = project_root / "data" / f"{dataset_name}.yaml"

    # --- 1. Validation ---
    if not source_images_dir.is_dir():
        print(f"Error: Source image directory not found at: {source_images_dir}")
        exit(1)
    if not source_labels_dir.is_dir():
        print(f"Error: Source label directory not found at: {source_labels_dir}")
        exit(1)
    if not source_yaml_path.is_file():
        print(f"Error: Source YAML file not found at: {source_yaml_path}")
        exit(1)

    # --- Anti-fool mechanism: Check if data is already partitioned ---
    if output_dir.exists() and any(output_dir.iterdir()):
         print(f"Output directory '{output_dir}' already exists and is not empty. Skipping preparation.")
         return

    print(f"Reading images from: {source_images_dir}")
    image_files = sorted([f for f in os.listdir(source_images_dir) if f.endswith(('.jpg', '.png', '.jpeg'))])
    
    if not image_files:
        print(f"Error: No images found in {source_images_dir}")
        exit(1)

    print(f"Found {len(image_files)} total images.")

    # --- 2. Shuffle and Split ---
    random.seed(seed)
    random.shuffle(image_files)
    file_chunks = [image_files[i::num_clients] for i in range(num_clients)]
    print(f"Successfully shuffled and split data into {num_clients} chunks.")

    # --- 3. Create Output Directories and Symlinks ---
    print(f"Creating base output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(source_yaml_path, 'r') as f:
        original_yaml_data = yaml.safe_load(f)
        # Resolve the original validation path relative to the source dataset directory
        original_val_path = source_dir / original_yaml_data['val']

    for i in range(1, num_clients + 1):
        client_id = f"c{i}"
        client_dir = output_dir / client_id
        client_images_dir = client_dir / "images" / "train"
        client_labels_dir = client_dir / "labels" / "train"

        print(f"Processing {client_id}...")
        client_images_dir.mkdir(parents=True, exist_ok=True)
        client_labels_dir.mkdir(parents=True, exist_ok=True)

        # Create symlinks
        for image_name in file_chunks[i-1]:
            base_name = Path(image_name).stem
            label_name = f"{base_name}.txt"

            source_image_path = source_images_dir / image_name
            source_label_path = source_labels_dir / label_name

            if source_label_path.exists():
                os.symlink(source_image_path.resolve(), client_images_dir / image_name)
                os.symlink(source_label_path.resolve(), client_labels_dir / label_name)
            else:
                print(f"Warning: Label file not found for {image_name}, skipping.")
        
        print(f"  - Symlinked {len(file_chunks[i-1])} images and labels for {client_id}.")

        # --- 4. Generate Client YAML file ---
        client_yaml_data = original_yaml_data.copy()
        # Use absolute paths for both path and val (like the previous working version)
        client_yaml_data['path'] = str(client_dir)
        client_yaml_data['train'] = 'images/train'
        client_yaml_data['val'] = str(project_root / "datasets" / dataset_name / "images" / "val")
        
        client_yaml_path = output_dir / f"{client_id}.yaml"
        with open(client_yaml_path, 'w') as f:
            yaml.dump(client_yaml_data, f, sort_keys=False, default_flow_style=False)
        
        print(f"  - Generated YAML config at: {client_yaml_path}")

    print("--- Data preparation complete. ---")


if __name__ == "__main__":
    # This script is in 'src/', so the project root is one level up.
    PROJECT_ROOT = Path(__file__).resolve().parent.parent

    parser = argparse.ArgumentParser(description="Partition a dataset for Federated Learning.")
    
    parser.add_argument('--dataset-name', type=str, required=True, help='Name of the dataset (e.g., kitti).')
    parser.add_argument('--num-clients', type=int, default=4, help='Number of clients.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility.')
    
    # Optional arguments for custom paths, useful for testing
    parser.add_argument('--source-dir', type=Path, default=None, help='(Optional) Path to the source dataset directory.')
    parser.add_argument('--output-dir', type=Path, default=None, help='(Optional) Path to the output directory.')
    
    args = parser.parse_args()

    prepare_data(
        dataset_name=args.dataset_name,
        num_clients=args.num_clients,
        seed=args.seed,
        project_root=PROJECT_ROOT,
        source_dir=args.source_dir,
        output_dir=args.output_dir,
    )
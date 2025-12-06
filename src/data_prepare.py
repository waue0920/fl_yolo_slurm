import argparse
import numpy as np
import os
import random
import yaml
import shutil
import json
from pathlib import Path


def split_data_dirichlet(images_list, labels_list, client_num, alpha):
    """
    Perform Non-IID data splitting using Dirichlet distribution.
    
    This is an independent implementation, not relying on FedML, to avoid unnecessary dependencies.
    
    Args:
        images_list: List of image filenames
        labels_list: List of corresponding class labels
        client_num: Number of clients
        alpha: Dirichlet parameter (smaller means less uniform)
    
    Returns:
        client_dict: {client_id: [image_filenames]}
    """
    labels_array = np.array(labels_list)
    num_classes = len(np.unique(labels_array))
    
    # Build index dict for each class
    class_indices = {}
    for label in np.unique(labels_array):
        class_indices[label] = np.where(labels_array == label)[0].tolist()
    
    # Initialize index list for each client
    client_indices = [[] for _ in range(client_num)]
    
    # Use Dirichlet distribution for each class
    for label, indices in class_indices.items():
        np.random.shuffle(indices)
        
        # Generate Dirichlet proportions
        proportions = np.random.dirichlet(np.repeat(alpha, client_num))
        
        # Calculate sample count for each client
        counts = (proportions * len(indices)).astype(int)
        
        # Correct total count deviation due to rounding
        diff = len(indices) - counts.sum()
        if diff > 0:
            # Add to the top diff clients with largest proportions
            top_clients = np.argsort(proportions)[-diff:]
            for client_id in top_clients:
                counts[client_id] += 1
        elif diff < 0:
            # Reduce from the client with the most samples
            for _ in range(abs(diff)):
                valid_clients = np.where(counts > 0)[0]
                if len(valid_clients) > 0:
                    client_to_reduce = valid_clients[np.argmax(counts[valid_clients])]
                    counts[client_to_reduce] -= 1
        
        # Assign indices to clients
        start_idx = 0
        for client_id in range(client_num):
            end_idx = start_idx + counts[client_id]
            client_indices[client_id].extend(indices[start_idx:end_idx])
            start_idx = end_idx
    
    # Convert to {client_id: [image_filenames]} format
    client_dict = {}
    for client_id, indices in enumerate(client_indices):
        np.random.shuffle(indices)
        client_dict[client_id] = [images_list[idx] for idx in indices]
    
    return client_dict
def prepare_data(
    dataset_name: str,
    num_clients: int,
    seed: int,
    project_root: Path,
    source_dir: Path = None,
    output_dir: Path = None,
    lda_alpha: float = 100.0,
):
    """
    Splits a YOLOv9-compatible dataset into multiple partitions for federated learning.

    This function creates symlinks to the original data to save space.
    Supports: COCO (special handling) + standard YOLOv9 datasets (KITTI, SIM10K, Foggy, Cityscapes, BDD100K)

    Args:
        dataset_name: The name of the dataset (e.g., 'coco', 'kitti', 'bdd100k').
        num_clients: The number of clients to partition the data for.
        seed: The random seed for shuffling to ensure reproducibility.
        project_root: The absolute path to the project's root directory.
        source_dir: Optional. The path to the source dataset. 
                    Defaults to '{project_root}/datasets/{dataset_name}' or 'coco' for COCO.
        output_dir: Optional. The path to the output directory for federated data.
                    Defaults to '{project_root}/federated_data/{dataset_name}_{num_clients}'.
        lda_alpha: Dirichlet alpha for Non-IID split. >= 100 for IID, < 100 for Non-IID.
    """
    print(f"--- Starting data preparation for dataset: {dataset_name} ---")
    print(f"--- Number of clients: {num_clients} ---")

    # Configure paths based on dataset type
    is_coco = (dataset_name.lower() == "coco")
    
    if is_coco:
        # COCO dataset: special directory structure
        if source_dir is None:
            source_dir = project_root / "coco"
        source_images_dir = source_dir / "images" / "train2017"
        source_labels_dir = source_dir / "labels" / "train2017"
        source_yaml_path = project_root / "gyolo" / "data" / f"{dataset_name}.yaml"
        train_subdir = "train2017"  # COCO uses train2017
    else:
        # Standard YOLOv9 datasets (KITTI, SIM10K, Foggy, Cityscapes, BDD100K)
        if source_dir is None:
            source_dir = project_root / "datasets" / dataset_name
        source_images_dir = source_dir / "images" / "train"
        source_labels_dir = source_dir / "labels" / "train"
        source_yaml_path = project_root / "data" / f"{dataset_name}.yaml"
        train_subdir = "train"  # Standard datasets use train
        
    if output_dir is None:
        # Alpha >= 100 is considered IID, < 100 is Non-IID
        if lda_alpha >= 100.0:
            out_name = f"{dataset_name}_{num_clients}"
        else:
            # Format alpha value (e.g. 0.1 -> A010, 1.0 -> A100)
            alpha_str = ("%.2f" % lda_alpha).replace(".", "")
            out_name = f"{dataset_name}A{alpha_str}_{num_clients}"
        output_dir = project_root / "federated_data" / out_name

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
    
    if lda_alpha >= 100.0:
        # IID split: simple round-robin allocation
        print(f"[IID] Using round-robin split (alpha={lda_alpha})")
        file_chunks = [image_files[i::num_clients] for i in range(num_clients)]
        for i, chunk in enumerate(file_chunks):
            print(f"  Client {i+1}: {len(chunk)} images")
    else:
        # Non-IID split: using Dirichlet split
        print(f"[Non-IID] Using Dirichlet split (alpha={lda_alpha})")
        
        # Choose label reading method based on dataset type
        if is_coco:
            # COCO dataset: read from instances_train2017.json
            coco_ann_path = project_root / "coco" / "annotations" / "instances_train2017.json"
            
            if not coco_ann_path.is_file():
                print(f"[ERROR] COCO instances file not found at {coco_ann_path}")
                print("[ERROR] Non-IID split requires instances_train2017.json")
                exit(1)
            
            print(f"[Non-IID] Loading annotations from {coco_ann_path}")
            with open(coco_ann_path, 'r') as f:
                coco_instances = json.load(f)
            
            # Create image_id -> filename mapping
            image_id_to_filename = {img['id']: img['file_name'] for img in coco_instances['images']}
            
            # Create image_id -> category_ids mapping (one image may have multiple objects)
            image_to_categories = {}
            for ann in coco_instances['annotations']:
                img_id = ann['image_id']
                cat_id = ann['category_id']
                if img_id not in image_to_categories:
                    image_to_categories[img_id] = set()
                image_to_categories[img_id].add(cat_id)
            
            # Choose primary category for each image (take smallest category_id)
            filename_to_label = {}
            for img_id, filename in image_id_to_filename.items():
                if img_id in image_to_categories:
                    filename_to_label[filename] = min(image_to_categories[img_id])
                else:
                    filename_to_label[filename] = 0
        else:
            # Standard YOLOv9 datasets (KITTI, SIM10K, Foggy, Cityscapes, BDD100K): read from YOLO format label files
            print(f"[Non-IID] Reading labels from YOLO format files in {source_labels_dir}")
            filename_to_label = {}
            
            for img_file in image_files:
                base_name = Path(img_file).stem
                label_file = source_labels_dir / f"{base_name}.txt"
                
                if label_file.exists():
                    # Read first number from YOLO format label (class ID)
                    with open(label_file, 'r') as f:
                        lines = f.readlines()
                        if lines:
                            # YOLO format: class_id x_center y_center width height
                            class_id = int(lines[0].split()[0])
                            filename_to_label[img_file] = class_id
                        else:
                            # Empty label file, set to class 0
                            filename_to_label[img_file] = 0
                else:
                    # No label file, set to class 0
                    filename_to_label[img_file] = 0
        
        # Prepare data for FedML
        images_list = []
        labels_list = []
        for img_file in image_files:
            images_list.append(img_file)
            labels_list.append(filename_to_label.get(img_file, 0))
        
        labels_array = np.array(labels_list)
        num_classes = len(np.unique(labels_array))
        print(f"[Non-IID] Found {num_classes} unique classes in dataset")
        
        # Use Dirichlet split
        print(f"[Non-IID] Partitioning with FedML split_data_dirichlet...")
        np.random.seed(seed)
        
        # split_data_dirichlet returns dict: {client_id: [image_filenames]}
        client_dict = split_data_dirichlet(
            images_list=images_list,
            labels_list=labels_list,
            client_num=num_clients,
            alpha=lda_alpha
        )
        
        # Convert to file_chunks format
        file_chunks = []
        for client_id in range(num_clients):
            client_files = client_dict[client_id]
            file_chunks.append(client_files)
            print(f"  Client {client_id+1}: {len(client_files)} images")
        
        print(f"[Non-IID] Dirichlet split completed successfully")

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
        print(f"Processing {client_id}...")

        # Create directory structure based on dataset type
        client_images_dir = client_dir / "images" / train_subdir
        client_images_dir.mkdir(parents=True, exist_ok=True)
        
        client_labels_dir = client_dir / "labels" / train_subdir
        client_labels_dir.mkdir(parents=True, exist_ok=True)
        
        # COCO-specific: stuff labels and annotations
        if is_coco:
            source_stuff_labels_dir = project_root / "coco" / "stuff" / "train2017"
            client_stuff_labels_dir = client_dir / "stuff" / "train2017"
            if source_stuff_labels_dir.is_dir():
                client_stuff_labels_dir.mkdir(parents=True, exist_ok=True)
            
            client_annotations_dir = client_dir / "annotations"
            client_annotations_dir.mkdir(parents=True, exist_ok=True)
        
        # Create symlinks for images and labels
        symlink_img, symlink_label, symlink_stuff = 0, 0, 0
        client_image_names = set()  # For COCO captions generation
        
        for image_name in file_chunks[i-1]:
            base_name = Path(image_name).stem
            label_name = f"{base_name}.txt"
            source_image_path = source_images_dir / image_name
            source_label_path = source_labels_dir / label_name

            # Image symlink
            if source_image_path.exists():
                os.symlink(source_image_path.resolve(), client_images_dir / image_name)
                symlink_img += 1
                client_image_names.add(base_name)
            else:
                print(f"Warning: Image file not found for {image_name}, skipping.")

            # Label symlink
            if source_label_path.exists():
                os.symlink(source_label_path.resolve(), client_labels_dir / label_name)
                symlink_label += 1
            else:
                print(f"Warning: Label file not found for {image_name}, skipping.")

            # COCO-specific: stuff label symlink
            if is_coco:
                source_stuff_label_path = source_stuff_labels_dir / label_name
                if source_stuff_labels_dir.is_dir() and source_stuff_label_path.exists():
                    os.symlink(source_stuff_label_path.resolve(), client_stuff_labels_dir / label_name)
                    symlink_stuff += 1
        
        print(f"  - Created {symlink_img} image symlinks")
        print(f"  - Created {symlink_label} label symlinks")
        if is_coco and symlink_stuff > 0:
            print(f"  - Created {symlink_stuff} stuff symlinks")

        # COCO-specific: Generate client-specific captions_train2017.json
        if is_coco:
            coco_ann_path = project_root / "coco" / "annotations" / "captions_train2017.json"
            client_ann_path = client_annotations_dir / "captions_train2017.json"
            if coco_ann_path.is_file():
                with open(coco_ann_path, 'r') as f:
                    coco_ann = json.load(f)
                # Build image_id map
                image_id_map = {str(img['file_name']).split('.')[0]: img['id'] for img in coco_ann['images']}
                client_image_ids = set([image_id_map[name] for name in client_image_names if name in image_id_map])
                # Filter images/annotations
                client_images = [img for img in coco_ann['images'] if img['id'] in client_image_ids]
                client_annotations = [ann for ann in coco_ann['annotations'] if ann['image_id'] in client_image_ids]
                # Write client captions_train.json
                client_coco_ann = coco_ann.copy()
                client_coco_ann['images'] = client_images
                client_coco_ann['annotations'] = client_annotations
                with open(client_ann_path, 'w') as f:
                    json.dump(client_coco_ann, f)
                print(f"  - Generated client captions_train2017.json at {client_ann_path}")

        # --- 4. Generate Client YAML file ---
        client_yaml_data = original_yaml_data.copy()
        client_yaml_data['path'] = str(client_dir)
        client_yaml_data['train'] = f'images/{train_subdir}'
        
        if is_coco:
            # COCO-specific paths
            client_yaml_data['val'] = str(project_root / "coco" / "images" / "val2017")
            client_yaml_data['stuff'] = 'stuff/train2017'
            client_yaml_data['test'] = str(project_root / "coco" / "test-dev2017.txt")
        else:
            # Standard YOLOv9 datasets: use original validation path
            client_yaml_data['val'] = str(original_val_path)
        
        client_yaml_data.pop('download', None)  # Remove download field

        client_yaml_path = output_dir / f"{client_id}.yaml"
        with open(client_yaml_path, 'w') as f:
            yaml.dump(client_yaml_data, f, sort_keys=False, default_flow_style=False)

        print(f"  - Generated YAML config at: {client_yaml_path}")

    print("--- Data preparation complete. ---")


if __name__ == "__main__":
    # This script is in 'src/', so the project root is one level up.
    PROJECT_ROOT = Path(__file__).resolve().parent.parent

    parser = argparse.ArgumentParser(description="Partition a dataset for Federated Learning.")
    
    parser.add_argument('--dataset-name', type=str, required=True, help='Name of the dataset (e.g., coco).')
    parser.add_argument('--num-clients', type=int, default=4, help='Number of clients.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility.')
    parser.add_argument('--LDA-alpha', type=float, default=100.0, 
                        help='Dirichlet alpha for Non-IID split. Use 100 for IID, 0.1-1.0 for Non-IID.')
    
    # Optional arguments for custom paths
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
        lda_alpha=args.LDA_alpha,
    )
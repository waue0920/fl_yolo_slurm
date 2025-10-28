import os
import sys
from collections import defaultdict

CLASS_MAPPINGS = {
    'bdd100k': {
        0: 'Bike',
        1: 'Bus', 
        2: 'Car', 
        3: 'Motor', 
        4: 'Person', 
        5: 'Rider', 
        6: 'Traffic Light', 
        7: 'Traffic Sign',
        8: 'Train',
        9: 'Truck'
    },
    'kitti': {
        0: 'Car',
        1: 'Van',
        2: 'Truck',
        3: 'Pedestrian',
        4: 'Person Sitting',
        5: 'Cyclist',
        6: 'Tram',
        7: 'Misc'
    }
}

def count_labels_all_lines(label_dir):
    counts = defaultdict(int)
    for fname in os.listdir(label_dir):
        if not fname.endswith('.txt'):
            continue
        path = os.path.join(label_dir, fname)
        with open(path, 'r') as f:
            for line in f:
                if not line.strip():
                    continue
                class_id = int(line.strip().split()[0])
                counts[class_id] += 1
    return counts

def main(root_folder, dataset):
    if dataset not in CLASS_MAPPINGS:
        print(f"Dataset type must be one of {list(CLASS_MAPPINGS.keys())}")
        exit(2)
    class_names = CLASS_MAPPINGS[dataset]
    client_list = ['c1', 'c2', 'c3', 'c4']
    result = {class_name: [0]*len(client_list) for class_name in class_names.values()}
    unknown_classes = set()
    
    for idx, client in enumerate(client_list):
        label_train_dir = os.path.join(root_folder, client, 'labels', 'train')
        client_counts = count_labels_all_lines(label_train_dir)
        for class_id, count in client_counts.items():
            if class_id in class_names:
                result[class_names[class_id]][idx] = count
            else:
                unknown_classes.add(class_id)
        print(f"[DEBUG] {client}各 class_id 分布: {dict(client_counts)}")
    
    output_file = f"{root_folder.rstrip('/')}_DataDesc.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        head = "\t".join(["Class"] + [f'C{i+1}' for i in range(len(client_list))])
        f.write(f"{head}\n")
        for class_name in class_names.values():
            counts = "\t".join(str(num) for num in result[class_name])
            f.write(f"{class_name}\t{counts}\n")
        if unknown_classes:
            f.write(f"\n[Warning] Detected unknown class IDs: {sorted(list(unknown_classes))}\n")
    print(f"Done! Table output to {output_file}")
    if unknown_classes:
        print(f"[Warning] Detected unknown class IDs in your dataset: {sorted(list(unknown_classes))}")

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python data_desc.py <root_folder> <dataset_type>")
        print("Example: python data_desc.py ./bdd100kA010_4/ bdd100k")
        print("         python data_desc.py ./kittiA010_4/ kitti")
        exit(1)
    main(sys.argv[1], sys.argv[2])


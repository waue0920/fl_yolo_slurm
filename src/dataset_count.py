import os
import sys
from collections import defaultdict

def count_labels(label_dir):
    counts = defaultdict(int)
    for fname in os.listdir(label_dir):
        if not fname.endswith('.txt'):
            continue
        with open(os.path.join(label_dir, fname), 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = line.split()
                if data:
                    class_id = int(data[0])
                    counts[class_id] += 1
    return counts

def main(root_folder):
    # 假設所有標註在 <root_folder>/labels/ 或 <root_folder>/labels/train/ 下
    label_dirs = [os.path.join(root_folder, 'labels'), os.path.join(root_folder, 'labels', 'train')]
    all_counts = defaultdict(int)
    found = False
    for label_dir in label_dirs:
        if not os.path.exists(label_dir):
            continue
        found = True
        folder_counts = count_labels(label_dir)
        for class_id, cnt in folder_counts.items():
            all_counts[class_id] += cnt

    if not found:
        print("No labels directory found!")
        exit(2)
    
    print("ClassID\tCount")
    for class_id in sorted(all_counts):
        print(f"{class_id}\t{all_counts[class_id]}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python dataset_count.py <dataset_folder>")
        exit(1)
    main(sys.argv[1])


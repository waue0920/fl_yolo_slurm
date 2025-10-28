#!/usr/bin/env python3
"""
單獨用來批次驗證yolov9訓練後 runs/..../weights/*.pt 檔
將算出  mAP 等benchmark
使用方式：
python val_yolo_weights.py --weights-dir /path/to/weights --data data/kitti.yaml
"""

import argparse
import glob
import os
from pathlib import Path
import subprocess
import pandas as pd
from datetime import datetime

def parse_args():
    parser = argparse.ArgumentParser(description='Benchmark multiple YOLOv9 weights')
    parser.add_argument('--weights-dir', type=str, required=True, help='Directory containing .pt files')
    parser.add_argument('--data', type=str, required=True, help='dataset.yaml path')
    parser.add_argument('--img', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--batch-size', type=int, default=32, help='batch size')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--workers', type=int, default=8, help='max dataloader workers')
    parser.add_argument('--project', default='runs/benchmark', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--output-csv', default='benchmark_results.csv', help='output CSV file')
    return parser.parse_args()

def get_weight_files(weights_dir):
    """取得所有 .pt 檔案並排序"""
    weight_files = glob.glob(os.path.join(weights_dir, '*.pt'))
    # 排序：best.pt, last.pt, epoch0.pt, epoch10.pt, ...
    def sort_key(path):
        name = os.path.basename(path)
        if name == 'best.pt':
            return (0, 0)
        elif name == 'last.pt':
            return (1, 0)
        elif name.startswith('epoch'):
            epoch_num = int(name.replace('epoch', '').replace('.pt', ''))
            return (2, epoch_num)
        else:
            return (3, name)
    
    weight_files.sort(key=sort_key)
    return weight_files

def run_validation(weight_file, args):
    """執行單個權重檔的驗證"""
    print(f"\n{'='*80}")
    print(f"Validating: {os.path.basename(weight_file)}")
    print(f"{'='*80}")
    
    # 構建驗證命令
    cmd = [
        'python', 'val_dual.py',
        '--weights', weight_file,
        '--data', args.data,
        '--img', str(args.img),
        '--batch-size', str(args.batch_size),
        '--device', args.device,
        '--workers', str(args.workers),
        '--project', args.project,
        '--name', f"{args.name}_{os.path.basename(weight_file).replace('.pt', '')}",
        '--save-json',
        '--verbose'
    ]
    
    try:
        # 執行驗證
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        output = result.stdout
        
        # 解析結果（從輸出中提取 mAP）
        # 尋找類似 "all   748   4197   0.906   0.763   0.862   0.625" 的行
        for line in output.split('\n'):
            if 'all' in line and len(line.split()) >= 7:
                parts = line.split()
                try:
                    metrics = {
                        'precision': float(parts[3]),
                        'recall': float(parts[4]),
                        'mAP_0.5': float(parts[5]),
                        'mAP_0.5:0.95': float(parts[6])
                    }
                    return metrics
                except (ValueError, IndexError):
                    continue
        
        print(f"Warning: Could not parse metrics from output")
        return None
        
    except subprocess.CalledProcessError as e:
        print(f"Error validating {weight_file}: {e}")
        print(f"stderr: {e.stderr}")
        return None

def main():
    args = parse_args()
    
    # 檢查權重目錄
    if not os.path.isdir(args.weights_dir):
        print(f"Error: Weights directory not found: {args.weights_dir}")
        return
    
    # 取得所有權重檔案
    weight_files = get_weight_files(args.weights_dir)
    if not weight_files:
        print(f"Error: No .pt files found in {args.weights_dir}")
        return
    
    print(f"\nFound {len(weight_files)} weight files:")
    for wf in weight_files:
        print(f"  - {os.path.basename(wf)}")
    
    # 逐一驗證
    results = []
    for weight_file in weight_files:
        file_name = os.path.basename(weight_file)
        file_size_mb = os.path.getsize(weight_file) / (1024 * 1024)
        
        metrics = run_validation(weight_file, args)
        
        if metrics:
            results.append({
                'Weight File': file_name,
                'Size (MB)': f"{file_size_mb:.1f}",
                'Precision': f"{metrics['precision']:.4f}",
                'Recall': f"{metrics['recall']:.4f}",
                'mAP@0.5': f"{metrics['mAP_0.5']:.4f}",
                'mAP@0.5:0.95': f"{metrics['mAP_0.5:0.95']:.4f}"
            })
        else:
            results.append({
                'Weight File': file_name,
                'Size (MB)': f"{file_size_mb:.1f}",
                'Precision': 'N/A',
                'Recall': 'N/A',
                'mAP@0.5': 'N/A',
                'mAP@0.5:0.95': 'N/A'
            })
    
    # 生成結果表格
    df = pd.DataFrame(results)
    
    print(f"\n{'='*80}")
    print("BENCHMARK RESULTS")
    print(f"{'='*80}")
    print(df.to_string(index=False))
    
    # 儲存 CSV
    output_path = os.path.join(args.project, args.name, args.output_csv)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")
    
    # 找出最佳模型
    valid_results = [r for r in results if r['mAP@0.5'] != 'N/A']
    if valid_results:
        best_model = max(valid_results, key=lambda x: float(x['mAP@0.5']))
        print(f"\nBest Model (by mAP@0.5): {best_model['Weight File']} with mAP@0.5 = {best_model['mAP@0.5']}")

if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Validation script for federated learning models.
Evaluates the performance of aggregated models on arbitrary datasets via YOLOv9 core components.
"""

import argparse
import os
import sys
import torch
import json
from pathlib import Path
import json
import numpy as np

# ensure we can import yolov9 modules
SRC_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC_ROOT / 'yolov9'))
# ensure we can import yolov9 package
# add workspace root to path so 'yolov9' module is importable
sys.path.insert(0, str(SRC_ROOT))
from yolov9.utils.dataloaders import create_dataloader
from yolov9.utils.general import check_img_size
from yolov9.models.common import DetectMultiBackend
from yolov9.utils.torch_utils import select_device
from yolov9.utils.general import check_dataset, check_img_size, non_max_suppression, box_iou, xywh2xyxy
from yolov9.utils.metrics import ap_per_class
from yolov9.utils.dataloaders import create_dataloader
import numpy as np

def process_batch(detections, labels, iouv):
    """
    Return correct predictions matrix. Both sets of boxes are in (x1, y1, x2, y2) format.
    Arguments:
        detections (Array[N, 6]), x1, y1, x2, y2, conf, class
        labels (Array[M, 5]), class, x1, y1, x2, y2
    Returns:
        correct (Array[N, 10]), for 10 IoU levels
    """
    correct = torch.zeros(detections.shape[0], iouv.shape[0], dtype=torch.bool, device=iouv.device)
    iou = box_iou(labels[:, 1:], detections[:, :4])
    x = torch.where((iou >= iouv[0]) & (labels[:, 0:1] == detections[:, 5]))  # IoU above threshold and classes match
    if x[0].shape[0]:
        matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()  # [label, detection, iou]
        if x[0].shape[0] > 1:
            matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
            matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        matches = torch.tensor(matches, device=iouv.device)
        correct[matches[:, 1].long()] = matches[:, 2:3] >= iouv
    return correct

def run_validation(model_path, data_config, output_dir, model_name="model", device_str=''):
    """
    Run YOLOv9 validation on the specified model and dataset.
    
    Args:
        model_path: Path to the model weights (.pt file)
        data_config: Path to the dataset configuration (.yaml file)
        output_dir: Directory to save validation results
        model_name: Name identifier for this model (for organizing results)
        device_str: Device string for PyTorch (e.g., '0' for GPU 0, 'cpu', etc.)
    
    Returns:
        dict: Validation metrics (mAP, etc.)
    """
    
    # Ensure output directory exists
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create model-specific output directory
    model_output_dir = output_dir / f"validation_{model_name}"
    model_output_dir.mkdir(parents=True, exist_ok=True)
    
    # --- Direct YOLOv9 inference and metric computation ---
    # Device and model setup
    device = select_device(device_str)
    model = DetectMultiBackend(model_path, device=device)
    gs = int(model.stride)  # grid size
    img_size = check_img_size(640, s=gs)

    # Dataset loader (validation split)
    data_dict = check_dataset(data_config)
    # create DataLoader for validation
    dataloader, dataset = create_dataloader(
        data_dict['val'], img_size, 16, gs,
        single_cls=False, rect=True
    )

    # Metrics
    conf_thres, iou_thres = 0.001, 0.6
    stats, seen = [], 0
    # 使用資料集配置檔案的類別定義，而不是模型的
    names = data_dict['names']  # 從 yaml 檔案載入
    nc = len(names)  # number of classes
    iouv = torch.linspace(0.5, 0.95, 10)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()
    
    # 準備統計數據收集
    jdict, stats, ap, ap_class = [], [], [], []
    
    # 資料集類別統計收集
    target_class_counts = np.zeros(nc, dtype=int)
    target_classes_present = set()
    
    # Iterate batches
    for batch_i, (imgs, targets, paths, shapes) in enumerate(dataloader):
        imgs = imgs.to(device).float() / 255.0  # 正規化到 0-1
        nb, _, height, width = imgs.shape  # batch size, channels, height, width
        
        with torch.no_grad():
            # 推論
            pred = model(imgs)
            # 處理 DetectMultiBackend 可能返回 list 的情況
            if isinstance(pred, list):
                pred = pred[0]  # 取第一個輸出（通常是主要的預測結果）
            
            # NMS
            targets[:, 2:] *= torch.tensor((width, height, width, height))  # to pixels
            pred = non_max_suppression(pred, conf_thres, iou_thres, labels=[], multi_label=True, agnostic=False)

        # 處理每張圖片的預測結果
        for si, pred_img in enumerate(pred):
            labels = targets[targets[:, 0] == si, 1:]
            nl = len(labels)
            tcls = labels[:, 0].tolist() if nl else []  # target class
            path = Path(paths[si])
            shape = shapes[si][0]
            seen += 1
            
            # 統計實際的 target classes
            for cls_idx in tcls:
                cls_idx = int(cls_idx)
                if 0 <= cls_idx < nc:
                    target_class_counts[cls_idx] += 1
                    target_classes_present.add(cls_idx)

            if len(pred_img) == 0:
                if nl:
                    stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
                continue

            # 預測結果 - 過濾無效類別
            predn = pred_img.clone()
            # 只保留在資料集類別範圍內的預測 (0 到 nc-1)
            valid_mask = (predn[:, 5] >= 0) & (predn[:, 5] < nc)
            predn = predn[valid_mask]
            
            if len(predn) == 0:
                if nl:
                    stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
                continue
            
            # 計算 metrics
            if nl:
                tbox = xywh2xyxy(labels[:, 1:5])  # target boxes
                labelsn = torch.cat((labels[:, 0:1], tbox), 1)  # native-space labels
                # 確保所有 tensor 都在同一個 device 上
                labelsn = labelsn.to(device)
                predn = predn.to(device)
                iouv = iouv.to(device)
                correct = process_batch(predn, labelsn, iouv)
            else:
                correct = torch.zeros(predn.shape[0], niou, dtype=torch.bool)
            
            # 確保統計數據的長度一致 - 使用過濾後的 predn
            stats.append((correct.cpu(), predn[:, 4].cpu(), predn[:, 5].cpu(), tcls))

    # 計算指標
    stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy
    if len(stats) and stats[0].any():
        tp, fp, p, r, f1, ap, ap_class = ap_per_class(*stats, plot=False, save_dir="", names=names)
        ap50, ap75, ap = ap[:, 0], ap[:, 5], ap.mean(1)  # AP@0.5, AP@0.75, AP@0.5:0.95
        mp, mr, map50, map75, map = p.mean(), r.mean(), ap50.mean(), ap75.mean(), ap.mean()
        nt = np.bincount(stats[3].astype(np.int64), minlength=nc)  # number of targets per class
    else:
        nt = torch.zeros(1)
        mp, mr, map50, map75, map = 0.0, 0.0, 0.0, 0.0, 0.0
        ap, ap_class = [], []

    # 收集詳細指標
    metrics = {
        'mAP@0.5': float(map50),
        'mAP@0.75': float(map75), 
        'mAP@0.5:0.95': float(map),
        'Precision': float(mp),
        'Recall': float(mr),
        'per_class_ap50': {},
        'per_class_ap75': {},
        'per_class_ap': {},
        'dataset_analysis': {
            'total_classes_defined': nc,
            'classes_with_samples': len(target_classes_present),
            'classes_present': sorted(list(target_classes_present)),
            'class_counts': target_class_counts.tolist(),
            'missing_classes': []
        }
    }
    
    # 分析缺失的類別
    missing_classes = []
    for class_idx in range(nc):
        if class_idx not in target_classes_present:
            if isinstance(names, dict):
                class_name = names.get(class_idx, f"class_{class_idx}")
            else:
                class_name = names[class_idx] if class_idx < len(names) else f"class_{class_idx}"
            missing_classes.append({
                'index': class_idx,
                'name': class_name,
                'count': 0
            })
    metrics['dataset_analysis']['missing_classes'] = missing_classes
    
    # 初始化所有類別的 AP 為 0
    for class_idx, class_name in enumerate(names):
        metrics['per_class_ap50'][class_name] = 0.0
        metrics['per_class_ap75'][class_name] = 0.0
        metrics['per_class_ap'][class_name] = 0.0
    
    # 更新有實際 AP 值的類別
    if len(ap) and len(ap_class):
        for i, c in enumerate(ap_class):
            class_idx = int(c)
            # 確保類別索引在有效範圍內
            if 0 <= class_idx < len(names):
                class_name = names[class_idx]
                metrics['per_class_ap50'][class_name] = float(ap50[i]) if i < len(ap50) else 0.0
                metrics['per_class_ap75'][class_name] = float(ap75[i]) if i < len(ap75) else 0.0  
                metrics['per_class_ap'][class_name] = float(ap[i]) if i < len(ap) else 0.0

    print(f"✓ Validation completed for {model_name}")
    print(f"  Classes detected: {len([k for k, v in metrics['per_class_ap50'].items() if v > 0])}/{len(names)}")
    
    # Save results
    (model_output_dir / f"{model_name}_metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics

def parse_validation_output(output_text):
    """
    Parse validation metrics from YOLOv9 output text.
    
    Args:
        output_text: The stdout from val.py
        
    Returns:
        dict: Parsed metrics
    """
    metrics = {}
    
    lines = output_text.split('\n')
    for line in lines:
        # Look for mAP results
        if 'all' in line and 'mAP@0.5' in line:
            # Example: "all   1000  0.852  0.745  0.801  0.456"
            parts = line.split()
            if len(parts) >= 6:
                try:
                    metrics['mAP@0.5'] = float(parts[3])
                    metrics['mAP@0.5:0.95'] = float(parts[4])
                    metrics['precision'] = float(parts[2])
                    metrics['recall'] = float(parts[1]) if parts[1] != 'all' else float(parts[2])
                except (ValueError, IndexError):
                    pass
        
        # Look for speed metrics
        if 'Speed:' in line:
            # Example: "Speed: 2.1ms pre-process, 15.2ms inference, 1.3ms NMS per image"
            metrics['speed_info'] = line.strip()
    
    return metrics

def summary_results(results_dict, output_dir, class_names=None, dataset_info=None):
    """
    Generate a comprehensive summary of validation results and dataset analysis.
    
    Args:
        results_dict: Dictionary of {model_name: metrics}
        output_dir: Directory to save comparison results
        class_names: List of class names to ensure all classes are shown
        dataset_info: Dictionary containing dataset analysis information
    """
    
    output_dir = Path(output_dir)
    comparison_file = output_dir / "model_comparison.json"
    summary_file = output_dir / "validation_summary.txt"
    
    # Save detailed comparison as JSON
    with open(comparison_file, 'w') as f:
        json.dump(results_dict, f, indent=2)
    
    # Generate human-readable summary
    with open(summary_file, 'w') as f:
        
        
        # 資料集分析部分（如果提供了資料集資訊）
        if dataset_info:
            f.write("\n### Dataset Analysis ###\n")
            f.write(f"資料集定義：{dataset_info['nc']} 個類別 {dataset_info['class_names_list']}\n\n")
            f.write("實際驗證集內容：\n")
            
            # 顯示每個類別的樣本數量
            for class_idx in range(dataset_info['nc']):
                count = dataset_info['target_class_counts'][class_idx]
                class_name = dataset_info['class_names_list'][class_idx] if class_idx < len(dataset_info['class_names_list']) else f"class_{class_idx}"
                status = "✓" if count > 0 else "✗"
                f.write(f"{status} Class {class_idx} ({class_name}): {count:,} samples\n")
            
            # 顯示缺失類別摘要
            if dataset_info['missing_classes']:
                f.write(f"\n缺失的類別：{', '.join(dataset_info['missing_classes'])} ({len(dataset_info['missing_classes'])}/{dataset_info['nc']} 類別缺失)\n")
            else:
                f.write(f"\n✓ 所有 {dataset_info['nc']} 個定義的類別都有驗證樣本\n")
            
            f.write("\n" + "="*60 + "\n\n")
        
        # 整體指標表格
        f.write("### Federated Learning Model Validation Summary ###\n\n")

        f.write(f"{'Model':<15} {'mAP@0.5':<10} {'mAP@0.75':<10} {'mAP@0.5:0.95':<12} {'Precision':<10} {'Recall':<10}\n")
        f.write("-" * 80 + "\n")
        
        # Sort models by name for consistent ordering
        for model_name in sorted(results_dict.keys()):
            metrics = results_dict[model_name]
            if metrics:
                map50 = f"{metrics.get('mAP@0.5', 0):.3f}"
                map75 = f"{metrics.get('mAP@0.75', 0):.3f}"
                map5095 = f"{metrics.get('mAP@0.5:0.95', 0):.3f}"
                precision = f"{metrics.get('Precision', 0):.3f}"
                recall = f"{metrics.get('Recall', 0):.3f}"
                
                f.write(f"{model_name:<15} {map50:<10} {map75:<10} {map5095:<12} {precision:<10} {recall:<10}\n")
            else:
                f.write(f"{model_name:<15} {'FAILED':<10} {'FAILED':<10} {'FAILED':<12} {'FAILED':<10} {'FAILED':<10}\n")
        
        f.write("\n")
        
        # 每類別 mAP@0.5 表格
        f.write("=== Per-Class mAP@0.5 ===\n")
        # 使用完整的類別列表，確保顯示所有類別（包括 AP=0 的）
        if class_names:
            names_to_show = class_names
        else:
            # 如果沒有提供類別名稱，則從結果中收集
            all_classes = set()
            for metrics in results_dict.values():
                if metrics and 'per_class_ap50' in metrics:
                    for key in metrics['per_class_ap50'].keys():
                        if isinstance(key, str) and not key.isdigit():
                            all_classes.add(key)
            names_to_show = sorted(list(all_classes))
        
        if names_to_show:
            # 表格標頭
            header = f"{'Model':<15}"
            for cls_name in names_to_show:
                header += f"{cls_name:<12}"
            f.write(header + "\n")
            f.write("-" * len(header) + "\n")
            
            # 每個模型的數據
            for model_name in sorted(results_dict.keys()):
                metrics = results_dict[model_name]
                if metrics and 'per_class_ap50' in metrics:
                    row = f"{model_name:<15}"
                    for cls_name in names_to_show:
                        ap_val = metrics['per_class_ap50'].get(cls_name, 0)
                        row += f"{ap_val:<12.3f}"
                    f.write(row + "\n")
            f.write("\n")
        
        # 每類別 mAP@0.5:0.95 表格
        f.write("=== Per-Class mAP@0.5:0.95 ===\n")
        if names_to_show:
            # 表格標頭
            header = f"{'Model':<15}"
            for cls_name in names_to_show:
                header += f"{cls_name:<12}"
            f.write(header + "\n")
            f.write("-" * len(header) + "\n")
            
            # 每個模型的數據
            for model_name in sorted(results_dict.keys()):
                metrics = results_dict[model_name]
                if metrics and 'per_class_ap' in metrics:
                    row = f"{model_name:<15}"
                    for cls_name in names_to_show:
                        ap_val = metrics['per_class_ap'].get(cls_name, 0)
                        row += f"{ap_val:<12.3f}"
                    f.write(row + "\n")
            f.write("\n")
        
        # Add improvement analysis
        if len(results_dict) > 1:
            f.write("=== Performance Analysis ===\n")
            baseline_key = min(results_dict.keys())  # Assume first model is baseline
            baseline_metrics = results_dict[baseline_key]
            
            if baseline_metrics and 'mAP@0.5' in baseline_metrics:
                baseline_map50 = baseline_metrics['mAP@0.5']
                baseline_map = baseline_metrics['mAP@0.5:0.95']
                f.write(f"Baseline Model ({baseline_key}):\n")
                f.write(f"  mAP@0.5 = {baseline_map50:.3f}\n")
                f.write(f"  mAP@0.5:0.95 = {baseline_map:.3f}\n\n")
                
                for model_name, metrics in results_dict.items():
                    if model_name != baseline_key and metrics and 'mAP@0.5' in metrics:
                        current_map50 = metrics['mAP@0.5']
                        current_map = metrics['mAP@0.5:0.95']
                        
                        improvement50 = current_map50 - baseline_map50
                        improvement = current_map - baseline_map
                        improvement50_pct = (improvement50 / baseline_map50) * 100 if baseline_map50 > 0 else 0
                        improvement_pct = (improvement / baseline_map) * 100 if baseline_map > 0 else 0
                        
                        f.write(f"{model_name}:\n")
                        f.write(f"  mAP@0.5 = {current_map50:.3f} (Δ = {improvement50:+.3f}, {improvement50_pct:+.1f}%)\n")
                        f.write(f"  mAP@0.5:0.95 = {current_map:.3f} (Δ = {improvement:+.3f}, {improvement_pct:+.1f}%)\n\n")
    
    print(f"\n --- [ Validation Summary ] ---\n")
    print(f"Detailed results: {comparison_file}")
    print(f"Summary report: {summary_file}")
    
    # Display summary on console
    with open(summary_file, 'r') as f:
        print(f.read())

def main():
    parser = argparse.ArgumentParser(description="Validate federated learning models")
    parser.add_argument("--device", type=str, default="0", help="cuda device e.g. '0' or '0,1' or 'cpu'")
    parser.add_argument("--experiment-dir", type=Path, required=True,
                        help="Path to experiment directory (e.g., experiments/9_kitti_4C_2R_*)")
    parser.add_argument("--data-config", type=Path, default="data/kitti.yaml",
                        help="Path to dataset configuration file")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Output directory for validation results (default: experiment_dir/validation)")
    parser.add_argument("--baseline-model", type=Path, default="yolov9-c.pt",
                        help="Path to baseline model for comparison")
    
    args = parser.parse_args()
    
    # Set default output directory
    if args.output_dir is None:
        args.output_dir = args.experiment_dir / "validation"
    
    print("=== Federated Learning Model Validation ===")
    print(f"Experiment: {args.experiment_dir}")
    print(f"Dataset: {args.data_config}")
    print(f"Output: {args.output_dir}")
    print()
    
    # Find available models to validate
    models_to_test = {}
    
    # Baseline model
    if args.baseline_model.exists():
        models_to_test["baseline"] = args.baseline_model
    
    # Aggregated weights from federated learning rounds
    aggregated_dir = args.experiment_dir / "aggregated_weights"
    if aggregated_dir.exists():
        for weight_file in sorted(aggregated_dir.glob("w_s_r*.pt")):
            round_num = weight_file.stem.split('_')[-1].replace('r', '')
            model_name = f"round_{round_num}"
            models_to_test[model_name] = weight_file
    
    if not models_to_test:
        print("Error: No models found to validate!")
        print(f"  Checked baseline: {args.baseline_model}")
        print(f"  Checked aggregated weights: {aggregated_dir}")
        return 1
    
    print(f"Found {len(models_to_test)} models to validate:")
    for name, path in models_to_test.items():
        print(f"  {name}: {path}")
    print()
    
    # Load dataset config to get class names
    from yolov9.utils.general import check_dataset
    data_dict = check_dataset(args.data_config)
    # Extract class names as a list of strings
    if isinstance(data_dict['names'], dict):
        class_names = [data_dict['names'][i] for i in sorted(data_dict['names'].keys())]
    else:
        class_names = list(data_dict['names'])
    
    # 快速載入資料集來分析類別分布（只收集資料，不顯示）
    dataset_info = None
    try:
        dataloader, dataset = create_dataloader(
            data_dict['val'], 640, 16, 32,
            single_cls=False, rect=True
        )
        
        names = data_dict['names']
        nc = len(names)
        
        # 準備類別名稱列表
        if isinstance(names, dict):
            class_names_list = [names[i] for i in sorted(names.keys())]
        else:
            class_names_list = list(names)
        
        # 統計類別分布
        target_class_counts = np.zeros(nc, dtype=int)
        target_classes_present = set()
        
        for batch_i, (imgs, targets, paths, shapes) in enumerate(dataloader):
            for target in targets:
                cls_idx = int(target[1])  # class index
                if 0 <= cls_idx < nc:
                    target_class_counts[cls_idx] += 1
                    target_classes_present.add(cls_idx)
        
        # 分析缺失類別
        missing_classes = []
        for class_idx in range(nc):
            if class_idx not in target_classes_present:
                if isinstance(names, dict):
                    class_name = names.get(class_idx, f"class_{class_idx}")
                else:
                    class_name = names[class_idx] if class_idx < len(names) else f"class_{class_idx}"
                missing_classes.append(class_name)
        
        # 收集資料集資訊供後續使用
        dataset_info = {
            'nc': nc,
            'class_names_list': class_names_list,
            'target_class_counts': target_class_counts.tolist(),
            'missing_classes': missing_classes
        }
        
    except Exception as e:
        print(f"無法分析資料集類別分布: {e}")
        print()
    
    # Run validation for each model
    results = {}
    for model_name, model_path in models_to_test.items():
        metrics = run_validation(model_path, args.data_config, args.output_dir, model_name, args.device)
        results[model_name] = metrics
        print()
    
    # Generate comparison report
    summary_results(results, args.output_dir, class_names, dataset_info)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
Draw Aggregated Bounding Boxes - YOLO 模型預測結果視覺化與差異分析工具

===========
#功能說明：
本工具提供兩種模式：
1. 掃描模式 (--scan): 計算兩個模型在驗證集上的差異指標，輸出 CSV 報告
2. 繪圖模式 (預設): 批次繪製多個模型在指定圖片上的偵測結果，以顏色標示 GT/TP/FP

===========
#條件：目錄結構需求：
  tools/draw_models/          # 模型檔案
    ├── cityscapes_fedavg.pt
    ├── cityscapes_fedyoga.pt
    ├── kittiO_fedavg.pt
    └── kittiO_fedyoga.pt
  
  tools/draw_pics/            # 圖片與標籤（按資料集分類）
    ├── cityscapes/
    │   ├── frankfurt_000001_002759_leftImg8bit.jpg
    │   ├── frankfurt_000001_002759_leftImg8bit.txt
    │   ├── munster_000047_000019_leftImg8bit.jpg
    │   └── munster_000047_000019_leftImg8bit.txt
    └── kittiO/
        ├── 004094.png
        ├── 004094.txt
        ├── 007404.png
        └── 007404.txt
  
  tools/draw_output/          # 輸出結果（自動建立）
    ├── draw-cityscapes-frankfurt_000001_002759_leftImg8bit-fedavg.png
    ├── draw-cityscapes-frankfurt_000001_002759_leftImg8bit-fedyoga.png
    ├── draw-kittiO-004094-fedavg.png
    └── draw-kittiO-004094-fedyoga.png

===========
#使用方式：
-----------
##【模式 1：掃描模式 - 計算模型差異指標】
  python tools/draw_agg_bbox.py --scan --main-model fedyoga

輸出：
自動掃描所有資料集配對（fedavg vs fedyoga），計算每張驗證圖片的 TP/FP/FN/AP50，
並輸出 CSV 檔案，按 diff_score 排序找出差異最大的圖片。
  - score_cityscapes.csv (492 張圖片統計)
  - score_foggy.csv (500 張圖片統計)
  - score_kittiO.csv (748 張圖片統計)
  - score_sim10k.csv (1999 張圖片統計)

-----------
##【模式 2：繪圖模式 - 視覺化預測結果】
  python tools/draw_agg_bbox.py

輸出檔名格式：
為指定的圖片批次繪製多個模型的預測結果，自動標註 GT（綠色）、TP（黃色）、FP（紅色）。
圖片右上角顯示統計資訊：Model name, AP50, GT, TP, FP, FN。
  draw-{dataset}-{image_name}-{algo}.png
  例如：draw-cityscapes-frankfurt_000001_002759_leftImg8bit-fedavg.png

===========
# 統計指標：
  GT  - Ground Truth 總數
  TP  - True Positive（IoU≥0.5 且類別正確）
  FP  - False Positive（預測錯誤或 IoU<0.5）
  FN  - False Negative（未被偵測到的 GT）
  AP50 - Average Precision at IoU=0.5
"""

import argparse
import os
import sys
from pathlib import Path
import torch
import numpy as np
import cv2
from tqdm import tqdm

# 添加專案路徑
TOOL_ROOT = Path(__file__).resolve().parent
WORK_ROOT = TOOL_ROOT.parent
sys.path.insert(0, str(WORK_ROOT / 'yolov9'))
sys.path.insert(0, str(WORK_ROOT))

from yolov9.utils.general import check_dataset, check_img_size, non_max_suppression, box_iou, xywh2xyxy, scale_boxes
from yolov9.utils.metrics import ap_per_class
from yolov9.models.common import DetectMultiBackend
from yolov9.utils.torch_utils import select_device
from yolov9.utils.dataloaders import LoadImages
import yaml

# 信心閾值設定
DRAW_CONF_THRES = 0.5  # 畫圖模式：過濾低信心預測


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
    if len(labels) == 0:
        return correct
    
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


def load_label(label_path):
    """載入單張圖片的標籤"""
    if not label_path.exists():
        return np.zeros((0, 5), dtype=np.float32)
    
    labels = []
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                # YOLO format: class x_center y_center width height (normalized)
                cls = int(parts[0])
                x_center, y_center, w, h = map(float, parts[1:5])
                labels.append([cls, x_center, y_center, w, h])
    
    return np.array(labels, dtype=np.float32) if labels else np.zeros((0, 5), dtype=np.float32)


def extract_dataset_from_model_name(model_name):
    """從模型名稱提取資料集名稱
    例如：kittiO_fedavg → kittiO
          cityscapes_fedyoga → cityscapes
    """
    # 嘗試用常見分隔符分割
    for sep in ['_', '-', '.']:
        if sep in model_name:
            parts = model_name.split(sep)
            if len(parts) >= 2:
                return parts[0]
    
    # 如果沒有分隔符，返回完整名稱
    return model_name


def detect_dataset_from_model_name(model_name):
    """從模型名稱推測資料集"""
    model_lower = model_name.lower()
    
    # 優先檢查完整匹配
    if 'kittio' in model_lower or 'kittoa' in model_lower:
        return 'kittiO'
    elif 'bdd100k' in model_lower:
        return 'bdd100k'
    elif 'sim10k' in model_lower:
        return 'sim10k'
    elif 'foggy' in model_lower:
        return 'foggy'
    elif 'cityscapes' in model_lower:
        return 'cityscapes'
    elif 'kitti' in model_lower:
        return 'kittiO'  # 預設使用 kittiO
    else:
        # 預設
        return 'kittiO'


def draw_single_image(model, img_path, label_path, device, img_size, nc, class_names, 
                      model_name, output_path, iou_thres=0.6):
    """
    為單張圖片繪製偵測結果
    
    Args:
        model: 模型
        img_path: 圖片路徑
        label_path: 標籤路徑
        device: 設備
        img_size: 圖片大小
        nc: 類別數量
        class_names: 類別名稱字典
        model_name: 模型名稱（用於顯示）
        output_path: 輸出路徑
        iou_thres: NMS IoU 閾值
    
    Returns:
        bool: 是否成功繪製
    """
    # 載入真實標籤
    gt_labels = load_label(label_path)
    
    # 載入並推論圖片
    dataset = LoadImages(str(img_path), img_size=img_size)
    
    iouv = torch.linspace(0.5, 0.95, 10).to(device)
    
    for path, img, im0s, vid_cap, s in dataset:
        im_draw = im0s.copy()
        h0, w0 = im0s.shape[:2]
        img_tensor = torch.from_numpy(img).to(device).float() / 255.0
        if img_tensor.ndimension() == 3:
            img_tensor = img_tensor.unsqueeze(0)
        
        # 取得 resize 後的圖片尺寸
        _, _, img_h, img_w = img_tensor.shape
        
        # 推論
        with torch.no_grad():
            pred = model(img_tensor)
            if isinstance(pred, list):
                pred = pred[0]
        pred = non_max_suppression(pred, DRAW_CONF_THRES, iou_thres, labels=[], multi_label=True, agnostic=False)
        detections = pred[0] if len(pred) else torch.empty((0, 6))
        
        # 準備真實標籤 (轉換到 resize 後的座標空間)
        if len(gt_labels) > 0:
            labels_pixel = gt_labels.copy()
            labels_pixel[:, 1] *= img_w  # x_center
            labels_pixel[:, 2] *= img_h  # y_center
            labels_pixel[:, 3] *= img_w  # width
            labels_pixel[:, 4] *= img_h  # height
            
            # 轉換 xywh 到 xyxy
            tbox = xywh2xyxy(torch.from_numpy(labels_pixel[:, 1:5]).float()).to(device)
            tcls = torch.from_numpy(labels_pixel[:, 0:1]).float().to(device)
            labels_tensor = torch.cat((tcls, tbox), 1)
            
            # 繪製真實標籤 (綠框)
            gt_boxes_scaled = scale_boxes(img_tensor.shape[2:], tbox, im0s.shape).round()
            for idx, (box, cls) in enumerate(zip(gt_boxes_scaled, labels_tensor[:, 0])):
                x1, y1, x2, y2 = map(int, box.cpu().tolist())
                cls_int = int(cls.item())
                cls_name = class_names.get(cls_int, str(cls_int))
                color = (0, 255, 0)  # 綠色 - 真實標籤
                cv2.rectangle(im_draw, (x1, y1), (x2, y2), color, 2)
                label = f"GT_{cls_name}"
                cv2.putText(im_draw, label, (x1, max(y1-5,0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        else:
            labels_tensor = torch.empty((0, 5)).to(device)
            labels_pixel = np.zeros((0, 5), dtype=np.float32)
        
        # 處理預測框
        if len(detections) > 0:
            # 計算哪些預測是正確的
            if len(labels_tensor) > 0:
                correct = process_batch(detections, labels_tensor, iouv)
                is_correct = correct[:, 0]  # IoU@0.5 的結果
            else:
                is_correct = torch.zeros(len(detections), dtype=torch.bool)
            
            # 尺寸縮放回原圖
            det_scaled = detections.clone()
            det_scaled[:, :4] = scale_boxes(img_tensor.shape[2:], det_scaled[:, :4], im0s.shape).round()
            
            for idx, (*box, conf, cls) in enumerate(det_scaled.tolist()):
                x1, y1, x2, y2 = map(int, box)
                cls_int = int(cls)
                cls_name = class_names.get(cls_int, str(cls_int))
                
                # 根據正確性選擇顏色
                if is_correct[idx]:
                    color = (0, 255, 255)  # 黃色 - 正確預測
                    label_prefix = "TP"
                else:
                    color = (0, 0, 255)  # 紅色 - 錯誤預測
                    label_prefix = "FP"
                
                cv2.rectangle(im_draw, (x1, y1), (x2, y2), color, 2)
                label = f"{label_prefix}_{cls_name}:{conf:.2f}"
                cv2.putText(im_draw, label, (x1, max(y1-5,0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        else:
            is_correct = torch.zeros(0, dtype=torch.bool)
        
        # 計算統計數據
        num_gt = len(gt_labels)
        num_tp = is_correct.sum().item() if len(detections) > 0 else 0
        num_fp = len(detections) - num_tp if len(detections) > 0 else 0
        num_fn = num_gt - num_tp
        
        # 計算 AP50
        if num_gt > 0 and len(detections) > 0 and len(labels_pixel) > 0:
            tcls_list = labels_pixel[:, 0].astype(int).tolist()
            stats = [(correct.cpu(), detections[:, 4].cpu(), detections[:, 5].cpu(), tcls_list)]
            stats = [np.concatenate(x, 0) for x in zip(*stats)]
            if len(stats) and stats[0].any():
                try:
                    names_dict = {i: str(i) for i in range(nc)}
                    tp, fp, p, r, f1, ap, ap_class = ap_per_class(*stats, plot=False, save_dir="", names=names_dict)
                    if len(ap) > 0:
                        ap50_value = ap[:, 0].mean()
                    else:
                        ap50_value = 0.0
                except:
                    ap50_value = 0.0
            else:
                ap50_value = 0.0
        else:
            ap50_value = 0.0
        
        # 在圖片右上角繪製統計資訊
        img_h_display, img_w_display = im_draw.shape[:2]
        
        # 設定文字參數
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        line_height = 25
        padding = 10
        
        # 準備文字內容
        texts = [
            f"Model: {model_name}",
            f"AP50: {ap50_value:.4f}",
            f"GT: {num_gt}",
            f"TP: {num_tp}",
            f"FP: {num_fp}",
            f"FN: {num_fn}"
        ]
        
        # 計算背景框大小
        max_width = 0
        for text in texts:
            (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)
            max_width = max(max_width, text_w)
        
        box_width = max_width + 2 * padding
        box_height = len(texts) * line_height + padding
        
        # 繪製半透明背景
        overlay = im_draw.copy()
        x1_box = img_w_display - box_width - 10
        y1_box = 10
        x2_box = img_w_display - 10
        y2_box = y1_box + box_height
        cv2.rectangle(overlay, (x1_box, y1_box), (x2_box, y2_box), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, im_draw, 0.4, 0, im_draw)
        
        # 繪製文字
        y_offset = y1_box + padding + 20
        for i, text in enumerate(texts):
            # 根據內容選擇顏色
            if i == 0:  # Model
                color = (255, 255, 255)  # 白色
            elif i == 1:  # AP50
                color = (255, 255, 0)  # 青色
            elif i == 2:  # GT
                color = (0, 255, 0)  # 綠色
            elif i == 3:  # TP
                color = (0, 255, 255)  # 黃色
            elif i == 4:  # FP
                color = (0, 0, 255)  # 紅色
            else:  # FN
                color = (128, 128, 128)  # 灰色
            
            cv2.putText(im_draw, text, (x1_box + padding, y_offset), 
                       font, font_scale, color, thickness, cv2.LINE_AA)
            y_offset += line_height
        
        # 儲存圖片
        cv2.imwrite(str(output_path), im_draw)
        return True
    
    return False


def compute_single_image_stats(model, img_path, label_path, device, img_size, nc, iou_thres=0.6):
    """
    計算單張圖片的統計數據（不繪圖）
    
    Args:
        model: 模型
        img_path: 圖片路徑
        label_path: 標籤路徑
        device: 設備
        img_size: 圖片大小
        nc: 類別數量
        iou_thres: NMS IoU 閾值
    
    Returns:
        dict: 統計數據 {'tp': int, 'fp': int, 'fn': int, 'ap50': float}
    """
    # 載入真實標籤
    gt_labels = load_label(label_path)
    
    # 如果沒有標籤，返回全 0
    if len(gt_labels) == 0:
        return {
            'tp': 0,
            'fp': 0,
            'fn': 0,
            'ap50': 0.0
        }
    
    # 載入並推論圖片
    dataset = LoadImages(str(img_path), img_size=img_size)
    
    iouv = torch.linspace(0.5, 0.95, 10).to(device)
    
    # 初始化變數
    num_tp = 0
    num_fp = 0
    num_fn = len(gt_labels)
    ap50_value = 0.0
    
    for path, img, im0s, vid_cap, s in dataset:
        img_tensor = torch.from_numpy(img).to(device).float() / 255.0
        if img_tensor.ndimension() == 3:
            img_tensor = img_tensor.unsqueeze(0)
        
        # 取得 resize 後的圖片尺寸
        _, _, img_h, img_w = img_tensor.shape
        
        # 推論
        with torch.no_grad():
            pred = model(img_tensor)
            if isinstance(pred, list):
                pred = pred[0]
        pred = non_max_suppression(pred, DRAW_CONF_THRES, iou_thres, labels=[], multi_label=True, agnostic=False)
        detections = pred[0] if len(pred) else torch.empty((0, 6))
        
        # 準備真實標籤 (轉換到 resize 後的座標空間)
        labels_pixel = gt_labels.copy()
        labels_pixel[:, 1] *= img_w  # x_center
        labels_pixel[:, 2] *= img_h  # y_center
        labels_pixel[:, 3] *= img_w  # width
        labels_pixel[:, 4] *= img_h  # height
        
        # 轉換 xywh 到 xyxy
        tbox = xywh2xyxy(torch.from_numpy(labels_pixel[:, 1:5]).float()).to(device)
        tcls = torch.from_numpy(labels_pixel[:, 0:1]).float().to(device)
        labels_tensor = torch.cat((tcls, tbox), 1)
        
        # 處理預測框
        if len(detections) > 0:
            # 計算哪些預測是正確的
            correct = process_batch(detections, labels_tensor, iouv)
            is_correct = correct[:, 0]  # IoU@0.5 的結果
            
            # 計算統計數據
            num_gt = len(gt_labels)
            num_tp = is_correct.sum().item()
            num_fp = len(detections) - num_tp
            num_fn = num_gt - num_tp
            
            # 計算 AP50
            tcls_list = labels_pixel[:, 0].astype(int).tolist()
            stats = [(correct.cpu(), detections[:, 4].cpu(), detections[:, 5].cpu(), tcls_list)]
            stats = [np.concatenate(x, 0) for x in zip(*stats)]
            if len(stats) and stats[0].any():
                try:
                    names_dict = {i: str(i) for i in range(nc)}
                    tp, fp, p, r, f1, ap, ap_class = ap_per_class(*stats, plot=False, save_dir="", names=names_dict)
                    if len(ap) > 0:
                        ap50_value = ap[:, 0].mean()
                except:
                    ap50_value = 0.0
        # else: 沒有預測時，使用初始化的值（FN = GT數量）
    
    return {
        'tp': num_tp,
        'fp': num_fp,
        'fn': num_fn,
        'ap50': ap50_value
    }


def main():
    parser = argparse.ArgumentParser(
        description='批次繪製多個模型在多張圖片上的偵測結果，或掃描差異指標',
        usage='python draw_agg_bbox.py --scan 或 python draw_agg_bbox.py --models_dir ./draw_models/ --images_dir ./draw_pics/'
    )
    parser.add_argument('--scan', action='store_true', help='啟用掃描模式：計算兩個模型的差異指標並輸出 CSV')
    parser.add_argument('--main-model', type=str, default='', help='指定主要模型關鍵字（如 fedyoga），用於計算有向差異分數')
    parser.add_argument('--model1', type=str, default='', help='模型1名稱（不含.pt，例如：kittiO_fedavg）')
    parser.add_argument('--model2', type=str, default='', help='模型2名稱（不含.pt，例如：kittiO_fedyoga）')
    parser.add_argument('--models_dir', type=str, default='./tools/draw_models/', help='模型檔案目錄')
    parser.add_argument('--images_dir', type=str, default='./tools/draw_pics/', help='圖片檔案目錄')
    parser.add_argument('--output_dir', type=str, default='./tools/draw_output/', help='輸出目錄')
    parser.add_argument('--img_size', type=int, default=640, help='推論圖片大小 (預設: 640)')
    parser.add_argument('--device', type=str, default='', help='計算設備 (預設: 自動選擇)')
    parser.add_argument('--iou_thres', type=float, default=0.6, help='NMS IoU 閾值 (預設: 0.6)')
    
    args = parser.parse_args()
    
    if args.scan:
        # ===== 掃描模式：計算差異指標 =====
        
        # 設定模型目錄
        models_dir = Path(args.models_dir)
        if not models_dir.exists():
            print(f"錯誤: 找不到模型目錄: {models_dir}")
            return
        
        # 自動掃描目錄中的模型
        model_files = sorted(list(models_dir.glob('*.pt')))
        if len(model_files) < 2:
            print(f"錯誤: 在 {models_dir} 中至少需要2個模型檔案")
            return
        
        print(f"找到 {len(model_files)} 個模型:")
        for i, mf in enumerate(model_files):
            print(f"  [{i}] {mf.name}")
        
        # 自動偵測或使用指定的模型配對
        if args.model1 and args.model2:
            # 使用者指定單一配對
            model_pairs = [(args.model1, args.model2)]
        else:
            # 自動識別所有資料集的模型
            model_dict = {}
            for mf in model_files:
                dataset = extract_dataset_from_model_name(mf.stem)
                if dataset not in model_dict:
                    model_dict[dataset] = []
                model_dict[dataset].append(mf.stem)
            
            # 建立配對清單：將每個資料集的所有模型組合成一組
            dataset_model_groups = []
            for dataset, models in model_dict.items():
                if len(models) >= 2:
                    # 如果指定了 main_model，確保它在第一個位置
                    if args.main_model:
                        main = next((m for m in models if args.main_model.lower() in m.lower()), None)
                        if main:
                            # main_model 放第一個，其他模型放後面
                            others = [m for m in models if m != main]
                            sorted_models = [main] + sorted(others)
                        else:
                            print(f"警告: 在 {dataset} 中找不到包含 '{args.main_model}' 的模型")
                            sorted_models = sorted(models)
                    else:
                        sorted_models = sorted(models)
                    
                    dataset_model_groups.append((dataset, sorted_models))
            
            if len(dataset_model_groups) == 0:
                print("錯誤: 無法找到任何模型配對")
                return
            
            print(f"\n自動識別到 {len(dataset_model_groups)} 個資料集:")
            for dataset, models in dataset_model_groups:
                print(f"  {dataset}: {len(models)} 個模型 - {', '.join([m.split('_')[-1] for m in models])}")
                if args.main_model:
                    print(f"    主模型: {models[0].split('_')[-1]}, 比較對象: {models[1].split('_')[-1]}")
        
        # 處理每一個資料集組
        for dataset_name, model_names in dataset_model_groups:
            print(f"\n{'='*60}")
            print(f"處理資料集: {dataset_name}")
            print(f"模型數量: {len(model_names)}")
            print(f"模型清單: {', '.join([m.split('_')[-1] for m in model_names])}")
            if args.main_model and len(model_names) >= 2:
                print(f"Diff 計算: {model_names[0].split('_')[-1]} vs {model_names[1].split('_')[-1]}")
            print(f"{'='*60}")
            
            # 輸出檔名
            if args.main_model and len(model_names) >= 2:
                algo1 = model_names[0].split('_')[-1]
                algo2 = model_names[1].split('_')[-1]
                output_csv = Path(f'./score_{dataset_name}_{algo1}_vs_{algo2}.csv')
            else:
                output_csv = Path(f'./score_{dataset_name}.csv')
                output_csv = Path(f'./score_{dataset_name}.csv')
            
            # 檢查所有模型檔案
            model_paths = []
            for model_name in model_names:
                model_path = models_dir / f'{model_name}.pt'
                if not model_path.exists():
                    print(f"錯誤: 找不到模型 {model_path}")
                    break
                model_paths.append(model_path)
            
            if len(model_paths) != len(model_names):
                print(f"跳過資料集 {dataset_name}（模型檔案不完整）")
                continue
            
            # 載入資料集配置
            data_yaml_path = WORK_ROOT / 'data' / f'{dataset_name}.yaml'
            if not data_yaml_path.exists():
                print(f"錯誤: 找不到資料集配置 {data_yaml_path}")
                continue
            
            with open(data_yaml_path, 'r') as f:
                data_dict = yaml.safe_load(f)
            
            data_dict = check_dataset(str(data_yaml_path))
            nc = data_dict['nc']
            
            # 取得驗證圖片路徑
            val_path = Path(data_dict['path']) / data_dict['val']
            if not val_path.exists():
                # 嘗試相對路徑
                val_path = WORK_ROOT / 'datasets' / dataset_name / 'images' / 'val'
            
            if not val_path.exists():
                print(f"錯誤: 找不到驗證圖片路徑: {val_path}")
                continue
            
            val_images = sorted(list(val_path.glob('*.png')) + list(val_path.glob('*.jpg')))
            if len(val_images) == 0:
                print(f"錯誤: 在 {val_path} 中找不到圖片")
                continue
            
            print(f"找到 {len(val_images)} 張驗證圖片")
            
            # 設定設備
            device = select_device(args.device)
            print(f"使用設備: {device}")
            
            # 載入所有模型
            print("載入模型...")
            models = {}
            img_sizes = {}
            for model_name, model_path in zip(model_names, model_paths):
                try:
                    model = DetectMultiBackend(str(model_path), device=device)
                    gs = int(model.stride)
                    img_size = check_img_size(args.img_size, s=gs)
                    models[model_name] = model
                    img_sizes[model_name] = img_size
                    print(f"  ✓ 載入: {model_name}")
                except Exception as e:
                    print(f"  ✗ 載入失敗 {model_name}: {e}")
                    break
            
            if len(models) != len(model_names):
                print(f"跳過資料集 {dataset_name}（模型載入失敗）")
                continue
            
            # 收集結果
            results = []
            
            print("開始掃描...")
            for img_path in tqdm(val_images, desc=f"處理 {dataset_name} 圖片"):
                image_name = img_path.name
                
                # 標籤路徑 (從 images/val 往上兩層到 dataset root，然後進入 labels/val)
                label_path = img_path.parent.parent.parent / 'labels' / 'val' / (img_path.stem + '.txt')
                
                # 計算所有模型的統計數據
                all_stats = {}
                for model_name in model_names:
                    stats = compute_single_image_stats(
                        models[model_name], img_path, label_path, device, 
                        img_sizes[model_name], nc, args.iou_thres
                    )
                    all_stats[model_name] = stats
                
                # 建立結果字典（基本資訊）
                result = {
                    'dataset': dataset_name,
                    'image': image_name
                }
                
                # 加入所有模型的統計數據
                for model_name in model_names:
                    algo_name = model_name.split('_')[-1]
                    stats = all_stats[model_name]
                    result[f'{algo_name}_tp'] = stats['tp']
                    result[f'{algo_name}_fp'] = stats['fp']
                    result[f'{algo_name}_fn'] = stats['fn']
                    result[f'{algo_name}_ap50'] = stats['ap50']
                
                # 計算 diff_score 和 signed_diff（使用前兩個模型）
                if len(model_names) >= 2:
                    stats1 = all_stats[model_names[0]]
                    stats2 = all_stats[model_names[1]]
                    
                    diff_score = abs(stats1['tp'] - stats2['tp']) + abs(stats1['fp'] - stats2['fp']) + abs(stats1['fn'] - stats2['fn'])
                    
                    # signed_diff: main_model 表現好為正值
                    signed_diff = (stats1['tp'] - stats2['tp']) - (stats1['fp'] - stats2['fp']) - (stats1['fn'] - stats2['fn'])
                    
                    result['diff_score'] = diff_score
                    result['signed_diff'] = signed_diff
                    result['compare_pair'] = f"{model_names[0].split('_')[-1]}_vs_{model_names[1].split('_')[-1]}"
                else:
                    result['diff_score'] = 0
                    result['signed_diff'] = 0
                    result['compare_pair'] = 'N/A'
                
                results.append(result)
            
            # 輸出 CSV
            import pandas as pd
            df = pd.DataFrame(results)
            df.to_csv(output_csv, index=False)
            
            print(f"✓ 掃描完成！輸出至: {output_csv}")
            print(f"  共處理 {len(results)} 張圖片")
            
            # 顯示統計摘要
            if args.main_model and len(model_names) >= 2:
                # 顯示 main_model 表現較好的前10張
                df_better = df[df['signed_diff'] > 0].sort_values('signed_diff', ascending=False)
                print(f"\n{model_names[0].split('_')[-1]} 表現較好的圖片數: {len(df_better)}")
                print(f"\n前10張 {model_names[0].split('_')[-1]} 優勢最大的圖片:")
                for i, row in df_better.head(10).iterrows():
                    algo1 = model_names[0].split('_')[-1]
                    algo2 = model_names[1].split('_')[-1]
                    print(f"  {row['image']}: signed_diff={row['signed_diff']:.0f} "
                          f"(TP:{row[f'{algo1}_tp']:.0f}-{row[f'{algo2}_tp']:.0f}, "
                          f"FP:{row[f'{algo1}_fp']:.0f}-{row[f'{algo2}_fp']:.0f}, "
                          f"FN:{row[f'{algo1}_fn']:.0f}-{row[f'{algo2}_fn']:.0f})")
            else:
                # 顯示單純差異最大的前10張
                df_sorted = df.sort_values('diff_score', ascending=False)
                print("\n前10個差異最大的圖片:")
                if len(model_names) >= 2:
                    for i, row in df_sorted.head(10).iterrows():
                        algo1 = model_names[0].split('_')[-1]
                        algo2 = model_names[1].split('_')[-1]
                        print(f"  {row['image']}: diff_score={row['diff_score']:.0f} "
                              f"(TP:{row[f'{algo1}_tp']:.0f}-{row[f'{algo2}_tp']:.0f}, "
                              f"FP:{row[f'{algo1}_fp']:.0f}-{row[f'{algo2}_fp']:.0f}, "
                              f"FN:{row[f'{algo1}_fn']:.0f}-{row[f'{algo2}_fn']:.0f})")
        
        print(f"\n{'='*60}")
        print("✓ 所有資料集掃描完成！")
        print(f"{'='*60}")
        return
    
    # ===== 原有繪圖模式 =====
    # 設定路徑
    models_dir = Path(args.models_dir)
    images_dir = Path(args.images_dir)
    output_dir = Path(args.output_dir)
    
    # 檢查目錄
    if not models_dir.exists():
        print(f"錯誤: 找不到模型目錄: {models_dir}")
        return
    
    if not images_dir.exists():
        print(f"錯誤: 找不到圖片目錄: {images_dir}")
        return
    
    # 建立輸出目錄
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 設定設備
    device = select_device(args.device)
    print(f"使用設備: {device}")
    
    # 掃描模型檔案
    model_files = sorted(list(models_dir.glob('*.pt')))
    if len(model_files) == 0:
        print(f"錯誤: 在 {models_dir} 中找不到 .pt 模型檔案")
        return
    
    print(f"\n找到 {len(model_files)} 個模型:")
    for mf in model_files:
        print(f"  - {mf.name}")
    
    # 掃描圖片檔案（掃描所有子目錄）
    total_images = 0
    for subdir in images_dir.iterdir():
        if subdir.is_dir():
            img_count = len(list(subdir.glob('*.png'))) + len(list(subdir.glob('*.jpg')))
            total_images += img_count
    
    if total_images == 0:
        print(f"錯誤: 在 {images_dir} 的子目錄中找不到圖片檔案")
        return
    
    print(f"\n找到 {total_images} 張圖片（分布在各資料集目錄）")
    
    # 處理每個模型
    # 先計算總任務數（需要遍歷一次）
    total_tasks = 0
    for model_file in model_files:
        model_name = model_file.stem
        dataset_name = extract_dataset_from_model_name(model_name)
        
        # 找對應目錄
        for subdir in images_dir.iterdir():
            if subdir.is_dir() and subdir.name.lower() == dataset_name.lower():
                img_count = len(list(subdir.glob('*.png'))) + len(list(subdir.glob('*.jpg')))
                total_tasks += img_count
                break
    
    print(f"\n總共需要處理: {total_tasks} 個任務")
    
    with tqdm(total=total_tasks, desc="處理進度") as pbar:
        for model_file in model_files:
            model_name = model_file.stem  # 例如: kittiO_fedyoga
            
            # 從模型名稱提取資料集名稱
            dataset_name = extract_dataset_from_model_name(model_name)
            print(f"\n處理模型: {model_name} (提取資料集: {dataset_name})")
            
            # 找到對應的圖片目錄（不區分大小寫）
            images_base_dir = None
            for subdir in images_dir.iterdir():
                if subdir.is_dir() and subdir.name.lower() == dataset_name.lower():
                    images_base_dir = subdir
                    break
            
            if images_base_dir is None or not images_base_dir.exists():
                print(f"  警告: 找不到圖片目錄 {images_dir / dataset_name}，跳過此模型")
                continue
            
            print(f"  圖片目錄: {images_base_dir}")
            
            # 掃描該資料集目錄下的圖片
            image_files_for_model = sorted(list(images_base_dir.glob('*.png')) + list(images_base_dir.glob('*.jpg')))
            
            if len(image_files_for_model) == 0:
                print(f"  警告: 在 {images_base_dir} 中找不到圖片，跳過此模型")
                continue
            
            print(f"  找到 {len(image_files_for_model)} 張圖片")
            
            # 載入資料集配置
            data_yaml_path = WORK_ROOT / 'data' / f'{dataset_name}.yaml'
            if not data_yaml_path.exists():
                print(f"  警告: 找不到資料集配置 {data_yaml_path}，跳過此模型")
                pbar.update(len(image_files_for_model))
                continue
            
            with open(data_yaml_path, 'r') as f:
                data_dict = yaml.safe_load(f)
            
            data_dict = check_dataset(str(data_yaml_path))
            nc = data_dict['nc']
            class_names = data_dict['names']
            
            # 載入模型
            try:
                model = DetectMultiBackend(str(model_file), device=device)
                gs = int(model.stride)
                img_size = check_img_size(args.img_size, s=gs)
            except Exception as e:
                print(f"  錯誤: 無法載入模型 {model_file}: {e}")
                pbar.update(len(image_files_for_model))
                continue
            
            # 處理每張圖片
            for image_file in image_files_for_model:
                image_name = image_file.stem  # 例如: 006508
                
                # 標籤檔案在同一目錄
                label_file = images_base_dir / f"{image_name}.txt"
                
                # 從 model_name 提取演算法名稱 (例如: cityscapes_fedavg -> fedavg)
                algo_name = model_name.split('_')[-1] if '_' in model_name else model_name
                
                # 輸出檔名: draw-{dataset}-{image_name}-{algo}.png
                output_file = output_dir / f"draw-{dataset_name}-{image_name}-{algo_name}.png"
                
                # 繪製
                try:
                    success = draw_single_image(
                        model, image_file, label_file, device, img_size, nc, class_names,
                        model_name, output_file, args.iou_thres
                    )
                    if not success:
                        print(f"  警告: 處理失敗 {image_file.name}")
                except Exception as e:
                    print(f"  錯誤: 處理 {image_file.name} 時發生錯誤: {e}")
                
                pbar.update(1)
    
    print(f"\n✓ 完成！輸出目錄: {output_dir}")
    print(f"  共生成 {len(list(output_dir.glob('draw-*.png')))} 張圖片")


if __name__ == '__main__':
    main()

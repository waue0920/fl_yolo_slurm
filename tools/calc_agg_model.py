#!/usr/bin/env python3
"""
Calculate Aggregated Model Performance - YOLO 模型 mAP50 性能矩陣計算工具

===========
#功能說明：
本工具提供兩種模式：
1. 計算模式 (預設): 計算多個模型在驗證集上的 mAP50 性能矩陣，輸出 CSV 報告
2. 視覺化模式 (--show): 為指定圖片繪製各模型的偵測結果，以顏色標示 GT/TP/FP

===========
#條件：目錄結構需求：
  experiments/60_kittiOA010_fedyoga_4C_12R_202511041631/
    └── aggregated_weights/
        ├── w_s_r1.pt     # Round 1 聚合權重
        ├── w_s_r4.pt     # Round 4 聚合權重
        └── w_s_r9.pt     # Round 9 聚合權重
  
  yolov9-c.pt             # 初始權重（專案根目錄）
  
  datasets/kittiO/
    ├── images/val/       # 驗證圖片
    └── labels/val/       # YOLO 格式標籤

===========
#使用方式：
-----------
##【模式 1：計算模式 - 計算 mAP50 性能矩陣】


輸出：
計算初始權重和指定 round 模型在所有驗證圖片上的 mAP50，輸出 CSV 矩陣。
  - calc_agg_model_map50_60_kittiOA010_fedyoga_4C_12R_202511041631.csv
  
CSV 格式：
  image, yolov9-c, w_s_r1, w_s_r4, w_s_r9
  000023.png, 0.8500, 0.8700, 0.9100, 0.9200
  000025.png, 0.7200, 0.7500, 0.7800, 0.8000
  ...

統計資訊：
  - 各模型的平均 mAP50
  - 進步最多的前 10 張圖（比較最後兩個 round）

-----------
##【模式 2：視覺化模式 - 單張圖片畫框輸出】
  python tools/calc_agg_model.py experiments/60_kittiOA010_fedyoga_4C_12R_202511041631/ --model_id 1,4,9 --show 005824.png

輸出檔名格式：
為指定圖片使用各模型進行推論，繪製彩色標註框（GT 綠色、TP 黃色、FP 紅色）。
  show_{experiment}_{image_stem}_w{round}.png
  例如：show_60_kittiOA010_fedyoga_4C_12R_202511041631_005824_w0.png   # 初始權重
       show_60_kittiOA010_fedyoga_4C_12R_202511041631_005824_w1.png   # Round 1
       show_60_kittiOA010_fedyoga_4C_12R_202511041631_005824_w4.png   # Round 4
       show_60_kittiOA010_fedyoga_4C_12R_202511041631_005824_w9.png   # Round 9

進階參數：
  --num_images N          # 使用前 N 張驗證圖片（預設：-1 表示全部）
  --img_size 640          # 推論圖片大小（預設：640）
  --device cuda:0         # 計算設備（預設：自動選擇）
  --iou_thres 0.6         # NMS IoU 閾值（預設：0.6）

===========
# 統計指標：
  mAP50 - Mean Average Precision at IoU=0.5（主要評估指標）
  GT    - Ground Truth 總數
  TP    - True Positive（IoU≥0.5 且類別正確）
  FP    - False Positive（預測錯誤或 IoU<0.5）
  FN    - False Negative（未被偵測到的 GT）
  AP50  - Average Precision at IoU=0.5（單張圖片）

===========
# 信心閾值設定：
  CALC_CONF_THRES = 0.5   # 計算模式：用於 mAP 計算
  DRAW_CONF_THRES = 0.5   # 視覺化模式：過濾低信心預測
"""

import argparse
import os
import sys
from pathlib import Path
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

# 添加專案路徑
TOOL_ROOT = Path(__file__).resolve().parent
WORK_ROOT = TOOL_ROOT.parent
sys.path.insert(0, str(WORK_ROOT / 'yolov9'))
sys.path.insert(0, str(WORK_ROOT))




from yolov9.utils.general import check_dataset, check_img_size, non_max_suppression, box_iou, xywh2xyxy
from yolov9.utils.metrics import ap_per_class
from yolov9.models.common import DetectMultiBackend
from yolov9.utils.torch_utils import select_device
from yolov9.utils.dataloaders import LoadImages
import yaml

# 信心閾值設定
CALC_CONF_THRES = 0.5  # 算分模式（mAP計算）：用於輔助視覺化理解，非正式評估指標
DRAW_CONF_THRES = 0.5  # 畫圖模式（視覺化）：過濾低信心預測以展示實際應用效果


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


def load_label(label_path, img_size):
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


def calculate_map50_single_image(model, img_path, label_path, device, img_size, nc, conf_thres=0.001, iou_thres=0.6, debug=False):
    """
    計算單張圖片的 mAP50
    
    Args:
        model: 載入的模型
        img_path: 圖片路徑
        label_path: 標籤路徑
        device: 計算設備
        img_size: 圖片大小
        nc: 類別數量
        conf_thres: 信心閾值
        iou_thres: NMS IoU 閾值
        debug: 是否輸出調試信息
    
    Returns:
        float: 該圖片的 mAP50
    """
    # 載入並預處理圖片
    dataset = LoadImages(str(img_path), img_size=img_size)
    
    # 載入標籤 (YOLO 格式: class x_center y_center width height, 都是歸一化的)
    labels = load_label(label_path, img_size)
    
    # 如果沒有標籤，返回 NaN (表示無法計算)
    if len(labels) == 0:
        if debug:
            print(f"  {img_path.name}: 無標籤")
        return np.nan
    
    if debug:
        print(f"  {img_path.name}: {len(labels)} 個標籤")
    
    iouv = torch.linspace(0.5, 0.95, 10).to(device)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()
    
    # 進行推論
    for path, img, im0s, vid_cap, s in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.float() / 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        
        # 取得原始圖片尺寸 (im0s 是原始未 resize 的圖片)
        height, width = im0s.shape[:2]
        
        # 取得 resize 後的圖片尺寸 (img 已經被 resize)
        _, _, img_h, img_w = img.shape
        
        if debug:
            print(f"    原始尺寸: {width}x{height}, Resize後: {img_w}x{img_h}")
        
        # 推論
        with torch.no_grad():
            pred = model(img)
            if isinstance(pred, list):
                pred = pred[0]
        
        # NMS (pred 的座標是相對於 resize 後的圖片)
        pred = non_max_suppression(pred, conf_thres, iou_thres, labels=[], multi_label=True, agnostic=False)
        
        # 處理預測結果
        if len(pred) > 0 and len(pred[0]) > 0:
            pred = pred[0]  # 取第一張圖片的預測
            
            if debug:
                print(f"    推論得到 {len(pred)} 個預測")
            
            # 過濾有效類別
            valid_mask = (pred[:, 5] >= 0) & (pred[:, 5] < nc)
            pred = pred[valid_mask]
            
            if debug:
                print(f"    過濾後剩餘 {len(pred)} 個有效預測")
            
            if len(pred) == 0:
                # 有標籤但沒有有效預測，返回 0
                return 0.0
            
            # 準備標籤 (從歸一化座標轉換為像素座標)
            # 注意：pred 的座標是相對於 resize 後的圖片 (img_w x img_h)
            # 所以標籤也應該轉換到 resize 後的尺寸
            # labels 格式: [class, x_center_norm, y_center_norm, w_norm, h_norm]
            labels_pixel = labels.copy()
            labels_pixel[:, 1] *= img_w   # x_center (使用 resize 後的寬度)
            labels_pixel[:, 2] *= img_h   # y_center (使用 resize 後的高度)
            labels_pixel[:, 3] *= img_w   # width
            labels_pixel[:, 4] *= img_h   # height
            
            # 轉換 xywh 到 xyxy
            tbox = xywh2xyxy(torch.from_numpy(labels_pixel[:, 1:5]).float()).to(device)
            tcls = torch.from_numpy(labels_pixel[:, 0:1]).float().to(device)
            labels_tensor = torch.cat((tcls, tbox), 1)
            
            # 計算 correct predictions
            correct = process_batch(pred, labels_tensor, iouv)
            
            if debug:
                print(f"    pred classes: {pred[:, 5].unique().cpu().numpy()}")
                print(f"    true classes: {labels_tensor[:, 0].unique().cpu().numpy()}")
                print(f"    correct matches: {correct.sum().item()}/{len(correct)}")
                # 顯示前幾個預測的 confidence
                print(f"    pred conf (top 5): {pred[:5, 4].cpu().numpy()}")
                print(f"    pred bbox sample: {pred[0, :4].cpu().numpy()}")
                print(f"    true bbox sample: {labels_tensor[0, 1:].cpu().numpy()}")
            
            # 收集統計數據
            tcls_list = labels_pixel[:, 0].astype(int).tolist()
            stats = [(correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls_list)]
            
            # 計算 AP
            stats = [np.concatenate(x, 0) for x in zip(*stats)]
            if len(stats) and stats[0].any():
                try:
                    # names 需要是 dict
                    names_dict = {i: str(i) for i in range(nc)}
                    tp, fp, p, r, f1, ap, ap_class = ap_per_class(*stats, plot=False, save_dir="", names=names_dict)
                    if len(ap) > 0:
                        ap50 = ap[:, 0]  # AP@0.5
                        map50 = ap50.mean()
                        if debug:
                            print(f"    mAP50 = {map50:.4f}")
                        return float(map50)
                except Exception as e:
                    # 如果計算失敗，返回 0
                    if debug:
                        print(f"    計算 AP 失敗: {e}")
                    return 0.0
        else:
            if debug:
                print(f"    無預測結果")
        
        # 有標籤但沒有預測，返回 0
        return 0.0
    
    return 0.0


def main():
    parser = argparse.ArgumentParser(
        description='計算多個模型在指定圖片上的 mAP50 矩陣',
        usage='python calc_agg_model.py EXPERIMENT_DIR'
    )
    parser.add_argument('experiment_dir', type=str, help='實驗資料夾路徑 (例如: ./experiments/60_kittiOA010_fedyoga_4C_12R_202511041631/)')
    parser.add_argument('--num_images', type=int, default=-1, help='使用的驗證圖片數量 (預設: -1 表示全部)')
    parser.add_argument('--model_id', type=str, default='1,5,15', help='要驗證的模型 round ID，逗號分隔 (預設: 1,5,15)')
    parser.add_argument('--show', type=str, default='', help='指定單一驗證圖片檔名 (例如 005824.png) 進行各模型畫框輸出')
    parser.add_argument('--img_size', type=int, default=640, help='推論圖片大小 (預設: 640)')
    parser.add_argument('--device', type=str, default='', help='計算設備 (預設: 自動選擇)')
    parser.add_argument('--iou_thres', type=float, default=0.6, help='NMS IoU 閾值 (預設: 0.6)')
    
    args = parser.parse_args()
    
    # 解析模型 ID
    model_ids = [int(x.strip()) for x in args.model_id.split(',')]
    
    # 建立模型列表（初始權重 + 指定的 round 模型）
    val_models_list = [
        ('yolov9-c.pt', WORK_ROOT),  # 初始權重在專案根目錄
    ]
    # 加入指定的 round 模型
    for mid in model_ids:
        val_models_list.append((f'w_s_r{mid}.pt', None))  # None 表示在 aggregated_weights 目錄
    
    # 解析實驗資料夾路徑
    experiment_dir = Path(args.experiment_dir)
    if not experiment_dir.is_absolute():
        experiment_dir = WORK_ROOT / experiment_dir
    
    if not experiment_dir.exists():
        print(f"錯誤: 找不到實驗資料夾: {experiment_dir}")
        return
    
    # 建立模型基礎路徑
    val_models_base_dir = experiment_dir / 'aggregated_weights'
    
    if not val_models_base_dir.exists():
        print(f"錯誤: 找不到 aggregated_weights 資料夾: {val_models_base_dir}")
        return
    
    print(f"實驗資料夾: {experiment_dir.name}")
    print(f"模型資料夾: {val_models_base_dir}")
    
    # 設定設備
    device = select_device(args.device)
    print(f"使用設備: {device}")
    
    # 固定使用 kittiO 資料集 (從實驗資料夾名稱判斷，或直接使用 kittiO)
    # 從資料夾名稱提取資料集名稱
    exp_name = experiment_dir.name
    if 'kittiOA' in exp_name:
        dataset_name = 'kittiO'  # kittiOA010 等變體都使用 kittiO 的驗證集
    elif 'kittiO' in exp_name:
        dataset_name = 'kittiO'
    elif 'bdd100k' in exp_name:
        dataset_name = 'bdd100k'
    elif 'sim10k' in exp_name:
        dataset_name = 'sim10k'
    elif 'foggy' in exp_name:
        dataset_name = 'foggy'
    elif 'cityscapes' in exp_name:
        dataset_name = 'cityscapes'
    else:
        dataset_name = 'kittiO'  # 預設使用 kittiO
    
    print(f"使用資料集: {dataset_name}")
    
    # 載入資料集設定
    data_yaml_path = WORK_ROOT / 'data' / f'{dataset_name}.yaml'
    if not data_yaml_path.exists():
        print(f"錯誤: 找不到資料集設定檔: {data_yaml_path}")
        return
    
    with open(data_yaml_path, 'r') as f:
        data_dict = yaml.safe_load(f)
    
    # 檢查資料集
    data_dict = check_dataset(str(data_yaml_path))
    nc = data_dict['nc']
    names = data_dict['names']
    
    # 取得驗證圖片路徑
    val_path = Path(data_dict['path']) / data_dict['val']
    if not val_path.exists():
        # 嘗試相對路徑
        val_path = WORK_ROOT / 'datasets' / dataset_name / 'images' / 'val'
    
    if not val_path.exists():
        print(f"錯誤: 找不到驗證圖片路徑: {val_path}")
        return
    
    # 取得驗證圖片
    all_val_images = sorted(list(val_path.glob('*.png')) + list(val_path.glob('*.jpg')))
    
    if len(all_val_images) == 0:
        print(f"錯誤: 在 {val_path} 中找不到圖片")
        return
    
    # 如果指定數量，取前 N 張；否則使用全部
    if args.num_images > 0:
        val_images = all_val_images[:args.num_images]
        print(f"找到 {len(all_val_images)} 張驗證圖片，使用前 {len(val_images)} 張")
    else:
        val_images = all_val_images
        print(f"找到 {len(val_images)} 張驗證圖片，使用全部")
    
    # 載入所有模型
    models = []
    model_names = []
    print("\n載入模型:")
    for i, (model_file, base_dir) in enumerate(val_models_list):
        # 根據 base_dir 決定模型路徑
        if base_dir is None:
            model_path = val_models_base_dir / model_file
        else:
            model_path = base_dir / model_file
        
        if not model_path.exists():
            print(f"  警告: 找不到模型檔案 {model_path}，跳過")
            continue
        
        print(f"  [{i}] {model_file}")
        model = DetectMultiBackend(str(model_path), device=device)
        gs = int(model.stride)
        img_size = check_img_size(args.img_size, s=gs)
        models.append((model, img_size))
        model_names.append(model_path.stem)
    
    if len(models) == 0:
        print("\n錯誤: 沒有成功載入任何模型")
        return
    
    print(f"\n成功載入 {len(models)} 個模型")
    
    # 模式判斷：有 --show 參數時只做畫框，否則計算 mAP
    if args.show:
        # ===== 模式2: 單張圖片畫框輸出 =====
        target_name = args.show.strip()
        # 從完整驗證集搜尋該圖片
        all_val_images = sorted(list(val_path.glob('*.png')) + list(val_path.glob('*.jpg')))
        target_matches = [p for p in all_val_images if p.name == target_name]
        if not target_matches:
            print(f"\n[show] 找不到圖片 {target_name} 於完整驗證集，跳過畫框輸出")
            return
        
        show_img_path = target_matches[0]
        print(f"\n[show] 為 {target_name} 生成各模型偵測結果圖 ...")
        print(f"  使用信心閾值: {DRAW_CONF_THRES} (畫圖模式)")
        
        # 取得類別名稱字典
        class_names = names  # 從 data_dict 載入的類別名稱
        
        # 載入真實標籤
        dataset_root = Path(data_dict['path'])
        label_path = dataset_root / 'labels' / 'val' / show_img_path.with_suffix('.txt').name
        
        if not label_path.exists():
            print(f"  警告: 找不到標籤檔案 {label_path}")
            gt_labels = np.zeros((0, 5), dtype=np.float32)
        else:
            gt_labels = load_label(label_path, args.img_size)
        
        print(f"  真實標籤數量: {len(gt_labels)}")
        
        # 載入原始圖片
        import cv2
        from yolov9.utils.general import scale_boxes
        
        # IoU 閾值設定
        iouv = torch.linspace(0.5, 0.95, 10).to(device)
        
        for mi, (model, img_size) in enumerate(models):
            # 推論該單張圖片
            dataset = LoadImages(str(show_img_path), img_size=img_size)
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
                pred = non_max_suppression(pred, DRAW_CONF_THRES, args.iou_thres, labels=[], multi_label=True, agnostic=False)
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
                        cls_name = class_names.get(cls_int, str(cls_int))  # 取得類別名稱，若找不到則用數字
                        color = (0, 255, 0)  # 綠色 - 真實標籤
                        cv2.rectangle(im_draw, (x1, y1), (x2, y2), color, 2)
                        label = f"GT_{cls_name}"
                        cv2.putText(im_draw, label, (x1, max(y1-5,0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
                else:
                    labels_tensor = torch.empty((0, 5)).to(device)
                
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
                        cls_name = class_names.get(cls_int, str(cls_int))  # 取得類別名稱
                        
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
                
                # 計算 AP50（使用與 mAP 計算相同的邏輯）
                if num_gt > 0 and len(detections) > 0:
                    # 準備統計數據計算 AP
                    tcls_list = labels_pixel[:, 0].astype(int).tolist() if len(gt_labels) > 0 else []
                    stats = [(correct.cpu(), detections[:, 4].cpu(), detections[:, 5].cpu(), tcls_list)]
                    stats = [np.concatenate(x, 0) for x in zip(*stats)]
                    if len(stats) and stats[0].any():
                        try:
                            names_dict = {i: str(i) for i in range(nc)}
                            tp, fp, p, r, f1, ap, ap_class = ap_per_class(*stats, plot=False, save_dir="", names=names_dict)
                            if len(ap) > 0:
                                ap50_value = ap[:, 0].mean()  # AP@0.5
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
                
                # 準備文字內容（移除 Model 名稱）
                texts = [
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
                    if i == 0:  # AP50
                        color = (255, 255, 0)  # 青色
                    elif i == 1:  # GT
                        color = (0, 255, 0)  # 綠色
                    elif i == 2:  # TP
                        color = (0, 255, 255)  # 黃色
                    elif i == 3:  # FP
                        color = (0, 0, 255)  # 紅色
                    else:  # FN
                        color = (128, 128, 128)  # 灰色
                    
                    cv2.putText(im_draw, text, (x1_box + padding, y_offset), 
                               font, font_scale, color, thickness, cv2.LINE_AA)
                    y_offset += line_height
                
                # 檔名規則: show_<experiment>_<stem>_w<round>.png 其中初始模型 round=0
                if mi == 0:
                    round_id = '0'
                else:
                    round_id = str(model_ids[mi-1])
                out_name = f"show_{experiment_dir.name}_{show_img_path.stem}_w{round_id}.png"
                cv2.imwrite(str(Path.cwd() / out_name), im_draw)
                print(f"  輸出: {out_name} (GT:{len(gt_labels)}, Pred:{len(detections)}, TP:{is_correct.sum().item() if len(detections) > 0 else 0})")
        print("\n✓ 畫框輸出完成")
        return
    
    # ===== 模式1: 計算 mAP50 =====
    print(f"\n使用信心閾值: {CALC_CONF_THRES} (算分模式)")
    # 初始化結果矩陣 (num_images x num_models)
    results = np.zeros((len(val_images), len(models)), dtype=np.float32)
    
    # 計算每張圖片在每個模型上的 mAP50
    print("\n開始計算 mAP50...")
    for img_idx, img_path in enumerate(tqdm(val_images, desc="處理圖片")):
        # 找到對應的標籤檔案
        # img_path 格式: .../kittiO/images/val/000023.png
        # label_path 應該是: .../kittiO/labels/val/000023.txt
        dataset_root = img_path.parent.parent.parent  # 從 images/val 往上兩層到資料集根目錄
        label_path = dataset_root / 'labels' / 'val' / (img_path.stem + '.txt')
        
        if not label_path.exists():
            print(f"\n警告: 找不到標籤檔案 {label_path}")
            continue
        
        # 對每個模型計算 mAP50
        for model_idx, (model, img_size) in enumerate(models):
            map50 = calculate_map50_single_image(
                model, img_path, label_path, device, img_size, nc,
                conf_thres=CALC_CONF_THRES, iou_thres=args.iou_thres,
                debug=False  # 關閉 debug
            )
            results[img_idx, model_idx] = map50
    
    # 建立 DataFrame 並儲存
    df = pd.DataFrame(results, columns=model_names)
    df.insert(0, 'image', [img.name for img in val_images])
    
    # 儲存到當前執行目錄
    output_filename = f'calc_agg_model_map50_{experiment_dir.name}.csv'
    output_path = Path.cwd() / output_filename
    df.to_csv(output_path, index=False, float_format='%.4f')
    
    print(f"\n✓ 結果已儲存至: {output_path}")
    print(f"  矩陣大小: {len(val_images)} 張圖片 × {len(models)} 個模型")
    
    # 顯示統計資訊
    print("\n=== 統計資訊 ===")
    for i, model_name in enumerate(model_names):
        mean_map50 = np.nanmean(results[:, i])  # 使用 nanmean 忽略 NaN 值
        print(f"{model_name}: 平均 mAP50 = {mean_map50:.4f}")
    
    # 計算進步最多的前10張圖 (改成倒數兩個比)
    if len(models) >= 3:  # 至少要有初始權重 + 兩個 round 模型
        print("\n=== 進步最多的前 10 張圖 ===")
        # 計算進步幅度 (改成倒數兩個比)
        improvements = results[:, -1] - results[:, -2]
        
        # 過濾掉 NaN 值
        valid_mask = ~np.isnan(improvements)
        valid_improvements = improvements[valid_mask]
        valid_indices = np.where(valid_mask)[0]
        
        if len(valid_improvements) > 0:
            # 取得前10個進步最多的索引
            top_k = min(10, len(valid_improvements))
            top_indices = valid_indices[np.argsort(valid_improvements)[-top_k:][::-1]]
            
            print(f"{'排名':<6} {'圖片':<20} {model_names[-2]:<12} {model_names[-1]:<12} {'進步幅度':<12}")
            print("-" * 70)
            for rank, idx in enumerate(top_indices, 1):
                img_name = val_images[idx].name
                score_first = results[idx, -2]  # 倒數第二個模型
                score_last = results[idx, -1]  # 最後一個模型
                improvement = improvements[idx]
                print(f"{rank:<6} {img_name:<20} {score_first:>10.4f}  {score_last:>10.4f}  +{improvement:>9.4f}")
        else:
            print("  無有效資料")


if __name__ == '__main__':
    main()

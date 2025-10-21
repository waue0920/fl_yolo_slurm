## 修改說明 README
## 2024-10-21 由 waue0920 修改  
基本上沒修改 yolov9 的主要邏輯，
但是為了避免每次訓練時都把模型上傳到 wandb，因為模型檔案通常很大，這樣會浪費很多時間和頻寬。
- yolov9/utils/loggers/wandb/wandb_utils.py
- yolov9/utils/loggers/__init__.py 
  
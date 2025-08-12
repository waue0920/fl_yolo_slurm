# Datasets

目前測試支援五種交通資料集
* KITTI
* SIM10K
* BDD100K
* Cityscapes
* Foggy

可自行至官方網站下載，或使用此測試包 
* yolo_traffic_datasets_5in1_26G.tar ([twcc-cos載點](https://cos.twcc.ai/wauehpcproject/yolo_traffic_datasets_5in1_26G.tar))


``` bash
./
├── data
│   ├── bdd.yaml
│   ├── cityscapes.yaml
│   ├── foggy.yaml
│   ├── kitti.yaml
│   └── sim10k.yaml
├── datasets
│   ├── bdd100k
│   │   ├── VOC
│   │   ├── images
│   │   └── labels
│   ├── cityscapes
│   │   ├── VOC
│   │   ├── images
│   │   └── labels
│   ├── foggy
│   │   ├── VOC
│   │   ├── images
│   │   └── labels
│   ├── kitti
│   │   ├── VOC
│   │   ├── images
│   │   └── labels
│   └── sim10k
│       ├── VOC
│       ├── images
│       └── labels
```
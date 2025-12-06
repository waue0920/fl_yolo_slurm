# Datasets

Currently supports 5 traffic datasets for testing:
* KITTI
* SIM10K
* BDD100K
* Cityscapes
* Foggy

You can download them from their official websites or use this test package:
* yolo_traffic_datasets_5in1_26G.tar ([TWCC-COS Download Link](https://cos.twcc.ai/wauehpcproject/yolo_traffic_datasets_5in1_26G.tar))

```bash
./
├── data
│   ├── bdd.yaml
│   ├── cityscapes.yaml
│   ├── foggy.yaml
│   ├── kitti.yaml
│   └── sim10k.yaml
├── datasets
│   ├── bdd100k
│   │   ├── VOC
│   │   ├── images
│   │   └── labels
│   ├── cityscapes
│   │   ├── VOC
│   │   ├── images
│   │   └── labels
│   ├── foggy
│   │   ├── VOC
│   │   ├── images
│   │   └── labels
│   ├── kitti
│   │   ├── VOC
│   │   ├── images
│   │   └── labels
│   └── sim10k
│       ├── VOC
│       ├── images
│       └── labels
```
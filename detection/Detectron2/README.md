## DETECTRON 사용법

- detection 개 짱짱 편함

|   | |
|------|--|
|환경    | 2080 Ti 2EA   |
|cuda    | v11.8         |
|python  | python 3.10.6   |
|pytorch  | 2.1.1+cu118   |


### Install
[detectron2 official installation link](https://detectron2.readthedocs.io/en/latest/tutorials/install.html)
+
``` pip install fvscore, cloudpickle, pycocotools ```
### ⭐ Dataset
1. coco 형식으로 맞춰서 datasets 폴더에 저장
    ```
        datasets
        │  ├─ COCO
        │  │  ├─ annotations
        │  │  │  ├─ captions_train2014.json
        │  │  │  ├─ captions_val2014.json
        │  │  │  ├─ instances_train2014.json
        │  │  │  ├─ instances_val2014.json
        │  │  │  ├─ person_keypoints_train2014.json
        │  │  │  └─ person_keypoints_val2014.json
        │  │  |
        │  │  ├─ train2014
        │  │  │  ├─ COCO_train2014_000000000151.jpg
        │  │  │  ├─ COCO_train2014_000000000260.jpg
        │  │  │  ├─ COCO_train2014_000000000307.jpg
        │  │  │  ├─ COCO_train2014_000000000404.jpg
        │  │  │  ├─ COCO_train2014_000000000450.jpg
    ```
2. `train_net.py`에 데이터셋 register
    ```python
        dataset_root = 'datasets/'  #
        # 학습용 dataset 등록
        register_coco_instances("COCO_train", meta, dataset_root + "COCO/annotations/person_keypoints_train2014.json", dataset_root + "COCO/train2014")

        # 테스트용 dataset 등록
        register_coco_instances("COCO_test", meta, dataset_root + "COCO/annotations/person_keypoints_val2014.json", dataset_root + "COCO/val2014")
    ```
    - register_coco_instances({name}, {metadata}, {annotation 파일 경로}, {image root 경로})

### Training
- `python_train_net.py --config {cofig} --num-gpus {n_gpu} --resume`
    - config : `./config/ ~~ .yaml`
        학습, 모델에 대한 config 정의된 파일
    - n_gpu : 학습에 사용할 gpu 개수(for DDP)

> bug
    - if. port가 이미 사용중이다? => `--dist-url tcp://127.0.0.1:12345` 추가
# Introduction
### Astroformer: More Data Might not be all you need for Classifaction

# Results Overview

### CIFAR-100

| Model Name   | Top-1 Accuracy | FLOPs | Params |
|--------------|----------------|-------|--------|
| Astroformer-3| 87.65          | 31.36 | 161.95 |
| Astroformer-4| 93.36          | 60.54 | 271.68 |
| Astroformer-5| 89.38          | 115.97| 655.34 |

### CIFAR-10

| Model Name   | Top-1 Accuracy | FLOPs | Params |
|--------------|----------------|-------|--------|
| Astroformer-3| 99.12          | 31.36 | 161.75 |
| Astroformer-4| 98.93          | 60.54 | 271.54 |
| Astroformer-5| 93.23          | 115.97| 655.04 |

### Tiny Imagenet

| Model Name   | Top-1 Accuracy | FLOPs | Params |
|--------------|----------------|-------|--------|
| Astroformer-3| 86.86          | 24.84 | 150.39 |
| Astroformer-4| 91.12          | 40.38 | 242.58 |
| Astroformer-5| 92.98          | 89.88 | 595.55 |

### Galaxy10 DECals

| Model Name   | Top-1 Accuracy | FLOPs | Params |
|--------------|----------------|-------|--------|
| Astroformer-3| 92.39          | 31.36 | 161.75 |
| Astroformer-4| 94.86          | 60.54 | 271.54 |
| Astroformer-5| 94.81          | 105.9 | 681.25 |

## Dataset setting

``` sh
|-DATASET
    |-train
    | |-class1
    | | |-img1.jpg
    | | |-img2.jpg
    | | |-...
    | |-class2
    | | |-img3.jpg
    | | |-...
    | |-class3
    | | |-img4.jpg
    | | |-...
    | |-...
    |-validation
      |-class1
      | |-img5.jpg
      | |-...
      |-class2
      | |-img6.jpg
      | |-...
      |-class3
      | |-img7.jpg
      | |-...
    |-test
      |-class1
      | |-img5.jpg
      | |-...
      |-class2
      | |-img6.jpg
      | |-...
      |-class3
      | |-img7.jpg
      | |-...
      
```


## Training

Astroformer-5은 엄청 큰 모델이니, 3-4을 사용하는 걸 추천함.

```sh
sh distributed_train.sh 8 [데이터셋 ROOT PATH] 
    --train-split [your_train_dir] 
    --val-split [your_val_dir] 
    --model astroformer_5 # model setting
    --num-classes 10
    --img-size 256
    --in-chans 3
    --input-size 3 256 256
    --batch-size 256
    --grad-accum-steps 1
    --opt adamw
    --sched cosine
    --lr-base 2e-5
    --lr-cycle-decay 1e-2
    --lr-k-decay 1
    --warmup-lr 1e-5
    --epochs 300
    --warmup-epochs 5
    --mixup 0.8
    --smoothing 0.1
    --drop 0.1
    --save-images
    --amp
    --amp-impl apex # apex 연산을 원하면 추가 세팅 필요 e.g) pip install apex
    --output result_ours/"NAME"
    --log-wandb
```





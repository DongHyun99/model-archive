# Introduction
### CvT: Introducing Convolutions to Vision Transformers

![](figures/pipeline.svg)

# 가중치 파일 (Pretrained model)
## Models pre-trained on ImageNet-1k
| Model  | Resolution | Param | GFLOPs | Top-1 |
|--------|------------|-------|--------|-------|
| CvT-13 | 224x224    | 20M   | 4.5    | 81.6  |
| CvT-21 | 224x224    | 32M   | 7.1    | 82.5  |
| CvT-13 | 384x384    | 20M   | 16.3   | 83.0  |
| CvT-21 | 384x384    | 32M   | 24.9   | 83.3  |

## Models pre-trained on ImageNet-22k
| Model   | Resolution | Param | GFLOPs | Top-1 |
|---------|------------|-------|--------|-------|
| CvT-13  | 384x384    | 20M   | 16.3   | 83.3  |
| CvT-21  | 384x384    | 32M   | 24.9   | 84.9  |
| CvT-W24 | 384x384    | 277M  | 193.2  | 87.6  |

You can download all the models from our [model zoo](https://1drv.ms/u/s!AhIXJn_J-blW9RzF3rMW7SsLHa8h?e=blQ0Al).


# Quick start
## Installation

``` sh
pip install -r requirements.txt 
```

해당 코드는 torch 1.7.1, python 3.7 ~ 3.8 사이에서 모든 과정을 수행하길 권장함.

## Data preparation
데이터를 아래 구조대로 구축하는 걸 권장함.
``` sh
|-DATASET
  |-imagenet # 이름이 imagenet이어야 함. 바꾼다면 code를 참고하여 수정하길 권장
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
    |-val
      |-class1
      | |-img5.jpg
      | |-...
      |-class2
      | |-img6.jpg
      | |-...
      |-class3
      | |-img7.jpg
      | |-...
      |-...
```

## Run
experiments 폴더 내에 각 가중치 별 config.yaml이 존재함.

``` sh
experiments
|-imagenet
| |- cvt-13-224x224.yaml
| |- cvt-13-384x384.yaml
| |- cvt-21-224x224.yaml
| |- cvt-21-384x384.yaml
| |- cvt-w24-384x384.yaml
|-...
```

local에서 학습하는 방법

``` sh
bash run.sh -g 2 -t train --cfg experiments/imagenet/cvt/cvt-13-224x224.yaml
# -g : gpu 개수, -t : train, test mode, --cfg: 사용할 config 파일 설정
```

local에서 test하는 방법
``` sh
bash run.sh -t test --cfg experiments/imagenet/cvt/cvt-13-224x224.yaml TEST MODEL_FILE ${PRETRAINED_MODLE_FILE}
```
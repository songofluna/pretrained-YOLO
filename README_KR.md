# YOLO Object Detection, Tracking, and Line Counting

<p align="center">
  <img src="results/figures/hero.png" width="900">
</p>

<p align="center">
  <b>Ultralytics YOLO · Object Detection · Confidence Thresholds · Embeddings · Multi-Object Tracking · Trajectory Visualization · Line Crossing Counter</b>
</p>

<p align="center">
  <a href="README.md">English README</a>
</p>

## Overview

이 프로젝트는 pretrained YOLO model을 한 번 실행해 보는 것뿐 아니라, 실제 detection output을 분석한 뒤 video tracking과 object counting까지 구현해 보는 것을 목표로 한다.

먼저 한 장의 image에서 bounding box, class, confidence score를 직접 확인하고 confidence threshold와 IoU가 prediction에 어떤 영향을 주는지 살펴본다.

그 다음 pretrained model에서 image embedding을 추출하고 cosine similarity를 비교한다. 마지막에는 video에 YOLO tracking을 적용하여 object ID와 trajectory를 시각화하고, 가상의 선을 통과하는 object를 자동으로 세는 line-crossing counter를 구현한다.

전체 흐름은 다음과 같다.

```text
Object Detection
      ↓
Prediction Analysis
      ↓
Embedding Inspection
      ↓
Multi-Object Tracking
      ↓
Trajectory Visualization
      ↓
Line-Crossing Counter
```

## Project Structure

```text
yolo-detection-tracking/
├── README.md
├── README_KR.md
│
├── notebooks/
│   └── 01_pretrained_yolo.ipynb
│
├── assets/
│   └── input/
│       ├── mixkit-cars-driving-by-on-road-2022-hd-ready.mp4
│       └── mixkit-young-bearded-man-walking-down-the-street-4888-hd-ready.mp4
│
└── results/
    ├── figures/
    │   ├── hero.png
    │   ├── detection_result.png
    │   ├── object_crops.png
    │   ├── confidence_thresholds.png
    │   ├── embedding_similarity.png
    │   └── tracking_frame.png
    │
    └── videos/
        ├── tracking_result.mp4
        ├── tracking_trajectory.mp4
        └── line_counter.mp4
```

## 1. Pretrained Object Detection

먼저 pretrained YOLO model을 한 장의 image에 적용한다.

각 detection에는 다음과 같은 정보가 포함된다.

* bounding box 좌표
* predicted class ID
* class name
* confidence score

<p align="center">
  <img src="results/figures/detection_result.png" width="850">
</p>

단순히 bounding box가 그려진 결과만 보는 것이 아니라 model이 실제로 반환하는 prediction object 내부를 직접 확인했다.

이를 통해 object detection을 단순한 시각화 기능이 아니라 구조화된 prediction 문제로 이해할 수 있다.

## 2. Detected Object Crop

각 bounding box의 좌표를 이용하여 original image에서 detected object를 직접 crop하였다.

<p align="center">
  <img src="results/figures/object_crops.png" width="900">
</p>

이 과정에서 model의 bounding-box output이 실제 image indexing과 어떻게 연결되는지 확인했다.

또한 detection output을 이후의 다른 computer-vision task에 그대로 사용할 수 있다는 점을 확인했다.

## 3. Confidence Threshold

Confidence threshold를 바꾸면서 detection 결과가 어떻게 달라지는지 비교하였다.

사용한 threshold는 다음과 같다.

```text
conf = 0.10
conf = 0.25
conf = 0.50
conf = 0.80
```

실험에서는 다음과 같은 결과가 나왔다.

```text
conf=0.10 -> {'person': 5, 'bus': 1, 'stop sign': 1}
conf=0.25 -> {'person': 4, 'bus': 1}
conf=0.50 -> {'person': 4, 'bus': 1}
conf=0.80 -> {'person': 3, 'bus': 1}
```

<p align="center">
  <img src="results/figures/confidence_thresholds.png" width="950">
</p>

Threshold를 낮추면 더 많은 detection을 남길 수 있지만 confidence가 낮은 prediction까지 포함된다.

반대로 threshold를 높이면 prediction이 더 보수적으로 변하지만 실제 object까지 제거될 수 있다.

이를 통해 confidence threshold가 단순한 옵션이 아니라 recall과 precision 사이의 trade-off를 조절하는 값이라는 것을 직접 확인했다.

## 4. IoU와 Overlapping Boxes

Bounding box 사이의 overlap을 직접 확인하기 위해 pairwise IoU를 계산하였다.

두 box $A$와 $B$에 대해 IoU는 다음과 같이 정의된다.

$$
\mathrm{IoU}(A,B) = \frac{|A \cap B|}{|A \cup B|}
$$

실험 중 두 person detection 사이에서 다음과 같이 매우 높은 IoU가 관찰되었다.

```text
Maximum IoU: 0.9109
```

이 결과를 통해 거의 동일한 위치를 가리키는 여러 prediction이 왜 문제가 되는지 확인할 수 있었다.

또한 IoU를 NMS와 box matching을 이해하기 위한 기하학적 기준으로 연결해서 살펴보았다.

Notebook에서는 end-to-end prediction과 traditional NMS 방식의 설정도 비교하였다.

## 5. Image Embedding

Pretrained model을 이용하여 image에서 256-dimensional embedding을 추출하였다.

```text
Embedding shape: torch.Size([256])
```

Original image와 여러 transformation을 적용한 image 사이의 cosine similarity를 비교하였다.

실험 결과의 예시는 다음과 같다.

```text
Original vs Flip       : 0.9952
Original vs Grayscale  : 0.9994
Original vs Blur       : 0.9911
Original vs Different  : 0.9504
Original vs Black      : 0.8590
Original vs White      : 0.8322
Original vs Noise      : 0.8408
```

<p align="center">
  <img src="results/figures/embedding_similarity.png" width="800">
</p>

Flip, grayscale, blur를 적용하더라도 embedding similarity가 매우 높게 유지되었다.

이를 통해 pretrained vision model의 feature representation이 여러 image transformation에 상당히 invariant할 수 있다는 것을 확인했다.

동시에 완전히 다른 image도 높은 similarity를 보일 수 있었다.

따라서 detector에서 얻은 global embedding을 그대로 image identity나 semantic retrieval을 위한 정교한 similarity metric으로 해석하면 안 된다는 한계도 확인했다.

## 6. Multi-Object Tracking

다음 단계에서는 single image detection을 video로 확장하였다.

각 frame을 독립적으로 detection하는 것에서 끝나지 않고, 동일한 object에 frame 사이에서 지속되는 track ID를 부여한다.

<p align="center">
  <img src="results/figures/tracking_frame.png" width="900">
</p>

Tracking의 핵심 구조는 다음과 같다.

```text
Detection
    ↓
Frame-to-frame association
    ↓
Persistent track ID
```

결과 영상:

[tracking_result.mp4 보기](results/videos/tracking_result.mp4)

Detection이 "이 frame에 무엇이 있는가?"를 해결한다면, tracking은 "이 object가 다음 frame에서도 같은 object인가?"를 추가로 해결한다.

## 7. Trajectory Visualization

각 tracked object의 중심 좌표를 frame마다 저장하고, 시간 순서대로 연결하여 trajectory를 시각화하였다.

결과 영상:

[tracking_trajectory.mp4 보기](results/videos/tracking_trajectory.mp4)

Trajectory는 단순히 결과를 보기 좋게 만드는 용도만 있는 것이 아니다.

Track path가 갑자기 다른 위치로 튀거나 비정상적인 경로를 만들면 tracker가 object identity를 잘못 연결한 **ID switch**가 발생했을 가능성을 확인할 수 있다.

따라서 trajectory visualization은 tracking quality를 눈으로 검사할 수 있는 diagnostic tool 역할도 한다.

## 8. Line-Crossing Object Counter

마지막 단계에서는 video 위에 virtual line을 만들고 tracked object가 그 선을 통과할 때 자동으로 count하도록 구현하였다.

전체 pipeline은 다음과 같다.

```text
Video frame
    ↓
YOLO detection
    ↓
Object tracking
    ↓
Track center calculation
    ↓
Line-crossing test
    ↓
Count update
```

결과 영상:

[line_counter.mp4 보기](results/videos/line_counter.mp4)

Track ID를 유지하고 있기 때문에 매 frame마다 동일한 object를 반복해서 세는 것이 아니라 실제로 선을 통과하는 event를 기준으로 count할 수 있다.

이 구조는 다음과 같은 실제 응용의 가장 기본적인 형태와 연결된다.

* 차량 통행량 측정
* 유동인구 계수
* CCTV analytics
* 출입 인원 측정
* smart camera system

## Key Concepts

| Topic                | 확인한 내용                                    |
| -------------------- | ----------------------------------------- |
| Object detection     | Bounding box, class, confidence           |
| COCO classes         | Pretrained detector가 인식할 수 있는 label space |
| Confidence threshold | Prediction을 남기는 기준                        |
| IoU                  | Bounding box 사이의 geometric overlap        |
| NMS                  | 겹치는 prediction을 제거하는 과정                   |
| Object crop          | Box 좌표를 실제 image processing에 사용           |
| Embedding            | Learned image representation 추출           |
| Cosine similarity    | Embedding vector 사이의 similarity           |
| Tracking             | Frame 사이에서 object identity 유지             |
| Track ID             | 동일 object를 시간 방향으로 연결                     |
| Trajectory           | Object의 이동 경로 시각화                         |
| ID switch            | Tracker가 identity를 잘못 연결하는 오류             |
| Line crossing        | Object movement를 event로 변환                |
| Object counting      | Tracking output을 이용한 video analytics      |

## What I Learned

이 프로젝트를 통해 서로 따로 공부하면 추상적으로 느껴질 수 있는 여러 computer-vision 개념을 하나의 pipeline 안에서 연결해볼 수 있었다.

직접 확인한 내용은 다음과 같다.

* YOLO의 plotted result가 아니라 실제 prediction object를 읽는 방법
* bounding-box coordinate와 confidence score의 의미
* confidence threshold가 prediction에 미치는 영향
* bounding box 사이의 IoU 계산
* IoU와 NMS의 관계
* detection 좌표를 이용한 object crop
* pretrained vision model에서 embedding 추출
* cosine similarity를 이용한 representation 비교
* generic embedding과 task-specific metric learning의 차이
* video object tracking
* frame 사이에서 track ID를 유지하는 방식
* trajectory visualization을 이용한 tracking 오류 진단
* line-crossing event detection
* detection과 tracking output으로 간단한 object counter 구현

## Main Takeaway

이 프로젝트에서 가장 중요한 점은 pretrained detector의 output을 단순한 최종 결과가 아니라 **다음 단계의 입력 데이터**로 다룬 것이다.

Bounding box는 crop에 사용할 수 있고, 서로 비교하여 IoU를 계산할 수 있으며, video에서는 track ID와 연결할 수 있다.

Tracking 결과는 다시 trajectory와 event detection으로 확장할 수 있다.

결국 하나의 pretrained model에서 출발하여 다음과 같은 application pipeline까지 만들었다.

```text
Object Detection
      ↓
Prediction Analysis
      ↓
Embedding Inspection
      ↓
Multi-Object Tracking
      ↓
Trajectory Analysis
      ↓
Line-Crossing Counter
```

즉, 이 프로젝트는 YOLO inference 자체보다 **model output을 이용해 더 높은 수준의 computer-vision application을 만드는 과정**을 확인하는 데 초점을 두었다.

## Tools

```text
Python
PyTorch
Ultralytics YOLO
OpenCV
Matplotlib
Torchvision
Jupyter Notebook
```

## Notes

* Pretrained model을 사용했으며 detector 자체를 training하거나 fine-tuning하지 않았다.
* Detector가 인식할 수 있는 object class는 pretrained label space에 제한된다.
* Tracking quality는 detection quality와 frame-to-frame association 모두의 영향을 받는다.
* Line counter는 별도의 학습 model이 아니라 tracking 결과 위에 만든 간단한 application logic이다.

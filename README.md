# YOLO Object Detection, Tracking, and Line Counting

<p align="center">
  <img src="results/figures/hero.png" width="900">
</p>

<p align="center">
  <b>Ultralytics YOLO · Object Detection · Confidence Thresholds · Embeddings · Multi-Object Tracking · Trajectory Visualization · Line Crossing Counter</b>
</p>

<p align="center">
  <a href="README_KR.md">한국어 README</a>
</p>

## Overview

This project explores a pretrained YOLO model beyond a single object-detection call.

The goal is to understand what a modern detection pipeline actually returns, how confidence thresholds affect predictions, how detected regions can be inspected, what global image embeddings look like, and how frame-by-frame detections can be extended into tracking and a simple video analytics application.

The project gradually moves from **image detection** to **video tracking**, then to **trajectory visualization** and finally to an **object line-crossing counter**.

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

A pretrained YOLO model is first applied to a single image.

For each detection, the model provides information such as:

- bounding box coordinates
- predicted class ID
- class name
- confidence score

<p align="center">
  <img src="results/figures/detection_result.png" width="850">
</p>

Instead of treating detection as only a plotted image, the notebook inspects the actual prediction objects returned by the model.

This makes it possible to understand the detector as a structured prediction system rather than a black-box visualization tool.

## 2. Cropping Detected Objects

The predicted bounding boxes are used to crop each detected object from the original image.

<p align="center">
  <img src="results/figures/object_crops.png" width="900">
</p>

This step connects bounding-box coordinates directly to image indexing and demonstrates how detector outputs can be reused in downstream computer-vision pipelines.

## 3. Confidence Threshold Analysis

The confidence threshold determines which predictions are retained.

Predictions were compared across several threshold values:

```text
conf = 0.10
conf = 0.25
conf = 0.50
conf = 0.80
```

One experiment produced:

```text
conf=0.10 -> {'person': 5, 'bus': 1, 'stop sign': 1}
conf=0.25 -> {'person': 4, 'bus': 1}
conf=0.50 -> {'person': 4, 'bus': 1}
conf=0.80 -> {'person': 3, 'bus': 1}
```

<p align="center">
  <img src="results/figures/confidence_thresholds.png" width="950">
</p>

Lower thresholds increase recall but also admit weaker predictions. Higher thresholds remove low-confidence detections but may discard valid objects.

The experiment makes the precision-recall tradeoff visible at the prediction level.

## 4. IoU and Overlapping Boxes

Intersection over Union (IoU) was examined directly using pairwise bounding-box comparisons.

For two boxes \(A\) and \(B\),

\[
\mathrm{IoU}(A,B)
=
\frac{|A \cap B|}
{|A \cup B|}
\]

A pair of person detections in the experiment had a very high overlap:

```text
Maximum IoU: 0.9109
```

This was useful for understanding why duplicate or highly overlapping detections matter and how IoU is related to suppression and box matching.

The notebook also compares end-to-end prediction behavior with a traditional NMS-style setting.

## 5. Image Embeddings

The pretrained model was also used to extract a 256-dimensional image embedding.

```text
Embedding shape: torch.Size([256])
```

Cosine similarity was then measured between the original image and modified versions of it.

Example results:

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

The experiment showed that the extracted representation is highly invariant to several image transformations.

It also demonstrated an important limitation: a generic global embedding from the detector should not automatically be interpreted as a metric specifically optimized for image identity or semantic retrieval.

## 6. Multi-Object Tracking

Detection was then extended from a single image to video.

Instead of independently detecting objects in every frame, tracking assigns persistent IDs to objects across time.

<p align="center">
  <img src="results/figures/tracking_frame.png" width="900">
</p>

Tracking turns a sequence of detections into temporal object identities:

```text
Detection
    ↓
Frame-to-frame association
    ↓
Persistent track ID
```

Video result:

[View tracking result](results/videos/tracking_result.mp4)

## 7. Trajectory Visualization

The center position of each tracked object is stored over time and connected to form a trajectory.

This makes the temporal behavior of the tracker directly visible.

[View trajectory result](results/videos/tracking_trajectory.mp4)

Trajectory visualization is not only cosmetic. Sudden jumps or implausible paths can reveal tracking failures such as an **ID switch**, where the tracker incorrectly transfers one object's identity to another object.

## 8. Line-Crossing Object Counter

The final application adds a virtual line to the video and counts tracked objects when they cross it.

The basic pipeline is:

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

[View line counter result](results/videos/line_counter.mp4)

Because each object has a persistent track ID, the system can avoid simply counting the same object again in every frame.

This is the basic structure behind practical applications such as:

- traffic flow measurement
- pedestrian counting
- CCTV analytics
- entrance/exit monitoring
- simple smart-camera systems

## Key Concepts Explored

| Topic | What was examined |
|---|---|
| Object detection | Bounding boxes, classes, confidence |
| COCO classes | Understanding the detector's pretrained label space |
| Confidence threshold | Effect on retained predictions |
| IoU | Geometric overlap between bounding boxes |
| NMS | Suppression of overlapping predictions |
| Cropping | Reusing detector coordinates for downstream processing |
| Embeddings | Extracting and comparing learned image representations |
| Cosine similarity | Similarity between embedding vectors |
| Tracking | Maintaining object identity across frames |
| Track IDs | Associating detections through time |
| Trajectories | Visualizing motion history |
| ID switch | Diagnosing tracking association errors |
| Line crossing | Event detection from tracked motion |
| Object counting | A simple video analytics application |

## What I Learned

This project helped connect several concepts that are easy to study separately but much clearer when combined in one pipeline.

I learned how to:

- inspect YOLO prediction objects instead of only displaying them
- interpret bounding-box coordinates and confidence scores
- understand how confidence thresholds change model output
- calculate and inspect IoU between predictions
- connect overlapping boxes to NMS behavior
- crop detected objects using box coordinates
- extract learned embeddings from a pretrained vision model
- evaluate embedding similarity with cosine similarity
- distinguish generic feature similarity from task-specific metric learning
- run object tracking on video
- maintain and use track IDs across frames
- visualize trajectories as a diagnostic tool
- detect line-crossing events
- build a simple object counter from detection and tracking outputs

## Main Takeaway

A pretrained detector becomes much more useful once its outputs are treated as data.

Bounding boxes can be cropped, compared, tracked, accumulated over time, and converted into higher-level events.

The progression in this project was therefore:

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

The final result is a small but complete example of how a modern computer-vision model can be turned into an application pipeline rather than used only for one-shot inference.

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

- The project uses a pretrained model; no detector training or fine-tuning is performed.
- The detector can only predict classes included in its pretrained label space.
- Tracking quality depends on both detection quality and frame-to-frame association.
- The line counter is a simple application built on top of tracking rather than a separate learned model.

# Pedestrian Detection System using YOLO and IMX500

## Overview
This project implements a pedestrian detection system designed for automotive safety applications, such as detecting pedestrians near zebra crossings. The system leverages a lightweight YOLO-based object detection model and is optimized for embedded deployment on the Sony IMX500 AI vision sensor. It can also be tested in a desktop environment.

## Dataset
- **Custom dataset**: Collected manually and annotated using Label Studio.
- **Total labeled images**: 513
  - **Training**: 443 images (86%)
  - **Validation**: 70 images (14%)
- **Classes**: Pedestrian-related scenarios relevant to zebra crossings.
- **Annotation format**: YOLO bounding boxes.

## Image Preprocessing
- Images of varying resolutions are resized by YOLO during training.
- Training image size: 640 × 640.
- Aspect ratio preservation is done internally using padding (letterboxing).

## Model Training
- **Model**: YOLOv4 Nano (lightweight and embedded-friendly).
- **Pretrained on COCO dataset** and fine-tuned on the custom pedestrian dataset.
- **Training parameters**:
  - Epochs: 100
  - Batch size: 16
  - Image size: 640 × 640
  - Training script: `train.py`
  - Dataset configuration: `yolo_config.yaml`

## Evaluation Metrics
- **Precision (P)**: Measures the correctness of pedestrian detections.
- **Recall (R)**: Measures how many actual pedestrians were detected.
- **mAP@0.5**: Mean Average Precision at IoU threshold 0.5.

### Training Outputs:
- Loss curves (Box loss, classification loss, and DFL loss for training and validation).
- Precision-Recall curves.
- Confusion matrices.
- Training summaries and final trained weights (`best.pt`).

## Deployment

### IMX500 Deployment
- The trained model is exported to IMX format for deployment on the Sony IMX500 AI vision sensor.
- **Export process**: Convert `best.pt` to IMX format using `yolo_export.py`.
- Deployment scripts located in the `IMX500/` directory.

### Desktop Testing
- The system can be tested in a desktop environment using:
  - Python
  - OpenCV
  - Ultralytics YOLO framework

## Limitations
- Reduced performance under low-light conditions.
- Single-camera monocular setup.
- No temporal tracking across frames.

## Possible Improvements
- Expand the dataset to include more environmental diversity (e.g., night, foggy, shadow conditions).
- Add temporal tracking for video streams.
- Integrate with additional ADAS perception modules.

## Conclusion
This project demonstrates an end-to-end pedestrian detection pipeline, from dataset annotation to training and embedded deployment on IMX500 hardware. It meets all project requirements and serves as a solid foundation for automotive safety applications.

---
**Author**  
Niraliben Yash Jani – 22402416  
Satyajit Sushant Pardeshi - 22408966  
Master’s Student – Automotive / AI Systems

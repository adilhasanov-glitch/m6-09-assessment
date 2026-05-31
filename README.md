# Cat Detection Project (YOLO + ONNX + Docker)

## Overview

This project implements a cat detection system using a YOLO-based object detection model exported to ONNX format. The system is fully containerized using Docker to ensure reproducibility across different environments.

The model takes images as input and outputs bounding box predictions for detected cats.

---

## Project Structure

```
container/
  app/
    cli.py
    detector.py
  models/
    best.onnx
requirements.txt
Dockerfile
STUDENT.json

input/
output/
runs/
images/
labels/

m6-04-assessment.ipynb
data.yaml
README.md
.gitignore
```

---

## Model Details

- Model type: YOLO object detection
- Export format: ONNX
- Task: Detect cats in images
- Postprocessing: Confidence thresholding + Non-Max Suppression (NMS)

---

## Docker Setup

### Build the image

```
docker build -t cat-detector:final -f container/Dockerfile .
```

---

### Run info command

```
docker run --rm cat-detector:final info
```

This prints student information from STUDENT.json.

---

### Run prediction

Make sure the input folder contains images.

```
docker run --rm -v ${PWD}\input:/data/input -v ${PWD}\output:/data/output cat-detector:final predict
```

---

## Output

After running prediction, results are saved in:

```
output/predictions.csv
```

Format:

```
image,boxes
001.jpg,[[x1,y1,x2,y2,confidence,class], ...]
```

---

## Notes

- The model runs entirely inside Docker
- Input/output folders are mounted at runtime
- ONNX Runtime is used for inference
- Predictions are filtered using confidence threshold + NMS

---

## Reproducibility

To reproduce results:

1. Clone repository
2. Build Docker image
3. Add images to input/
4. Run prediction command
5. Check output/predictions.csv

## Image for leaderboard

```
docker pull adilhasanov/cat-detector:final
Image: adilhasanov/cat-detector:final
Student: Adil Hasanov
```

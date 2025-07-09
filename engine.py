# Imports 
import os
import cv2
import json
import torch
import matplotlib.pyplot as plt
from collections import defaultdict
from pathlib import Path

# Load YOLOv5 modrl
model = torch.hub.load('ultralytics/yolov5', 'yolov5x', pretrained=True)
model.eval()

# Directry Setup
INPUT_VIDEO = 'input/input_video.mp4'
OUTPUT_DIR = 'output'
FRAME_DIR = f'{OUTPUT_DIR}/frames'
JSON_DIR = f'{OUTPUT_DIR}/json_frames'
VIDEO_OUT = f'{OUTPUT_DIR}/output_video.mp4'
BAR_CHART_PATH = f'{OUTPUT_DIR}/object_frequency.png'

Path(FRAME_DIR).mkdir(parents=True, exist_ok=True)
Path(JSON_DIR).mkdir(parents=True, exist_ok=True)

# Video Processsing
cap = cv2.VideoCapture(INPUT_VIDEO)
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
interval = 5

frame_idx = 0
json_outputs = {}
object_counts = defaultdict(int)
diversity_per_frame = {}

annotated_frames = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    if frame_idx % interval == 0:
        results = model(frame)
        detections = results.xyxy[0]  # (x1, y1, x2, y2, conf, cls)

        frame_json = []
        unique_classes = set()

        # Save info to JSON
        for *box, conf, cls in detections.tolist():
            label = model.names[int(cls)]
            unique_classes.add(label)
            object_counts[label] += 1
            frame_json.append({
                'label': label,
                'bbox': [round(x, 2) for x in box],
                'confidence': round(conf, 3)
            })

            # Frame Annotations
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        # Save tp json, annotated frames etc.
        json_path = f"{JSON_DIR}/frame_{frame_idx}.json"
        with open(json_path, 'w') as f:
            json.dump(frame_json, f, indent=2)

        annotated_path = f"{FRAME_DIR}/frame_{frame_idx}.jpg"
        cv2.imwrite(annotated_path, frame)

        json_outputs[frame_idx] = frame_json
        diversity_per_frame[frame_idx] = len(unique_classes)
        annotated_frames.append(frame)

    frame_idx += 1

cap.release()

# Identify frame with most diverse classes
max_diverse_frame = max(diversity_per_frame, key=diversity_per_frame.get)
print(f"Frame with max class diversity: {max_diverse_frame}")

# Plot bar chart
plt.figure(figsize=(10,6))
plt.bar(object_counts.keys(), object_counts.values(), color='skyblue')
plt.xlabel("Object Classes")
plt.ylabel("Frequency")
plt.title("Total Object Counts per Class")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(BAR_CHART_PATH)
plt.close()

print("Detection Summary Engine Complete....")
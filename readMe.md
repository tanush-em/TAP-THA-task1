# Detection Summary Engine 

## 📌 Task Description

This project is built for an object detection pipeline as part of a computer vision task:

- Use a pretrained object detection model (YOLOv5x).
- Process every 5th frame of a short `.mp4` video (approx. 15–20 seconds).
- For each processed frame:
  - Detect and label objects.
  - Save JSON with class names, bounding boxes, and confidence scores.
- Keep track of:
  - Total object counts per class.
  - Frame with the maximum class diversity.
- Generate a bar chart showing object frequency.
- Optionally save annotated frames and a compiled output video.

---

## 📂 Project Structure
```
detection_summary_engine/
├── detect_summary.py          # Main detection script
├── requirements.txt           # Dependencies
├── README.md                  # Project documentation
├── input/
│   └── input_video.mp4        # Your input video (15-20 sec)
├── output/
│   ├── frames/                # Annotated frames (optional)
│   ├── json_frames/           # JSON outputs per frame
│   ├── object_frequency.png   # Bar chart of object counts
│   └── output_video.mp4       # (Optional) compiled video
```

## 🔁 Project Workflow

1. **Video Input**  
   The input video is loaded from the `input/` directory.

2. **Frame Sampling**  
   Every 5th frame is selected for object detection.

3. **Object Detection (YOLOv5x)**  
   The model detects objects and returns:
   - Bounding boxes
   - Confidence scores
   - Class labels

4. **Data Storage**  
   For each processed frame:
   - A `.json` file is created containing all detection details.
   - The frame is annotated and optionally saved to `output/frames/`.

5. **Summary Generation**  
   - Total object count per class is tracked.
   - The frame with the most unique classes is identified.
   - A bar chart is generated showing object frequency.

6. **Output Compilation (Optional)**  
   All annotated frames can be compiled into a single output video.

---

## 🚀 How to Run

### 1. Install Requirements

Make sure Python 3.7+ is installed. Then run:

```bash
pip install -r requirements.txt
```

Required packages include:

* `torch`
* `opencv-python`
* `matplotlib`
* `ultralytics` model loader

### 2. Add Your Video

Place your `.mp4` video in the `input/` directory and rename it to:

```
input/input_video.mp4
```

> Recommended: 15–20 seconds video for manageable output.

### 3. Run the Script

```bash
python engine.py
```

The script will process every 5th frame and generate all required outputs.

---

## 📦 Output Details

* `output/json_frames/frame_XX.json`: Detected objects per frame with class, bounding box, and confidence.
* `output/frames/frame_XX.jpg`: Optional annotated frames.
* `output/object_frequency.png`: Bar chart of object class counts.
* `output/output_video.mp4`: (Optional) compiled video with annotations.
* **Console Output**: Frame index with the highest number of distinct object classes.

Example JSON:

```json
[
  {
    "label": "person",
    "bbox": [130.5, 45.2, 320.7, 410.9],
    "confidence": 0.921
  },
  {
    "label": "dog",
    "bbox": [500.1, 300.4, 650.8, 470.2],
    "confidence": 0.873
  }
]
```

---

## 🧠 Notes & Suggestions

* Model used: `yolov5x` (highest accuracy variant from YOLOv5).
* If processing a long video, consider increasing the frame interval to reduce output size.
* The current setup is tuned for short clips (~15–20 seconds) with decent object presence.

---

## ✅ Completion Message

At the end of the script, the following is printed:

```
Frame with max class diversity: <frame_number>
Detection Summary Engine Complete....
```

---

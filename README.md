# Alertify 🚨

**Alertify** is a lightweight project for detecting accidents (or other custom objects) using YOLO-based models. It includes training and inference scripts, example datasets, pretrained weights, and a notification module to send alerts when an event is detected.

---

## 🔍 Features

- Train object detection models on a custom dataset (`data/`)
- Run inference/detection on images, videos, or live streams
- Send notifications/alerts via `notifications.py` when an event is detected
- Pretrained weights included for quick testing (`yolov3u.pt`, `yolov8n.pt`)

---

## 📁 Repository structure

- `train.py` - training script for model training
- `main.py`, `main1.py`, `main2.py` - inference / demo scripts (check each for options)
- `notifications.py` - module to send alerts (configure credentials / endpoints)
- `dataset.yaml` - dataset configuration (used by training scripts)
- `data/` - dataset folder with `train/`, `test/`, `valid/` subfolders
- `runs/` - output directory for runs (trained weights, detect outputs)
- `yolov3u.pt`, `yolov8n.pt` - example/pretrained model weights

---

## 🛠️ Requirements

Create a Python virtual environment and install dependencies. If you don't have a `requirements.txt`, create one from your environment after installing packages.

Example:

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
# or install common deps
pip install torch torchvision opencv-python pyyaml pandas
# If using Ultralytics YOLOv8:
pip install ultralytics
```

Tip: To generate `requirements.txt` after you install packages locally:

```bash
pip freeze > requirements.txt
```

---

## 🚀 Usage

### Training

Check `train.py` for available CLI arguments. Typical usage:

```bash
python train.py --data dataset.yaml --epochs 50 --img 640
```

- Ensure `dataset.yaml` points to the correct `data/` folders and class names.
- Trained weights and logs will be saved under `runs/`.

### Inference / Detection

Run any of the `main*.py` scripts (they may have slightly different options).

Example:

```bash
python main.py --source path/to/image_or_video --weights yolov8n.pt --conf 0.25
```

- `--source` accepts image, video, or directory paths. Use `0` for default webcam.
- Output (predictions / annotated images) will be saved to `runs/detect/`.

### Notifications

`notifications.py` is used to send alerts when detections meet your criteria.

- Edit or configure credentials/endpoint (e.g., API keys, email or SMS settings) before use.
- Use environment variables or a config file to keep secrets out of source control.

---

## 📦 Dataset

The `data/` folder follows a typical YOLO layout:

```
data/
  train/
    images/
    labels/
  test/
    images/
    labels/
  valid/
    images/
    labels/
```

Label files should follow YOLO format: `<class> <x_center> <y_center> <width> <height>` (normalized).


---

## 🤝 Contributing

Contributions are welcome! Consider:

- Opening an issue for bugs or feature requests
- Submitting a pull request with clear description and tests if applicable
- Adding or updating documentation

---







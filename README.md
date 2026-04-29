# CheatSpot

CheatSpot is a Flask-based computer vision web app for detecting cheating behavior during exams using YOLO object detection, a custom CNN+ViT hybrid detector, and GradCAM-based explainability.

It supports image and video uploads, shows annotated results, displays live alerts, and provides an analytics dashboard for model evaluation.

## Features

- Upload image or video files for cheating analysis.
- Detect cheating using a YOLOv8s model.
- Generate annotated output images and videos.
- View recent detection alerts in the dashboard.
- Adjust confidence threshold from the dashboard.
- Generate GradCAM explanations for detected frames.
- Compare multiple models in the analytics page.
- View model performance metrics such as mAP, precision, recall, and F1 score.

## Project Structure

```bash
CheatSpot_Individual-Dissertation-main/
├── app.py
├── launcher.py
├── xai.py
├── requirements.txt
├── README.md
├── templates/
├── static/
├── uploads/
├── output/
├── yolov8s/
├── yolov8m/
├── yolov10s/
├── yolov10m/
├── yolov11s/
├── yolov11m/
├── cnn_vit_hybrid/
├── vit_standalone/
├── custom_detector/
├── one_class_svm/
├── train/
├── valid/
└── test/
```

## Requirements

- Python 3.10+
- Flask
- OpenCV
- Ultralytics
- NumPy

Install dependencies with:

```bash
pip install -r requirements.txt
```

If needed, install additional packages manually:

```bash
pip install flask opencv-python ultralytics numpy
```

## Setup

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd CheatSpot_Individual-Dissertation-main
```

### 2. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

On macOS/Linux, use:

```bash
source .venv/bin/activate
```

On Windows, use:

```bash
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Make sure model weights exist

The app expects trained model weights such as:

- `yolov8s/weights/best.pt`

Make sure the correct weight file is available before running the app.

## Running the App

Start the Flask server with:

```bash
python app.py
```

Or, if you use the launcher:

```bash
python launcher.py
```

Then open the app in your browser:

```bash
http://127.0.0.1:5001
```

## How to Use

### Dashboard

The dashboard shows:

- live video feed,
- current confidence threshold,
- session alerts,
- quick actions for upload and XAI explanation.

### Upload Media

1. Go to the upload page.
2. Select an image or video file.
3. Click **Analyse Cheating**.
4. Wait for the model to process the file.
5. View the annotated result page.

### GradCAM Explainability

For image results, click **Explain with GradCAM** to generate an explanation heatmap showing where the model focused.

### Analytics

The analytics page provides:

- model comparison charts,
- training loss curves,
- radar charts,
- head-to-head metrics,
- custom detector and ViT performance summaries.

## API Endpoints

The app exposes a few Flask endpoints for UI updates and alerts:

- `GET /api/alerts`
- `GET /api/get_conf`
- `POST /api/set_conf`
- `GET /explain/<filename>`
- `GET /explain_dashboard`

## Configuration

Important settings are defined in `app.py`:

- `UPLOAD_FOLDER`
- `OUTPUT_FOLDER`
- `CONF_THRESHOLD`

You can adjust the confidence threshold from the dashboard UI or directly in code.

## Notes

- Uploaded files are saved in `uploads/`.
- Generated results are saved in `output/`.
- The app uses a YOLO model for detection and a GradCAM module for explanations.
- Make sure the `static/cheatspot_result.mp4` file exists if you want the dashboard live feed and dashboard explanation feature to work correctly.

## Troubleshooting

- If the model does not load, confirm the `.pt` file path is correct.
- If video output fails, check that OpenCV and video codecs are installed properly.
- If GradCAM generation fails, verify that `xai.py` is working and the input file exists.
- If the dashboard shows no alerts, make sure a detection has already been generated.

## Maintainer

Created by Nithish.

## License

Add your preferred license here.
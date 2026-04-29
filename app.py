import base64
import mimetypes
import os
import shutil
import threading
from datetime import datetime

import cv2
import numpy as np
from flask import Flask, jsonify, redirect, render_template, request, send_file, url_for
from ultralytics import YOLO

from xai import generate_gradcam


app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "output"
MODEL_PATH = "yolov8s/weights/best.pt"
DASHBOARD_VIDEO_PATH = os.path.join("static", "cheatspot_result.mp4")

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["OUTPUT_FOLDER"] = OUTPUT_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

model = YOLO(MODEL_PATH)
model.names[0] = "students_cheating"

ALERTS = []
CONF_THRESHOLD = 0.60


def is_video_file(filename):
    filetype, _ = mimetypes.guess_type(filename)
    return filetype and filetype.startswith("video")


def encode_frame_to_base64(frame):
    _, buffer = cv2.imencode(".jpg", frame)
    return base64.b64encode(buffer).decode("utf-8")


def append_alert(frame, conf):
    ALERTS.append(
        {
            "time": datetime.now().strftime("%H:%M:%S"),
            "message": f"🚨 Cheating detected {conf:.1%}",
            "conf": int(conf * 100),
            "frame": encode_frame_to_base64(frame),
        }
    )

    if len(ALERTS) > 10:
        ALERTS.pop(0)


def clear_output_folder():
    if os.path.exists(app.config["OUTPUT_FOLDER"]):
        shutil.rmtree(app.config["OUTPUT_FOLDER"])
    os.makedirs(app.config["OUTPUT_FOLDER"], exist_ok=True)


def predict_frame(frame):
    return model.predict(
        frame,
        conf=CONF_THRESHOLD,
        classes=[0],
        verbose=False,
    )


def draw_custom_boxes(results, img):
    annotated_img = img.copy()

    for r in results:
        boxes = r.boxes
        if boxes is None:
            continue

        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0].cpu().numpy())
            label = f"students_cheating {conf:.1%}"

            color = (0, 0, 255) if conf >= 0.60 else (0, 140, 255)
            thickness = max(3, int(conf * 6))

            cv2.rectangle(
                annotated_img,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                color,
                thickness,
            )

            label_size = cv2.getTextSize(
                label,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                2,
            )[0]

            cv2.rectangle(
                annotated_img,
                (int(x1), int(y1) - label_size[1] - 10),
                (int(x1) + label_size[0], int(y1)),
                color,
                -1,
            )

            cv2.putText(
                annotated_img,
                label,
                (int(x1), int(y1) - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

            if conf >= CONF_THRESHOLD:
                append_alert(annotated_img, conf)

    return annotated_img


def preprocess_dashboard_video():
    if not os.path.exists(DASHBOARD_VIDEO_PATH):
        print("⚠️ Dashboard video not found, skipping pre-processing")
        return

    print("🎬 Pre-processing dashboard video for alerts...")
    cap = cv2.VideoCapture(DASHBOARD_VIDEO_PATH)
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % 15 == 0:
            results = predict_frame(frame)
            draw_custom_boxes(results, frame)

        frame_count += 1

    cap.release()
    print(f"✅ Dashboard pre-processing done! {len(ALERTS)} alerts found.")


threading.Thread(target=preprocess_dashboard_video, daemon=True).start()


@app.route("/")
def dashboard():
    return render_template("dashboard.html")


@app.route("/api/alerts")
def api_alerts():
    return jsonify(ALERTS[-10:])


@app.route("/api/set_conf", methods=["POST"])
def set_conf():
    global CONF_THRESHOLD

    data = request.get_json()
    val = float(data.get("conf", 0.60))
    CONF_THRESHOLD = max(0.25, min(0.75, val))

    print(f"🎚️ Threshold updated → {CONF_THRESHOLD:.0%}")
    return jsonify({"conf": CONF_THRESHOLD})


@app.route("/api/get_conf")
def get_conf():
    return jsonify({"conf": CONF_THRESHOLD})


@app.route("/upload", methods=["GET"])
def upload_form():
    return render_template("upload.html")


@app.route("/upload", methods=["POST"])
def upload_file():
    file = request.files.get("file")
    if not file or file.filename == "":
        return "No file uploaded", 400

    filename = file.filename
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    clear_output_folder()

    if is_video_file(filepath):
        cap = cv2.VideoCapture(filepath)
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fourcc = cv2.VideoWriter_fourcc(*"avc1")
        output_path = os.path.join(app.config["OUTPUT_FOLDER"], "cheatspot_result.mp4")
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = predict_frame(frame)
            annotated = draw_custom_boxes(results, frame)
            out.write(annotated)

            frame_count += 1
            if frame_count % 30 == 0:
                print(f"Processed {frame_count} frames...")

        cap.release()
        out.release()
        display_filename = "cheatspot_result.mp4"

    else:
        img = cv2.imread(filepath)
        results = predict_frame(img)
        annotated_img = draw_custom_boxes(results, img)
        display_filename = filename
        output_path = os.path.join(app.config["OUTPUT_FOLDER"], display_filename)
        cv2.imwrite(output_path, annotated_img)

    return redirect(url_for("show_result", filename=display_filename))


@app.route("/result/<filename>")
def show_result(filename):
    filetype = "video" if filename.endswith(".mp4") else "image"
    return render_template("result.html", filename=filename, filetype=filetype)


@app.route("/output/<filename>")
def output_file(filename):
    output_path = os.path.join(app.config["OUTPUT_FOLDER"], filename)
    return send_file(output_path)


@app.route("/analytics")
def analytics():
    return render_template("analytics.html")


@app.route("/explain/<filename>")
def explain(filename):
    try:
        input_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        output_filename = f"gradcam_{filename}"
        output_path = os.path.join(app.config["OUTPUT_FOLDER"], output_filename)

        generate_gradcam(input_path, output_path)
        return jsonify({"gradcam_url": f"/output/{output_filename}"})

    except Exception as e:
        print(f"❌ GradCAM error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/explain_dashboard")
def explain_dashboard():
    try:
        cap = cv2.VideoCapture(DASHBOARD_VIDEO_PATH)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames // 2)

        ret, frame = cap.read()
        cap.release()

        if not ret:
            return jsonify({"error": "Could not read video frame"}), 500

        frame_path = os.path.join(app.config["UPLOAD_FOLDER"], "dashboard_frame.jpg")
        cv2.imwrite(frame_path, frame)

        output_filename = "gradcam_dashboard.jpg"
        output_path = os.path.join(app.config["OUTPUT_FOLDER"], output_filename)
        generate_gradcam(frame_path, output_path)

        return jsonify({"gradcam_url": f"/output/{output_filename}"})

    except Exception as e:
        print(f"❌ Dashboard GradCAM error: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5001)
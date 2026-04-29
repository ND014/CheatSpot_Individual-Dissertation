from flask import Flask, render_template, request, redirect, url_for, send_file, jsonify
import os, cv2, mimetypes, shutil, threading, base64
import numpy as np
from ultralytics import YOLO
from datetime import datetime
from xai import generate_gradcam

# ========================
# CONFIG
# ========================
app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'output'

app.config.update({
    'UPLOAD_FOLDER': UPLOAD_FOLDER,
    'OUTPUT_FOLDER': OUTPUT_FOLDER
})

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

model = YOLO('yolov8s/weights/best.pt')
model.names[0] = 'students_cheating'

CONF_THRESHOLD = 0.60
ALERTS = []


# ========================
# UTILITIES
# ========================
def is_video_file(filename):
    filetype, _ = mimetypes.guess_type(filename)
    return filetype and filetype.startswith("video")


def encode_frame(frame):
    _, buffer = cv2.imencode('.jpg', frame)
    return base64.b64encode(buffer).decode('utf-8')


def add_alert(frame, conf):
    ALERTS.append({
        'time': datetime.now().strftime("%H:%M:%S"),
        'message': f"🚨 Cheating detected {conf:.1%}",
        'conf': int(conf * 100),
        'frame': encode_frame(frame)
    })
    if len(ALERTS) > 10:
        ALERTS.pop(0)


def clear_output_folder():
    if os.path.exists(OUTPUT_FOLDER):
        shutil.rmtree(OUTPUT_FOLDER)
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# ========================
# CORE LOGIC
# ========================
def draw_boxes(results, img):
    annotated = img.copy()

    for r in results:
        if not r.boxes:
            continue

        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            conf = float(box.conf[0].cpu().numpy())

            label = f"students_cheating {conf:.1%}"
            color = (0, 0, 255) if conf >= CONF_THRESHOLD else (0, 140, 255)
            thickness = max(3, int(conf * 6))

            # Draw box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)

            # Label background
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(annotated, (x1, y1 - h - 10), (x1 + w, y1), color, -1)

            # Label text
            cv2.putText(annotated, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Alert trigger
            if conf >= CONF_THRESHOLD:
                add_alert(annotated, conf)

    return annotated


def process_video(input_path, output_path):
    cap = cv2.VideoCapture(input_path)

    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*'avc1'),
        fps,
        (width, height)
    )

    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(frame, conf=CONF_THRESHOLD, classes=[0], verbose=False)
        annotated = draw_boxes(results, frame)

        out.write(annotated)

        frame_count += 1
        if frame_count % 30 == 0:
            print(f"Processed {frame_count} frames...")

    cap.release()
    out.release()


def process_image(input_path, output_path):
    img = cv2.imread(input_path)
    results = model.predict(img, conf=CONF_THRESHOLD, classes=[0], verbose=False)
    annotated = draw_boxes(results, img)
    cv2.imwrite(output_path, annotated)


# ========================
# DASHBOARD PREPROCESS
# ========================
def preprocess_dashboard_video():
    video_path = os.path.join('static', 'cheatspot_result.mp4')

    if not os.path.exists(video_path):
        print("⚠️ Dashboard video not found")
        return

    print("🎬 Preprocessing dashboard video...")
    cap = cv2.VideoCapture(video_path)

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % 15 == 0:
            results = model.predict(frame, conf=CONF_THRESHOLD, classes=[0], verbose=False)
            draw_boxes(results, frame)

        frame_count += 1

    cap.release()
    print(f"✅ Done! {len(ALERTS)} alerts generated")


threading.Thread(target=preprocess_dashboard_video, daemon=True).start()


# ========================
# ROUTES
# ========================
@app.route('/')
def dashboard():
    return render_template('dashboard.html')


@app.route('/upload', methods=['GET'])
def upload_form():
    return render_template('upload.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files.get('file')

    if not file or file.filename == '':
        return 'No file uploaded', 400

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    clear_output_folder()

    if is_video_file(filepath):
        output_name = 'cheatspot_result.mp4'
        process_video(filepath, os.path.join(OUTPUT_FOLDER, output_name))
    else:
        output_name = file.filename
        process_image(filepath, os.path.join(OUTPUT_FOLDER, output_name))

    return redirect(url_for('show_result', filename=output_name))


@app.route('/result/<filename>')
def show_result(filename):
    filetype = 'video' if filename.endswith('.mp4') else 'image'
    return render_template('result.html', filename=filename, filetype=filetype)


@app.route('/output/<filename>')
def output_file(filename):
    return send_file(os.path.join(OUTPUT_FOLDER, filename))


# ========================
# API
# ========================
@app.route('/api/alerts')
def api_alerts():
    return jsonify(ALERTS[-10:])


@app.route('/api/set_conf', methods=['POST'])
def set_conf():
    global CONF_THRESHOLD

    val = float(request.json.get('conf', 0.60))
    CONF_THRESHOLD = max(0.25, min(0.75, val))

    print(f"🎚️ Threshold → {CONF_THRESHOLD:.0%}")
    return jsonify({'conf': CONF_THRESHOLD})


@app.route('/api/get_conf')
def get_conf():
    return jsonify({'conf': CONF_THRESHOLD})


# ========================
# XAI / GRADCAM
# ========================
@app.route('/explain/<filename>')
def explain(filename):
    try:
        input_path = os.path.join(UPLOAD_FOLDER, filename)
        output_name = f'gradcam_{filename}'
        output_path = os.path.join(OUTPUT_FOLDER, output_name)

        generate_gradcam(input_path, output_path)
        return jsonify({'gradcam_url': f'/output/{output_name}'})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/explain_dashboard')
def explain_dashboard():
    try:
        video_path = os.path.join('static', 'cheatspot_result.mp4')
        cap = cv2.VideoCapture(video_path)

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.set(cv2.CAP_PROP_POS_FRAMES, total // 2)

        ret, frame = cap.read()
        cap.release()

        if not ret:
            return jsonify({'error': 'Frame read failed'}), 500

        frame_path = os.path.join(UPLOAD_FOLDER, 'dashboard_frame.jpg')
        cv2.imwrite(frame_path, frame)

        output_name = 'gradcam_dashboard.jpg'
        output_path = os.path.join(OUTPUT_FOLDER, output_name)

        generate_gradcam(frame_path, output_path)

        return jsonify({'gradcam_url': f'/output/{output_name}'})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ========================
# RUN
# ========================
if __name__ == '__main__':
    app.run(debug=True, port=5001)
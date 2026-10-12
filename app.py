from flask import Flask, render_template, Response, request, redirect, session, url_for
import cv2
import numpy as np
from ultralytics import YOLO
import sqlite3
from werkzeug.security import generate_password_hash, check_password_hash
import os

app = Flask(__name__)
app.secret_key = "YourSecretKey123"

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Store uploaded videos
UPLOADED_VIDEOS = []     # list of video paths


# -------------------- DATABASE SETUP --------------------
def init_db():
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE,
            password TEXT
        )
    """)
    conn.commit()
    conn.close()

init_db()


# -------------------- YOLO CONFIG --------------------
MODEL_PATH = "yolov8n.pt"
CONF_THRESHOLD = 0.4
PIXEL_TO_METER = 0.30
VEHICLE_CLASSES = {"car", "bus", "truck", "bicycle", "motorcycle"}

SLOW_SPEED = 30
MEDIUM_SPEED = 60
MAX_SPEED = 150
FONT = cv2.FONT_HERSHEY_SIMPLEX

model = YOLO(MODEL_PATH)


# -------------------- VIDEO PROCESSING --------------------
def generate_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    FRAME_TIME = 1 / fps if fps > 0 else 0.03

    prev_positions = {}

    while True:
        success, frame = cap.read()
        if not success:
            break

        results = model.track(frame, persist=True, conf=CONF_THRESHOLD)

        if results[0].boxes.id is not None:
            for box, cls_id, tid in zip(
                results[0].boxes.xyxy,
                results[0].boxes.cls,
                results[0].boxes.id
            ):
                cname = model.names[int(cls_id)]
                if cname not in VEHICLE_CLASSES:
                    continue

                x1, y1, x2, y2 = map(int, box)
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                tid = int(tid)

                speed = 0
                if tid in prev_positions:
                    dist_pixels = np.linalg.norm(
                        np.array((cx, cy)) - np.array(prev_positions[tid])
                    )
                    dist_meters = dist_pixels * PIXEL_TO_METER
                    speed = (dist_meters / FRAME_TIME) * 3.6

                speed = min(max(speed, 0), MAX_SPEED)
                prev_positions[tid] = (cx, cy)

                color = (0, 255, 0) if speed <= SLOW_SPEED else \
                        (0, 165, 255) if speed <= MEDIUM_SPEED else (0, 0, 255)

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{cname} {speed:.1f} km/h",
                            (x1, y1 - 10), FONT, 0.6, color, 2)

        _, buffer = cv2.imencode(".jpg", frame)
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")


# -------------------- ROUTES --------------------
@app.route("/")
def home():
    return redirect("/login")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        conn = sqlite3.connect("users.db")
        c = conn.cursor()
        c.execute("SELECT * FROM users WHERE username=?", (username,))
        user = c.fetchone()
        conn.close()

        if user and check_password_hash(user[2], password):
            session["user"] = username
            return redirect("/upload")

        return render_template("login.html", error="Invalid credentials")

    return render_template("login.html")


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form["username"]
        password = generate_password_hash(request.form["password"])

        try:
            conn = sqlite3.connect("users.db")
            c = conn.cursor()
            c.execute("INSERT INTO users VALUES (NULL, ?, ?)", (username, password))
            conn.commit()
            conn.close()
            return redirect("/login")
        except:
            return render_template("register.html", error="User exists")

    return render_template("register.html")


# -------- UPLOAD 3 VIDEOS --------
@app.route("/upload", methods=["GET", "POST"])
def upload():
    global UPLOADED_VIDEOS

    if "user" not in session:
        return redirect("/login")

    if request.method == "POST":
        files = request.files.getlist("video")
        UPLOADED_VIDEOS = []

        if len(files) != 3:
            return render_template("upload.html", error="Upload exactly 3 videos")

        for file in files:
            if file.filename == "":
                continue
            unique_name = f"{len(UPLOADED_VIDEOS)}_{file.filename}"
            path = os.path.join(UPLOAD_FOLDER, unique_name)
            file.save(path)
            UPLOADED_VIDEOS.append(path)

        return redirect("/dashboard")
    

    return render_template("upload.html")


# -------- DASHBOARD --------
@app.route("/dashboard")
def dashboard():
    if "user" not in session:
        return redirect("/login")

    return render_template("result.html", videos=UPLOADED_VIDEOS)


@app.route("/video_feed/<int:vid_id>")
def video_feed(vid_id):
    if "user" not in session:
        return redirect("/login")

    if vid_id >= len(UPLOADED_VIDEOS):
        return "Video not found"

    return Response(
        generate_frames(UPLOADED_VIDEOS[vid_id]),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/logout")
def logout():
    session.clear()
    return redirect("/login")


if __name__ == "__main__":
    app.run(debug=True, threaded=True)


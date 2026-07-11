# Face Recognition Attendance System

A complete, working attendance system that uses your webcam to register faces,
train a recognition model, and automatically mark attendance in CSV files.

Built with **OpenCV** (Haar Cascade for face detection + LBPH for face
recognition) — no dlib/CMake build headaches, works on Windows/Mac/Linux with
a plain `pip install`.

## How it works

1. **Register** — capture ~60 photos of each person's face from your webcam.
2. **Train** — build a recognition model (LBPH) from all registered faces.
3. **Recognize** — open the webcam, detect + identify faces in real time, and
   log attendance (ID, name, date, time) to a daily CSV file.
4. **View** — print or export attendance for any date.

A simple desktop GUI (Tkinter) ties all four steps into one window if you'd
rather not use the command line.

## Project structure

```
face_attendance_system/
├── app/
│   ├── config.py                # all settings (camera index, thresholds, paths)
│   ├── attendance_store.py      # student registry + attendance CSV read/write
│   ├── capture_faces.py         # Step 1: register + capture face samples
│   ├── train_model.py           # Step 2: train the LBPH recognizer
│   ├── recognize_attendance.py  # Step 3: live recognition + attendance marking
│   ├── view_attendance.py       # Step 4: view/export attendance records
│   └── gui.py                   # optional all-in-one Tkinter GUI
├── dataset/                     # captured face images (created automatically)
├── trainer/                     # trained model + label map (created automatically)
├── attendance/                  # daily attendance CSVs (created automatically)
├── students.csv                 # registry of enrolled people (created automatically)
├── requirements.txt
└── README.md
```

## Setup

Requires **Python 3.8+** and a webcam.

```bash
cd face_attendance_system
pip install -r requirements.txt
```

> `opencv-contrib-python` includes the `cv2.face` module (LBPH recognizer)
> that plain `opencv-python` does not — make sure you install the contrib
> package, not `opencv-python`.

## Usage

### 1. Register people (repeat for each person)

```bash
python -m app.capture_faces --id 101 --name "Jane Doe"
```

This opens your webcam, detects your face, and saves ~60 cropped face images
to `dataset/101_Jane_Doe/`. Press `q` to stop early if needed. Register
everyone you want the system to recognize before moving on.

### 2. Train the model

```bash
python -m app.train_model
```

Trains an LBPH recognizer on everything in `dataset/` and saves
`trainer/trainer.yml` + `trainer/labels.json`. Re-run this any time you add
or remove people.

### 3. Run live attendance

```bash
python -m app.recognize_attendance
```

Opens the webcam, draws a box around each detected face with the predicted
name, and marks attendance (once per person per session) into
`attendance/Attendance_YYYY-MM-DD.csv`. Faces the model doesn't recognize
confidently are labeled "Unknown" and are not marked. Press `q` to quit.

### 4. View attendance

```bash
python -m app.view_attendance                     # today
python -m app.view_attendance --date 2026-07-11    # specific date
python -m app.view_attendance --all                # list all dates on file
```

### Optional: one-window GUI

```bash
python -m app.gui
```

Lets you register, train, start attendance, and view today's log from a
single Tkinter window instead of the command line.

## Configuration

Open `app/config.py` to tune:

- `CAMERA_INDEX` — which webcam to use if you have more than one.
- `SAMPLES_PER_PERSON` — how many face images to capture per person (default 60).
- `CONFIDENCE_THRESHOLD` — LBPH match strictness. **Lower = stricter.** LBPH
  confidence is a distance, so smaller values mean a closer match. Default is
  `70`; lower it (e.g. `50`) if you're getting false positives, raise it if
  real people aren't being recognized.
- `MIN_SECONDS_BETWEEN_DUPLICATE_MARKS` — cooldown to avoid writing duplicate
  rows while someone lingers in front of the camera.

## Tips for good accuracy

- Capture registration photos in good, even lighting, facing the camera.
- Capture a few samples with slight head turns/expressions for robustness.
- Retrain (`train_model`) after adding or removing anyone.
- If lighting during recognition is very different from registration, accuracy
  drops — LBPH is lightweight but sensitive to lighting changes. For higher
  accuracy in production, consider swapping the recognizer for a deep
  embedding model (e.g. `face_recognition`/dlib or `insightface`), which this
  project's structure makes easy since detection/training/recognition are
  cleanly separated in `app/`.

## Extending this project

- Swap the CSV backend in `attendance_store.py` for SQLite/MySQL for a
  multi-user or networked deployment.
- Add an email/Slack notification when someone is marked present.
- Add a `--export-excel` option to `view_attendance.py` using `openpyxl`.
- Deploy `recognize_attendance.py`'s loop inside a Flask app with a video
  stream endpoint for a browser-based UI.

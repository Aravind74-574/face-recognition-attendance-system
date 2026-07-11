"""
Step 3: Run real-time face recognition from the webcam and mark attendance.

Usage:
    python -m app.recognize_attendance

Press 'q' to quit.
"""
import json
import os
import sys

import cv2

from app import config
from app.attendance_store import mark_attendance, load_students


def load_label_map():
    if not os.path.exists(config.LABELS_FILE):
        print("[!] No trained model found. Run app.train_model first.")
        sys.exit(1)
    with open(config.LABELS_FILE, "r") as f:
        return {int(k): v for k, v in json.load(f).items()}


def parse_id_name(folder_name: str):
    """'101_Jane_Doe' -> ('101', 'Jane Doe')"""
    student_id, _, rest = folder_name.partition("_")
    return student_id, rest.replace("_", " ")


def run():
    label_map = load_label_map()
    students = load_students()

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(config.TRAINER_FILE)

    detector = cv2.CascadeClassifier(config.CASCADE_PATH)
    cam = cv2.VideoCapture(config.CAMERA_INDEX)
    if not cam.isOpened():
        print("[!] Could not open webcam. Check CAMERA_INDEX in app/config.py.")
        sys.exit(1)

    print("[*] Recognition started. Press 'q' to quit.")
    marked_this_session = set()

    while True:
        ok, frame = cam.read()
        if not ok:
            print("[!] Failed to read frame from camera.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(80, 80))

        for (x, y, w, h) in faces:
            face_crop = cv2.resize(gray[y:y + h, x:x + w], config.FACE_SIZE)
            label, confidence = recognizer.predict(face_crop)

            if confidence <= config.CONFIDENCE_THRESHOLD and label in label_map:
                student_id, name = parse_id_name(label_map[label])
                name = students.get(student_id, name)
                display = f"{name} ({confidence:.0f})"
                color = (0, 200, 0)

                if student_id not in marked_this_session:
                    written = mark_attendance(student_id, name)
                    if written:
                        marked_this_session.add(student_id)
                        print(f"[+] Attendance marked: {name} ({student_id})")
            else:
                display = f"Unknown ({confidence:.0f})"
                color = (0, 0, 255)

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, display, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        cv2.imshow("Attendance - press 'q' to quit", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cam.release()
    cv2.destroyAllWindows()
    print(f"[*] Session ended. {len(marked_this_session)} people marked present.")


if __name__ == "__main__":
    run()

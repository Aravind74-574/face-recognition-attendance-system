"""
Step 1: Register a person and capture face samples from the webcam.

Usage:
    python -m app.capture_faces --id 101 --name "Jane Doe"

Captures config.SAMPLES_PER_PERSON grayscale face crops and stores them in
dataset/<id>_<name>/img_XX.jpg. Press 'q' at any time to stop early.
"""
import argparse
import os
import sys

import cv2

from app import config
from app.attendance_store import add_student


def sanitize(name: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in name).strip("_")


def capture_faces(student_id: str, name: str, samples: int = config.SAMPLES_PER_PERSON):
    try:
        add_student(student_id, name)
    except ValueError as e:
        print(f"[!] {e}")
        proceed = input("Continue capturing more samples for this existing ID anyway? [y/N]: ")
        if proceed.strip().lower() != "y":
            sys.exit(1)

    person_dir = os.path.join(config.DATASET_DIR, f"{student_id}_{sanitize(name)}")
    os.makedirs(person_dir, exist_ok=True)

    detector = cv2.CascadeClassifier(config.CASCADE_PATH)
    cam = cv2.VideoCapture(config.CAMERA_INDEX)
    if not cam.isOpened():
        print("[!] Could not open webcam. Check CAMERA_INDEX in app/config.py.")
        sys.exit(1)

    print(f"[*] Capturing {samples} face samples for '{name}' (ID: {student_id}).")
    print("[*] Look at the camera. Press 'q' to stop early.")

    count = 0
    while count < samples:
        ok, frame = cam.read()
        if not ok:
            print("[!] Failed to read frame from camera.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(80, 80))

        for (x, y, w, h) in faces:
            count += 1
            face_crop = cv2.resize(gray[y:y + h, x:x + w], config.FACE_SIZE)
            filepath = os.path.join(person_dir, f"img_{count:03d}.jpg")
            cv2.imwrite(filepath, face_crop)

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"Samples: {count}/{samples}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            break  # only take one face per frame

        cv2.imshow("Registering Face - press 'q' to quit early", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        if count >= samples:
            break

    cam.release()
    cv2.destroyAllWindows()
    print(f"[+] Done. Captured {count} samples into '{person_dir}'.")
    print("[*] Next step: python -m app.train_model")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Register a person and capture face samples.")
    parser.add_argument("--id", required=True, help="Unique ID for this person (e.g. roll number/employee id)")
    parser.add_argument("--name", required=True, help="Full name of the person")
    parser.add_argument("--samples", type=int, default=config.SAMPLES_PER_PERSON, help="Number of face samples to capture")
    args = parser.parse_args()

    capture_faces(args.id, args.name, args.samples)

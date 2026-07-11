"""
Step 2: Train the LBPH face recognizer on everything in dataset/.

Usage:
    python -m app.train_model
"""
import json
import os
import sys

import cv2
import numpy as np

from app import config


def train():
    if not os.path.isdir(config.DATASET_DIR) or not os.listdir(config.DATASET_DIR):
        print("[!] No data found in dataset/. Run app.capture_faces first for at least one person.")
        sys.exit(1)

    recognizer = cv2.face.LBPHFaceRecognizer_create()

    faces = []
    labels = []
    label_to_name = {}   # numeric label -> "id_name" folder, used to recover id/name later
    next_label = 0
    id_name_to_label = {}

    for person_dir in sorted(os.listdir(config.DATASET_DIR)):
        full_dir = os.path.join(config.DATASET_DIR, person_dir)
        if not os.path.isdir(full_dir):
            continue

        if person_dir not in id_name_to_label:
            id_name_to_label[person_dir] = next_label
            label_to_name[next_label] = person_dir
            next_label += 1
        label = id_name_to_label[person_dir]

        for filename in os.listdir(full_dir):
            if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            path = os.path.join(full_dir, filename)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, config.FACE_SIZE)
            faces.append(img)
            labels.append(label)

    if not faces:
        print("[!] No valid face images found to train on.")
        sys.exit(1)

    print(f"[*] Training on {len(faces)} images across {len(label_to_name)} people...")
    recognizer.train(faces, np.array(labels))
    recognizer.save(config.TRAINER_FILE)

    with open(config.LABELS_FILE, "w") as f:
        json.dump(label_to_name, f, indent=2)

    print(f"[+] Model saved to {config.TRAINER_FILE}")
    print(f"[+] Label map saved to {config.LABELS_FILE}")
    print("[*] Next step: python -m app.recognize_attendance")


if __name__ == "__main__":
    train()

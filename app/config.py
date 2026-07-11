"""
Central configuration for the Face Recognition Attendance System.
Edit values here rather than hunting through every script.
"""
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATASET_DIR = os.path.join(BASE_DIR, "dataset")        # captured face images
TRAINER_DIR = os.path.join(BASE_DIR, "trainer")         # trained model + label map
ATTENDANCE_DIR = os.path.join(BASE_DIR, "attendance")   # daily CSV logs
STUDENTS_CSV = os.path.join(BASE_DIR, "students.csv")   # registry of enrolled people

TRAINER_FILE = os.path.join(TRAINER_DIR, "trainer.yml")
LABELS_FILE = os.path.join(TRAINER_DIR, "labels.json")

# Haar cascade shipped with opencv-python
# Haar cascade — using a local copy since the pip package's bundled
# copy isn't reliably found on some OpenCV versions/platforms.
CASCADE_PATH = os.path.join(BASE_DIR, "haarcascade_frontalface_default.xml")# Camera device index (0 = default webcam). Change to 1/2 if you have multiple cameras.
CAMERA_INDEX = 0

# Number of face samples to capture per person during registration
SAMPLES_PER_PERSON = 60

# Faces are resized to this size before training/recognition
FACE_SIZE = (200, 200)

# LBPH prediction returns a "confidence" that is actually a DISTANCE:
# lower = more confident match. Anything above this is treated as "Unknown".
CONFIDENCE_THRESHOLD = 70

# Minimum seconds between two attendance marks for the same person on the
# same day (prevents spamming the CSV while they stand in front of the camera)
MIN_SECONDS_BETWEEN_DUPLICATE_MARKS = 5

for d in (DATASET_DIR, TRAINER_DIR, ATTENDANCE_DIR):
    os.makedirs(d, exist_ok=True)

"""
Small helpers shared across scripts: managing the student registry and
writing/reading attendance CSV files.
"""
import csv
import os
from datetime import datetime

from app import config


def ensure_students_csv():
    if not os.path.exists(config.STUDENTS_CSV):
        with open(config.STUDENTS_CSV, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "name"])


def add_student(student_id: str, name: str):
    ensure_students_csv()
    existing = load_students()
    if student_id in existing:
        raise ValueError(f"ID '{student_id}' is already registered to '{existing[student_id]}'.")
    with open(config.STUDENTS_CSV, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([student_id, name])


def load_students() -> dict:
    """Returns {id: name}"""
    ensure_students_csv()
    students = {}
    with open(config.STUDENTS_CSV, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            students[row["id"]] = row["name"]
    return students


def _today_csv_path(date: datetime = None) -> str:
    date = date or datetime.now()
    filename = f"Attendance_{date.strftime('%Y-%m-%d')}.csv"
    return os.path.join(config.ATTENDANCE_DIR, filename)


def mark_attendance(student_id: str, name: str) -> bool:
    """
    Appends an attendance row for today if this person hasn't already been
    marked within MIN_SECONDS_BETWEEN_DUPLICATE_MARKS. Returns True if a new
    row was written, False if it was a duplicate/skip.
    """
    path = _today_csv_path()
    now = datetime.now()

    rows = []
    if os.path.exists(path):
        with open(path, "r", newline="") as f:
            rows = list(csv.DictReader(f))

    for row in reversed(rows):
        if row["id"] == student_id:
            last_time = datetime.strptime(row["time"], "%H:%M:%S")
            last_dt = now.replace(hour=last_time.hour, minute=last_time.minute, second=last_time.second, microsecond=0)
            if (now - last_dt).total_seconds() < config.MIN_SECONDS_BETWEEN_DUPLICATE_MARKS:
                return False
            break  # already marked once today; still fine to allow re-marks (e.g. leave time), just not spammy ones

    write_header = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["id", "name", "date", "time"])
        writer.writerow([student_id, name, now.strftime("%Y-%m-%d"), now.strftime("%H:%M:%S")])
    return True


def already_marked_today(student_id: str) -> bool:
    path = _today_csv_path()
    if not os.path.exists(path):
        return False
    with open(path, "r", newline="") as f:
        for row in csv.DictReader(f):
            if row["id"] == student_id:
                return True
    return False

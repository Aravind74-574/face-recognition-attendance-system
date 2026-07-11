"""
View attendance records.

Usage:
    python -m app.view_attendance                 # show today's attendance
    python -m app.view_attendance --date 2026-07-11
    python -m app.view_attendance --all            # list every date on file
"""
import argparse
import csv
import os
from datetime import datetime

from app import config


def show(date_str: str):
    path = os.path.join(config.ATTENDANCE_DIR, f"Attendance_{date_str}.csv")
    if not os.path.exists(path):
        print(f"[!] No attendance file found for {date_str}.")
        return
    with open(path, "r", newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        print(f"No entries for {date_str}.")
        return

    print(f"\nAttendance for {date_str}  ({len(rows)} entries)")
    print("-" * 50)
    print(f"{'ID':<10}{'Name':<20}{'Time':<10}")
    print("-" * 50)
    for row in rows:
        print(f"{row['id']:<10}{row['name']:<20}{row['time']:<10}")


def list_all_dates():
    if not os.path.isdir(config.ATTENDANCE_DIR):
        print("No attendance records yet.")
        return
    files = sorted(f for f in os.listdir(config.ATTENDANCE_DIR) if f.startswith("Attendance_"))
    if not files:
        print("No attendance records yet.")
        return
    for f in files:
        date_str = f.replace("Attendance_", "").replace(".csv", "")
        print(date_str)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="View attendance records.")
    parser.add_argument("--date", help="Date in YYYY-MM-DD format (default: today)")
    parser.add_argument("--all", action="store_true", help="List all dates with records")
    args = parser.parse_args()

    if args.all:
        list_all_dates()
    else:
        date_str = args.date or datetime.now().strftime("%Y-%m-%d")
        show(date_str)

"""
Simple desktop GUI that ties the whole pipeline together.

Usage:
    python -m app.gui
"""
import subprocess
import sys
import threading
import tkinter as tk
from tkinter import messagebox, ttk

from app import config
from app.attendance_store import load_students
from app.view_attendance import show
import io
import contextlib
from datetime import datetime


class AttendanceApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Face Recognition Attendance System")
        self.geometry("480x420")
        self.resizable(False, False)

        title = tk.Label(self, text="Face Recognition Attendance System",
                          font=("Helvetica", 16, "bold"))
        title.pack(pady=15)

        # --- Registration frame ---
        reg_frame = ttk.LabelFrame(self, text="1. Register a new person")
        reg_frame.pack(fill="x", padx=20, pady=10)

        ttk.Label(reg_frame, text="ID:").grid(row=0, column=0, padx=5, pady=5, sticky="e")
        self.id_entry = ttk.Entry(reg_frame, width=15)
        self.id_entry.grid(row=0, column=1, padx=5, pady=5)

        ttk.Label(reg_frame, text="Name:").grid(row=0, column=2, padx=5, pady=5, sticky="e")
        self.name_entry = ttk.Entry(reg_frame, width=15)
        self.name_entry.grid(row=0, column=3, padx=5, pady=5)

        ttk.Button(reg_frame, text="Capture Faces", command=self.capture_faces).grid(
            row=0, column=4, padx=10, pady=5)

        # --- Actions frame ---
        action_frame = ttk.LabelFrame(self, text="2. Train & Run")
        action_frame.pack(fill="x", padx=20, pady=10)

        ttk.Button(action_frame, text="Train Model", command=self.train_model).pack(
            side="left", expand=True, fill="x", padx=10, pady=10)
        ttk.Button(action_frame, text="Start Attendance", command=self.start_recognition).pack(
            side="left", expand=True, fill="x", padx=10, pady=10)

        # --- Attendance view frame ---
        view_frame = ttk.LabelFrame(self, text="3. Today's Attendance")
        view_frame.pack(fill="both", expand=True, padx=20, pady=10)

        self.text_area = tk.Text(view_frame, height=10, width=50)
        self.text_area.pack(padx=10, pady=10, fill="both", expand=True)

        ttk.Button(self, text="Refresh Attendance", command=self.refresh_attendance).pack(pady=5)

        self.refresh_attendance()

    def capture_faces(self):
        student_id = self.id_entry.get().strip()
        name = self.name_entry.get().strip()
        if not student_id or not name:
            messagebox.showwarning("Missing info", "Please enter both ID and Name.")
            return

        def run():
            subprocess.run([sys.executable, "-m", "app.capture_faces",
                             "--id", student_id, "--name", name])
            messagebox.showinfo("Done", f"Finished capturing samples for {name}.")

        threading.Thread(target=run, daemon=True).start()

    def train_model(self):
        def run():
            result = subprocess.run([sys.executable, "-m", "app.train_model"],
                                     capture_output=True, text=True)
            messagebox.showinfo("Training", result.stdout or result.stderr)

        threading.Thread(target=run, daemon=True).start()

    def start_recognition(self):
        def run():
            subprocess.run([sys.executable, "-m", "app.recognize_attendance"])
            self.refresh_attendance()

        threading.Thread(target=run, daemon=True).start()

    def refresh_attendance(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            show(datetime.now().strftime("%Y-%m-%d"))
        self.text_area.delete("1.0", tk.END)
        self.text_area.insert(tk.END, buf.getvalue() or "No attendance marked yet today.")


if __name__ == "__main__":
    app = AttendanceApp()
    app.mainloop()

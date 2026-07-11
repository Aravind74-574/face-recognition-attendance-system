"""
Live browser demo of the face recognition pipeline.

Runs entirely in the visitor's browser tab via WebRTC (no data leaves their
session, nothing is saved to disk/server). Visitors can:
  1. Register their own face (a few quick snapshots, right there in-browser)
  2. Instantly train a tiny per-session model
  3. See themselves recognized live, with attendance logged for the demo session

This is a self-contained showcase of the same detect -> train -> recognize
pipeline as the desktop app, adapted to run for anonymous visitors with no
persistent storage.
"""
import os
import time
import urllib.request
import uuid
from datetime import datetime

import av
import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoProcessorBase

# ---------------------------------------------------------------------------
# Setup: cascade file (downloaded once, cached on the server instance)
# ---------------------------------------------------------------------------
CASCADE_PATH = os.path.join(os.path.dirname(__file__), "haarcascade_frontalface_default.xml")
CASCADE_URL = (
    "https://raw.githubusercontent.com/opencv/opencv/master/data/"
    "haarcascades/haarcascade_frontalface_default.xml"
)

if not os.path.exists(CASCADE_PATH):
    urllib.request.urlretrieve(CASCADE_URL, CASCADE_PATH)

FACE_SIZE = (200, 200)
SAMPLES_NEEDED = 20
CONFIDENCE_THRESHOLD = 70

st.set_page_config(page_title="Face Recognition Attendance — Live Demo", page_icon="🎥")
st.title("🎥 Face Recognition Attendance — Live Demo")
st.caption(
    "Runs live in your browser via WebRTC. Nothing is saved after your session ends. "
    "This is a lightweight showcase of the full desktop pipeline — "
    "see the [GitHub repo](https://github.com/Aravind74-574/face-recognition-attendance-system) "
    "for the complete version."
)

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
# Streamlit re-runs this whole script top-to-bottom on every interaction, so a
# plain module-level variable would be wiped out before the background WebRTC
# thread ever reads it. st.cache_resource gives us an object that survives
# reruns. We key it per-visitor (a random id stashed in session_state) so
# different visitors' trained models never collide with each other.
if "uid" not in st.session_state:
    st.session_state.uid = str(uuid.uuid4())


@st.cache_resource
def _get_recognizer_store():
    return {}


_recognizer_store = _get_recognizer_store()  # {uid: trained cv2 recognizer}

if "samples" not in st.session_state:
    st.session_state.samples = []  # captured face crops (numpy arrays)
if "recognizer" not in st.session_state:
    st.session_state.recognizer = None

mode = st.sidebar.radio("Mode", ["1. Register your face", "2. Live recognition"])

detector = cv2.CascadeClassifier(CASCADE_PATH)


class RegisterProcessor(VideoProcessorBase):
    """Detects a face each frame and stashes crops into session_state when asked."""

    def __init__(self):
        self.latest_face = None

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(80, 80))

        for (x, y, w, h) in faces:
            self.latest_face = cv2.resize(gray[y:y + h, x:x + w], FACE_SIZE)
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            break

        return av.VideoFrame.from_ndarray(img, format="bgr24")


class RecognizeProcessor(VideoProcessorBase):
    """
    Runs live recognition against the visitor's trained model.

    IMPORTANT: both __init__ (via the factory) and recv() run on a background
    thread managed by streamlit-webrtc, where st.session_state and other
    st.* calls are not safe. This class only ever touches plain Python
    objects (the recognizer passed in, and its own instance attributes).
    """

    def __init__(self, recognizer=None):
        self.recognizer = recognizer
        self.log = []
        self.last_marked = 0.0

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(80, 80))

        for (x, y, w, h) in faces:
            face_crop = cv2.resize(gray[y:y + h, x:x + w], FACE_SIZE)
            if self.recognizer is not None:
                label, confidence = self.recognizer.predict(face_crop)
                if confidence <= CONFIDENCE_THRESHOLD:
                    text, color = f"You ({confidence:.0f})", (0, 200, 0)
                    now = time.time()
                    if now - self.last_marked > 5:
                        self.log.append(datetime.now().strftime("%H:%M:%S"))
                        self.last_marked = now
                else:
                    text, color = f"Unknown ({confidence:.0f})", (0, 0, 255)
            else:
                text, color = "No model trained yet", (0, 165, 255)

            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")


RTC_CONFIGURATION = {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}

# ---------------------------------------------------------------------------
# Mode 1: Register
# ---------------------------------------------------------------------------
if mode == "1. Register your face":
    st.subheader("Step 1 — Capture a few samples of your face")
    st.write(
        f"Allow camera access below, then click **Capture sample** a few times "
        f"({SAMPLES_NEEDED} recommended) while moving your head slightly."
    )

    ctx = webrtc_streamer(
        key="register",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_processor_factory=RegisterProcessor,
        media_stream_constraints={"video": True, "audio": False},
    )

    col1, col2 = st.columns(2)
    with col1:
        if st.button("📸 Capture sample", disabled=not ctx.state.playing):
            if ctx.video_processor and ctx.video_processor.latest_face is not None:
                st.session_state.samples.append(ctx.video_processor.latest_face)
                st.success(f"Captured {len(st.session_state.samples)}/{SAMPLES_NEEDED}")
            else:
                st.warning("No face detected yet — center your face in frame and try again.")
    with col2:
        if st.button("🗑️ Reset samples"):
            st.session_state.samples = []
            st.session_state.recognizer = None
            _recognizer_store.pop(st.session_state.uid, None)
            st.rerun()

    st.progress(min(len(st.session_state.samples) / SAMPLES_NEEDED, 1.0))

    if len(st.session_state.samples) >= SAMPLES_NEEDED:
        st.success("Enough samples captured!")
        if st.button("🧠 Train model now"):
            recognizer = cv2.face.LBPHFaceRecognizer_create()
            labels = np.zeros(len(st.session_state.samples), dtype=np.int32)
            recognizer.train(st.session_state.samples, labels)
            st.session_state.recognizer = recognizer
            _recognizer_store[st.session_state.uid] = recognizer
            st.success("Model trained! Switch to '2. Live recognition' in the sidebar.")

# ---------------------------------------------------------------------------
# Mode 2: Recognize
# ---------------------------------------------------------------------------
else:
    st.subheader("Step 2 — Live recognition")
    if st.session_state.recognizer is None:
        st.warning("Train a model first in '1. Register your face'.")
    else:
        st.write("You should see a green box with **You** once recognized.")

        # Captured here (main thread) as a plain value — safe for the
        # factory to use even though streamlit-webrtc invokes it on a
        # background thread, since it never touches st.* itself.
        my_recognizer = _recognizer_store.get(st.session_state.uid)

        def _make_recognize_processor():
            return RecognizeProcessor(recognizer=my_recognizer)

        ctx = webrtc_streamer(
            key="recognize",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTC_CONFIGURATION,
            video_processor_factory=_make_recognize_processor,
            media_stream_constraints={"video": True, "audio": False},
        )

        st.subheader("Session attendance log")
        if st.button("🔄 Refresh log"):
            st.rerun()

        if ctx.video_processor and ctx.video_processor.log:
            for t in ctx.video_processor.log:
                st.write(f"✅ Marked present at {t}")
        else:
            st.write("No attendance marked yet this session. Click 'Refresh log' after being recognized.")

st.divider()
st.caption(
    "This demo trains a temporary model in your browser session only — nothing is "
    "stored server-side and everything resets when you close the tab. The full "
    "desktop version (persistent multi-user database, CSV logs, GUI) is on GitHub."
)

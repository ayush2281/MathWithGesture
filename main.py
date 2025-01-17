import cvzone
import cv2
import mediapipe as mp
import numpy as np
from cvzone.HandTrackingModule import HandDetector
from PIL import Image
import streamlit as st
import google.generativeai as genai

# Streamlit setup
st.markdown("<h1 style='color:red;'>Virtual Math Calculator</h1>", unsafe_allow_html=True)
col1, col2 = st.columns([2, 1])

with col1:
    run = st.checkbox('Run', value=True)
    FRAME_WINDOW = st.image([])

with col2:
    st.title("Answer")
    st.markdown("", unsafe_allow_html=True)

# Persistent state variables
if "prev_pos" not in st.session_state:
    st.session_state.prev_pos = None
if "canvas" not in st.session_state:
    st.session_state.canvas = None
if "output_text" not in st.session_state:
    st.session_state.output_text = ""

# AI setup
genai.configure(api_key="YOUR_API_KEY")
model = genai.GenerativeModel("gemini-1.5-flash")

# Webcam setup
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)
detector = HandDetector(detectionCon=0.5, minTrackCon=0.5)

# Function for gesture detection
def getHandInfo(img):
    hands, img = detector.findHands(img, draw=False)
    if hands:
        return detector.fingersUp(hands[0]), hands[0]["lmList"]
    return None

# Function for drawing
def draw(info):
    fingers, lmList = info
    if fingers == [0, 1, 0, 0, 0]:  # Index finger up
        current_pos = tuple(lmList[8][0:2])
        if st.session_state.prev_pos:
            cv2.line(st.session_state.canvas, st.session_state.prev_pos, current_pos, (0, 0, 255), 10)
        st.session_state.prev_pos = current_pos
    elif fingers == [1, 0, 0, 0, 0]:  # Thumb up
        st.session_state.canvas = np.zeros_like(st.session_state.canvas)

# Main app logic
if run:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    if st.session_state.canvas is None:
        st.session_state.canvas = np.zeros_like(img)

    info = getHandInfo(img)
    if info:
        draw(info)

    combined_img = cv2.addWeighted(img, 0.7, st.session_state.canvas, 0.3, 0)
    FRAME_WINDOW.image(combined_img, channels="BGR")
else:
    cap.release()

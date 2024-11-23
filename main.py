import cvzone
import cv2
import mediapipe as mp
import numpy as np
from cvzone.HandTrackingModule import HandDetector
import google.generativeai as genai
from PIL import Image
import streamlit as st


# set the title
st.markdown("<h1 style='color:red;'>Virtual Math Calculator</h1>", unsafe_allow_html=True)




col1, col2 = st.columns([2,1])

with col1:
    run = st.checkbox('Run', value = True)
    FRAME_WINDOW = st.image([])

with col2:
    output_text_area = st.title("Answer")
    output_text_area = st.markdown("", unsafe_allow_html=True)  # Use markdown to allow HTML for color
genai.configure(api_key="AIzaSyDwsoIen-GpXYPgQr0LAnM4aSNfBDVp1ZI" )
model = genai.GenerativeModel("gemini-1.5-flash")



# Initialize the webcam to capture video
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

# Initialize the HandDetector class with the given parameters
detector = HandDetector(staticMode=False, maxHands=2, modelComplexity=1, detectionCon=0.5, minTrackCon=0.5)

#
def getHandInfo(img):
    hands, img = detector.findHands(img, draw=False, flipType=True)
    if hands:
        hand = hands[0]
        lmList = hand["lmList"]
        fingers = detector.fingersUp(hand)
        print(fingers)
        return fingers, lmList
    else:
        return None


# drawing part
def draw(info, prev_pos, canvas):
    fingers, lmList = info
    current_pos = None
    if fingers == [0,1,0,0,0]:
        current_pos = tuple(lmList[8][0:2])  # Convert to tuple
        if prev_pos is None:
            prev_pos = current_pos
        else:
            # Smooth the line by averaging the positions
            current_pos = ((current_pos[0] + prev_pos[0]) // 2, (current_pos[1] + prev_pos[1]) // 2)  # Smoothing
        cv2.line(canvas, current_pos, prev_pos, color=(0,0,255), thickness=10)
      # restart the drawing
    elif fingers == [1,0,0,0,0]:
        canvas = np.zeros_like(img)




    return current_pos, canvas



def sendToAI(model, canvas, fingers):
    if fingers == [0,1,1,1,1]:
        pil_image = Image.fromarray(canvas)
        response = model.generate_content(["Solve this Math Problem", pil_image])
        # response = model.generate_content("Which is the largest ocean")
        return response.text

prev_pos = None
canvas = None
image_combined = None
output_text = ""

while True:
    success, img = cap.read()
    img = cv2.flip(img, flipCode=1)
    if canvas is None:
        canvas = np.zeros_like(img)

    info = getHandInfo(img)
    if info:
        fingers, lmlist = info
        # print(fingers)
        prev_pos, canvas = draw(info, prev_pos, canvas)
        output_text = sendToAI(model,canvas, fingers)
    image_combined = cv2.addWeighted(img, 0.7, canvas, 0.3, 0)
    FRAME_WINDOW.image(image_combined, channels='BGR')  # Increased the size of the camera feed

    if output_text:
      output_text_area.text(output_text)

    # Display only the combined image
    # cv2.imshow("Image Combined", image_combined)  # Display only combined image
    # cv2.imshow("Canvas", canvas)
    # cv2.imshow("Image Combined", image_combined)


    cv2.waitKey(1)

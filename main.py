import cvzone  # Importing cvzone for hand tracking and gesture recognition
import cv2  # Importing OpenCV for video capture and image processing
import mediapipe as mp  # Importing MediaPipe for hand detection
import numpy as np  # Importing NumPy for array operations
from cvzone.HandTrackingModule import HandDetector  # Importing HandDetector from cvzone
import google.generativeai as genai  # Importing Google's Generative AI for content generation
from PIL import Image  # Importing PIL for image processing
import streamlit as st  # Importing Streamlit for creating the web app

# Set the title of the Streamlit app with red color
st.markdown("<h1 style='color:red;'>Virtual Math Calculator</h1>", unsafe_allow_html=True)

col1, col2 = st.columns([2, 1])  # Creating two columns with a 2:1 ratio

with col1:
    run = st.checkbox('Run', value=True)  # Checkbox to start/stop the webcam feed
    FRAME_WINDOW = st.image([])  # Placeholder for displaying the webcam feed

with col2:
    output_text_area = st.title("Answer")  # Title for the answer section
    output_text_area = st.markdown("", unsafe_allow_html=True)  # Placeholder for displaying the answer

genai.configure(api_key="AIzaSyDwsoIen-GpXYPgQr0LAnM4aSNfBDVp1ZI")  # Configuring the Generative AI with API key
model = genai.GenerativeModel("gemini-1.5-flash")  # Initializing the Generative AI model

# Initialize the webcam to capture video
cap = cv2.VideoCapture(0)
cap.set(3, 1280)  # Setting the width of the video capture
cap.set(4, 720)  # Setting the height of the video capture

# Initialize the HandDetector class with the given parameters
detector = HandDetector(staticMode=False, maxHands=2, modelComplexity=1, detectionCon=0.5, minTrackCon=0.5)

def getHandInfo(img):
    hands, img = detector.findHands(img, draw=False, flipType=True)  # Detecting hands in the image
    if hands:
        hand = hands[0]
        lmList = hand["lmList"]  # Getting the list of landmarks
        fingers = detector.fingersUp(hand)  # Getting the status of fingers (up/down)
        print(fingers)
        return fingers, lmList
    else:
        return None

def draw(info, prev_pos, canvas):
    fingers, lmList = info
    current_pos = None
    if fingers == [0, 1, 0, 0, 0]:  # If index finger is up
        current_pos = tuple(lmList[8][0:2])  # Get the position of the index finger
        if prev_pos is None:
            prev_pos = current_pos
        else:
            # Smooth the line by averaging the positions
            current_pos = ((current_pos[0] + prev_pos[0]) // 2, (current_pos[1] + prev_pos[1]) // 2)
        cv2.line(canvas, current_pos, prev_pos, color=(0, 0, 255), thickness=10)  # Draw a line on the canvas
    elif fingers == [1, 0, 0, 0, 0]:  # If thumb is up
        canvas = np.zeros_like(img)  # Clear the canvas
    return current_pos, canvas

def sendToAI(model, canvas, fingers):
    if fingers == [0, 1, 1, 1, 1]:  # If only thumb is down and four fingers are up
        pil_image = Image.fromarray(canvas)  # Convert the canvas to a PIL image
        response = model.generate_content(["Solve this Math Problem", pil_image])  # Send the image to the AI model
        return response.text

prev_pos = None
canvas = None
image_combined = None
output_text = ""

while run:
    success, img = cap.read()  # Capture a frame from the webcam
    img = cv2.flip(img, flipCode=1)  # Flip the image horizontally
    if canvas is None:
        canvas = np.zeros_like(img)  # Initialize the canvas

    info = getHandInfo(img)  # Get hand information from the image
    if info:
        fingers, lmList = info
        prev_pos, canvas = draw(info, prev_pos, canvas)  # Draw on the canvas based on hand gestures
        output_text = sendToAI(model, canvas, fingers)  # Send the canvas to the AI model for solving
    image_combined = cv2.addWeighted(img, 0.7, canvas, 0.3, 0)  # Combine the original image and the canvas
    FRAME_WINDOW.image(image_combined, channels='BGR')  # Display the combined image in the Streamlit app

    if output_text:
        output_text_area.markdown(f"<span style='color:blue;'>{output_text}</span>", unsafe_allow_html=True)  # Display the answer

    cv2.waitKey(1)  # Wait for 1 millisecond

cap.release()  # Release the webcam
cv2.destroyAllWindows()  # Close all OpenCV windows

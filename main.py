import cvzone
import cv2
import mediapipe as mp
import numpy as np
from cvzone.HandTrackingModule import HandDetector
from flask import Flask, request, jsonify
from PIL import Image
import google.generativeai as genai

# AI setup
genai.configure(api_key="AIzaSyDwsoIen-GpXYPgQr0LAnM4aSNfBDVp1ZI")  # Replace with your API key
model = genai.GenerativeModel("gemini-1.5-flash")

# Flask setup
app = Flask(__name__)

# Initialize hand detector
detector = HandDetector(detectionCon=0.5, minTrackCon=0.5)

# Persistent state variables
prev_pos = None
canvas = np.zeros((720, 1280, 3), dtype=np.uint8)


# Function for gesture detection
def getHandInfo(img):
    hands, img = detector.findHands(img, draw=False)
    if hands:
        return detector.fingersUp(hands[0]), hands[0]["lmList"]
    return None


# Function for drawing
def draw(info):
    global prev_pos, canvas
    fingers, lmList = info
    if fingers == [0, 1, 0, 0, 0]:  # Index finger up
        current_pos = tuple(lmList[8][0:2])
        if prev_pos:
            cv2.line(canvas, prev_pos, current_pos, (0, 0, 255), 10)
        prev_pos = current_pos
    elif fingers == [1, 0, 0, 0, 0]:  # Thumb up
        canvas = np.zeros_like(canvas)


# Function to get AI-generated answer from model
def generate_answer(input_text):
    response = model.generate(text=input_text)
    return response.result


@app.route('/predict', methods=['POST'])
def predict():
    global prev_pos, canvas
    # Get the image data from the request
    file = request.files['image']
    img = Image.open(file)
    img = np.array(img)

    # Process the image with hand gestures
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    info = getHandInfo(img)
    if info:
        draw(info)

    combined_img = cv2.addWeighted(img, 0.7, canvas, 0.3, 0)

    # Convert the image back to a format that can be returned in the response
    _, img_encoded = cv2.imencode('.png', combined_img)
    img_bytes = img_encoded.tobytes()

    # Generate AI-based answer from the model
    input_text = "Generated from hand gesture drawing"  # Or modify based on your use case
    generated_text = generate_answer(input_text)

    # Return the processed image and AI answer as response
    return jsonify({
        'prediction': 'Image processed successfully',
        'generated_text': generated_text
    }), 200


if __name__ == '__main__':
    app.run(debug=True)

import cv2
import mediapipe as mp
import math

from mediapipe.tasks.python.vision import GestureRecognizer, GestureRecognizerOptions
from mediapipe.tasks.python import BaseOptions

import webbrowser
import time
import pyautogui

# Initialize Mediapipe Hands
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands



options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path="gesture_recognizer.task"),
    num_hands=1
)
gesture_recognizer = GestureRecognizer.create_from_options(options)

# Define a function to calculate the distance between two points
def calculate_distance(point1, point2):
    return math.hypot(point2[0]-point1[0], point2[1]-point1[1])

def recognize_thumb_left(hand_landmarks):
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    thumb_ip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP]
    wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
    index_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]

    thumb_extended = thumb_tip.x < thumb_ip.x  # tip is more left than IP joint  
    fingers_curled = all(
        hand_landmarks.landmark[finger_tip].y > hand_landmarks.landmark[finger_pip].y
        for finger_tip, finger_pip in [
            (mp_hands.HandLandmark.INDEX_FINGER_TIP, mp_hands.HandLandmark.INDEX_FINGER_PIP),
            (mp_hands.HandLandmark.MIDDLE_FINGER_TIP, mp_hands.HandLandmark.MIDDLE_FINGER_PIP),
            (mp_hands.HandLandmark.RING_FINGER_TIP, mp_hands.HandLandmark.RING_FINGER_PIP),
            (mp_hands.HandLandmark.PINKY_TIP, mp_hands.HandLandmark.PINKY_PIP)
        ]
    )

    pointing_left = thumb_tip.x < wrist.x and thumb_tip.x < index_mcp.x - 0.05

    if thumb_extended and fingers_curled and pointing_left:
        return "Thumb_Left"
    return None


def recognize_thumb_right(hand_landmarks):
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    thumb_ip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP]
    wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
    index_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]

    thumb_extended = thumb_tip.x > thumb_ip.x  # tip more right than IP joint  
    fingers_curled = all(
        hand_landmarks.landmark[finger_tip].y > hand_landmarks.landmark[finger_pip].y
        for finger_tip, finger_pip in [
            (mp_hands.HandLandmark.INDEX_FINGER_TIP, mp_hands.HandLandmark.INDEX_FINGER_PIP),
            (mp_hands.HandLandmark.MIDDLE_FINGER_TIP, mp_hands.HandLandmark.MIDDLE_FINGER_PIP),
            (mp_hands.HandLandmark.RING_FINGER_TIP, mp_hands.HandLandmark.RING_FINGER_PIP),
            (mp_hands.HandLandmark.PINKY_TIP, mp_hands.HandLandmark.PINKY_PIP)
        ]
    )

    pointing_right = thumb_tip.x > wrist.x and thumb_tip.x > index_mcp.x + 0.05

    if thumb_extended and fingers_curled and pointing_right:
        return "Thumb_Right"
    return None

def main():
    # Initialize video capture
    cap = cv2.VideoCapture(0)  # 0 is the default webcam

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as hands:

        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue

            # Flip the image horizontally and convert the BGR image to RGB.
            image = cv2.flip(image, 1)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # To improve performance, optionally mark the image as not writeable to pass by reference.
            image_rgb.flags.writeable = False
            results = hands.process(image_rgb)

            # Draw the hand annotations on the image.
            image_rgb.flags.writeable = True
            image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:

                    # Try to recognize custom gestures first
                    gesture = recognize_thumb_left(hand_landmarks)
                    if gesture:
                        pyautogui.press("a") # Move Left
                    else: # If first gesture isn't recognized, try the next one. 
                        gesture = recognize_thumb_right(hand_landmarks)
                        if gesture: 
                            pyautogui.press("d") # Move Right
                        else:
                            # If none of the custome gestures are recognized, check for canned gestures.
                            # We need to do a little bit of extra processing in order to extract the canned gestures.
                            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
                            result = gesture_recognizer.recognize(mp_image)
                            if result.gestures:
                                gesture = result.gestures[0][0].category_name
                                if gesture == "Open_Palm":
                                    webbrowser.open('https://freepacman.org/#google_vignette', new=2) # Changing endpoint to our game
                                    time.sleep(1)
                                    pyautogui.write("Game is Running!", interval=0.25)
                                    # Make sure to allow for time between recognized gestures so only one window is opened
                                    time.sleep(5)
                                if gesture == "Pointing_Up": # Move Up
                                    pyautogui.press("w")
                                if gesture == "Closed_Fist": # Move Down
                                    pyautogui.press("s")
                    
                    # Display gesture near hand location
                    cv2.putText(image, gesture, 
                                (int(hand_landmarks.landmark[0].x * image.shape[1]), 
                                 int(hand_landmarks.landmark[0].y * image.shape[0]) - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

            # Display the resulting image
            cv2.imshow('Gesture Recognition', image)

            if cv2.waitKey(5) & 0xFF == 27:
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
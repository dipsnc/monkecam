import cv2
import mediapipe as mp
from deepface import DeepFace
import numpy as np

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Load monkey images
thinking_monkey = cv2.imread("monk/thinking.jpg")
idea_monkey = cv2.imread("monk/idea.jpg")
surprised_monkey = cv2.imread("monk/surprise.jpg")
neutral_monkey = cv2.imread("monk/neutral.jpg")
phone_monkey = cv2.imread("monk/phone.jpg")

monkey_dict = {
    ("one_finger", "neutral"): thinking_monkey,
    ("one_finger", "happy"): idea_monkey,
    ("other", "surprise"): surprised_monkey,
    ("no_hand", "neutral"): neutral_monkey, 
    ("other", "neutral"): phone_monkey, 
}

# Hand gesture helper
def get_hand_state(landmarks):
    tips = [8, 12, 16, 20]
    pip = [6, 10, 14, 18]
    open_fingers = 0
    open_list = []

    for tip, joint in zip(tips, pip):
        if landmarks.landmark[tip].y < landmarks.landmark[joint].y:
            open_fingers += 1
            open_list.append(tip)

    if open_fingers == 1 and 8 in open_list:
        return "one_finger"
    else:
        return "other"

# Webcam
webcam = cv2.VideoCapture(0)

valid_emotions = ["happy", "neutral", "surprise"]

with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    current_emotion = "neutral"
    current_hand = "no_hand"

    while True:
        ret, frame = webcam.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        # detect hand gesture
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                current_hand = get_hand_state(hand_landmarks)
        else:
            current_hand = "no_hand"

        # detect emotion (limited)
        try:
            analysis = DeepFace.analyze(rgb, actions=["emotion"], enforce_detection=False, detector_backend="opencv")
            result = analysis[0] if isinstance(analysis, list) else analysis
            emotions = result["emotion"]
            top_emotion = max(emotions, key=emotions.get)
            if top_emotion in valid_emotions:
                current_emotion = top_emotion
            else:
                current_emotion = "neutral"
        except Exception as e:
            print("emotion error:", e)
            current_emotion = "neutral"

        # select matching monkey
        key = (current_hand, current_emotion)
        monkey = monkey_dict.get(key, neutral_monkey)

        # resize and combine side-by-side
        h, w, _ = frame.shape
        mh, mw, _ = monkey.shape
        scale = h / mh
        new_w = int(mw * scale)
        monkey_resized = cv2.resize(monkey, (new_w, h))

        combined = np.zeros((h, w + new_w, 3), dtype=np.uint8)
        combined[:, :w] = frame
        combined[:, w:w + new_w] = monkey_resized

        # label info
        cv2.putText(combined, f"monke", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # show one window only
        cv2.imshow("Cam", combined)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

webcam.release()
cv2.destroyAllWindows()

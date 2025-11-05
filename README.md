# 🐒 Monkey Mood Detector

A playful computer vision project that reacts to your face and hand gestures — just like a mirror, but with monkeys.

---

## 🎥 What It Does
- Detects your **hand gestures** using [MediaPipe Hands](https://developers.google.com/mediapipe/solutions/vision/hand_landmarker)
- Analyzes your **facial emotion** (happy, neutral, surprise, etc.) using [DeepFace](https://github.com/serengil/deepface)
- Displays different **monkey images** on screen depending on your expression and gesture combo.

### 🧩 Monkey Reactions Table

| Hand Gesture | Emotion   | Monkey Shown |
|---------------|-----------|---------------|
| One finger    | Neutral   | 🤔 Thinking Monkey |
| One finger    | Surprise  | 💡 Idea Monkey |
| Fist          | Surprise  | 😲 Surprised Monkey |
| No hand       | Neutral   | 😐 Neutral Monkey |

---

## 🧠 Tech Stack
- **Python 3.9+**
- **OpenCV**
- **MediaPipe**
- **DeepFace**
- **NumPy**

---

## 💬 Credits
Made purely for fun and curiosity 🐵  
Inspired by a similar face-reaction project I came across on Instagram ✨  
Built with ❤️ using **OpenCV**, **MediaPipe**, and **DeepFace**


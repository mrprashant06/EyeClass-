import cv2
import numpy as np
from tensorflow.keras.models import load_model
import datetime

print("="*50)
print("EyeClass - Real-Time Student Focus & Emotion")
print("="*50)

# ----- 1. Load model -----
print("\n[1/3] Loading emotion model...")
model_path = "models/emotion_model.h5"
try:
    model = load_model(model_path)
    print(f"      Model loaded: {model_path}")
except:
    print(f"      ERROR: Model not found!")
    exit()

# Correct order matching your training folders
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# ----- 2. Face detector -----
print("[2/3] Initializing face detector...")
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# ----- 3. Start webcam -----
print("[3/3] Starting webcam...")
print("\nControls: Q = quit | S = screenshot\n" + "="*50 + "\n")

cap = cv2.VideoCapture(0)

def get_focus_status(emotion, confidence):
    """Simple focus estimation."""
    if emotion in ['Happy', 'Neutral', 'Surprise'] and confidence > 0.5:
        return "FOCUSED", (0, 255, 0)
    elif emotion in ['Sad', 'Fear', 'Disgust']:
        return "DISTRACTED", (0, 165, 255)
    else:
        return "UNENGAGED", (0, 0, 255)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(100, 100))

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (48, 48))
        roi = roi_gray.astype('float32') / 255.0
        roi = np.expand_dims(roi, axis=(0, -1))

        preds = model.predict(roi, verbose=0)[0]
        emotion_idx = np.argmax(preds)
        emotion = emotion_labels[emotion_idx]
        confidence = np.max(preds)

        focus_text, color = get_focus_status(emotion, confidence)

        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, f"{emotion} ({confidence:.2f})", (x, y-30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.putText(frame, focus_text, (x, y-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(frame, timestamp, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    cv2.putText(frame, "Q:Quit | S:Screenshot", (10, frame.shape[0]-10), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

    cv2.imshow('EyeClass - Student Focus & Emotion', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        ss_name = f"screenshot_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        cv2.imwrite(ss_name, frame)
        print(f"Screenshot saved: {ss_name}")

cap.release()
cv2.destroyAllWindows()
print("\nEyeClass closed.")
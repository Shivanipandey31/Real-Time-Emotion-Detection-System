from flask import Flask, render_template, request, redirect, url_for, Response
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os

app = Flask(__name__)

# Load trained model (with check)
MODEL_PATH = 'models/emotion_model.keras'

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")
else:
    print(f"✅ Loading model from {MODEL_PATH}")

model = load_model(MODEL_PATH)
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# OpenCV face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def predict_emotion_from_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    predicted_emotion = "Unknown"
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (48, 48))
        roi = roi_gray.astype('float32') / 255.0
        roi = np.expand_dims(roi, axis=0)
        roi = np.expand_dims(roi, axis=-1)

        prediction = model.predict(roi, verbose=0)
        predicted_emotion = emotion_labels[np.argmax(prediction)]

        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(image, predicted_emotion, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

        break  # Only detect the first face for simplicity

    return image, predicted_emotion


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files.get('image')
    if not file:
        return redirect(url_for('index'))

    if not file.filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        return "Invalid file format. Please upload a JPG or PNG image."

    img_path = os.path.join('static', 'uploaded_image.jpg')
    file.save(img_path)

    image = cv2.imread(img_path)
    output_img, emotion = predict_emotion_from_image(image)

    result_path = os.path.join('static', 'result.jpg')
    cv2.imwrite(result_path, output_img)

    print(f"[INFO] Predicted Emotion: '{emotion}'")  # Debug print

    return render_template('result.html', user_image='result.jpg', emotion=emotion)


def gen_frames():
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            frame = predict_emotion_from_image(frame)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/webcam')
def webcam():
    return render_template('webcam.html')

@app.route('/video')
def video():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))

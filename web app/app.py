from flask import Flask,render_template, request, session,Response
import os
import cv2
import shutil
from flask import jsonify
import numpy as np
from keras.models import model_from_json
app = Flask(__name__,static_folder='staticFiles')
app.secret_key='asdfg'
UPLOAD_FOLDER = os.path.join('staticFiles', 'uploads')
app.config['UPLOAD_FOLDER']=UPLOAD_FOLDER
camera=cv2.VideoCapture("staticFiles\\uploads\\video.mp4")
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

# load json and create model
json_file = open('.\model\emotion_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
emotion_model = model_from_json(loaded_model_json)

# load weights into new model
emotion_model.load_weights(".\model\emotion_model.h5")
print("Loaded model from disk")
def generate_frames():
    while True:
        ret,frame=camera.read()
        frame = cv2.resize(frame, (1280, 720))
        if not ret:
            break
        else:
            face_detector = cv2.CascadeClassifier('.\\haarcascades\\haarcascade_frontalface_default.xml')
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # detect faces available on camera
            num_faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

            # take each face available on the camera and Preprocess it
            for (x, y, w, h) in num_faces:
                cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (0, 255, 0), 4)
                roi_gray_frame = gray_frame[y:y + h, x:x + w]
                cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)

                # predict the emotions
                emotion_prediction = emotion_model.predict(cropped_img)
                maxindex = int(np.argmax(emotion_prediction))
                cv2.putText(frame, emotion_dict[maxindex], (x+5, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
            
                ret,buffer=cv2.imencode('.jpg',frame)
                frame=buffer.tobytes()

        yield(b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            
@app.route("/")
def index():
    return render_template("index.html")
@app.route('/cam')
def web():
    return render_template('video.html')
@app.route("/up")
def ss():
    return render_template("upload.html")
@app.route('/video')
def video():
    return Response(generate_frames(),mimetype='multipart/x-mixed-replace; boundary=frame')
@app.route("/resule")
def result():
    return render_template('result.html')

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['video']
    if file: 
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], 'video.mp4'))
        return jsonify(success=True)
    return jsonify(success=False)
if __name__ == "__main__":  
   app.run(debug=True)
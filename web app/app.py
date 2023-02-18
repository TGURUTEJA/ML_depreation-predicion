from flask import Flask,render_template, request, session,Response
import os
import cv2
import shutil
from flask import jsonify
import numpy as np
from keras.models import model_from_json
import threading
import numpy as np
from keras.models import model_from_json
from moviepy.editor import *
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from time import perf_counter



app = Flask(__name__,static_folder='staticFiles')
app.secret_key='asdfg'
UPLOAD_FOLDER = os.path.join('staticFiles', 'uploads')
app.config['UPLOAD_FOLDER']=UPLOAD_FOLDER



#<--------------------------------------------------------Video--------------------------------------------------------------------------->

emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

# load json and create model
json_file = open('S:\\depreation predicion\\ML_depreation-predicion\\web app\\staticFiles\\emotion_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
emotion_model = model_from_json(loaded_model_json)

# load weights into new model
emotion_model.load_weights("S:\\depreation predicion\\ML_depreation-predicion\\web app\\staticFiles\\emotion_model.h5")
print("Loaded model from disk")

def generate_frames():
    camera=cv2.VideoCapture('staticFiles\\uploads\\video.mp4')
    clips=[]
    pre = { "Angry":0, "Disgusted":0, "Fearful":0,"Happy":0, "Neutral":0, "Sad":0,"Surprised":0}
    total=1
    while True:
        success,frame=camera.read()
        t1_start = perf_counter()
        if not success:
            break
        else:
            face_detector = cv2.CascadeClassifier('S:\\depreation predicion\\ML_depreation-predicion\\web app\\staticFiles\\haarcascade_frontalface_default.xml')
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # detect faces available on camera
            num_faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)
            for (x, y, w, h) in num_faces:
                cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (0, 255, 0), 4)
                roi_gray_frame = gray_frame[y:y + h, x:x + w]
                cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)

                # predict the emotions
                emotion_prediction = emotion_model.predict(cropped_img)
                maxindex = int(np.argmax(emotion_prediction))
                cv2.putText(frame, emotion_dict[maxindex], (x+5, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                total+=1
                pre[emotion_dict[maxindex]]+=1
            t1_stop = perf_counter()
            clips.append(ImageClip(np.array(frame)).set_duration((t1_stop-t1_start)))
    d_pre=(pre["Angry"]*20+pre["Disgusted"]*10+pre["Fearful"]*30+pre["Sad"]*40)/total
    print(total)
    video_clip=concatenate_videoclips(clips,method='compose')
    video_clip.write_videofile('staticFiles\\uploads\\video_output.mp4',fps=24)
    video_clip.close()
    return d_pre*10


#<--------------------------------------------------------audio----------------------------------------------------------------------------->



import requests
import time
API_KEY_ASSEMBLYAI="e1bceda9131140eaabc1bc63f2c26ad9"
upload_endpoint = 'https://api.assemblyai.com/v2/upload'
transcript_endpoint = 'https://api.assemblyai.com/v2/transcript'
headers_auth_only = {'authorization': API_KEY_ASSEMBLYAI}

headers = {
    "authorization": API_KEY_ASSEMBLYAI,
    "content-type": "application/json"
}

CHUNK_SIZE = 5_242_880  # 5MB


def upload(filename):
    def read_file(filename):
        with open(filename, 'rb') as f:
            while True:
                data = f.read(CHUNK_SIZE)
                if not data:
                    break
                yield data

    upload_response = requests.post(upload_endpoint, headers=headers_auth_only, data=read_file(filename))
    return upload_response.json()['upload_url']


def transcribe(audio_url):
    transcript_request = {
        'audio_url': audio_url
    }

    transcript_response = requests.post(transcript_endpoint, json=transcript_request, headers=headers)
    return transcript_response.json()['id']

        
def poll(transcript_id):
    polling_endpoint = transcript_endpoint + '/' + transcript_id
    polling_response = requests.get(polling_endpoint, headers=headers)
    return polling_response.json()


def get_transcription_result_url(url):
    transcribe_id = transcribe(url)
    while True:
        data = poll(transcribe_id)
        if data['status'] == 'completed':
            return data, None
        elif data['status'] == 'error':
            return data, data['error']
            
        print("waiting for 30 seconds")
        time.sleep(30)
        
        
def save_transcript(url, title):
    data, error = get_transcription_result_url(url)
    
    if data:
        return data['text'],None
    elif error:
        return data['text'],error
def start(filename):
    audio_url = upload(filename)
    return save_transcript(audio_url, 'file_title')



#<--------------------------------------------------------------WEB_SERVER----------------------------------------------------------------->
#<---HOME_ROOT--->
@app.route("/")
def index():
    return render_template("home.html")

#<---WEB_CAM_ROOT--->
@app.route('/cam')
def web():
    return render_template('video.html')

#<---VIDEO_UPLOADE_ROOT--->
@app.route("/up")
def ss():
    return render_template("upload.html")


#<---UPLOAD--->
@app.route('/upload', methods=['POST'])
def uploa():
    file = request.files['video']
    if file: 
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], 'video.mp4'))
        return jsonify(success=True)
    return jsonify(success=False)



#<---Result--->
@app.route("/resule")
def result():
    a=generate_frames()
    a=0
    error=None
    data,error=start('staticFiles\\uploads\\video.mp4')
    ana=SentimentIntensityAnalyzer()
    v=ana.polarity_scores(data)
    p_text=(70*v['neg']+30*v['neu'])
    p_video=a
    p=(p_text+p_video)/2
    if error!=None:
        data=error
    return render_template('result.html',r=data,per=p)

@app.route('/find')
def find():
    return render_template('find.html')






if __name__ == "__main__":  
   app.run(debug=True)
const videoButton = document.getElementById('main__video-button');
const video = document.getElementById('main__video');
const Data_Extracting=document.getElementById('Data_Extracting')
let mediaRecorder;
var data
const parts=[]
async function init() {

    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            audio: true,
            video: true
        });
        startWebcam(stream);
    } catch (err) {
        console.log('Error retrieving a media device.');
        console.log(err);
    }
}

function startWebcam(stream) {
    window.stream = stream;
    video.srcObject = stream;
}

videoButton.onclick = () => {
    let a=videoButton.textContent;
    console.log('Record');

    switch (videoButton.textContent) {
        
        case 'Record':
            startRecording();
            
            videoButton.textContent = 'Stop';
            break;
        case 'Stop':
            videoButton.textContent = 'Record';
            mediaRecorder.stop();
            break;
    }

}

function startRecording() {
    if (video.srcObject === null) {
        video.srcObject = window.stream;
    }
    mediaRecorder = new MediaRecorder(window.stream, {mimeType: 'video/webm;codecs=vp9,opus'});
    mediaRecorder.start();
    mediaRecorder.ondataavailable = recordVideo;
}

function recordVideo(event) {
    if (event.data && event.data.size > 0) {
        video.srcObject = null;
        let videoUrl = URL.createObjectURL(event.data);
        video.src = videoUrl;
        parts.push(event.data);
    }
}

function stopRecording() {
    mediaRecorder.stop();
}
function down(){
  const blob= new Blob(parts,{
    type : 'video/webm'
  })
  let formData = new FormData();
  formData.append('file','filenmae')
  formData.append('video', blob, "video.mp4");
 
  fetch('/upload', {
    method: 'POST',
    body: formData

});
}




init();
#Load Important Packages
import cv2
import os
import numpy as np
from tensorflow.keras.models import model_from_json

root_dir = os.getcwd()
# Load Face Detection Model
face_cascade = cv2.CascadeClassifier("models/haarcascade_frontalface_default.xml")
# Load Anti-Spoofing Model graph
json_file = open('models/antispoofing_model.json','r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load antispoofing model weights 
model.load_weights('models/antispoofing_model.h5')
print("Model loaded from disk")

video = cv2.VideoCapture(0)

while True:
    try:
        ret,frame = video.read()
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray,1.3,5)
        
        for (x,y,w,h) in faces:  
            face = frame[y-5:y+h+5,x-5:x+w+5]
            resized_face = cv2.resize(face,(160,160))
            resized_face = resized_face.astype("float") / 255.0
           
            # resized_face = img_to_array(resized_face)
            resized_face = np.expand_dims(resized_face, axis=0)
            
            # model to determine if the face is "real" or "fake"
            preds = model.predict(resized_face)[0]
            print(preds)
            
            if preds> 0.5:
                label = 'Fake Face'
                cv2.putText(frame, label, (x,y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
                cv2.rectangle(frame, (x, y), (x+w,y+h),
                    (0, 0, 255), 2)
            else:
                label = 'Real Face'
                cv2.putText(frame, label, (x,y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
                cv2.rectangle(frame, (x, y), (x+w,y+h),
                (0, 255, 0), 2)
        
        cv2.imshow('frame', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    except Exception as e:
        pass

video.release()        
cv2.destroyAllWindows()
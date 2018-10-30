import cv2
import numpy as np
import dlib
from scipy.spatial import distance as dist


PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"  
# Create the haar cascade  
predictor = dlib.shape_predictor(PREDICTOR_PATH)  
cap = cv2.VideoCapture(0)
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

while True:
    check, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        gray,
        1.1,
        5)
   #print(faces)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        faceCenter = (int((2*x+w)/2), int((2*y+h)/2))
        dlib_rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))

        landmarks = np.matrix([[p.x, p.y] for p in predictor(gray, dlib_rect).parts()])  
    
        # landmarks_display = landmarks[RIGHT_EYE_POINTS]
        A = dist.euclidean(landmarks[36],landmarks[30])
        # B = dist.euclidean(landmarks[45], landmarks[30])
        B = dist.euclidean(landmarks[8],landmarks[30])
        print(A)
        
        if(A<40):
             cv2.putText(frame, "Right", (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        elif (A>65):
             cv2.putText(frame, "Left", (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

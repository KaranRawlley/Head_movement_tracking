import cv2
import os
import pyautogui as pya


HAAR_CASCADE_PATH = "haarcascade_frontalface_alt.xml"
CAMERA_INDEX = 0
DISTURBANCE_TOLERANCE_HORI = 20.0  # Sensitivity
DISTURBANCE_TOLERANCE_UP = 15.0  # Sensitivity
DISTURBANCE_TOLERANCE_DOWN = -10.0
CAMERA_FPS = 20



def run():
    cap = cv2.VideoCapture(0)  
    faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
    originCenter = None
    faceArray = []
    faceCenter = None
    i = 0

    while True:
        check, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray,1.1,5)
        cv2.namedWindow('me1', cv2.WINDOW_NORMAL)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            faceCenter = (int((2*x+w)/2), int((2*y+h)/2))
            cv2.circle(frame, faceCenter, 7, (0, 0, 255),2)
            pya.moveTo(faceCenter)
            #print(pya.size())
            # print("mouse",pya.position())
            # print("center",faceCenter)
        cv2.resizeWindow('me1',600,600)
        cv2.imshow('me1', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
run()

# <<<<<<< HEAD

# =======
# >>>>>>> dc11f117d97a5ea19b4ab71beb613881784f6eba

import cv2 as cv
import numpy as np

# setup 
haar_cascade = cv.CascadeClassifier('haar_face.xml')
features = np.load('features.npy', allow_pickle=True)
lables = np.load('labels.npy')
face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('face_trained.yml')

people=['Boris Lee', 'Tan Liu', 'Thomas Peng']
def rescale (img, scale=0.75):

    width = int(img.shape[1] * scale)
    height = int(img.shape[0] * scale)

    dimensions = (width,height)

    return cv.resize(img, dimensions, interpolation=cv.INTER_AREA)
# catch the camera
cap = cv.VideoCapture(0)  # which camera

while True:
    ret, frame = cap.read(0) # ret: True/False
    if ret:
        
        #cv.imshow('frame', frame)    # show the video
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        faces_rectangle = haar_cascade.detectMultiScale(gray,1.1,15)

        for (x,y,w,h) in faces_rectangle:
            faces_roi = gray[y:y+h,x:x+w]
            label, confidence = face_recognizer.predict(faces_roi)
            cv.rectangle(frame, (x,y), (x+w,y+h), (0,255,0),thickness=2)

            cv.putText(frame, str(people[label]),(x,y-2), cv.FONT_HERSHEY_COMPLEX,
            0.9, (0,255,0), thickness = 2)
            cv.putText(frame, str(round(confidence,3)),(x+w,y+h), cv.FONT_HERSHEY_COMPLEX,
            0.8, (0,255,0), thickness = 2)
            
        cv.imshow('Detected Faces',frame)

        if cv.waitKey(1) & 0xFF == ord('q'):  # exit by press the Q buttom
            break
    else:
        print('CameraError: Can not capture camera')
        break

cap.release()

cv.destroyAllWindows()

import cv2
import os
import time

directory = r'C:\\Users\\Tan Liu\Desktop\\Visual Studio Code\\face_train\\Tan Liu'
org_pic = len(os.listdir(directory))
haar_cascade = cv2.CascadeClassifier('haar_face.xml')

# catch the camera
cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)  # which camera

def face_dectect (img,sensetive):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces_rectangle = haar_cascade.detectMultiScale(gray, 1.1, minNeighbors=sensetive)

    if len(faces_rectangle) == 1:
        a,b,c,d = 0,0,0,0
        for (x,y,w,h) in faces_rectangle:
            #print((x,y,x+w,y+h))
            cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0),thickness=2)
            a,b,c,d = x,y,w,h

        return img, True, a,b,c,d  

    else:
        cv2.putText(img, 'Saved Failed',(0,img.shape[0]//2)\
        , cv2.FONT_HERSHEY_COMPLEX,\
        3.0, (0,0,255), thickness = 2)
        #print('Too much Faces are Dectected')

        return img, False, 0,0,0,0

while True:
    ret, frame = cap.read(0) # ret: True/False
    ret, photo_frame = cap.read(0)

    if ret:
        cv2.imshow('frame', frame)    # show the video
        
        if cv2.waitKey(1) & 0xFF == ord('q'):  # exit by press the Q buttom
            break

        if cv2.waitKey(1) & 0xFF == ord('a'):  # photo by press the A buttom
            d = 2
            img, dectected, x,y,w,h = face_dectect(photo_frame,d)

            while dectected == False and d < 20:
                ret, photo_frame = cap.read(0)
                img, dectected, x,y,w,h = face_dectect(photo_frame,d)
                d += 2

            if dectected and x!=0 and y!=0 and w !=0 and h!=0:
                cv2.destroyAllWindows()
                os.chdir(directory)                          # change to direct root
                filename =  f'{int(time.time())}.jpg'        # use time as file name
                cropped = frame[y:y+h,x:x+w]                 # crop the faces
                cv2.imwrite(filename, cropped)               # cv2.imwrite to save the file 
                cv2.imshow('Successfully saved',photo_frame)
                time.sleep(0.3)

            else:
                cv2.destroyAllWindows()
                cv2.imshow('Saved Failed',photo_frame)
                time.sleep(0.1)

            pass
    else:

        print('CameraError: Can not capture camera')
        break

cap.release()

cv2.destroyAllWindows()

end_pic = len(os.listdir(directory))

print(f'Totally take {end_pic-org_pic} pictures')
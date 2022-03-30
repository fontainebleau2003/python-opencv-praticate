import cv2 as cv
import numpy as np

path = 'photos/Mr.Bean_04.jpg'  #路徑放這裡
pic=cv.imread(path)

gray = cv.cvtColor(pic, cv.COLOR_BGR2GRAY)

haar_cascade = cv.CascadeClassifier('haar_face.xml')

faces_rectangle = haar_cascade.detectMultiScale(gray,scaleFactor = 1.1, minNeighbors=5)

for (x,y,w,h) in faces_rectangle:
    #cv.rectangle(pic, (x,y), (x+w,y+h), (0,255,0),thickness=2)
    cropped = pic[y:y+h, x:x+w]

gray = cv.cvtColor(cropped, cv.COLOR_BGR2GRAY)

img = cropped
cv.imshow('img',img)
rows,cols,channels = img.shape
print(rows,cols,channels)


hsv = cv.cvtColor(img,cv.COLOR_BGR2HSV)
#cv.imshow('hsv',hsv)

# pink
lower = np.array([130,0,220])
upper = np.array([255,255,255])

mask = cv.inRange(hsv, lower, upper)
#cv.imshow('mask',mask)

erode = cv.erode(mask,None,iterations=1)
#cv.imshow('erode',erode)

dilate = cv.dilate(erode,None,iterations=1)
#cv.imshow('dilate',dilate)

for i in range(rows):
    for j in range(cols):
        if erode[i,j]==255: 
            img[i,j]=(255,255,255) # Replace the color by BGR passageway
 
#img = cv.resize(img,None,fx=2,fy=2)

cv.imshow('res',img)

cv.waitKey(0)

cv.destroyAllWindows()
import cv2 as cv

def rescale (img, scale=0.3):

    width = int(img.shape[1] * scale)
    height = int(img.shape[0] * scale)

    dimensions = (width,height)

    return cv.resize(img, dimensions, interpolation=cv.INTER_AREA)

#img = cv.imread('photos/Mr.Bean_04.jpg')
img = rescale(cv.imread(r'C:\\Users\\Tan Liu\Desktop\\Visual Studio Code\\face_train\\Tan Liu\\1647868339.jpg'))
#cv.imshow('Mr.Bean', img)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
#cv.imshow('Gray',gray)

haar_cascade = cv.CascadeClassifier('haar_face.xml')

faces_rectangle = haar_cascade.detectMultiScale(gray,scaleFactor = 1.1, minNeighbors=13)

print(f'Number of Face found = {len(faces_rectangle)}')


for (x,y,w,h) in faces_rectangle:
    print((x,y,x+w,y+h))
    cv.rectangle(img, (x,y), (x+w,y+h), (0,255,0),thickness=2)

cv.imshow('Detected Faces',img)

cv.waitKey(0)

cv.destroyAllWindows()
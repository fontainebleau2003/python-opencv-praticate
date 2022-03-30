import os
import cv2 as cv
import numpy as np

people_name = []
DIR = 'face_train'

for i in os.listdir(DIR):
    people_name.append(i)

print(people_name)

haar_cascade = cv.CascadeClassifier('haar_face.xml')

features = []
labels = []

def create_train():
    for person in people_name:
        path = os.path.join(DIR, person)
        label = people_name.index(person)

        for img in os.listdir((path)):
            img_path = os.path.join(path,img)

            img_array = cv.imread(img_path)
            gray = cv.cvtColor(img_array, cv.COLOR_BGR2GRAY)

            faces_rect = haar_cascade.detectMultiScale\
            (gray,scaleFactor = 1.1, minNeighbors=7)

            for (x,y,w,h) in faces_rect:
                faces_reg = gray[y:y+h,x:x+w]
                features.append(faces_reg)
                labels.append(label)

create_train()
print(f'Length of the features = {len(features)}')
print(f'Length of the labels = {len(labels)}')

features = np.array(features, dtype='object')
labels = np.array(labels)

# Train regonizer
face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.train(features,labels)

face_recognizer.save('face_trained.yml')
np.save('features.npy', features)
np.save('labels.npy', labels)

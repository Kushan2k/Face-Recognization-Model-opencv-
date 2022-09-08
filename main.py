
import numpy as np
import os
import cv2 as cv

ROOT_DIR='./faces'



peopls=os.listdir(ROOT_DIR)

feature_images=[]
labels=[]

recognizer=cv.CascadeClassifier('face.xml')
MODEL=cv.face.LBPHFaceRecognizer_create()



def create_train():
  for person in peopls:

    path=os.path.join(ROOT_DIR,person)
    label=peopls.index(person)

    for img in os.listdir(path):
      img_path=os.path.join(path,img)

      img_file=cv.imread(img_path)
      imgGray=cv.cvtColor(img_file,cv.COLOR_BGR2GRAY)

      face_rect=recognizer.detectMultiScale(imgGray,scaleFactor=1.1,minNeighbors=4)

      for (x,y,w,h) in face_rect:
        face=imgGray[y:y+h,x:x+w]

        feature_images.append(face)
        labels.append(label)
        


create_train()

feature_images=np.array(feature_images,dtype='object')
labels=np.array(labels)

MODEL.train(feature_images,labels)
MODEL.save('face_model.yml')

np.save('features.npy',feature_images)
np.save('labels.npy',labels)
